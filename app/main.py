"""
VidMash — FastAPI Application Entry Point.

Provides the web interface and REST API for submitting YouTube videos,
tracking processing progress, and downloading generated documents.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.models.schemas import (
    DocumentReport,
    ProcessingStage,
    ProcessingStatus,
    VideoInput,
)

logger = logging.getLogger("vidmash.app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
)

# Stage display labels
STAGE_LABELS: dict[ProcessingStage, str] = {
    ProcessingStage.QUEUED: "Queued",
    ProcessingStage.EXTRACTING: "Extracting Video",
    ProcessingStage.SEGMENTING: "Segmenting Content",
    ProcessingStage.ANALYZING: "Analyzing Density",
    ProcessingStage.BUILDING_GRAPH: "Building Knowledge Graph",
    ProcessingStage.GENERATING_VISUALS: "Generating Visuals",
    ProcessingStage.WRITING_NARRATIVE: "Writing Narrative",
    ProcessingStage.RENDERING: "Rendering Document",
    ProcessingStage.COMPLETED: "Completed",
    ProcessingStage.FAILED: "Failed",
}

# Ordered list of active processing stages (for the UI progress tracker)
PROCESSING_STAGES_ORDER: list[ProcessingStage] = [
    ProcessingStage.EXTRACTING,
    ProcessingStage.SEGMENTING,
    ProcessingStage.ANALYZING,
    ProcessingStage.BUILDING_GRAPH,
    ProcessingStage.GENERATING_VISUALS,
    ProcessingStage.WRITING_NARRATIVE,
    ProcessingStage.RENDERING,
]

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
jobs: dict[str, ProcessingStatus] = {}

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hook — ensure output directory exists."""
    settings = get_settings()
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    logger.info("VidMash started — output dir: %s", settings.OUTPUT_DIR)
    yield
    logger.info("VidMash shutting down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VidMash",
    description="Video-to-Document Analysis Tool — Watch Less. Know More.",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files
BASE_DIR = Path(__file__).resolve().parent
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

# Jinja2 templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ---------------------------------------------------------------------------
# Background processing task
# ---------------------------------------------------------------------------

async def process_video(job_id: str, url: str) -> None:
    """Run the full VidMash pipeline as a background task.

    Updates the global *jobs* dict at every stage so the front-end can
    poll for progress.
    """
    from app.core.pipeline import VidMashPipeline

    status = jobs[job_id]
    settings = get_settings()
    output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    async def _progress(
        stage: ProcessingStage,
        progress: float,
        message: str,
    ) -> None:
        status.stage = stage
        status.progress = int(min(progress, 100))
        status.message = message

    def _on_metadata(meta) -> None:
        status.video_title = meta.title

    try:
        pipeline = VidMashPipeline()
        pdf_path, html_path = await pipeline.run(
            url=url,
            output_dir=output_dir,
            progress_callback=_progress,
            on_metadata=_on_metadata,
        )
        status.pdf_path = pdf_path
        status.html_path = html_path
        status.stage = ProcessingStage.COMPLETED
        status.progress = 100
        status.message = "Processing complete!"
        logger.info("Job %s completed successfully", job_id)

    except Exception as exc:
        logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        status.stage = ProcessingStage.FAILED
        status.error = str(exc)
        status.message = f"Processing failed: {exc}"


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request) -> HTMLResponse:
    """Render the VidMash landing page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/processing/{job_id}", response_class=HTMLResponse)
async def processing_page(request: Request, job_id: str) -> HTMLResponse:
    """Render the real-time processing status page."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    status = jobs[job_id]
    return templates.TemplateResponse(
        "processing.html",
        {
            "request": request,
            "job_id": job_id,
            "url": status.video_url,
            "video_title": "Processing video…",
            "stages": [
                {
                    "key": s.value,
                    "label": STAGE_LABELS[s],
                }
                for s in PROCESSING_STAGES_ORDER
            ],
        },
    )


@app.get("/report/{job_id}", response_class=HTMLResponse)
async def report_page(request: Request, job_id: str) -> HTMLResponse:
    """Serve the generated inline HTML report."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    status = jobs[job_id]
    if status.stage != ProcessingStage.COMPLETED or not status.html_path:
        raise HTTPException(status_code=400, detail="Report not ready yet")

    html_path = Path(status.html_path)
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML report file not found")

    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/report/{job_id}/{filename}")
async def report_asset(job_id: str, filename: str) -> FileResponse:
    """Serve visual assets (PNG charts/diagrams) referenced by the HTML report."""
    settings = get_settings()
    file_path = Path(settings.OUTPUT_DIR) / job_id / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path=str(file_path))


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.post("/api/process")
async def start_processing(
    payload: VideoInput,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Accept a video URL and start background processing.

    Returns the ``job_id`` and a redirect URL for the processing page.
    """
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        video_url=url,
        stage=ProcessingStage.QUEUED,
        progress=0,
        message="Job created — waiting to start…",
    )

    background_tasks.add_task(process_video, job_id, url)
    logger.info("Job %s created for URL: %s", job_id, url)

    return JSONResponse(
        content={"job_id": job_id, "redirect": f"/processing/{job_id}"}
    )


@app.get("/api/status/{job_id}")
async def get_status(job_id: str) -> JSONResponse:
    """Return the current processing status for a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    status = jobs[job_id]
    return JSONResponse(
        content={
            "job_id": status.job_id,
            "url": status.video_url,
            "video_title": status.video_title,
            "stage": status.stage.value,
            "progress": status.progress,
            "message": status.message,
            "error": status.error,
            "pdf_path": status.pdf_path,
            "html_path": status.html_path,
        }
    )


@app.get("/api/download/pdf/{job_id}")
async def download_pdf(job_id: str) -> FileResponse:
    """Download the generated PDF report."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    status = jobs[job_id]
    if not status.pdf_path or not Path(status.pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF file not found")

    return FileResponse(
        path=status.pdf_path,
        media_type="application/pdf",
        filename=f"vidmash_report_{job_id}.pdf",
    )


@app.get("/api/download/html/{job_id}")
async def download_html(job_id: str) -> FileResponse:
    """Download the generated HTML report."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    status = jobs[job_id]
    if not status.html_path or not Path(status.html_path).exists():
        raise HTTPException(status_code=404, detail="HTML file not found")

    return FileResponse(
        path=status.html_path,
        media_type="text/html",
        filename=f"vidmash_report_{job_id}.html",
    )
