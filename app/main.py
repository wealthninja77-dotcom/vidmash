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

def load_existing_jobs() -> None:
    """Scan OUTPUT_DIR and load all completed reports into the in-memory jobs cache."""
    settings = get_settings()
    out_dir = Path(settings.OUTPUT_DIR)
    if not out_dir.exists():
        return
    for item in out_dir.iterdir():
        if item.is_dir():
            report_json = item / "report.json"
            if report_json.exists():
                try:
                    with open(report_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    meta = data.get("metadata", {})
                    job_id = item.name
                    pdf_file = item / "report.pdf"
                    html_file = item / "report.html"
                    jobs[job_id] = ProcessingStatus(
                        job_id=job_id,
                        video_url=f"https://www.youtube.com/watch?v={meta.get('video_id', '')}" if meta.get('video_id') else "",
                        video_title=meta.get("title", "Research Dossier"),
                        stage=ProcessingStage.COMPLETED,
                        progress=100,
                        message="Processing complete",
                        pdf_path=str(pdf_file) if pdf_file.exists() else "",
                        html_path=str(html_file) if html_file.exists() else "",
                    )
                except Exception as exc:
                    logger.debug("Could not index existing job %s: %s", item.name, exc)


def _resolve_job(identifier: str) -> tuple[str, Path] | None:
    """Resolve identifier (either job_id or video_id) to (real_job_id, job_dir)."""
    settings = get_settings()
    out_dir = Path(settings.OUTPUT_DIR)

    # 1. Direct match on directory name (job_id)
    direct = out_dir / identifier
    if direct.is_dir() and (direct / "report.json").exists():
        return identifier, direct

    # 2. Check in-memory jobs
    if identifier in jobs:
        d = out_dir / identifier
        if d.is_dir():
            return identifier, d

    # 3. Match against video_id in all report.json files
    if out_dir.exists():
        for d in out_dir.iterdir():
            if d.is_dir() and (d / "report.json").exists():
                try:
                    with open(d / "report.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("metadata", {}).get("video_id") == identifier:
                        return d.name, d
                except Exception:
                    continue
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hook — ensure output directory exists and index past dossiers."""
    settings = get_settings()
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    load_existing_jobs()
    logger.info("VidMash started — loaded %d historical dossiers", len(jobs))
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
    """Serve the generated inline HTML report with dynamic template rendering fallback."""
    resolved = _resolve_job(job_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Dossier not found")

    real_job_id, job_dir = resolved
    html_file = job_dir / "report.html"
    report_json = job_dir / "report.json"

    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))

    if report_json.exists():
        try:
            with open(report_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            from app.models.schemas import DocumentReport
            rep = DocumentReport.model_validate(data)
            return templates.TemplateResponse(
                "report.html",
                {
                    "request": request,
                    "report": rep,
                    "job_id": real_job_id,
                    "now": datetime.now(),
                },
            )
        except Exception as exc:
            logger.error("Failed to dynamically render report: %s", exc)

    raise HTTPException(status_code=404, detail="Report file not found")


@app.get("/report/{job_id}/{filename}")
async def report_asset(job_id: str, filename: str) -> FileResponse:
    """Serve visual assets (PNG charts/diagrams) referenced by the HTML report."""
    resolved = _resolve_job(job_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Job not found")

    real_job_id, job_dir = resolved
    clean_filename = Path(filename).name
    file_path = job_dir / clean_filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Asset {clean_filename} not found")
    return FileResponse(path=str(file_path))


@app.get("/api/library")
async def get_library() -> JSONResponse:
    """Return all completed research dossiers in the local workspace archive."""
    settings = get_settings()
    out_dir = Path(settings.OUTPUT_DIR)
    items = []

    if out_dir.exists():
        subdirs = sorted(
            [d for d in out_dir.iterdir() if d.is_dir() and (d / "report.json").exists()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        for d in subdirs:
            try:
                with open(d / "report.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                sec_count = len(data.get("sections", []))
                vis_count = len(data.get("visuals", []))
                takeaway_count = len(data.get("key_takeaways", []))
                dur = meta.get("duration", 0)
                mins = dur // 60
                secs = dur % 60
                dur_fmt = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
                read_mins = max(1, round(dur / 180))

                items.append({
                    "job_id": d.name,
                    "video_id": meta.get("video_id", ""),
                    "title": meta.get("title", "Untitled Research Dossier"),
                    "channel": meta.get("channel", "Unknown Channel"),
                    "duration": dur,
                    "duration_formatted": dur_fmt,
                    "read_time": f"{read_mins} min read",
                    "thumbnail_url": meta.get("thumbnail_url", ""),
                    "upload_date": meta.get("upload_date", ""),
                    "sections_count": sec_count,
                    "figures_count": vis_count,
                    "takeaways_count": takeaway_count,
                    "compression_ratio": data.get("compression_ratio", "85% compression"),
                    "pdf_available": (d / "report.pdf").exists(),
                    "created_at": datetime.fromtimestamp(d.stat().st_mtime).strftime("%b %d, %Y"),
                })
            except Exception:
                continue

    return JSONResponse(content={"dossiers": items, "count": len(items)})


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
    """Download the generated publication-quality PDF report."""
    resolved = _resolve_job(job_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Dossier not found")

    real_job_id, job_dir = resolved
    pdf_path = job_dir / "report.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"vidmash_whitepaper_{real_job_id}.pdf",
    )


@app.get("/api/download/html/{job_id}")
async def download_html(job_id: str) -> FileResponse:
    """Download the generated HTML report."""
    resolved = _resolve_job(job_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Dossier not found")

    real_job_id, job_dir = resolved
    html_path = job_dir / "report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML file not found")

    return FileResponse(
        path=str(html_path),
        media_type="text/html",
        filename=f"vidmash_dossier_{real_job_id}.html",
    )


from pydantic import BaseModel as PyBaseModel


class ChatQuery(PyBaseModel):
    message: str


@app.post("/api/chat/{job_id}")
async def chat_with_video(job_id: str, payload: ChatQuery) -> JSONResponse:
    """Live interactive AI Copilot Q&A grounded in the video's intelligence dossier."""
    resolved = _resolve_job(job_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Job not found or report not ready")

    real_job_id, output_dir = resolved
    report_json_path = output_dir / "report.json"

    query = payload.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not report_json_path.exists():
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found or report not ready")
        return JSONResponse(
            content={
                "answer": "This document is still being compiled by the Cortex Engine.",
                "section": "Processing",
                "timestamp": "00:00",
                "seconds": 0,
            }
        )

    try:
        with open(report_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read report data: {exc}")

    meta = data.get("metadata", {})
    title = meta.get("title", "the video")
    sections = data.get("sections", [])
    summary = data.get("executive_summary", "")

    import re
    words = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
    scored_sections = []
    for s in sections:
        content = (s.get("title", "") + " " + s.get("content", "")).lower()
        score = sum(1 for w in words if w in content)
        scored_sections.append((score, s))

    scored_sections.sort(key=lambda x: x[0], reverse=True)
    best_sec = scored_sections[0][1] if scored_sections else {}
    best_title = best_sec.get("title", "Key Discussion")
    start_t = int(best_sec.get("timestamp_start", 0))
    start_m, start_s = divmod(start_t, 60)
    time_str = f"{start_m:02d}:{start_s:02d}"

    from app.services.ai_service import AIService
    ai = AIService()
    answer = ""

    if not ai.is_local_mode:
        context_text = f"Title: {title}\nSummary: {summary[:500]}\n\n"
        for _, sec in scored_sections[:2]:
            context_text += f"Section: {sec.get('title')} [{sec.get('timestamp_start', 0)}s]:\n{sec.get('content')[:1200]}\n\n"

        prompt = (
            f"You are VidMash Intelligence Copilot. Answer the question about '{title}' "
            f"concisely and authoritatively using ONLY the context provided. "
            f"Always include the timestamp [{time_str}] where the topic is addressed.\n\n"
            f"CONTEXT:\n{context_text}\n\nQUESTION: {query}"
        )
        try:
            answer = await ai.generate(prompt)
        except Exception:
            answer = ""

    if not answer:
        sentences = [st.strip() for st in best_sec.get("content", "").split(".") if len(st.strip()) > 25]
        matching_sentences = [st for st in sentences if any(w in st.lower() for w in words)]
        if matching_sentences:
            core_quote = ". ".join(matching_sentences[:2]) + "."
        elif sentences:
            core_quote = ". ".join(sentences[:2]) + "."
        else:
            core_quote = summary[:200]

        answer = (
            f"According to **{title}** at **[{time_str}]** in section *'{best_title}'*:\n\n"
            f"> {core_quote}\n\n"
            f"This addresses your inquiry directly without requiring you to watch the video."
        )

    return JSONResponse(
        content={
            "answer": answer,
            "section": best_title,
            "timestamp": time_str,
            "seconds": start_t,
        }
    )


@app.get("/api/download/markdown/{job_id}")
async def download_markdown(job_id: str) -> FileResponse:
    """Download the generated Markdown Research Dossier."""
    resolved = _resolve_job(job_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Dossier not found")

    real_job_id, job_dir = resolved
    file_path = job_dir / "report.md"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Markdown report not found")
    return FileResponse(
        path=str(file_path),
        media_type="text/markdown",
        filename=f"vidmash_dossier_{real_job_id}.md",
    )


@app.get("/api/download/json/{job_id}")
async def download_json(job_id: str) -> FileResponse:
    """Download the raw structured JSON knowledge dataset."""
    resolved = _resolve_job(job_id)
    if not resolved:
        raise HTTPException(status_code=404, detail="Dossier not found")

    real_job_id, job_dir = resolved
    file_path = job_dir / "report.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="JSON report not found")
    return FileResponse(
        path=str(file_path),
        media_type="application/json",
        filename=f"vidmash_data_{real_job_id}.json",
    )
