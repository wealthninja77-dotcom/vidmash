"""
Layer 1: Extraction Cortex
Downloads video metadata and transcript using multiple adaptive fallback tiers:
1. Fast YouTube transcript API
2. yt-dlp subtitle download
3. Direct audio stream download + local neural transcription via faster-whisper.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine, Optional

from app.config import get_settings
from app.models.schemas import TranscriptChunk, VideoMetadata
from app.services.transcript_service import TranscriptService

logger = logging.getLogger(__name__)

# Shared thread pool for blocking yt-dlp operations
_executor = ThreadPoolExecutor(max_workers=4)


class ExtractionCortex:
    """Extracts video metadata and transcript from a given URL."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._transcript_service = TranscriptService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(
        self,
        url: str,
        progress_callback: Optional[Callable[[float, str], Coroutine[Any, Any, None]]] = None,
    ) -> tuple[VideoMetadata, list[TranscriptChunk]]:
        """Run the full extraction pipeline for a video URL.

        Args:
            url: The video URL to process.
            progress_callback: Optional async callback receiving (progress_pct, message).

        Returns:
            A tuple of (VideoMetadata, list[TranscriptChunk]).
        """
        logger.info("Starting extraction for URL: %s", url)

        async def _notify(pct: float, msg: str) -> None:
            if progress_callback:
                try:
                    await progress_callback(pct, msg)
                except Exception as cb_err:
                    logger.debug("Extraction progress callback error: %s", cb_err)

        video_id = self._extract_video_id(url)
        await _notify(5.0, "Connecting to video source & fetching metadata...")

        # 1. Fetch metadata first with fast timeout & fallback
        try:
            metadata = await asyncio.wait_for(self._get_metadata(url), timeout=12.0)
        except Exception as meta_err:
            logger.warning("Metadata fetch timed out or failed (%s). Using fallback metadata.", meta_err)
            metadata = VideoMetadata(
                video_id=video_id,
                title="YouTube Video",
                channel="YouTube Creator",
                duration=0,
                description="",
                thumbnail_url=f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg" if video_id else "",
            )

        max_duration = self._settings.MAX_VIDEO_DURATION
        if metadata.duration > max_duration:
            raise ValueError(
                f"Video duration ({metadata.duration}s) exceeds maximum allowed ({max_duration}s)."
            )

        short_title = metadata.title[:35] + ("..." if len(metadata.title) > 35 else "")
        await _notify(7.0, f"Identified '{short_title}' — locating transcript...")

        # 2. Fetch transcript with multi-tier fallback
        chunks = await self._get_transcript(video_id, url, _notify)

        logger.info(
            "Extraction complete: '%s' — %d chunks, %ds duration",
            metadata.title,
            len(chunks),
            metadata.duration,
        )
        await _notify(15.0, f"Extraction complete ({len(chunks)} speech chunks processed)")
        return metadata, chunks

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    async def _get_metadata(self, url: str) -> VideoMetadata:
        """Extract video metadata without downloading the video."""
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(_executor, self._ytdlp_extract_info, url)

        chapters: list[dict] = []
        raw_chapters = info.get("chapters") or []
        for ch in raw_chapters:
            chapters.append(
                {
                    "title": ch.get("title", ""),
                    "start_time": ch.get("start_time", 0),
                    "end_time": ch.get("end_time", 0),
                }
            )

        raw_date = info.get("upload_date") or ""
        upload_date = ""
        if raw_date and len(raw_date) == 8:
            upload_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

        video_id = info.get("id") or self._extract_video_id(url)

        return VideoMetadata(
            video_id=video_id,
            title=info.get("title") or "Untitled Video",
            channel=info.get("channel") or info.get("uploader") or "Unknown",
            duration=int(info.get("duration") or 0),
            description=info.get("description") or "",
            thumbnail_url=info.get("thumbnail") or "",
            upload_date=upload_date,
            chapters=chapters,
            tags=info.get("tags") or [],
        )

    @staticmethod
    def _ytdlp_extract_info(url: str) -> dict[str, Any]:
        """Run yt-dlp extract_info without downloading or format constraints."""
        import yt_dlp

        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
            "socket_timeout": 8,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False) or {}
        except Exception as exc:
            logger.warning("yt-dlp extract_info failed: %s. Trying oEmbed fallback...", exc)
            try:
                import httpx
                r = httpx.get(f"https://www.youtube.com/oembed?url={url}&format=json", timeout=6.0)
                if r.status_code == 200:
                    data = r.json()
                    return {
                        "id": ExtractionCortex._extract_video_id(url),
                        "title": data.get("title", "Untitled Video"),
                        "channel": data.get("author_name", "Unknown Channel"),
                        "thumbnail": data.get("thumbnail_url", ""),
                        "duration": 0,
                    }
            except Exception as oe_err:
                logger.warning("oEmbed fallback failed: %s", oe_err)
            raise RuntimeError(f"Failed to extract video info: {exc}") from exc

    # ------------------------------------------------------------------
    # Transcript extraction with 3-tier fallback
    # ------------------------------------------------------------------

    async def _get_transcript(
        self,
        video_id: str,
        url: str,
        notify_cb: Callable[[float, str], Coroutine[Any, Any, None]],
    ) -> list[TranscriptChunk]:
        """Retrieve transcript with automatic fallback from API to subtitles to audio whisper."""
        # Tier 1: youtube_transcript_api (Fastest: < 2 sec)
        try:
            raw_entries = await asyncio.wait_for(
                self._transcript_service.get_youtube_transcript(video_id),
                timeout=4.0,
            )
            if raw_entries:
                return [
                    TranscriptChunk(
                        text=c["text"],
                        start=c["start"],
                        end=c["start"] + c.get("duration", 2.0),
                    )
                    for c in raw_entries
                ]
        except Exception as exc:
            logger.info("youtube_transcript_api skipped/failed (%s), trying tier 2 subtitles...", exc)

        # Tier 2: yt-dlp subtitle download
        await notify_cb(8.5, "Searching video subtitle tracks...")
        try:
            raw_subs = await asyncio.wait_for(
                self._transcript_service.get_subtitles_yt_dlp(url),
                timeout=6.0,
            )
            if raw_subs:
                return [
                    TranscriptChunk(
                        text=s["text"],
                        start=s["start"],
                        end=s["start"] + s.get("duration", 2.0),
                    )
                    for s in raw_subs
                ]
        except Exception as exc:
            logger.info("yt-dlp subtitles skipped/failed (%s), advancing to neural audio transcription...", exc)

        # Tier 3: Direct audio stream download + local faster-whisper transcription
        await notify_cb(10.0, "Downloading audio stream for local AI speech recognition...")
        try:
            return await self._transcribe_via_audio_stream(url, video_id, notify_cb)
        except Exception as exc:
            logger.error("Audio neural transcription failed: %s", exc)
            raise RuntimeError(
                f"Could not retrieve video transcript or audio stream. Please check that the URL is public and valid: {exc}"
            ) from exc

    async def _transcribe_via_audio_stream(
        self,
        url: str,
        video_id: str,
        notify_cb: Callable[[float, str], Coroutine[Any, Any, None]],
    ) -> list[TranscriptChunk]:
        """Download tiny audio stream and transcribe locally with faster-whisper."""
        import tempfile
        import yt_dlp
        import shutil

        temp_dir = tempfile.mkdtemp()
        out_tmpl = os.path.join(temp_dir, f"{video_id}.%(ext)s")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "ba[ext=m4a]/ba/bestaudio",
            "outtmpl": out_tmpl,
            "socket_timeout": 15,
        }

        loop = asyncio.get_running_loop()

        def _download_audio() -> None:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        await loop.run_in_executor(_executor, _download_audio)

        files = [
            os.path.join(temp_dir, f)
            for f in os.listdir(temp_dir)
            if not f.endswith(".part")
        ]

        if not files:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError("Failed to download audio stream for transcription")

        audio_file = files[0]
        await notify_cb(12.5, "Running local neural Whisper speech-to-text model...")

        try:
            raw_chunks = await self._transcript_service.transcribe_audio_whisper(audio_file)
            return [
                TranscriptChunk(
                    text=c["text"],
                    start=float(c["start"]),
                    end=float(c["start"]) + float(c.get("duration", 2.0)),
                )
                for c in raw_chunks
            ]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _extract_video_id(url: str) -> str:
        """Extract the YouTube video ID from various URL formats."""
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
            r"(?:embed\/)([0-9A-Za-z_-]{11})",
            r"(?:shorts\/)([0-9A-Za-z_-]{11})",
        ]
        for pat in patterns:
            match = re.search(pat, url)
            if match:
                return match.group(1)
        return url
