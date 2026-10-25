"""
Layer 1: Extraction Cortex
Downloads video audio, extracts transcript, captures metadata.

This is the first stage of the VidMash 7-layer Cortex pipeline.
It takes a video URL, extracts metadata using yt-dlp, and retrieves
the transcript using youtube_transcript_api with yt-dlp as a fallback.
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.config import get_settings
from app.models.schemas import TranscriptChunk, VideoMetadata
from app.services.transcript_service import TranscriptService

logger = logging.getLogger(__name__)

# Shared thread pool for blocking yt-dlp operations
_executor = ThreadPoolExecutor(max_workers=4)


class ExtractionCortex:
    """Extracts video metadata and transcript from a given URL.

    Orchestrates metadata retrieval via yt-dlp and transcript fetching
    via youtube_transcript_api (with yt-dlp subtitle fallback).
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._transcript_service = TranscriptService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, url: str) -> tuple[VideoMetadata, list[TranscriptChunk]]:
        """Run the full extraction pipeline for a video URL.

        Args:
            url: The video URL to process.

        Returns:
            A tuple of (VideoMetadata, list[TranscriptChunk]).

        Raises:
            ValueError: If the URL is invalid or unsupported.
            RuntimeError: If both metadata and transcript extraction fail.
        """
        logger.info("Starting extraction for URL: %s", url)

        # Run metadata and transcript extraction concurrently
        video_id = self._extract_video_id(url)
        metadata_task = self._get_metadata(url)
        transcript_task = self._get_transcript(video_id, url)

        metadata, chunks = await asyncio.gather(
            metadata_task,
            transcript_task,
        )

        # Validate duration against configured maximum
        max_duration = self._settings.MAX_VIDEO_DURATION
        if metadata.duration > max_duration:
            raise ValueError(
                f"Video duration ({metadata.duration}s) exceeds maximum "
                f"allowed ({max_duration}s)."
            )

        logger.info(
            "Extraction complete: '%s' — %d chunks, %ds duration",
            metadata.title,
            len(chunks),
            metadata.duration,
        )
        return metadata, chunks

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    async def _get_metadata(self, url: str) -> VideoMetadata:
        """Extract video metadata without downloading the video.

        Uses yt-dlp in extract-info mode (no download) run inside a
        thread executor to avoid blocking the event loop.

        Args:
            url: The video URL.

        Returns:
            Populated VideoMetadata instance.
        """
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(_executor, self._ytdlp_extract_info, url)

        # Parse chapters if available
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

        # Build upload date string (yt-dlp gives YYYYMMDD)
        raw_date = info.get("upload_date") or ""
        upload_date = ""
        if raw_date and len(raw_date) == 8:
            upload_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

        video_id = (
            info.get("id")
            or self._extract_video_id(url)
        )

        metadata = VideoMetadata(
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
        logger.info("Metadata extracted: '%s' by %s", metadata.title, metadata.channel)
        return metadata

    @staticmethod
    def _ytdlp_extract_info(url: str) -> dict[str, Any]:
        """Blocking helper: run yt-dlp extract_info.

        Args:
            url: The video URL.

        Returns:
            The raw info dict from yt-dlp.

        Raises:
            RuntimeError: If yt-dlp fails to extract info.
        """
        try:
            import yt_dlp  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "yt-dlp is not installed. Run: pip install yt-dlp"
            ) from exc

        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
            # Don't download anything, just extract info
            "format": "best",
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info: dict[str, Any] = ydl.extract_info(url, download=False) or {}
                return info
        except Exception as exc:
            logger.error("yt-dlp extract_info failed for %s: %s", url, exc)
            raise RuntimeError(f"Failed to extract video info: {exc}") from exc

    # ------------------------------------------------------------------
    # Transcript extraction
    # ------------------------------------------------------------------

    async def _get_transcript(
        self, video_id: str, url: str
    ) -> list[TranscriptChunk]:
        """Retrieve the transcript for a video.

        Strategy:
        1. Try youtube_transcript_api (fastest, most reliable for YouTube).
        2. Fall back to yt-dlp subtitle download.
        3. If both fail, raise RuntimeError.

        Args:
            video_id: The extracted video ID.
            url: The original video URL.

        Returns:
            List of TranscriptChunk instances.

        Raises:
            RuntimeError: If no transcript source succeeds.
        """
        # Attempt 1: youtube_transcript_api via TranscriptService
        try:
            chunks = await self._transcript_service.get_youtube_transcript(video_id)
            if chunks:
                logger.info(
                    "Transcript fetched via youtube_transcript_api: %d chunks",
                    len(chunks),
                )
                return chunks
        except Exception as exc:
            logger.warning(
                "youtube_transcript_api failed for %s: %s — trying yt-dlp subtitles",
                video_id,
                exc,
            )

        # Attempt 2: yt-dlp subtitle extraction
        try:
            chunks = await self._get_transcript_ytdlp(url)
            if chunks:
                logger.info(
                    "Transcript fetched via yt-dlp subtitles: %d chunks",
                    len(chunks),
                )
                return chunks
        except Exception as exc:
            logger.warning("yt-dlp subtitle extraction failed: %s", exc)

        raise RuntimeError(
            f"No transcript available for video '{video_id}'. "
            "The video may lack subtitles/captions."
        )

    async def _get_transcript_ytdlp(self, url: str) -> list[TranscriptChunk]:
        """Fall-back transcript extraction using yt-dlp subtitle download.

        Args:
            url: The video URL.

        Returns:
            List of TranscriptChunk instances parsed from subtitles.
        """
        loop = asyncio.get_running_loop()
        raw_subs = await loop.run_in_executor(
            _executor, self._ytdlp_download_subs, url
        )
        return self._parse_subtitle_data(raw_subs)

    @staticmethod
    def _ytdlp_download_subs(url: str) -> list[dict[str, Any]]:
        """Blocking helper: download subtitles via yt-dlp.

        Args:
            url: The video URL.

        Returns:
            List of raw subtitle dicts with 'text', 'start', 'duration' keys.

        Raises:
            RuntimeError: If yt-dlp fails to retrieve subtitles.
        """
        try:
            import yt_dlp  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("yt-dlp is not installed.") from exc

        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "subtitlesformat": "json3",
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False) or {}

            # yt-dlp stores subtitles in 'requested_subtitles' or 'subtitles'
            subtitles: dict[str, Any] = info.get("subtitles") or {}
            auto_subs: dict[str, Any] = info.get("automatic_captions") or {}

            # Prefer manual subtitles, fall back to automatic
            sub_data: list[dict[str, Any]] = []
            for lang_key in ["en", "en-US", "en-GB"]:
                if lang_key in subtitles:
                    sub_entries = subtitles[lang_key]
                    for entry in sub_entries:
                        if entry.get("ext") == "json3":
                            # Fetch the actual subtitle content
                            sub_url = entry.get("url", "")
                            if sub_url:
                                import urllib.request

                                import json

                                with urllib.request.urlopen(sub_url, timeout=30) as resp:
                                    raw = json.loads(resp.read().decode("utf-8"))
                                    events = raw.get("events", [])
                                    for evt in events:
                                        segs = evt.get("segs", [])
                                        text = "".join(
                                            s.get("utf8", "") for s in segs
                                        ).strip()
                                        if text and text != "\n":
                                            start_ms = evt.get("tStartMs", 0)
                                            dur_ms = evt.get("dDurationMs", 0)
                                            sub_data.append(
                                                {
                                                    "text": text,
                                                    "start": start_ms / 1000.0,
                                                    "duration": dur_ms / 1000.0,
                                                }
                                            )
                                    if sub_data:
                                        return sub_data
                    break

            # Try automatic captions
            for lang_key in ["en", "en-US", "en-GB", "en-orig"]:
                if lang_key in auto_subs:
                    sub_entries = auto_subs[lang_key]
                    for entry in sub_entries:
                        if entry.get("ext") == "json3":
                            sub_url = entry.get("url", "")
                            if sub_url:
                                import urllib.request

                                import json

                                with urllib.request.urlopen(sub_url, timeout=30) as resp:
                                    raw = json.loads(resp.read().decode("utf-8"))
                                    events = raw.get("events", [])
                                    for evt in events:
                                        segs = evt.get("segs", [])
                                        text = "".join(
                                            s.get("utf8", "") for s in segs
                                        ).strip()
                                        if text and text != "\n":
                                            start_ms = evt.get("tStartMs", 0)
                                            dur_ms = evt.get("dDurationMs", 0)
                                            sub_data.append(
                                                {
                                                    "text": text,
                                                    "start": start_ms / 1000.0,
                                                    "duration": dur_ms / 1000.0,
                                                }
                                            )
                                    if sub_data:
                                        return sub_data
                    break

            if not sub_data:
                raise RuntimeError("No English subtitles found via yt-dlp.")
            return sub_data

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"yt-dlp subtitle download failed: {exc}") from exc

    @staticmethod
    def _parse_subtitle_data(raw_subs: list[dict[str, Any]]) -> list[TranscriptChunk]:
        """Convert raw subtitle dicts into TranscriptChunk instances.

        Args:
            raw_subs: List of dicts with 'text', 'start', 'duration' keys.

        Returns:
            List of TranscriptChunk instances.
        """
        chunks: list[TranscriptChunk] = []
        for entry in raw_subs:
            text = entry.get("text", "").strip()
            if not text:
                continue
            start = float(entry.get("start", 0))
            duration = float(entry.get("duration", 0))
            chunks.append(
                TranscriptChunk(
                    text=text,
                    start=start,
                    end=start + duration,
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # URL parsing
    # ------------------------------------------------------------------

    def _extract_video_id(self, url: str) -> str:
        """Extract the video ID from various URL formats.

        Supported formats:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID
        - https://www.youtube.com/shorts/VIDEO_ID
        - https://www.youtube.com/live/VIDEO_ID
        - https://vimeo.com/VIDEO_ID
        - Generic fallback: last path segment

        Args:
            url: The video URL to parse.

        Returns:
            The extracted video ID string.

        Raises:
            ValueError: If the URL cannot be parsed.
        """
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string.")

        url = url.strip()

        # YouTube patterns
        youtube_patterns: list[str] = [
            # Standard watch URL: ?v=ID or &v=ID
            r"(?:youtube\.com/watch\?.*?v=)([\w-]{11})",
            # Short URL: youtu.be/ID
            r"(?:youtu\.be/)([\w-]{11})",
            # Embed URL: youtube.com/embed/ID
            r"(?:youtube\.com/embed/)([\w-]{11})",
            # Old embed: youtube.com/v/ID
            r"(?:youtube\.com/v/)([\w-]{11})",
            # Shorts: youtube.com/shorts/ID
            r"(?:youtube\.com/shorts/)([\w-]{11})",
            # Live: youtube.com/live/ID
            r"(?:youtube\.com/live/)([\w-]{11})",
        ]

        for pattern in youtube_patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                logger.debug("Extracted YouTube video ID: %s", video_id)
                return video_id

        # Vimeo patterns
        vimeo_match = re.search(r"vimeo\.com/(?:video/)?(\d+)", url)
        if vimeo_match:
            video_id = vimeo_match.group(1)
            logger.debug("Extracted Vimeo video ID: %s", video_id)
            return video_id

        # Generic fallback: use the last non-empty path segment
        from urllib.parse import urlparse

        parsed = urlparse(url)
        path_segments = [seg for seg in parsed.path.split("/") if seg]
        if path_segments:
            video_id = path_segments[-1]
            logger.debug("Extracted generic video ID from path: %s", video_id)
            return video_id

        raise ValueError(f"Unable to extract video ID from URL: {url}")
