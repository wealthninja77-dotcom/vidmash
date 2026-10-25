"""
Transcript Service — fetches and parses video transcripts.

Supports YouTube captions via ``youtube_transcript_api`` and
fallback subtitle extraction via ``yt-dlp`` with WebVTT parsing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi

from app.config import get_settings

logger = logging.getLogger(__name__)


class TranscriptService:
    """Retrieve and normalise video transcripts from multiple sources."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_youtube_transcript(self, video_id: str) -> list[dict[str, Any]]:
        """Fetch a YouTube transcript using the *youtube_transcript_api*.

        Args:
            video_id: The 11-character YouTube video identifier.

        Returns:
            A list of ``{"text": str, "start": float, "duration": float}`` dicts.

        Raises:
            RuntimeError: When no transcript can be retrieved.
        """
        loop = asyncio.get_running_loop()
        try:
            raw_entries = await loop.run_in_executor(
                None,
                lambda: YouTubeTranscriptApi.get_transcript(video_id),
            )
            result: list[dict[str, Any]] = []
            for entry in raw_entries:
                result.append(
                    {
                        "text": entry.get("text", "").strip(),
                        "start": float(entry.get("start", 0.0)),
                        "duration": float(entry.get("duration", 0.0)),
                    }
                )
            logger.info(
                "Retrieved %d transcript entries for video %s via youtube_transcript_api",
                len(result),
                video_id,
            )
            return result
        except Exception as exc:
            logger.warning(
                "youtube_transcript_api failed for %s: %s", video_id, exc
            )
            raise RuntimeError(
                f"Could not fetch transcript for video {video_id}: {exc}"
            ) from exc

    async def get_subtitles_yt_dlp(self, url: str) -> list[dict[str, Any]]:
        """Download auto-generated subtitles via *yt-dlp* and parse them.

        Falls back to this method when ``get_youtube_transcript`` fails.

        Args:
            url: Full video URL (YouTube or other supported site).

        Returns:
            A list of ``{"text": str, "start": float, "duration": float}`` dicts.
        """
        loop = asyncio.get_running_loop()
        tmpdir = tempfile.mkdtemp(prefix="vidmash_subs_")
        output_template = os.path.join(tmpdir, "subs")

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._run_yt_dlp(url, output_template, tmpdir),
            )
            logger.info(
                "Retrieved %d subtitle entries via yt-dlp for %s",
                len(result),
                url,
            )
            return result
        except Exception as exc:
            logger.error("yt-dlp subtitle extraction failed for %s: %s", url, exc)
            raise RuntimeError(
                f"yt-dlp subtitle extraction failed for {url}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_yt_dlp(
        self, url: str, output_template: str, tmpdir: str
    ) -> list[dict[str, Any]]:
        """Execute yt-dlp synchronously (called inside executor)."""
        import yt_dlp  # local import to keep top-level light

        ydl_opts: dict[str, Any] = {
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # Look for the VTT file that yt-dlp wrote
        vtt_content: str | None = None
        for fname in os.listdir(tmpdir):
            if fname.endswith(".vtt"):
                vtt_path = os.path.join(tmpdir, fname)
                with open(vtt_path, encoding="utf-8", errors="replace") as fh:
                    vtt_content = fh.read()
                break

        if not vtt_content:
            # Fallback: try to get subtitles from info dict
            subs: dict = info.get("requested_subtitles") or info.get("subtitles") or {}
            for lang_data in subs.values():
                if isinstance(lang_data, dict) and "data" in lang_data:
                    vtt_content = lang_data["data"]
                    break
                if isinstance(lang_data, list):
                    for entry in lang_data:
                        if isinstance(entry, dict) and "data" in entry:
                            vtt_content = entry["data"]
                            break
                    if vtt_content:
                        break

        if not vtt_content:
            raise RuntimeError("No VTT subtitle data found")

        return self._parse_vtt(vtt_content)

    def _parse_vtt(self, vtt_content: str) -> list[dict[str, Any]]:
        """Parse WebVTT content into a list of timed text dicts.

        Args:
            vtt_content: Raw ``.vtt`` file content.

        Returns:
            A list of ``{"text": str, "start": float, "duration": float}`` dicts.
        """
        # Pattern: 00:00:01.234 --> 00:00:05.678
        timestamp_re = re.compile(
            r"(\d{1,2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{1,2}:\d{2}:\d{2}\.\d{3})"
        )

        entries: list[dict[str, Any]] = []
        lines = vtt_content.strip().splitlines()

        i = 0
        while i < len(lines):
            match = timestamp_re.search(lines[i])
            if match:
                start_sec = self._vtt_ts_to_seconds(match.group(1))
                end_sec = self._vtt_ts_to_seconds(match.group(2))
                i += 1
                text_lines: list[str] = []
                while i < len(lines) and lines[i].strip():
                    # Strip VTT cue payload tags like <c>, </c>, <00:00:01.000>
                    cleaned = re.sub(r"<[^>]+>", "", lines[i]).strip()
                    if cleaned:
                        text_lines.append(cleaned)
                    i += 1

                text = " ".join(text_lines).strip()
                if text:
                    # De-duplicate consecutive identical entries
                    if entries and entries[-1]["text"] == text:
                        # Extend the previous entry's duration instead
                        prev_start = entries[-1]["start"]
                        entries[-1]["duration"] = end_sec - prev_start
                    else:
                        entries.append(
                            {
                                "text": text,
                                "start": start_sec,
                                "duration": round(end_sec - start_sec, 3),
                            }
                        )
            else:
                i += 1

        return entries

    @staticmethod
    def _vtt_ts_to_seconds(ts: str) -> float:
        """Convert a VTT timestamp ``HH:MM:SS.mmm`` to seconds."""
        parts = ts.split(":")
        if len(parts) == 3:
            hours, minutes, sec_ms = parts
        elif len(parts) == 2:
            hours = "0"
            minutes, sec_ms = parts
        else:
            return 0.0

        seconds_parts = sec_ms.split(".")
        secs = int(seconds_parts[0])
        millis = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

        return int(hours) * 3600 + int(minutes) * 60 + secs + millis / 1000.0
