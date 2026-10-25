"""
Transcript Service — fetches and parses video transcripts.

Supports YouTube captions via ``youtube_transcript_api``, fallback subtitle
extraction via ``yt-dlp``, and local neural transcription via ``faster-whisper``.
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


def clean_transcript_text(text: str) -> str:
    """Clean and deduplicate speech transcripts, removing rolling subtitle repeats and audio cues."""
    if not text:
        return ""
    
    # 1. Remove audio annotations: [music], [applause], [laughter], etc.
    text = re.sub(r"\[[^\]]*(?:music|applause|laughter|cheering|cough|inaudible)[^\]]*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*(?:music|applause|laughter|cheering)[^)]*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[♪♫♩♬]", " ", text)
    
    # 2. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # 3. Iteratively remove repeated words and rolling phrases
    prev = ""
    iterations = 0
    while prev != text and iterations < 4:
        prev = text
        # Remove repeated single words ("the the" -> "the", "that that" -> "that")
        text = re.sub(r"\b([a-zA-Z0-9'\"]+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
        
        # Remove repeated phrases from 12 words down to 2 words
        for n in range(12, 1, -1):
            pat = rf"\b((?:[a-zA-Z0-9'\"]+[\s\W]+){{{n-1}}}[a-zA-Z0-9'\"]+)(?:\s+\1\b)+"
            text = re.sub(pat, r"\1", text, flags=re.IGNORECASE)
        
        text = re.sub(r"\s+", " ", text).strip()
        iterations += 1
        
    return text


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
                cleaned = clean_transcript_text(entry.get("text", ""))
                if cleaned:
                    result.append(
                        {
                            "text": cleaned,
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

    async def transcribe_audio_whisper(self, audio_path: str) -> list[dict[str, Any]]:
        """Transcribe an audio file locally using faster-whisper.

        Args:
            audio_path: Absolute or relative path to the audio file.

        Returns:
            List of ``{"text": str, "start": float, "duration": float}`` dicts.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_whisper_sync, audio_path)

    @staticmethod
    def _run_whisper_sync(audio_path: str) -> list[dict[str, Any]]:
        """Run faster-whisper synchronously in a worker thread."""
        from faster_whisper import WhisperModel

        logger.info("Running faster-whisper transcription on %s ...", audio_path)
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, beam_size=1)
        results: list[dict[str, Any]] = []
        for s in segments:
            text = clean_transcript_text(s.text)
            if text:
                results.append(
                    {
                        "text": text,
                        "start": float(s.start),
                        "duration": float(max(0.1, s.end - s.start)),
                    }
                )
        logger.info("faster-whisper finished: %d segments extracted", len(results))
        return results

    async def get_subtitles_yt_dlp(self, url: str) -> list[dict[str, Any]]:
        """Download auto-generated subtitles via yt-dlp as WebVTT and parse them.

        Args:
            url: The video URL.

        Returns:
            List of ``{"text": str, "start": float, "duration": float}`` dicts.

        Raises:
            RuntimeError: When subtitle download or parsing fails.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ytdlp_subtitles_sync, url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ytdlp_subtitles_sync(self, url: str) -> list[dict[str, Any]]:
        """Blocking worker for yt-dlp subtitle download."""
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError("yt-dlp is required for subtitle fallback") from exc

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_tmpl = os.path.join(tmp_dir, "%(id)s.%(ext)s")
            ydl_opts: dict[str, Any] = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "vtt",
                "subtitleslangs": ["en.*", "en"],
                "outtmpl": out_tmpl,
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 8,
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as exc:
                raise RuntimeError(
                    f"yt-dlp failed to download subtitles: {exc}"
                ) from exc

            vtt_files = [
                os.path.join(tmp_dir, f)
                for f in os.listdir(tmp_dir)
                if f.endswith(".vtt")
            ]

            if not vtt_files:
                raise RuntimeError("No .vtt subtitle files found after download")

            best_file = self._select_best_vtt(vtt_files)
            with open(best_file, "r", encoding="utf-8", errors="replace") as fh:
                vtt_content = fh.read()

            return self._parse_vtt(vtt_content)

    def _parse_vtt(self, vtt_content: str) -> list[dict[str, Any]]:
        """Parse WebVTT content into structured transcript chunks, eliminating rolling duplicates."""
        cleaned = re.sub(r"WEBVTT[^\n]*\n", "", vtt_content)
        cleaned = re.sub(r"NOTE[^\n]*\n.*?\n\n", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"STYLE[^\n]*\n.*?\n\n", "", cleaned, flags=re.DOTALL)

        blocks = re.split(r"\n\s*\n", cleaned.strip())
        results: list[dict[str, Any]] = []

        timestamp_re = re.compile(
            r"(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}:)?(\d{2}):(\d{2})[.,](\d{3})"
        )

        last_block_lines: list[str] = []

        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue

            time_line_idx: int | None = None
            for idx, line in enumerate(lines):
                if "-->" in line:
                    time_line_idx = idx
                    break

            if time_line_idx is None:
                continue

            time_match = timestamp_re.search(lines[time_line_idx])
            if not time_match:
                continue

            start_sec = self._vtt_time_to_seconds(
                time_match.group(1),
                time_match.group(2),
                time_match.group(3),
                time_match.group(4),
            )
            end_sec = self._vtt_time_to_seconds(
                time_match.group(5),
                time_match.group(6),
                time_match.group(7),
                time_match.group(8),
            )
            duration = max(0.0, end_sec - start_sec)

            raw_text_lines = lines[time_line_idx + 1 :]
            # Filter out rolling lines that were already in the previous block
            unique_lines = []
            for l in raw_text_lines:
                clean_l = re.sub(r"<[^>]+>", "", l).strip()
                if clean_l and clean_l not in last_block_lines:
                    unique_lines.append(clean_l)

            # If all lines were identical to previous block, use the last line if not strictly identical
            if not unique_lines and raw_text_lines:
                clean_last = re.sub(r"<[^>]+>", "", raw_text_lines[-1]).strip()
                if clean_last and (not last_block_lines or clean_last != last_block_lines[-1]):
                    unique_lines.append(clean_last)

            last_block_lines = [re.sub(r"<[^>]+>", "", l).strip() for l in raw_text_lines]
            combined = " ".join(unique_lines).strip()
            text = clean_transcript_text(combined)

            if text:
                results.append(
                    {
                        "text": text,
                        "start": start_sec,
                        "duration": duration,
                    }
                )

        return results

    @staticmethod
    def _vtt_time_to_seconds(
        hours: str | None, minutes: str, seconds: str, millis: str
    ) -> float:
        """Convert VTT timestamp components to total seconds."""
        h = int(hours.rstrip(":")) if hours else 0
        m = int(minutes)
        s = int(seconds)
        ms = int(millis)
        return h * 3600 + m * 60 + s + ms / 1000.0

    @staticmethod
    def _select_best_vtt(vtt_files: list[str]) -> str:
        """Pick the best subtitle file from candidates."""
        for path in vtt_files:
            base = os.path.basename(path).lower()
            if ".en." in base and "orig" in base:
                return path
        for path in vtt_files:
            base = os.path.basename(path).lower()
            if ".en." in base and "auto" not in base:
                return path
        return vtt_files[0]
