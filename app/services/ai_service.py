"""
AI Service — Gemini API wrapper for VidMash.

Provides async methods for text generation, segment analysis,
and narrative writing using Google's Gemini 2.0 Flash model.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import google.generativeai as genai

from app.config import get_settings

logger = logging.getLogger(__name__)


class AIService:
    """Wrapper around the Google Gemini generative AI SDK.

    Handles configuration, prompt dispatch, retries with exponential back-off,
    and optional JSON-mode responses.
    """

    _MAX_RETRIES: int = 3
    _BASE_DELAY: float = 1.0  # seconds

    def __init__(self) -> None:
        """Initialise the Gemini SDK and select the model."""
        settings = get_settings()
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._model_name: str = "gemini-2.0-flash"
        logger.info("AIService initialised with model %s", self._model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        json_mode: bool = False,
    ) -> str:
        """Send *prompt* to Gemini and return the text response.

        Args:
            prompt: The user/task prompt.
            system_instruction: Optional system-level instruction prepended
                to the conversation.
            json_mode: When ``True`` the response MIME type is set to
                ``application/json`` so the model returns valid JSON.

        Returns:
            The raw text content of the model's response.
        """
        generation_config: dict[str, Any] = {}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        model_kwargs: dict[str, Any] = {"model_name": self._model_name}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction
        if generation_config:
            model_kwargs["generation_config"] = generation_config

        model = genai.GenerativeModel(**model_kwargs)

        last_exc: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                response = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: model.generate_content(prompt),
                )
                return response.text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = self._BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini attempt %d/%d failed (%s). Retrying in %.1fs …",
                    attempt,
                    self._MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Gemini API failed after {self._MAX_RETRIES} attempts"
        ) from last_exc

    async def analyze_segments(self, text: str) -> dict:
        """Ask Gemini to analyse *text* and return structured JSON.

        The returned dict contains keys such as ``topics``, ``entities``,
        ``importance_scores``, and ``segment_types``.
        """
        system = (
            "You are a world-class content analyst. Analyse the provided text "
            "and return a single JSON object with the following keys:\n"
            "  • topics — list of {name, description, start_offset, end_offset}\n"
            "  • entities — list of {name, type, mentions, context}\n"
            "  • importance_scores — list of {segment_index, score (0-1), reasoning}\n"
            "  • segment_types — list of {segment_index, type, confidence}\n"
            "  • key_themes — list of strings\n"
            "  • topic_hierarchy — {main_topic: [sub_topics …]}\n"
            "Return ONLY valid JSON. No markdown fences."
        )
        prompt = f"Analyse the following transcript text:\n\n{text}"

        raw = await self.generate(prompt, system_instruction=system, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Failed to parse Gemini analysis JSON — returning raw text wrapper")
            return {"raw": raw, "topics": [], "entities": [], "importance_scores": []}

    async def write_narrative(self, segments_data: dict, metadata: dict) -> dict:
        """Ask Gemini to compose a full narrative document from analysed data.

        Args:
            segments_data: Output from ``analyze_segments``.
            metadata: Video metadata dictionary.

        Returns:
            A dict with ``executive_summary``, ``sections``, ``key_takeaways``,
            ``action_items``, and ``key_quotes``.
        """
        system = (
            "You are a professional long-form writer and editor. "
            "Transform the structured analysis below into a polished, "
            "publication-quality document. Return a single JSON object:\n"
            "{\n"
            '  "executive_summary": "3-5 paragraphs of compelling prose",\n'
            '  "sections": [\n'
            '    {"title": "…", "content": "flowing prose — NO bullet points", "order": 1}\n'
            "  ],\n"
            '  "key_takeaways": ["takeaway 1", …],\n'
            '  "action_items": [\n'
            '    {"text": "…", "priority": "high|medium|low"}\n'
            "  ],\n"
            '  "key_quotes": [\n'
            '    {"text": "…", "speaker": "…", "timestamp": 0.0}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "• Write in professional, engaging prose.\n"
            "• Do NOT use bullet points for main section content.\n"
            "• Maintain the original speaker's intent.\n"
            "• Include specific data points, names, and facts.\n"
            "• The reader should need ZERO reason to watch the original video.\n"
            "• Return ONLY valid JSON."
        )

        prompt = (
            f"Video title: {metadata.get('title', 'Unknown')}\n"
            f"Channel: {metadata.get('channel', 'Unknown')}\n"
            f"Duration: {metadata.get('duration', 0)} seconds\n\n"
            f"Structured analysis:\n{json.dumps(segments_data, indent=2, default=str)}"
        )

        raw = await self.generate(prompt, system_instruction=system, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Failed to parse Gemini narrative JSON")
            return {
                "executive_summary": raw,
                "sections": [],
                "key_takeaways": [],
                "action_items": [],
                "key_quotes": [],
            }
