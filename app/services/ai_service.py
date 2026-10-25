"""
AI Service — Gemini API wrapper for VidMash.

Provides async methods for text generation, segment analysis,
and narrative writing with automatic failover to local synthesis.
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

    Handles configuration, prompt dispatch, fast timeouts,
    and automatic failover circuit-breaker to 100% local processing
    when quota limits are reached or network is unavailable.
    """

    def __init__(self) -> None:
        """Initialise the Gemini SDK and check API key status."""
        settings = get_settings()
        self._api_key: str = (settings.GEMINI_API_KEY or "").strip()
        self._has_valid_key: bool = bool(
            self._api_key and self._api_key not in ("your_gemini_api_key_here", "dummy", "")
        )
        self._local_fallback_active: bool = not self._has_valid_key

        if self._has_valid_key:
            try:
                genai.configure(api_key=self._api_key)
            except Exception as e:
                logger.warning("Could not configure Gemini SDK (%s). Using local mode.", e)
                self._has_valid_key = False
                self._local_fallback_active = True

        self._model_name: str = "gemini-flash-latest"
        logger.info(
            "AIService initialised (key present: %s, local mode: %s)",
            self._has_valid_key,
            self._local_fallback_active,
        )

    @property
    def is_local_mode(self) -> bool:
        """Return True if running in local heuristic mode (quota reached or no key)."""
        return not self._has_valid_key or self._local_fallback_active

    def enable_local_mode(self, reason: str = "") -> None:
        """Explicitly switch to local processing mode."""
        self._has_valid_key = False
        self._local_fallback_active = True
        logger.info("Switched to local synthesis mode: %s", reason or "manual switch")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        json_mode: bool = False,
    ) -> str:
        """Send *prompt* to Gemini with fast timeout and immediate fallback to local mode.

        Args:
            prompt: The user/task prompt.
            system_instruction: Optional system instruction.
            json_mode: Return JSON MIME type.

        Returns:
            The raw text content of the response.
        """
        if self.is_local_mode:
            raise RuntimeError(
                "Gemini AI quota reached or unavailable. Auto-switched to local engine."
            )

        generation_config: dict[str, Any] = {}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        model_kwargs: dict[str, Any] = {"model_name": self._model_name}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction
        if generation_config:
            model_kwargs["generation_config"] = generation_config

        model = genai.GenerativeModel(**model_kwargs)

        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: model.generate_content(
                        prompt,
                        request_options={"timeout": 8.0},
                    ),
                ),
                timeout=10.0,
            )
            return response.text
        except Exception as exc:  # noqa: BLE001
            err_str = str(exc)
            if "429" in err_str or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                logger.info(
                    "Gemini API quota reached (429). Seamlessly auto-switching to 100%% local AI engine."
                )
            else:
                logger.warning(
                    "Gemini API unavailable (%s). Seamlessly auto-switching to local AI engine.",
                    exc,
                )
            self.enable_local_mode(f"API call failed: {exc}")
            raise RuntimeError(f"Switched to local engine: {exc}") from exc

    async def analyze_segments(self, text: str) -> dict:
        """Ask Gemini to analyse *text* and return structured JSON."""
        system = (
            "You are a world-class content analyst. Analyse the provided text "
            "and return a single JSON object with topics, entities, importance_scores."
        )
        prompt = f"Analyse the following transcript text:\n\n{text}"

        raw = await self.generate(prompt, system_instruction=system, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw, "topics": [], "entities": [], "importance_scores": []}

    async def write_narrative(self, segments_data: dict, metadata: dict) -> dict:
        """Ask Gemini to compose a full narrative document from analysed data."""
        system = "Transform the structured analysis below into a polished, publication-quality document."
        prompt = f"Structured analysis:\n{json.dumps(segments_data, indent=2, default=str)}"

        raw = await self.generate(prompt, system_instruction=system, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "executive_summary": raw,
                "sections": [],
                "key_takeaways": [],
                "action_items": [],
                "key_quotes": [],
            }
