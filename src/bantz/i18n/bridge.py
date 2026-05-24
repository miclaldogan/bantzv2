"""
Bantz v2 — Language Bridge
TR↔EN translation via MarianMT. Lazy-loaded on first use.
~100ms on CPU, faster with GPU.

Models:
  TR→EN: Helsinki-NLP/opus-mt-tr-en
  EN→TR: Helsinki-NLP/opus-mt-tc-big-en-tr
"""
from __future__ import annotations

import asyncio
import logging
from typing import Literal

from bantz.config import config

logger = logging.getLogger(__name__)

Direction = Literal["tr2en", "en2tr"]

# Model ID'leri
_MODELS: dict[Direction, str] = {
    "tr2en": "Helsinki-NLP/opus-mt-tr-en",
    "en2tr": "Helsinki-NLP/opus-mt-tc-big-en-tr",
}

class _Translator:
    """Unidirectional translator. Lazy-loaded."""

    def __init__(self, direction: Direction) -> None:
        self.direction = direction
        self.model_id = _MODELS[direction]
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            logger.info(f"Loading MarianMT: {self.model_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            self._model.to("cpu")
            self._model.eval()
            self._torch = torch
            logger.info(f"MarianMT ready: {self.model_id}")
        except ImportError:
            raise RuntimeError(
                "transformers package not installed. "
                "Install with: pip install 'bantz[translation]'"
            )

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        self._load()
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=256,
                max_length=None,   # override model's built-in limit to avoid conflict
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def _chunk_translate(self, text: str) -> str:
        """Translate long text by splitting into paragraph-sized chunks.

        MarianMT has a 512-token input limit — this prevents silent
        truncation of longer butler responses.
        """
        import re
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text)]
        result_parts: list[str] = []
        for para in paragraphs:
            if not para:
                result_parts.append("")
                continue
            if len(para) <= 400:
                result_parts.append(self.translate(para))
            else:
                # Split on sentence endings, group into ≤400-char chunks
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                translated_sentences: list[str] = []
                for s in sentences:
                    if len(buf) + len(s) + 1 > 400 and buf:
                        translated_sentences.append(self.translate(buf.strip()))
                        buf = s
                    else:
                        buf = (buf + " " + s).strip() if buf else s
                if buf:
                    translated_sentences.append(self.translate(buf.strip()))
                result_parts.append(" ".join(translated_sentences))
        return "\n\n".join(result_parts)


# ── Global bridge ─────────────────────────────────────────────────────────────

class LanguageBridge:
    """
    Usage:
        from bantz.i18n.bridge import bridge
        en_text = await bridge.to_english("check my disk")
        tr_text = await bridge.to_turkish("Your disk is 76% full.")
    """

    def __init__(self) -> None:
        self._tr2en = _Translator("tr2en")
        self._en2tr = _Translator("en2tr")

    def is_enabled(self) -> bool:
        # Re-read config each call so .env changes take effect after restart.
        return config.translation_enabled and config.language == "tr"

    async def to_english(self, text: str) -> str:
        """TR → EN. Returns original text if bridge is disabled."""
        if not self.is_enabled():
            return text
        return await asyncio.get_running_loop().run_in_executor(
            None, self._tr2en.translate, text
        )

    async def to_turkish(self, text: str) -> str:
        """EN → TR. Returns original text if bridge is disabled."""
        if not self.is_enabled():
            return text
        return await asyncio.get_running_loop().run_in_executor(
            None, self._en2tr._chunk_translate, text
        )

    def preload(self) -> None:
        """Pre-load both models at startup (optional)."""
        if not self.is_enabled():
            return
        self._tr2en._load()
        self._en2tr._load()


bridge = LanguageBridge()