from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


_ONLY_PUNCT_RE = re.compile(r"^[^\w]+$", re.UNICODE)
_TOKEN_RE = re.compile(
    r"[0-9A-Za-zÀ-ÖØ-öø-ÿÆØÅæøå]+(?:[’'\-][0-9A-Za-zÀ-ÖØ-öø-ÿÆØÅæøå]+)*",
    re.UNICODE,
)
_TITLECASE_STOPWORDS = {
    "Det",
    "Den",
    "De",
    "En",
    "Et",
    "Hva",
    "Hvor",
    "Når",
    "Slik",
    "Her",
    "Nå",
    "Se",
    "Dette",
    "Denne",
    "Disse",
}


@dataclass(frozen=True)
class NerMention:
    start: int
    end: int
    surface: str
    ner_type: str
    ner_score: float


def is_valid_surface(surface: str) -> bool:
    text = surface.strip()
    if len(text) < 2:
        return False
    if text.isdigit():
        return False
    if _ONLY_PUNCT_RE.match(text):
        return False
    return True


def _is_all_caps_token(token: str) -> bool:
    letters = [ch for ch in token if ch.isalpha()]
    if len(letters) < 2:
        return False
    return all(ch.isupper() for ch in letters)


def _is_titlecase_token(token: str) -> bool:
    if not token:
        return False
    if token[0].isalpha() and token[0].isupper():
        return True
    return _is_all_caps_token(token)


def heuristic_mentions_from_title(
    text: str,
    *,
    max_mentions: int = 6,
    score: float = 0.45,
    max_span_chars: int = 60,
    max_span_tokens: int = 6,
    min_first_token_len: int = 0,
) -> list[NerMention]:
    """
    Conservative heuristic extractor to improve recall when model NER returns empty.

    Strategy: extract contiguous titlecase/all-caps token spans (e.g. "Erna Solberg", "NTB"),
    with light filtering to avoid common sentence-initial function words.
    """
    if not text:
        return []

    tokens: list[tuple[int, int, str]] = [(m.start(), m.end(), m.group(0)) for m in _TOKEN_RE.finditer(text)]
    if not tokens:
        return []

    mentions: list[NerMention] = []
    seen: set[tuple[int, int]] = set()

    i = 0
    while i < len(tokens):
        start_i, end_i, tok_i = tokens[i]
        if not _is_titlecase_token(tok_i):
            i += 1
            continue

        span_tokens = [tok_i]
        span_end = end_i
        j = i + 1
        while j < len(tokens):
            start_j, end_j, tok_j = tokens[j]
            if not _is_titlecase_token(tok_j):
                break
            if not text[span_end:start_j].isspace():
                break
            if (j - i + 1) > max_span_tokens:
                break
            span_end = end_j
            span_tokens.append(tok_j)
            j += 1

        surface = text[start_i:span_end]
        token_count = len(span_tokens)
        token0 = span_tokens[0] if span_tokens else ""

        if len(surface) <= max_span_chars and is_valid_surface(surface):
            accept = True
            # Filter the most common sentence-initial function words for single-token mentions.
            if token_count == 1:
                if token0 in _TITLECASE_STOPWORDS:
                    accept = False
                else:
                    is_caps = _is_all_caps_token(token0)
                    has_digit = any(ch.isdigit() for ch in token0)
                    has_dash = "-" in token0
                    # Avoid sentence-case first-word false positives.
                    if start_i == 0 and not (
                        is_caps
                        or has_digit
                        or has_dash
                        or (min_first_token_len > 0 and len(token0) >= min_first_token_len)
                    ):
                        accept = False
                    # Very short single tokens are often not useful for EL.
                    if len(token0) < 4 and not is_caps:
                        accept = False

            if accept:
                key = (start_i, span_end)
                if key not in seen:
                    seen.add(key)
                    mentions.append(
                        NerMention(
                            start=start_i,
                            end=span_end,
                            surface=surface,
                            ner_type="HEUR",
                            ner_score=float(score),
                        )
                    )

        i = j

    mentions.sort(key=lambda m: (m.start, m.end))
    if max_mentions and max_mentions > 0:
        mentions = mentions[: int(max_mentions)]
    return mentions


def load_token_classification_pipeline(
    *,
    model_name: str,
    device: int,
    max_length: int | None = None,
    aggregation_strategy: str = "simple",
) -> Any:
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'transformers'. Install it first, e.g. `pip install transformers`."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if max_length is not None and max_length > 0:
        tokenizer.model_max_length = int(max_length)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=aggregation_strategy,
        device=device,
    )


def mentions_from_pipeline_output(text: str, outputs: Iterable[dict[str, Any]]) -> list[NerMention]:
    mentions: list[NerMention] = []
    seen: set[tuple[int, int, str]] = set()
    for ent in outputs:
        start = int(ent.get("start", -1))
        end = int(ent.get("end", -1))
        if start < 0 or end <= start or end > len(text):
            continue
        surface = text[start:end]
        if not is_valid_surface(surface):
            continue
        ner_type = str(ent.get("entity_group") or ent.get("entity") or "")
        if not ner_type:
            continue
        ner_score = float(ent.get("score", 0.0))
        key = (start, end, ner_type)
        if key in seen:
            continue
        seen.add(key)
        mentions.append(
            NerMention(
                start=start,
                end=end,
                surface=surface,
                ner_type=ner_type,
                ner_score=ner_score,
            )
        )
    return mentions
