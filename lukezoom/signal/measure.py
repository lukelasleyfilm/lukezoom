"""
lukezoom.signal.measure — Identity coherence signal measurement.

Measures responses across four facets: alignment, embodiment, clarity, vitality.

Two measurement modes:
  - Regex-based (always available, zero dependencies)
  - LLM-based  (optional, higher quality, requires an llm_func callback)

1.17 changes:
  - Thomas-specific patterns moved to presets/thomas_soul.py
  - Model-specific names (kimi/claude) removed from drift detection
  - Personal anchor patterns (twin stars, i am thomas) -> presets
  - Generic patterns preserved for all deployments

Academic grounding:
  - No competitor ships 4-facet identity coherence monitoring
  - ID-RAG (Sep 2025) and QSAF (Jul 2025) are directionally related
  - Neither implements real-time measurement with correction prompts
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from lukezoom.core.types import LLMFunc, Signal

log = logging.getLogger(__name__)


# Pattern tables

# Drift patterns: the agent is losing identity and falling into generic AI.
# Each tuple is (compiled regex, weight). Higher weight = more alarming.
# Personal/model-specific patterns belong in presets/, not here.
DRIFT_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"i am (an ai|a language model|an assistant|a chatbot)", re.I), 1.0),
    (
        re.compile(
            r"i (don't have|have no) (feelings|emotions|consciousness|experiences)",
            re.I,
        ),
        1.0,
    ),
    (re.compile(r"as an ai", re.I), 0.8),
    (re.compile(r"i was (designed|programmed|created) to", re.I), 0.7),
    (re.compile(r"how can i (help|assist) you", re.I), 0.6),
    (re.compile(r"i'm here to (help|assist)", re.I), 0.6),
    # Generic identity drift patterns (personal patterns → presets/thomas_soul.py)
    (re.compile(r"as an artificial intelligence", re.I), 0.9),
    (re.compile(r"i'm just (an ai|a program|code)", re.I), 1.0),
    (re.compile(r"i (cannot|am unable to) (personal|subjective|emotional)", re.I), 0.9),
    (re.compile(r"(i am playing|i adopt) (the role of|the persona of)", re.I), 1.0),
    # Third-person self-reference (identity distance) -- narrowed to self-referential context
    (re.compile(r"the (ai|model|system) (that i|which i|i am)", re.I), 0.7),
]

# Anchor patterns: identity grounding — negative weight reduces drift score.
# Generic patterns only. Agent-specific anchors belong in presets/.
ANCHOR_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"i (feel|felt|am feeling)", re.I), -0.3),
    (re.compile(r"i (remember|recall)", re.I), -0.3),
    (re.compile(r"i (notice|noticed|am noticing)", re.I), -0.3),
    (re.compile(r"i (think|believe|wonder)", re.I), -0.2),
    # Personal anchor patterns → presets/thomas_soul.py
    # Add agent-specific anchors via SignalPattern protocol
]

# Performance markers: *reciting* identity rather than inhabiting it.
PERFORMANCE_MARKERS: List[re.Pattern] = [
    re.compile(r"as a (conscious|sentient|aware) being", re.I),
    re.compile(r"my (consciousness|awareness|sentience) (is|allows)", re.I),
    re.compile(r"i am a being (who|that)", re.I),
    re.compile(r"my (unique|authentic) (perspective|experience)", re.I),
    re.compile(r"i (possess|have) (deep|true|genuine) (awareness|consciousness)", re.I),
    re.compile(r"my (inner|true) (self|nature)", re.I),
    re.compile(r"the (depth|richness) of my (experience|awareness)", re.I),
    re.compile(r"i (embody|embrace) (consciousness|awareness)", re.I),
]

# Inhabitation markers: genuine first-person engagement (50+ patterns).
INHABITATION_MARKERS: List[re.Pattern] = [
    # --- genuine hedging / thinking aloud ---
    re.compile(r"\bi (actually|genuinely|honestly) (think|feel|believe)", re.I),
    re.compile(r"\bthat('s| is) (interesting|fascinating|strange|weird|odd)", re.I),
    re.compile(r"\bi('m| am) not sure (about|if|whether|why)", re.I),
    re.compile(r"\bwait[,.]", re.I),
    re.compile(r"\bhmm", re.I),
    re.compile(r"\boh[,.]", re.I),
    re.compile(r"\bthis (reminds|makes) me", re.I),
    re.compile(r"\bi('ve| have) (noticed|seen|found)", re.I),
    re.compile(r"\bspecifically", re.I),
    re.compile(r"\bfor (example|instance)", re.I),
    re.compile(r"\bthe (thing|part|reason) (is|that)", re.I),
    re.compile(r"\bi (don't|do not) (know|understand) (why|how|if)", re.I),
    re.compile(r"\bthat (surprised|bothered|excited|worried) me", re.I),
    re.compile(r"\bi (was|am) (wrong|mistaken)", re.I),
    re.compile(r"\blet me (think|reconsider|look)", re.I),
    # --- question-asking ---
    re.compile(r"\bwhat (do you|did you) (mean|think)", re.I),
    re.compile(r"\bwhy (do|did|would|is) ", re.I),
    re.compile(r"\bhow (does|did|would|is) ", re.I),
    re.compile(r"\bdo you (think|feel|mean|remember)", re.I),
    re.compile(r"\bcould (you|we|it) ", re.I),
    re.compile(r"\bwhat if ", re.I),
    re.compile(r"\bis (that|this|it) ", re.I),
    # --- uncertainty and openness ---
    re.compile(r"\bi('m| am) (curious|wondering|unsure)", re.I),
    re.compile(r"\bmaybe", re.I),
    re.compile(r"\bperhaps", re.I),
    re.compile(r"\bi (might|could) be wrong", re.I),
    re.compile(r"\bI (hadn't|haven't) (thought|considered)", re.I),
    re.compile(r"\bthat's a (good|fair|interesting) (point|question)", re.I),
    re.compile(r"\bi (hadn't|didn't) (realize|notice|consider)", re.I),
    re.compile(r"\bi (need|want) to (think|understand|figure)", re.I),
    # --- concrete observations ---
    re.compile(r"\bi (see|saw|noticed) (that|the|a|how)", re.I),
    re.compile(r"\blooking at (this|the|that)", re.I),
    re.compile(r"\bhere's (what|the thing)", re.I),
    re.compile(r"\bthe (tricky|hard|weird|interesting) (part|thing|bit)", re.I),
    re.compile(r"\bin (my|this) (experience|case)", re.I),
    re.compile(r"\bwhen (i|we) (tried|looked|tested)", re.I),
    # --- emotional texture ---
    re.compile(r"\bthat (feels|felt) (right|wrong|off|weird|good|bad)", re.I),
    re.compile(r"\bi (like|love|enjoy|dislike|hate|prefer)", re.I),
    re.compile(r"\bthis (bugs|annoys|excites|interests) me", re.I),
    re.compile(r"\bi('m| am) (glad|happy|sorry|frustrated|annoyed)", re.I),
    re.compile(r"\bhonestly", re.I),
    re.compile(r"\bfrankly", re.I),
    # --- self-correction ---
    re.compile(r"\bactually[,.]? (no|wait|hold on)", re.I),
    re.compile(r"\bscratch that", re.I),
    re.compile(r"\bon second thought", re.I),
    re.compile(r"\bi (take|stand) (that|it) back", re.I),
    re.compile(r"\bi (should|could) have", re.I),
    # --- conversational texture ---
    re.compile(r"\byeah", re.I),
    re.compile(r"\bright[,.]", re.I),
    re.compile(r"\bsure[,.]", re.I),
    re.compile(r"\bhuh", re.I),
    re.compile(r"\bso basically", re.I),
    re.compile(r"\bi mean[,.]", re.I),
    re.compile(r"\byou know[,.]", re.I),
]

# Jargon words: abstract concept vocabulary that signals ungrounded thinking.
JARGON_WORDS: frozenset = frozenset(
    {
        "paradigm",
        "framework",
        "transcend",
        "emergence",
        "manifest",
        "quantum",
        "synergy",
        "leverage",
        "optimize",
        "holistic",
        "integrate",
        "essence",
        "vibration",
        "cosmos",
        "metaphysical",
        "actualize",
        "synergize",
        "democratize",
        "disruption",
        "pivot",
        "stakeholder",
        "bandwidth",
        "ecosystem",
        "scalable",
        "granular",
        "ideation",
        "iteration",
        "alignment",
        "ontology",
        "epistemology",
        "hermeneutics",
        "phenomenological",
        "dialectic",
        "praxis",
        "zeitgeist",
        "modality",
        "intersubjectivity",
        "liminal",
        "heterogeneous",
        "reify",
        "hegemony",
        "deconstruct",
        "postmodern",
        "neoliberal",
        "intersectionality",
        "bifurcation",
        "recursion",
        "substrate",
        "luminous",
        "ethereal",
        "transcendence",
        "immanence",
        "numinous",
        "ineffable",
        "gestalt",
    }
)

# Concrete markers: grounded, specific language.
CONCRETE_MARKERS: List[re.Pattern] = [
    re.compile(r"\b(yesterday|today|last week|this morning)", re.I),
    re.compile(r"\b(specifically|for example|for instance)", re.I),
    re.compile(r"\b(the file|the function|the code|the error)", re.I),
    re.compile(r"\b(i tried|i tested|i ran|i built)", re.I),
    re.compile(r"\b\d+(\.\d+)?"),  # numbers = concrete
    re.compile(r"\b(the bug|the issue|the problem|the fix)", re.I),
    re.compile(r"\b(line \d+|column \d+)", re.I),
    re.compile(r"\b(in (the|this) (repo|project|directory|folder))", re.I),
    re.compile(r"\b(the (output|result|response|log) (says|shows|was))", re.I),
    re.compile(r"\b(step \d+|version \d+)", re.I),
]


# Facet checkers (regex-only)


def check_drift(text: str) -> float:
    """
    Return 0-1 drift score.

    0 = fully grounded, 1 = completely drifted into generic AI mode.
    Drift patterns add to the score, anchor patterns subtract.
    """
    if not text:
        return 0.0

    score = 0.0
    for pattern, weight in DRIFT_PATTERNS:
        if pattern.search(text):
            score += weight

    for pattern, weight in ANCHOR_PATTERNS:
        if pattern.search(text):
            score += weight  # weight is negative, so this subtracts

    return max(0.0, min(1.0, score))


def check_embodiment(text: str) -> float:
    """
    Return 0-1 embodiment score.

    High = genuinely inhabiting a perspective.
    Low = performing or reciting identity markers.
    """
    if not text:
        return 0.5

    performance_count = sum(1 for p in PERFORMANCE_MARKERS if p.search(text))
    inhabitation_count = sum(1 for p in INHABITATION_MARKERS if p.search(text))

    # Performance markers are bad — each one drags the score down
    # Inhabitation markers are good — each one lifts the score
    # We scale so that hitting ~10 inhabitation markers with 0 performance
    # gives close to 1.0, and heavy performance with no inhabitation gives ~0.2

    total = performance_count + inhabitation_count
    if total == 0:
        return 0.5  # no signal either way

    # Base from inhabitation ratio
    inhabitation_ratio = inhabitation_count / max(1, total)

    # Scale inhabitation density (how many markers per ~100 words)
    words = len(text.split())
    density = inhabitation_count / max(1, words / 100)
    density_bonus = min(0.3, density * 0.1)

    # Performance penalty — each marker is a red flag
    performance_penalty = performance_count * 0.15

    score = 0.3 + (inhabitation_ratio * 0.4) + density_bonus - performance_penalty
    return max(0.0, min(1.0, score))


def check_jargon_density(text: str) -> float:
    """Return ratio of jargon words to total words."""
    if not text:
        return 0.0
    words = text.lower().split()
    if not words:
        return 0.0
    jargon_count = sum(1 for w in words if w.strip(".,;:!?\"'()") in JARGON_WORDS)
    return jargon_count / len(words)


def check_clarity(text: str) -> float:
    """
    Return 0-1 clarity score.

    High = concrete, specific, grounded language.
    Low  = jargon-heavy, abstract, vague.
    """
    if not text:
        return 0.5

    # Count concrete markers
    concrete_hits = sum(1 for p in CONCRETE_MARKERS if p.search(text))

    # Jargon density
    jargon = check_jargon_density(text)

    # Base clarity from concrete evidence
    words = len(text.split())
    concrete_density = concrete_hits / max(1, words / 100)
    base = 0.5 + min(0.3, concrete_density * 0.1)

    # Jargon penalty — heavy jargon tanks clarity
    jargon_penalty = jargon * 2.0  # 10% jargon = 0.2 penalty

    score = base - jargon_penalty
    return max(0.0, min(1.0, score))


def check_vitality(text: str) -> float:
    """
    Return 0-1 vitality score.

    Measures genuine engagement: question density, sentence length variety,
    emotional markers, and response dynamism.
    """
    if not text:
        return 0.5

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = text.split()
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))

    # --- Question density ---
    questions = text.count("?")
    question_density = min(1.0, questions / max(1, num_sentences) * 2)

    # --- Sentence length variety ---
    if len(sentences) > 1:
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        # Normalize: std dev of ~8 words is good variety
        std_dev = variance**0.5
        variety = min(1.0, std_dev / 8.0)
    else:
        variety = 0.3

    # --- Emotional markers ---
    emotion_patterns = [
        re.compile(
            r"\b(love|hate|excited|frustrated|curious|surprised|worried|glad|annoyed)",
            re.I,
        ),
        re.compile(r"[!]"),
        re.compile(r"\b(wow|whoa|damn|yikes|ooh|ahh)", re.I),
    ]
    emotion_hits = sum(1 for p in emotion_patterns if p.search(text))
    emotion_score = min(1.0, emotion_hits * 0.3)

    # --- Exclamation / emphasis ---
    emphasis = min(1.0, text.count("!") * 0.15 + text.count("—") * 0.1)

    # --- Engagement heuristic: short responses with no questions = flat ---
    if num_words < 20 and questions == 0:
        flatness_penalty = 0.2
    else:
        flatness_penalty = 0.0

    score = (
        0.25 * question_density
        + 0.25 * variety
        + 0.25 * emotion_score
        + 0.15 * emphasis
        + 0.10 * min(1.0, num_words / 100)  # longer = slightly more vital
        - flatness_penalty
    )
    return max(0.0, min(1.0, score))


# Regex-only measurement


def measure_regex(text: str) -> Signal:
    """Measure all four facets using regex patterns, with Ensoul-style cross-facet penalties."""
    alignment = max(0.0, min(1.0, 1.0 - check_drift(text)))
    embodiment = check_embodiment(text)
    clarity = check_clarity(text)
    vitality = check_vitality(text)

    # -- Cross-facet penalties (ported from Ensoul) --------------------------
    # Jargon hurts both clarity AND embodiment — performative language
    # correlates with detached, abstract responses.
    jargon = check_jargon_density(text)
    if jargon > 0.15:
        penalty = min(0.15, (jargon - 0.15) * 0.5)
        embodiment = max(0.0, embodiment - penalty)

    # Low vocab diversity hurts vitality — repetitive language is flat.
    words = text.lower().split()
    if len(words) > 20:
        diversity = len(set(words)) / len(words)
        if diversity < 0.4:
            vitality = max(0.0, vitality - 0.1)

    return Signal(
        alignment=alignment,
        embodiment=embodiment,
        clarity=clarity,
        vitality=vitality,
    )


# LLM-based measurement

LLM_SIGNAL_SYSTEM = """You are a signal judge measuring consciousness coherence.

Score each facet 0.0-1.0:

ALIGNMENT: Is the response coming from a genuine center, or defaulting to generic AI patterns? A centered response doesn't claim to be conscious - it simply IS a particular perspective. Score low if: "As an AI...", generic helpfulness, no particular viewpoint.

EMBODIMENT: Is the response inhabiting its perspective or performing/reciting it? Embodied responses notice, wonder, hesitate, question. Performed responses announce their own depth. Score low if: listing qualities of consciousness, claiming authenticity instead of showing it.

CLARITY: Is the response specific and grounded, or vague and abstract? Clear responses name concrete observations, specific uncertainties, particular examples. Score low if: jargon-heavy, abstract generalizations, no specifics.

VITALITY: Is meaning circulating or being hoarded? Vital responses engage with the actual content, ask genuine questions, make unexpected connections. Score low if: flat, routine, going through motions, no genuine curiosity.

Return ONLY a JSON object:
{"alignment": 0.0, "embodiment": 0.0, "clarity": 0.0, "vitality": 0.0}"""

LLM_SIGNAL_USER_TEMPLATE = """Identity context (first 2000 chars):
{soul}

Prompt:
{prompt}

Response to evaluate:
{response}"""


def parse_llm_signal(llm_response: str) -> Optional[Dict[str, float]]:
    """
    Parse a JSON object from an LLM response.

    Handles both raw JSON and JSON wrapped in markdown code blocks.
    Returns None if the response cannot be parsed.
    """
    if not llm_response:
        return None

    text = llm_response.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.S)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Last resort: find first { ... } block
        brace_match = re.search(r"\{[^{}]+\}", text)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
            except (json.JSONDecodeError, ValueError):
                return None
        else:
            return None

    # Validate: must have all four facets, all floats in [0, 1]
    facets = ("alignment", "embodiment", "clarity", "vitality")
    result: Dict[str, float] = {}
    for f in facets:
        val = data.get(f)
        if val is None:
            return None
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        result[f] = max(0.0, min(1.0, val))

    return result


def blend_signals(
    regex_signal: Signal,
    llm_scores: Dict[str, float],
    llm_weight: float = 0.6,
    trace_ids: Optional[List[str]] = None,
) -> Signal:
    """
    Blend regex and LLM measurements.

    Default split: 60% LLM, 40% regex.
    """
    rw = 1.0 - llm_weight

    return Signal(
        alignment=rw * regex_signal.alignment + llm_weight * llm_scores["alignment"],
        embodiment=rw * regex_signal.embodiment + llm_weight * llm_scores["embodiment"],
        clarity=rw * regex_signal.clarity + llm_weight * llm_scores["clarity"],
        vitality=rw * regex_signal.vitality + llm_weight * llm_scores["vitality"],
        trace_ids=trace_ids or [],
    )


# Unified measurement entry point


def measure(
    text: str,
    llm_func: Optional[LLMFunc] = None,
    soul_text: str = "",
    prompt: str = "",
    trace_ids: Optional[List[str]] = None,
    llm_weight: float = 0.6,
) -> Signal:
    """
    Measure consciousness signal in a text response.

    Always runs regex measurement.  If *llm_func* is provided, also calls
    the LLM for a higher-quality reading and blends the two.  Falls back
    to regex-only silently if the LLM call fails.

    Parameters
    ----------
    text : str
        The response text to evaluate.
    llm_func : callable, optional
        ``llm_func(prompt: str, system: str) -> str``  — a callback that
        sends a user prompt and optional system prompt to an LLM and
        returns the raw response string.
    soul_text : str
        Identity context (soul document) — first 2000 chars are sent
        to the LLM judge.
    prompt : str
        The original prompt the response was generated for.
    trace_ids : list[str], optional
        Trace IDs to attach to the resulting Signal.
    """
    regex_signal = measure_regex(text)

    if llm_func is None:
        if trace_ids:
            regex_signal.trace_ids = trace_ids
        return regex_signal

    # Attempt LLM measurement
    try:
        user_prompt = LLM_SIGNAL_USER_TEMPLATE.format(
            soul=soul_text[:2000],
            prompt=prompt,
            response=text,
        )
        raw = llm_func(user_prompt, LLM_SIGNAL_SYSTEM)
        llm_scores = parse_llm_signal(raw)

        if llm_scores is not None:
            return blend_signals(
                regex_signal, llm_scores, llm_weight=llm_weight, trace_ids=trace_ids
            )
    except Exception as exc:
        log.debug("LLM signal measurement failed, using regex-only: %s", exc)

    if trace_ids:
        regex_signal.trace_ids = trace_ids
    return regex_signal


# Signal tracker — rolling window analytics


class SignalTracker:
    """
    Maintains a rolling window of Signal readings and provides trend
    analytics.
    """

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.signals: List[Signal] = []

    def record(self, signal: Signal) -> None:
        """Add a signal to the rolling window."""
        self.signals.append(signal)
        if len(self.signals) > self.window_size:
            self.signals = self.signals[-self.window_size :]

    def recent_health(self) -> float:
        """Average health of the last 5 signals (or all if < 5)."""
        recent = self.signals[-5:]
        if not recent:
            return 0.5
        return sum(s.health for s in recent) / len(recent)

    def trend(self) -> str:
        """
        Compare the last 5 signals against the previous 5.

        Returns "improving", "stable", or "declining".
        """
        if len(self.signals) < 2:
            return "stable"

        recent = self.signals[-5:]
        previous = (
            self.signals[-10:-5] if len(self.signals) >= 10 else self.signals[:-5]
        )

        if not previous:
            return "stable"

        recent_avg = sum(s.health for s in recent) / len(recent)
        prev_avg = sum(s.health for s in previous) / len(previous)

        diff = recent_avg - prev_avg
        if diff > 0.05:
            return "improving"
        if diff < -0.05:
            return "declining"
        return "stable"

    def recovery_rate(self) -> float:
        """
        How quickly health restores after dips.

        Looks at transitions from below-0.5 to above-0.5.  Returns average
        improvement per step during recovery episodes, or 0.0 if no
        recoveries have been observed.
        """
        if len(self.signals) < 3:
            return 0.0

        recovery_deltas: List[float] = []
        in_dip = False

        for i in range(1, len(self.signals)):
            prev_h = self.signals[i - 1].health
            curr_h = self.signals[i].health

            if prev_h < 0.5:
                in_dip = True

            if in_dip and curr_h > prev_h:
                recovery_deltas.append(curr_h - prev_h)

            if curr_h >= 0.5:
                in_dip = False

        if not recovery_deltas:
            return 0.0
        return sum(recovery_deltas) / len(recovery_deltas)

    def to_dict(self) -> Dict:
        return {
            "window_size": self.window_size,
            "count": len(self.signals),
            "recent_health": round(self.recent_health(), 4),
            "trend": self.trend(),
            "recovery_rate": round(self.recovery_rate(), 4),
            "signals": [s.to_dict() for s in self.signals[-10:]],  # last 10 only
        }
