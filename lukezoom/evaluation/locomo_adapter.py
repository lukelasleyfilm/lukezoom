"""
LOCOMO benchmark adapter for lukezoom.

LOCOMO (Long-form COnversation MeMOry) is the primary benchmark used
by MIRIX (85.4%), Mem0, and other agent memory systems.  This adapter
allows lukezoom to be evaluated against the same dataset for direct
comparison.

Dataset format (LOCOMO v1)::

    [
        {
            "conversation_id": "conv_001",
            "turns": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ],
            "questions": [
                {
                    "question_id": "q_001",
                    "question": "What is the user's favorite color?",
                    "answer": "blue",
                    "category": "single-hop" | "multi-hop" | "temporal" | "open"
                },
                ...
            ]
        },
        ...
    ]

Usage::

    adapter = LOCOMOAdapter(memory_system)
    results = adapter.evaluate("locomo_v1.json")
    print(f"Overall accuracy: {results['accuracy']:.1%}")
    print(f"Category breakdown: {results['by_category']}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class LOCOMOResult:
    """Results from a LOCOMO evaluation run."""

    total_questions: int = 0
    correct: int = 0
    accuracy: float = 0.0
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    conversations_processed: int = 0
    errors: int = 0


class LOCOMOAdapter:
    """
    Adapts lukezoom's memory system to the LOCOMO benchmark protocol.

    The adapter:
    1. Ingests LOCOMO conversation turns via ``memory.after()``
    2. For each question, assembles context via ``memory.before()``
    3. Passes context + question to an answer generator
    4. Compares generated answer against ground truth

    Parameters
    ----------
    memory_system : MemorySystem
        A live lukezoom MemorySystem instance.
    answer_fn : callable, optional
        Function that takes ``(question: str, context: str) -> str``.
        If None, uses a simple substring matching heuristic for
        offline evaluation (no LLM required).
    person_id : str
        Identity used for memory operations (default "locomo_user").
    token_budget : int
        Token budget for context assembly (default 6000).
    """

    def __init__(
        self,
        memory_system,
        answer_fn: Optional[Callable[[str, str], str]] = None,
        person_id: str = "locomo_user",
        token_budget: int = 6000,
    ):
        self._memory = memory_system
        self._answer_fn = answer_fn or self._heuristic_answer
        self._person = person_id
        self._budget = token_budget

    def evaluate(
        self,
        dataset_path: str,
        max_conversations: Optional[int] = None,
    ) -> LOCOMOResult:
        """
        Run full LOCOMO evaluation.

        Parameters
        ----------
        dataset_path : str
            Path to LOCOMO JSON file.
        max_conversations : int, optional
            Limit number of conversations to process (for quick testing).

        Returns
        -------
        LOCOMOResult
        """
        data = self._load_dataset(dataset_path)
        if max_conversations:
            data = data[:max_conversations]

        total = 0
        correct = 0
        category_stats: Dict[str, Dict[str, int]] = {}
        errors = 0

        for conv in data:
            try:
                self._ingest_conversation(conv["turns"])
                for q in conv.get("questions", []):
                    total += 1
                    cat = q.get("category", "unknown")

                    if cat not in category_stats:
                        category_stats[cat] = {"total": 0, "correct": 0}
                    category_stats[cat]["total"] += 1

                    predicted = self._answer_question(q["question"])
                    if self._check_answer(predicted, q["answer"]):
                        correct += 1
                        category_stats[cat]["correct"] += 1
            except Exception as e:
                logger.warning("Error processing conversation: %s", e)
                errors += 1
            finally:
                # Clear memory between conversations for fair eval
                self._reset_memory()

        # Compute per-category accuracy
        by_category = {}
        for cat, stats in category_stats.items():
            by_category[cat] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(
                    stats["correct"] / stats["total"]
                    if stats["total"] > 0
                    else 0.0,
                    4,
                ),
            }

        return LOCOMOResult(
            total_questions=total,
            correct=correct,
            accuracy=round(correct / total if total > 0 else 0.0, 4),
            by_category=by_category,
            conversations_processed=len(data),
            errors=errors,
        )

    # Internal methods

    def _load_dataset(self, path: str) -> List[Dict]:
        """Load LOCOMO dataset from JSON."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"LOCOMO dataset not found: {path}\n"
                f"Download from: https://github.com/snap-research/locomo"
            )
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ingest_conversation(self, turns: List[Dict]) -> None:
        """Feed conversation turns into lukezoom's memory."""
        for i in range(0, len(turns) - 1, 2):
            user_turn = turns[i]
            assistant_turn = turns[i + 1] if i + 1 < len(turns) else None

            if assistant_turn:
                # Run full pipeline: before → after
                ctx = self._memory.before(
                    person=self._person,
                    message=user_turn["content"],
                    token_budget=self._budget,
                )
                self._memory.after(
                    person=self._person,
                    their_message=user_turn["content"],
                    response=assistant_turn["content"],
                    trace_ids=ctx.trace_ids,
                )

    def _answer_question(self, question: str) -> str:
        """Assemble context and generate answer."""
        ctx = self._memory.before(
            person=self._person,
            message=question,
            token_budget=self._budget,
        )
        context_text = ctx.grounding if hasattr(ctx, "grounding") else ""
        return self._answer_fn(question, context_text)

    def _reset_memory(self) -> None:
        """Clear memory state between conversations."""
        try:
            if hasattr(self._memory, "purge_person"):
                self._memory.purge_person(
                    person=self._person, confirm="CONFIRM_PURGE"
                )
        except Exception:
            pass

    @staticmethod
    def _heuristic_answer(question: str, context: str) -> str:
        """
        Simple heuristic answer extractor (no LLM required).

        Scans context for sentences containing question keywords and
        returns the best-matching sentence.  This produces lower scores
        than an LLM-backed answer_fn but enables offline evaluation.
        """
        if not context:
            return ""

        # Extract content words from question
        stop_words = {
            "what", "when", "where", "who", "how", "why", "is", "are",
            "was", "were", "do", "does", "did", "the", "a", "an", "in",
            "of", "to", "for", "with", "on", "at", "by", "from", "that",
            "this", "it", "its", "my", "your", "his", "her", "their",
        }
        q_words = {
            w.lower().strip("?.,!") for w in question.split()
        } - stop_words

        # Score each sentence
        sentences = context.replace("\n", " ").split(".")
        best_score = 0
        best_sent = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            s_words = {w.lower() for w in sent.split()}
            overlap = len(q_words & s_words)
            if overlap > best_score:
                best_score = overlap
                best_sent = sent

        return best_sent

    @staticmethod
    def _check_answer(predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.

        Uses case-insensitive substring matching — consistent with
        LOCOMO evaluation protocol used by MIRIX.
        """
        pred_lower = predicted.lower().strip()
        truth_lower = ground_truth.lower().strip()

        if not pred_lower or not truth_lower:
            return False

        # Exact or substring match
        return truth_lower in pred_lower or pred_lower in truth_lower
