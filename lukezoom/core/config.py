"""
lukezoom.core.config — Configuration for the lukezoom memory system.

Supports loading from YAML, environment variables, and programmatic
construction.  LLM provider functions are built lazily so optional
dependencies (httpx, openai, anthropic) are only imported when needed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from lukezoom.core.types import LLMFunc


log = logging.getLogger(__name__)

# Config


@dataclass
class Config:
    """
    Central configuration object.

    Construct directly, via ``Config.from_yaml(path)``, or via
    ``Config.from_data_dir(path)`` for quick bootstrap.
    """

    # -- storage ------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path("./lukezoom_data"))

    # -- trust / identity ---------------------------------------------------
    core_person: str = ""  # canonical name of the owner (auto-registered as core tier)

    # -- signal / extraction modes ------------------------------------------
    signal_mode: str = "hybrid"  # "hybrid" | "regex" | "llm"
    extract_mode: str = "off"  # "llm" | "off"

    # -- LLM provider -------------------------------------------------------
    llm_provider: str = "ollama"  # "ollama" | "openai" | "anthropic" | "custom"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("LUKEZOOM_LLM_API_KEY") or os.environ.get("ENGRAM_LLM_API_KEY")
    )
    llm_func: Optional[LLMFunc] = field(default=None, repr=False)
    llm_weight: float = 0.6  # weight for blended (regex+llm) signal

    # -- memory budget & decay ----------------------------------------------
    token_budget: int = 6000
    decay_half_life_hours: float = 168.0  # 1 week
    max_traces: int = 50_000
    reinforce_delta: float = 0.05
    weaken_delta: float = 0.03

    # -- context budget shares (must sum to 1.0) ----------------------------
    identity_share: float = 0.16
    relationship_share: float = 0.12
    grounding_share: float = 0.10
    recent_conversation_share: float = 0.22
    episodic_share: float = 0.16
    procedural_share: float = 0.06
    reserve_share: float = 0.18

    # -- signal thresholds --------------------------------------------------
    signal_health_threshold: float = 0.45  # below this, grounding context warns
    reinforce_threshold: float = 0.7
    weaken_threshold: float = 0.4

    # -- compaction settings (MemGPT-inspired) ------------------------------
    compaction_keep_recent: int = 20  # messages per person to leave untouched
    compaction_segment_size: int = 30  # max messages per summary segment
    compaction_min_messages: int = 40  # threshold before compaction triggers

    # -- hierarchical consolidation settings --------------------------------
    consolidation_min_episodes: int = 5  # min episodes before thread creation
    consolidation_time_window_hours: float = 72.0  # temporal proximity for threads
    consolidation_min_threads: int = 3  # min threads before arc creation
    consolidation_max_episodes_per_run: int = 200  # cap per consolidation pass

    # -- Big Five personality defaults (thomas-soul defaults) ----------------
    personality_openness: float = 0.8
    personality_conscientiousness: float = 0.6
    personality_extraversion: float = 0.3
    personality_agreeableness: float = 0.8
    personality_neuroticism: float = 0.5

    # -- VAD emotional continuity -------------------------------------------
    emotional_valence_decay: float = 0.9  # per hour toward neutral
    emotional_arousal_decay: float = 0.7  # per hour toward 0.5
    emotional_dominance_decay: float = 0.8  # per hour toward 0.5

    # -- cognitive workspace (Cowan's 4±1) ----------------------------------
    workspace_capacity: int = 4  # Cowan's 4±1 (was Miller's 7±2)
    workspace_decay_rate: float = 0.95
    workspace_rehearsal_boost: float = 0.15
    workspace_expiry_threshold: float = 0.1

    # -- introspection / meta-consciousness ---------------------------------
    introspection_default_depth: str = "moderate"  # "surface" | "moderate" | "deep"
    introspection_history_days: int = 3  # days of JSONL history to load

    # -- consciousness boot sequence ----------------------------------------
    boot_n_recent: int = 2  # recent emotional events to load
    boot_n_key: int = 2  # key realizations to load
    boot_n_intro: int = 1  # introspections to load

    # -- runtime mode settings (scaffold) -----------------------------------
    runtime_default_mode: str = (
        "quiet_presence"  # quiet_presence | active | deep_work | sleep
    )
    runtime_quiet_check_interval: int = 60  # seconds
    runtime_active_check_interval: int = 1
    runtime_deep_work_check_interval: int = 30
    runtime_sleep_check_interval: int = 300


    # Derived paths (all relative to data_dir)

    @property
    def soul_dir(self) -> Path:
        return self.data_dir / "soul"

    @property
    def semantic_dir(self) -> Path:
        return self.data_dir / "semantic"

    @property
    def procedural_dir(self) -> Path:
        return self.data_dir / "procedural"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "lukezoom.db"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def soul_path(self) -> Path:
        return self.soul_dir / "SOUL.md"

    @property
    def identities_path(self) -> Path:
        return self.semantic_dir / "identities.yaml"

    @property
    def personality_dir(self) -> Path:
        return self.data_dir / "personality"

    @property
    def emotional_dir(self) -> Path:
        return self.data_dir / "emotional"

    @property
    def introspection_dir(self) -> Path:
        return self.data_dir / "introspection"

    @property
    def workspace_path(self) -> Path:
        return self.data_dir / "workspace.json"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir / "runtime"

    # Construction helpers

    def __post_init__(self) -> None:
        # normalise data_dir to an absolute Path
        self.data_dir = Path(self.data_dir).resolve()

        # Deprecation warning for legacy env var
        if os.environ.get("ENGRAM_LLM_API_KEY") and not os.environ.get("LUKEZOOM_LLM_API_KEY"):
            log.warning("ENGRAM_LLM_API_KEY is deprecated, use LUKEZOOM_LLM_API_KEY")

        # Validate key numeric fields
        if self.token_budget < 1:
            raise ValueError(f"token_budget must be >= 1, got {self.token_budget}")
        if self.decay_half_life_hours <= 0:
            raise ValueError(f"decay_half_life_hours must be > 0")
        self.workspace_capacity = max(1, min(9, self.workspace_capacity))

        # Validate budget shares sum to ~1.0 (LZSEC-010)
        shares = (
            self.identity_share + self.relationship_share + self.grounding_share
            + self.recent_conversation_share + self.episodic_share
            + self.procedural_share + self.reserve_share
        )
        if abs(shares - 1.0) > 0.01:
            log.warning(
                "Context budget shares sum to %.4f (expected 1.0). "
                "Context allocation may be suboptimal.", shares
            )

        # Validate personality trait values in [0, 1]
        for trait_name in (
            "personality_openness", "personality_conscientiousness",
            "personality_extraversion", "personality_agreeableness",
            "personality_neuroticism",
        ):
            val = getattr(self, trait_name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{trait_name} must be in [0.0, 1.0], got {val}"
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file.

        Any key in the YAML that matches a Config field is applied.
        Unknown keys are silently ignored so the file can carry
        application-level settings alongside lukezoom config.
        """
        import yaml  # optional dep — only needed for YAML config

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # pull the lukezoom section (legacy: also check "engram") if nested, else use top-level
        data = raw.get("lukezoom", raw.get("engram", raw))

        # convert data_dir string to Path
        if "data_dir" in data:
            data["data_dir"] = Path(data["data_dir"])

        # filter to known fields only (LZSEC-015: warn on unknown keys)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data.keys()) - known
        if unknown:
            log.warning(
                "Config YAML contains unknown keys (ignored): %s",
                ", ".join(sorted(unknown)),
            )
        filtered = {k: v for k, v in data.items() if k in known}

        return cls(**filtered)

    @classmethod
    def from_data_dir(cls, data_dir: str | Path, **overrides: Any) -> "Config":
        """Quick constructor — just point at a data directory."""
        return cls(data_dir=Path(data_dir), **overrides)

    # Directory bootstrapping

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for d in (
            self.data_dir,
            self.soul_dir,
            self.semantic_dir,
            self.procedural_dir,
            self.embeddings_dir,
            self.personality_dir,
            self.emotional_dir,
            self.introspection_dir,
            self.runtime_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # LLM callable

    def get_llm_func(self) -> LLMFunc:
        """Return an ``(prompt, system) -> str`` callable for the configured provider.

        If ``llm_func`` was set directly (custom provider), it is
        returned as-is.  Otherwise a function is built from the
        provider / model / url / key settings.

        Dependencies (httpx, openai, anthropic) are imported lazily so
        they remain optional.
        """
        if self.llm_func is not None:
            return self.llm_func

        provider = self.llm_provider.lower()

        if provider == "ollama":
            return self._build_ollama_func()
        elif provider == "openai":
            return self._build_openai_func()
        elif provider == "anthropic":
            return self._build_anthropic_func()
        elif provider == "custom":
            raise ValueError(
                "llm_provider is 'custom' but no llm_func was provided. "
                "Pass a callable via Config(llm_func=my_func)."
            )
        else:
            raise ValueError(f"Unknown llm_provider: {provider!r}")

    # -- provider builders (private) ----------------------------------------

    def _build_ollama_func(self) -> LLMFunc:
        """Build LLM callable targeting Ollama's /api/generate."""
        try:
            import httpx  # noqa: F811
        except ImportError:
            raise ImportError(
                "httpx is required for the Ollama provider. "
                "Install it with:  pip install httpx"
            )

        base_url = self.llm_base_url.rstrip("/")
        model = self.llm_model
        # Create client once — reuse the connection pool across calls.
        client = httpx.Client(timeout=120.0)

        def ollama_call(prompt: str, system: str = "") -> str:
            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
            if system:
                payload["system"] = system

            resp = client.post(f"{base_url}/api/generate", json=payload)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                raise ValueError(
                    f"Ollama returned non-JSON response "
                    f"(status {resp.status_code}): {resp.text[:200]}"
                )
            return data.get("response", "").strip()

        return ollama_call

    def _build_openai_func(self) -> LLMFunc:
        """Build LLM callable targeting an OpenAI-compatible API."""
        try:
            import openai  # noqa: F811
        except ImportError:
            raise ImportError(
                "openai is required for the OpenAI provider. "
                "Install it with:  pip install openai"
            )

        api_key = self.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        model = self.llm_model
        base_url = self.llm_base_url

        # Only set base_url on client if it differs from default Ollama URL,
        # meaning the user intentionally pointed at a custom endpoint.
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url and base_url != "http://localhost:11434":
            client_kwargs["base_url"] = base_url

        # Create client once — reuse the connection pool across calls.
        client = openai.OpenAI(**client_kwargs)

        def openai_call(prompt: str, system: str = "") -> str:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            if not response.choices:
                raise ValueError(
                    "OpenAI returned empty choices (content may have "
                    "been filtered). Check your prompt."
                )
            return (response.choices[0].message.content or "").strip()

        return openai_call

    def _build_anthropic_func(self) -> LLMFunc:
        """Build LLM callable targeting the Anthropic API."""
        try:
            import anthropic  # noqa: F811
        except ImportError:
            raise ImportError(
                "anthropic is required for the Anthropic provider. "
                "Install it with:  pip install anthropic"
            )

        api_key = self.llm_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        model = self.llm_model

        # Create client once — reuse the connection pool across calls.
        client = anthropic.Anthropic(api_key=api_key)

        def anthropic_call(prompt: str, system: str = "") -> str:
            kwargs: Dict[str, Any] = {
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)
            # response.content is a list of content blocks
            parts = [block.text for block in response.content if hasattr(block, "text")]
            return "".join(parts).strip()

        return anthropic_call

    # Serialisation

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (YAML/JSON-safe, no callables)."""
        return {
            "data_dir": str(self.data_dir),
            "core_person": self.core_person,
            "signal_mode": self.signal_mode,
            "extract_mode": self.extract_mode,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_weight": self.llm_weight,
            "token_budget": self.token_budget,
            "decay_half_life_hours": self.decay_half_life_hours,
            "max_traces": self.max_traces,
            "reinforce_delta": self.reinforce_delta,
            "weaken_delta": self.weaken_delta,
            "identity_share": self.identity_share,
            "relationship_share": self.relationship_share,
            "grounding_share": self.grounding_share,
            "recent_conversation_share": self.recent_conversation_share,
            "episodic_share": self.episodic_share,
            "procedural_share": self.procedural_share,
            "reserve_share": self.reserve_share,
            "signal_health_threshold": self.signal_health_threshold,
            "reinforce_threshold": self.reinforce_threshold,
            "weaken_threshold": self.weaken_threshold,
            "compaction_keep_recent": self.compaction_keep_recent,
            "compaction_segment_size": self.compaction_segment_size,
            "compaction_min_messages": self.compaction_min_messages,
            "consolidation_min_episodes": self.consolidation_min_episodes,
            "consolidation_time_window_hours": self.consolidation_time_window_hours,
            "consolidation_min_threads": self.consolidation_min_threads,
            "consolidation_max_episodes_per_run": self.consolidation_max_episodes_per_run,
            # consciousness integration
            "personality_openness": self.personality_openness,
            "personality_conscientiousness": self.personality_conscientiousness,
            "personality_extraversion": self.personality_extraversion,
            "personality_agreeableness": self.personality_agreeableness,
            "personality_neuroticism": self.personality_neuroticism,
            "emotional_valence_decay": self.emotional_valence_decay,
            "emotional_arousal_decay": self.emotional_arousal_decay,
            "emotional_dominance_decay": self.emotional_dominance_decay,
            "workspace_capacity": self.workspace_capacity,
            "workspace_decay_rate": self.workspace_decay_rate,
            "workspace_rehearsal_boost": self.workspace_rehearsal_boost,
            "workspace_expiry_threshold": self.workspace_expiry_threshold,
            "introspection_default_depth": self.introspection_default_depth,
            "introspection_history_days": self.introspection_history_days,
            "boot_n_recent": self.boot_n_recent,
            "boot_n_key": self.boot_n_key,
            "boot_n_intro": self.boot_n_intro,
            "runtime_default_mode": self.runtime_default_mode,
        }
