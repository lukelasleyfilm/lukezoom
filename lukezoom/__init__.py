"""
LukeZOOM 2.0 — Trust-gated cognitive memory for persistent AI agents.

2-function API: before() → Context, after() → AfterResult
Zero-LLM runtime. 1 dependency (pyyaml). MIT License.

    from lukezoom import Engine, Config
    engine = Engine(Config(core_person="luke"))
    ctx = engine.before(person="aidan", message="Hello")
    result = engine.after(person="aidan", their_message="Hello", response="Hi!")

Also available as MemorySystem (1.3 compat).
"""

__version__ = "2.1.0-TURBO"

from lukezoom.core.types import (
    AfterResult, Context, HealthBitmap, Signal, Trace, Message, MemoryStats, LLMFunc,
)
from lukezoom.core.config import Config
from lukezoom.trust import AccessPolicy, Tier, TrustGate
from lukezoom.system import MemorySystem, Engine, EngineBuilder

__all__ = [
    "Engine", "EngineBuilder", "MemorySystem", "Config",
    "Context", "AfterResult", "HealthBitmap", "Signal", "Trace", "Message", "MemoryStats",
    "Tier", "TrustGate", "AccessPolicy",
    "LLMFunc",
]
