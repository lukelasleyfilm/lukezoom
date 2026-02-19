# Changelog

## 2.1.0-TURBO — 2026-02-19

Red-team hardening release. 4-agent adversarial audit (simulated gov-level attack),
7 critical/high fixes applied. 138/138 tests passing. Blockchain integrity chain regenerated.

### Red-Team Fixes (7 findings)

- **LZRT-001** [P0] Hash truncation → full SHA-256 in influence log chain (was 64-bit, now 256-bit)
- **LZRT-002** [P0] Unicode confusable bypass → NFKC normalization in identity resolver + semantic store
- **LZRT-003** [P0] Stranger data side-channel → emotional/introspection gated on `skip_persistence`
- **LZRT-004** [P0] Path traversal prefix bypass → trailing separator in journal `read_entry()`
- **LZRT-005** [P1] Untrusted personality/emotional load → clamp + type-check all values on `_load()`
- **LZRT-006** [P1] .bak files survive GDPR purge → deleted in influence, semantic, and system purge paths
- **LZRT-007** [P1] Purged data in WAL/FTS indexes → VACUUM + WAL checkpoint + FTS rebuild after episodic purge

### Audit Methodology

4 parallel red-team agents (Claude Opus 4.6) with distinct attack vectors:
1. Trust escalation & identity spoofing (10 findings)
2. Injection & deserialization (15 findings)
3. Crypto weaknesses, race conditions, DoS (14 findings)
4. Data leakage & GDPR evasion (15 findings)

54 total findings, 7 fixed in this release, remainder documented in RED_TEAM_AUDIT.md
with accept-risk rationale or future-release tracking.

---

## 2.0.0 — 2026-02-19

Full security audit, rebrand, and hardening release.
Audited by Claude Opus 4.6 (Anthropic). 138/138 tests passing.
Blockchain integrity chain: 67 files, SHA-256 verified.

### Security Fixes (11 findings)

- **LZSEC-001** Anchoring beliefs override → merged with defaults (never replaced)
- **LZSEC-002** GDPR purge incomplete → all 7 stores now purge with word-boundary matching
- **LZSEC-003** SQLite thread safety → `threading.Lock` + `check_same_thread=False`
- **LZSEC-004** Vacuous trust test → assertions verify stranger data excluded
- **LZSEC-005** FTS5 injection → `_sanitize_fts_query()` strips special chars
- **LZSEC-006** Swallowed exceptions → `HealthBitmap` tracks all subsystem failures
- **LZSEC-008** Influence log integrity → SHA-256 hash chain with `verify_chain()`
- **LZSEC-009** Filename length → `MAX_NAME_LENGTH = 200` in semantic store
- **LZSEC-010** Config budget validation → warns if budget shares don't sum to ~1.0
- **LZSEC-013** Phantom module imports → removed (consciousness.identity, consciousness.boot)
- **LZSEC-015** Unknown config keys → warning on unrecognized YAML keys

### Error Handling

- Replaced 56 fire-and-forget `except: pass` blocks with `HealthBitmap` tracking
- `HealthBitmap` added to `Context` and `AfterResult` — failures visible, never silent
- All exceptions logged at `WARNING` level minimum

### GDPR Compliance

- `purge_person()` on all 7 stores: episodic, semantic, influence, injury, journal, personality, emotional
- Word-boundary regex matching (`\b`) prevents false positives (purging "al" won't delete "alice")
- `deep_purge()` includes consolidation content redaction
- Influence log hash chain rebuilt after purge to maintain integrity

### Rebrand

- `engram.*` → `lukezoom.*` across 30 files (docstrings, tool names, config defaults)
- 29 trust gate tool names: `engram_*` → `lukezoom_*`
- Default data dir: `./engram_data` → `./lukezoom_data` (legacy fallback preserved)
- Env var: `ENGRAM_LLM_API_KEY` → `LUKEZOOM_LLM_API_KEY` (legacy fallback preserved)

### Phantom Code Removal

- Deleted TYPE_CHECKING imports for nonexistent `consciousness.identity`, `consciousness.boot`
- Removed 14 `relic_*` config parameters, `consciousness_dir`, `relic_dir` properties
- Removed `relic_assessment` from `AfterResult`
- Removed stale version "1.22" reference

### New Features

- `HealthBitmap` dataclass — structured subsystem health tracking per API call
- `InfluenceLog.verify_chain()` — integrity verification for append-only safety log
- `PersonalitySystem.mean_revert_all()` — periodic personality drift correction
- `MemorySystem.full_maintenance()` — single call for all maintenance tasks
- `verify_integrity.py` — standalone blockchain integrity verification (zero dependencies)
- `INTEGRITY_CHAIN.json` — SHA-256 chain of all 67 source/test/config files

### Tests

- 138 tests (up from ~120)
- Fixed vacuous assertions in `test_integration.py` and `test_safety.py`
- Added `TestPurgeNewStores` — GDPR purge tests for all 5 newly-purge-capable stores
- Added `TestInjuryPersistence` — disk persistence and lifecycle transition tests

### Clinical Grounding

LukeZOOM's psychiatric concepts are grounded in the lived experience of its author
(bipolar type 1) and map to clinical domains practiced at CPMC / Sutter Health SF:

| System Concept | Clinical Parallel |
|---------------|-------------------|
| Signal states: coherent → drifting → dissociated | Mood episode progression |
| Injury lifecycle: fresh → processing → healing → healed | Episode recovery stages |
| 5 anchoring beliefs (injury.py) | CBT core beliefs for identity stability |
| Trust tiers with graduated access | Capacity evaluation (forensic psychiatry) |
| 4-facet identity coherence signal | Mental status exam domains |
| Emotional VAD with exponential decay | Affect regulation / return-to-baseline |

---

## 1.17.0

Initial public release. 2-function API, 5-tier trust gate, 4-layer memory,
consciousness signal tracking. 1 dependency (pyyaml).
