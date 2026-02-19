#!/usr/bin/env python3
"""
LukeZOOM 2.1.0-TURBO — Third-Party Integrity Verification Script

Usage:
    python3 verify_integrity.py

Reads INTEGRITY_CHAIN.json and re-computes every hash from source files.
Reports any tampering, missing files, or chain breaks.
"""

import hashlib
import json
import sys
from pathlib import Path


def verify():
    chain_path = Path("INTEGRITY_CHAIN.json")
    if not chain_path.exists():
        print("ERROR: INTEGRITY_CHAIN.json not found")
        return False

    audit = json.loads(chain_path.read_text(encoding="utf-8"))
    chain = audit["chain"]
    meta = audit["audit_metadata"]

    print(f"=== Verifying LukeZOOM {meta['version']} Integrity Chain ===")
    print(f"Original audit: {meta['audit_timestamp']}")
    print(f"Original auditor: {meta['auditor']}")
    print(f"Chain length: {len(chain)} blocks")
    print()

    errors = []
    prev_hash = "genesis"

    for block in chain:
        idx = block["index"]
        fpath = Path(block["file"])

        # Check file exists
        if not fpath.exists():
            errors.append(f"[{idx}] MISSING: {fpath}")
            prev_hash = block["block_hash"]
            continue

        # Verify file hash
        content = fpath.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != block["sha256"]:
            errors.append(
                f"[{idx}] TAMPERED: {fpath}\n"
                f"       Expected: {block['sha256']}\n"
                f"       Actual:   {actual_hash}"
            )

        # Verify chain linkage
        if block["prev_hash"] != prev_hash:
            errors.append(
                f"[{idx}] CHAIN BREAK at {fpath}\n"
                f"       Expected prev: {prev_hash}\n"
                f"       Recorded prev: {block['prev_hash']}"
            )

        # Verify block hash
        chain_input = (block["prev_hash"] + block["sha256"]).encode("utf-8")
        expected_block = hashlib.sha256(chain_input).hexdigest()
        if expected_block != block["block_hash"]:
            errors.append(f"[{idx}] BLOCK HASH MISMATCH: {fpath}")

        prev_hash = block["block_hash"]

    # Verify Merkle root
    all_hashes = "".join(b["sha256"] for b in chain)
    merkle = hashlib.sha256(all_hashes.encode("utf-8")).hexdigest()
    if merkle != meta["merkle_root"]:
        errors.append(
            f"MERKLE ROOT MISMATCH\n"
            f"  Expected: {meta['merkle_root']}\n"
            f"  Actual:   {merkle}"
        )

    # Report
    if errors:
        print(f"FAILED — {len(errors)} integrity violation(s):\n")
        for e in errors:
            print(f"  {e}\n")
        return False
    else:
        print(f"PASSED — All {len(chain)} files verified.")
        print(f"Merkle Root: {merkle}")
        print(f"Final Block: {chain[-1]['block_hash']}")
        return True


if __name__ == "__main__":
    ok = verify()
    sys.exit(0 if ok else 1)
