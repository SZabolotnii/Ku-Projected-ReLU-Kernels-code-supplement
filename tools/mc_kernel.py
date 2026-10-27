#!/usr/bin/env python3
"""Artifact-first Monte-Carlo / ablation runner with a persistent namespace.

The deterministic half of a long experiment sweep. TRACE L1: there is no model
call anywhere in this file, and none in the loop it drives. The runner decides
nothing about the science — it executes a grid a human wrote, writes every cell
to disk before it is allowed to exist in memory, and stops when a control edge
fails.

Three properties it exists to provide:

  artifact-first  Every cell is written to results/<run-id>/<cell>.json and
                  hashed into MANIFEST.sha256 BEFORE its value is cached. A
                  number that reached a manuscript can always be traced to a
                  file, which is what tools/coe_audit.py asks for. A value held
                  only in a kernel namespace is a cache, never a record.

  resumable       A cell whose artifact exists and whose hash checks out is not
                  recomputed. This is the whole reason for a persistent
                  namespace: an interrupted 4-hour sweep resumes where it
                  stopped, and a shared sub-computation (a 3M-sample marginal
                  simulation, say) is paid for once per run, not once per cell.

  self-halting    Controls run FIRST. If a control edge misses its expected
                  value the run aborts with exit 2 and the grid is not touched.
                  A sweep that confidently produces numbers while its own
                  self-test is broken is more dangerous than one that stops.

A spec module (passed as a path) defines the experiment:

    RUN_ID   : str                      identifies the results directory
    CONTROLS : list[dict]               cells that must pass, each with
                                        'expect': {key: (value, tol)}
    GRID     : list[dict]               the cells to sweep
    cell(params: dict, memo: dict) -> dict[str, float]

`memo` is the persistent namespace, handed to every cell in run order. A cell
may cache anything expensive in it under its own key; the runner never inspects
it and never persists it. Cache in memo, record on disk.

Usage:
    python tools/mc_kernel.py <spec.py> [--resume] [--only <substr>]
                                       [--prune] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class ControlFailure(RuntimeError):
    """A control edge missed its expected value. The grid does not run."""


def load_spec(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import spec {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    for attr in ("RUN_ID", "CONTROLS", "GRID", "cell"):
        if not hasattr(mod, attr):
            raise RuntimeError(f"spec {path.name} defines no {attr}")
    return mod


def cell_id(params: dict) -> str:
    """Stable identifier for a cell, from its parameters alone."""
    canon = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canon.encode()).hexdigest()[:8]
    parts = [f"{k}={params[k]}" for k in sorted(params)]
    stem = "_".join(parts).replace("/", "-").replace(" ", "")
    return f"{stem[:80]}__{digest}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_sha(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — not a git repo, no git, timeout
        return "unknown"


def write_artifact(results: Path, cid: str, params: dict, values: dict,
                   seconds: float) -> Path:
    """Write the cell, then hash it into MANIFEST.sha256. Disk before memory."""
    art = results / f"{cid}.json"
    art.write_text(json.dumps({
        "cell": cid,
        "params": params,
        "values": values,
        "seconds": round(seconds, 3),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, sort_keys=True) + "\n")
    manifest = results / "MANIFEST.sha256"
    line = f"{sha256_file(art)}  {art.name}\n"
    prior = manifest.read_text() if manifest.exists() else ""
    kept = [ln for ln in prior.splitlines(keepends=True)
            if not ln.rstrip().endswith(art.name)]
    manifest.write_text("".join(kept) + line)
    return art


def read_artifact(results: Path, cid: str) -> dict | None:
    """Return a cell's stored values, but only if its hash still checks out."""
    art = results / f"{cid}.json"
    manifest = results / "MANIFEST.sha256"
    if not art.exists() or not manifest.exists():
        return None
    recorded = {ln.split()[1]: ln.split()[0]
                for ln in manifest.read_text().splitlines() if ln.strip()}
    if recorded.get(art.name) != sha256_file(art):
        return None
    return json.loads(art.read_text())["values"]


def check_control(control: dict, values: dict) -> list[str]:
    misses = []
    for key, (want, tol) in control["expect"].items():
        if key not in values:
            misses.append(f"{key}: not returned by cell()")
            continue
        got = values[key]
        if abs(got - want) > tol:
            misses.append(f"{key}: got {got:.6g}, want {want:.6g} ± {tol:g}")
    return misses


def prune(results: Path, keep: set[str]) -> list[str]:
    """Drop artifacts no longer named by the grid, and their manifest lines.

    A cell removed from the spec (a validity predicate added, a range narrowed)
    leaves a stale file behind. Left in place it is indistinguishable from live
    evidence, so the committed artifact set would no longer match the grid that
    produced it.
    """
    manifest = results / "MANIFEST.sha256"
    dropped = []
    for art in sorted(results.glob("*.json")):
        if art.name == "RUN.json" or art.stem in keep:
            continue
        art.unlink()
        dropped.append(art.name)
    if dropped and manifest.exists():
        kept = [ln for ln in manifest.read_text().splitlines(keepends=True)
                if ln.strip() and ln.split()[1] not in dropped]
        manifest.write_text("".join(kept))
    return dropped


def run(spec_path: Path, resume: bool, only: str | None, dry_run: bool,
        do_prune: bool = False) -> int:
    mod = load_spec(spec_path)
    results = spec_path.parent / "results" / mod.RUN_ID
    results.mkdir(parents=True, exist_ok=True)
    memo: dict = {}

    grid = [c for c in mod.GRID
            if only is None or only in json.dumps(c, sort_keys=True)]
    print(f"spec     {spec_path}")
    print(f"results  {results}")
    print(f"controls {len(mod.CONTROLS)}   grid {len(grid)}"
          f"{'' if only is None else f' (filtered from {len(mod.GRID)})'}")
    if dry_run:
        for c in mod.CONTROLS + grid:
            print("  ", cell_id(c["params"] if "params" in c else c))
        return 0

    # --- L1 control edges. Nothing else runs until these pass. ---------------
    t_controls = time.time()
    for ctl in mod.CONTROLS:
        params = ctl["params"]
        cid = cell_id(params)
        t0 = time.time()
        values = mod.cell(params, memo)
        dt = time.time() - t0
        write_artifact(results, cid, params, values, dt)
        misses = check_control(ctl, values)
        label = ctl.get("name", cid)
        if misses:
            print(f"\nCONTROL FAILED  {label}")
            for m in misses:
                print(f"  {m}")
            print("\nEscalating to L4. The grid was not run: a sweep whose "
                  "self-test is broken must not produce numbers.")
            return 2
        print(f"  control ok   {label}  ({dt:.1f}s)")
    print(f"controls passed in {time.time() - t_controls:.1f}s\n")

    # --- the grid ------------------------------------------------------------
    rows, reused, computed, t_grid = [], 0, 0, time.time()
    for i, params in enumerate(grid, 1):
        cid = cell_id(params)
        cached = read_artifact(results, cid) if resume else None
        if cached is not None:
            values, dt, mark = cached, 0.0, "="
            reused += 1
        else:
            t0 = time.time()
            values = mod.cell(params, memo)
            dt = time.time() - t0
            write_artifact(results, cid, params, values, dt)
            computed += 1
            mark = " "
        rows.append({"params": params, "values": values})
        flat = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in values.items())
        head = "  ".join(f"{v}" for _, v in sorted(params.items()))
        print(f"{mark}[{i:3d}/{len(grid)}] {head:38s} {flat}   {dt:5.1f}s")

    elapsed = time.time() - t_grid
    if do_prune:
        live = {cell_id(c) for c in mod.GRID} | {cell_id(c["params"]) for c in mod.CONTROLS}
        dropped = prune(results, live)
        print(f"\npruned {len(dropped)} artifact(s) no longer in the grid")

    run_json = results / "RUN.json"
    run_json.write_text(json.dumps({
        "run_id": mod.RUN_ID,
        "spec": str(spec_path),
        "spec_git": git_sha(spec_path.parent),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "cells_total": len(grid),
        "cells_computed": computed,
        "cells_reused": reused,
        "grid_seconds": round(elapsed, 1),
        "controls": [c.get("name", "") for c in mod.CONTROLS],
    }, indent=2) + "\n")

    print(f"\n{computed} computed, {reused} reused, {elapsed:.1f}s")
    print(f"artifacts  {results}/  ({len(rows)} cells + MANIFEST.sha256 + RUN.json)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", type=Path, help="path to the experiment spec module")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose artifact exists and hashes clean")
    ap.add_argument("--only", help="run only cells whose params contain this substring")
    ap.add_argument("--dry-run", action="store_true", help="list cell ids and exit")
    ap.add_argument("--prune", action="store_true",
                    help="after the run, delete artifacts no longer named by the grid")
    a = ap.parse_args()
    if not a.spec.exists():
        print(f"no such spec: {a.spec}", file=sys.stderr)
        return 1
    try:
        return run(a.spec.resolve(), a.resume, a.only, a.dry_run, a.prune)
    except ControlFailure as e:
        print(f"control failure: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
