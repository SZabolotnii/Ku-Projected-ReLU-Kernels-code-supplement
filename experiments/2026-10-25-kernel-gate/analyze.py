"""Apply the frozen SPEC.md decision rules to the cell files of run 1 (and session 2 if present).

Reads only results/<run-id>/*.json whose hash matches MANIFEST.sha256; writes
analysis/<run-id>/ANALYSIS.json and ANALYSIS.md (generated, never edited by hand). The run
folder itself stays exactly as MANIFEST.sha256 describes it.
Usage: python analyze.py [--s2]
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
TARGET = 0.003
STATUS = {0: "PASS", 1: "NO_SPEED_GAIN", 2: "COMPETITOR_LIMITED", 3: "CANDIDATE_FAIL", 4: "REFERENCE_LIMITED"}


def load(run_id):
    run = HERE / "results" / run_id
    hashes = dict(reversed(line.split()) for line in (run / "MANIFEST.sha256").read_text().splitlines() if line.strip())
    cells = []
    for path in sorted(run.glob("*.json")):
        if path.name == "RUN.json":
            continue
        if hashes.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise SystemExit(f"{path.name}: hash does not match MANIFEST.sha256")
        cells.append(json.loads(path.read_text()))
    if not (run / "RUN.json").is_file():
        raise SystemExit(f"{run_id}: no RUN.json, the run did not finish")
    cells = [c for c in cells if "control" not in c["params"]]
    expected = 240 if run_id.endswith("s1") else 40
    if len(cells) != expected:
        raise SystemExit(f"{run_id}: {len(cells)} cells, expected {expected}")
    for c in cells:
        bad = [k for k, v in c["values"].items() if k.endswith(("_error", "_allowance", "h5_score", "u"))
               and not math.isfinite(v)]
        if bad:
            raise SystemExit(f"{c['cell']}: non-finite {bad}")
    return run, cells


def gate_verdicts(cells, competitors=None):
    """Per (law, d, n): PASS count. `competitors` restricts the eligible set (for K4)."""
    out = {}
    for c in cells:
        p, v = c["params"], c["values"]
        if p["part"] != "gate":
            continue
        status = int(v["status"])
        if competitors is not None and status in (0, 1, 2):
            secs = [v[f"{m}_seconds"] for m in competitors
                    if v[f"{m}_allowance"] <= TARGET and math.isfinite(v[f"{m}_seconds"])
                    and (m != "gaussian" or v["gaussian_mineig"] >= -1e-10)]
            status = 2 if not secs else (0 if v["candidate_seconds"] / min(secs) <= 0.8 else 1)
        key = f"{p['law']}_d{p['d']}_n{p['n']}"
        row = out.setdefault(key, dict(panels=0, passed=0, gaussian_fails=0, statuses={}, ratios=[]))
        row["panels"] += 1
        row["passed"] += status == 0
        row["gaussian_fails"] += int(v["gaussian_fails"])
        row["statuses"][STATUS[status]] = row["statuses"].get(STATUS[status], 0) + 1
        if math.isfinite(v["time_ratio"]):
            row["ratios"].append(v["time_ratio"])
    for key, row in out.items():
        if row["panels"] != 10:
            raise SystemExit(f"{key}: {row['panels']} panels, expected 10")
        row["s1"] = row["passed"] >= 9
        row["s2"] = row["gaussian_fails"] >= 9
        r = row.pop("ratios")
        row["ratio_range"] = [min(r), max(r)] if r else None
    return out


def auc(scores, labels):
    """AUC of -score for label True (low score predicts accurate); ties count 1/2."""
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p < q) + 0.5 * (p == q) for p in pos for q in neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", action="store_true", help="also read timing session 2")
    args = ap.parse_args()
    run, cells = load("kernel-gate-2026-10-25-s1")
    res = {}
    g1 = gate_verdicts(cells)
    res["gate_s1"] = g1
    sessions = [g1]
    if args.s2:
        _, cells2 = load("kernel-gate-2026-10-25-s2")
        res["gate_s2"] = gate_verdicts(cells2)
        sessions.append(res["gate_s2"])
    prim = {"H3": ("gamma1_d64_n32", "gamma1_d64_n128"), "H4": ("laplace_d16_n32", "laplace_d16_n128")}
    for h, keys in prim.items():
        s1 = all(s[k]["s1"] for s in sessions for k in keys)
        s2 = all(g1[k]["s2"] for k in keys)
        res[h] = dict(s1_all_sessions=s1, s2=s2, holds=s1 and s2,
                      session2_read=args.s2, note=None if args.s2 else "session 2 not run: S1 provisional")
    res["K2"] = not all(res[h]["s1_all_sessions"] for h in prim)
    old = gate_verdicts(cells, competitors=("gaussian", "sobol", "qcv"))
    # K4 per primary cell: S1 needs both n, as for H3/H4
    res["K4"] = {}
    for h, keys in prim.items():
        with_old = all(old[k]["s1"] for k in keys)
        with_all = all(g1[k]["s1"] for k in keys)
        res["K4"][h] = dict(s1_old_competitors=with_old, s1_all_competitors=with_all,
                            fires=with_old and not with_all)

    # (law, d) accuracy grid at n=32, dense: the 12 cells of SPEC (H2-only d=32, 256 excluded)
    grid = {}
    for c in cells:
        p, v = c["params"], c["values"]
        if p["n"] != 32 or p["geometry"] != "dense" or p["d"] not in (16, 64, 128):
            continue
        key = f"{p['law']}_d{p['d']}"
        row = grid.setdefault(key, dict(panels=0, cand_ok=0, gauss_ok=0, ref_ok=0, cand_allow=[], gauss_allow=[]))
        ok = bool(v["ref_ok"])
        row["panels"] += 1
        row["ref_ok"] += ok
        row["cand_ok"] += ok and v["candidate_allowance"] <= TARGET and v["candidate_mineig"] >= -1e-10
        row["gauss_ok"] += ok and v["gaussian_allowance"] <= TARGET
        row["cand_allow"].append(v["candidate_allowance"])
        row["gauss_allow"].append(v["gaussian_allowance"])
    for row in grid.values():
        row["cand_allow"] = [min(row["cand_allow"]), float(np.median(row["cand_allow"])), max(row["cand_allow"])]
        row["gauss_allow"] = [min(row["gauss_allow"]), float(np.median(row["gauss_allow"])), max(row["gauss_allow"])]
    res["grid"] = grid
    res["K3"] = all(r["gauss_ok"] >= 9 for r in grid.values() if r["cand_ok"] >= 9) and any(
        r["cand_ok"] >= 9 for r in grid.values())

    # H2 slope
    h2_cells = [c for c in cells
                if c["params"]["law"] == "gamma1" and c["params"]["geometry"] == "dense" and c["params"]["n"] == 32]
    pts = [(c["params"]["d"], c["values"]["candidate_error"], c["values"]["u"]) for c in h2_cells
           if c["values"]["ref_ok"] and c["values"]["candidate_error"] > 0]
    h2_excluded = len(h2_cells) - len(pts)

    def fit(rows):
        if len({r[0] for r in rows}) < 2:
            return None
        x = np.log([r[0] for r in rows])
        y = np.log([r[1] for r in rows])
        f = stats.linregress(x, y)
        tq = stats.t.ppf(0.975, len(rows) - 2)
        return dict(n=len(rows), slope=f.slope, upper95=f.slope + tq * f.stderr, lower95=f.slope - tq * f.stderr)

    h2 = fit(pts)
    if h2 is None:
        raise SystemExit("H2: fewer than two d values with a valid reference")
    resolved = [r for r in pts if r[1] >= 10 * r[2]]
    h2_res = fit(resolved) if len(resolved) > 2 else None
    res["H2"] = dict(all_panels=h2, resolved_panels=h2_res, falsified=h2["upper95"] > -1.2,
                     excluded_reference_limited=h2_excluded, d_values=sorted({r[0] for r in pts}))

    # H5: the rule must be the one frozen before this run
    rule = json.loads((HERE / "h5_rule.json").read_text())
    for name in ("kg", "spec"):
        if rule[f"{name}_sha256"] != hashlib.sha256((HERE / f"{name}.py").read_bytes()).hexdigest():
            raise SystemExit(f"{name}.py changed after h5_rule.json was frozen")
    tau = rule["tau"]
    test = [c["values"] for c in cells if c["params"]["n"] == 32 and c["params"]["law"] in ("gamma1", "laplace")
            and c["params"]["d"] in (16, 64)]
    sc = np.array([v["h5_score"] for v in test])
    al = np.array([v["candidate_allowance"] for v in test])
    lab = np.array([bool(v["ref_ok"]) and v["candidate_allowance"] <= TARGET for v in test])
    pred = sc <= tau
    a = auc(sc, lab)
    # an undefined AUC (one class empty) cannot support the rule: counted as falsified
    res["H5"] = dict(panels=len(test), tau=tau, auc=a, predicted_accurate_over_0006=int(np.sum(pred & (al > 0.006))),
                     reference_limited=int(sum(not v["ref_ok"] for v in test)),
                     falsified=bool(not math.isfinite(a) or a <= 0.75 or np.any(pred & (al > 0.006))),
                     confusion=dict(tp=int(np.sum(pred & lab)), fp=int(np.sum(pred & ~lab)),
                                    fn=int(np.sum(~pred & lab)), tn=int(np.sum(~pred & ~lab))))
    out = HERE / "analysis" / run.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "ANALYSIS.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    lines = ["# Kernel gate run 1 — analysis (generated by analyze.py)", ""]
    for h in ("H3", "H4"):
        lines.append(f"- **{h}**: holds={res[h]['holds']} (S1 all sessions={res[h]['s1_all_sessions']}, "
                     f"S2={res[h]['s2']}){'' if args.s2 else ' — session 2 not run, S1 provisional'}")
    lines += [f"- **K2** fires: {res['K2']}", f"- **K3** fires: {res['K3']}",
              f"- **K4** fires in: {[k for k, v in res['K4'].items() if v['fires']] or 'none'}",
              f"- **H2** slope {h2['slope']:.3f} (95% CI {h2['lower95']:.3f}..{h2['upper95']:.3f}, "
              f"{h2['n']} panels), falsified={res['H2']['falsified']}",
              f"- **H5** AUC {a:.3f}, τ={tau:.4g}, predicted-accurate with allowance>0.006: "
              f"{res['H5']['predicted_accurate_over_0006']}, falsified={res['H5']['falsified']}", "",
              "| cell | panels | PASS | K_G fails | statuses | time ratio range |", "|---|---:|---:|---:|---|---|"]
    for k, r in g1.items():
        rr = "—" if r["ratio_range"] is None else f"{r['ratio_range'][0]:.3f}–{r['ratio_range'][1]:.3f}"
        lines.append(f"| {k} | {r['panels']} | {r['passed']} | {r['gaussian_fails']} | {r['statuses']} | {rr} |")
    lines += ["", "| law, d (n=32 dense) | panels | ref ok | candidate ≤0.003 | K_G ≤0.003 | candidate allowance min/med/max | K_G allowance min/med/max |",
              "|---|---:|---:|---:|---:|---|---|"]
    for k, r in sorted(grid.items()):
        lines.append(f"| {k} | {r['panels']} | {r['ref_ok']} | {r['cand_ok']} | {r['gauss_ok']} | "
                     + " / ".join(f"{v:.4f}" for v in r["cand_allow"]) + " | "
                     + " / ".join(f"{v:.4f}" for v in r["gauss_allow"]) + " |")
    (out / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
