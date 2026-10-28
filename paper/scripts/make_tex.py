"""Tables, figures, number macros and the claim ledger of the manuscript.

Reads only the frozen result folders of runs 1 and 2 (and session 2); never reruns
anything. Writes paper/tables/*.tex, paper/figures/*.pdf, paper/numbers.tex,
paper/numbers-ledger.yaml (macro -> source field) and
experiments/paper-derived/DERIVED.json, which stores every derived value the paper
prints (medians, ranges, counts, slopes) under a descriptive key, so that a number in
the manuscript traces to a stored artifact rather than to a rerun of this script.
paper/claims.yaml belongs to /coe-audit and is not written here.

Run from the track root:  python paper/scripts/make_tex.py
"""
import glob
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm"})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

TRACK = Path(__file__).resolve().parents[2]
PAPER = TRACK / "paper"
GATE = "experiments/2026-10-25-kernel-gate"
S1 = f"{GATE}/results/kernel-gate-2026-10-25-s1"
S2 = f"{GATE}/results/kernel-gate-2026-10-25-s2"
ANALYSIS = f"{GATE}/analysis/kernel-gate-2026-10-25-s1/ANALYSIS.json"
H6 = "experiments/2026-10-26-h6-digits/results/h6-digits-2026-10-26"
SPECTRUM = "experiments/2026-10-28-h6-spectrum/results/h6-spectrum-2026-10-28"  # post hoc
TARGET = 0.003
LAWS = {"gamma1": "Gamma(1)", "gamma4": "Gamma(4)", "laplace": "Laplace", "uniform": "uniform"}

MACROS, CLAIMS, DERIVED = [], [], {}


def cells(run):
    out = []
    for f in sorted(glob.glob(str(TRACK / run / "*=*.json"))):
        d = json.load(open(f))
        if "control" in d["params"]:
            continue
        out.append(dict(d["params"], **d["values"], _file=f"{run}/{Path(f).name}"))
    return out


def macro(name, value, fmt, artifact, locator, context, note=""):
    """Define \\name as the formatted value and record it in the ledger."""
    if fmt == "sci":
        m, e = f"{value:.1e}".split("e")
        raw = f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"
    else:
        raw = fmt.format(value) if not isinstance(value, str) else value
        if raw.startswith("-"):
            raw = f"\\ensuremath{{{raw}}}"
    MACROS.append(f"\\newcommand{{\\{name}}}{{{raw}}}")
    DERIVED[f"macro.{name}"] = dict(value=value, printed=raw, source=artifact, field=locator)
    num = value if isinstance(value, (int, float)) else None
    CLAIMS.append(dict(
        id=hashlib.sha1(name.encode()).hexdigest()[:8], raw=raw, value=num, macro=name,
        context_class="narrative", location=dict(file="numbers.tex", macro=name),
        context=context, status="resolved",
        evidence=dict(kind="derived" if note else "artifact", artifact=f"Ku-Projected-ReLU-Kernels/{artifact}",
                      locator=locator, producer="paper/scripts/make_tex.py", confidence="high",
                      note=note or "read directly from the result file")))
    return raw


def rng(values, fmt="{:.2f}"):
    return f"[{fmt.format(min(values))}, {fmt.format(max(values))}]"


# ---------------------------------------------------------------- run 1 + session 2
c1, c2 = cells(S1), cells(S2)
an = json.load(open(TRACK / ANALYSIS))
gate1 = [c for c in c1 if c["part"] == "gate"]
key = lambda c: (c["law"], c["d"], c["n"])  # noqa: E731

rows = []
for law, d in (("gamma1", 64), ("laplace", 16)):
    for n in (32, 128):
        for sess, cc in (("1", gate1), ("2", c2)):
            sel = [c for c in cc if key(c) == (law, d, n)]
            tr = [c["time_ratio"] for c in sel if c["status"] == 0]
            rows.append((law, d, n, sess, sum(c["status"] == 0 for c in sel), len(sel),
                         sum(c["gaussian_fails"] == 1 for c in sel), float(np.median(tr)), min(tr), max(tr),
                         max(c["candidate_allowance"] for c in sel)))
lines = [r"\begin{tabular}{llrcrrc}", r"\toprule",
         r"Law & $d$ & $n$ & Session & PASS & $K_G$ fails & Time ratio, median [range] \\", r"\midrule"]
for law, d, n, sess, ps, tot, kg_f, med, lo, hi, _ in rows:
    cell = f"\\textbf{{{ps}/{tot}}}" if ps >= 9 else f"{ps}/{tot}"
    lines.append(f"{LAWS[law]} & {d} & {n} & {sess} & {cell} & {kg_f}/{tot} & {med:.2f} [{lo:.2f}, {hi:.2f}] \\\\")
    k = f"tab_gate.{law}_d{d}_n{n}_session{sess}"
    DERIVED[k] = dict(pass_panels=ps, panels=tot, kg_fails=kg_f, time_ratio_median=med,
                      time_ratio_min=lo, time_ratio_max=hi, source=S1 if sess == "1" else S2)
lines += [r"\bottomrule", r"\end{tabular}"]
(PAPER / "tables/tab_gate.tex").write_text("\n".join(lines) + "\n")

g_tr = [c["time_ratio"] for c in gate1 + c2 if c["law"] == "gamma1" and c["status"] == 0]
l_tr = [c["time_ratio"] for c in gate1 + c2 if c["law"] == "laplace" and c["status"] == 0]
all_tr = g_tr + l_tr
macro("trMin", min(all_tr), "{:.2f}", S1, "time_ratio, PASS panels, both sessions (min)", "time ratio lower end", "minimum over the PASS panels of S1 and S2 gate cells")
macro("trMax", max(all_tr), "{:.2f}", S1, "time_ratio, PASS panels, both sessions (max)", "time ratio upper end", "maximum over the PASS panels of S1 and S2 gate cells")
lap_fail = [c["candidate_allowance"] for c in gate1 + c2 if c["law"] == "laplace" and c["status"] == 3]
macro("lapFailLo", min(lap_fail), "{:.4f}", S1, "candidate_allowance of CANDIDATE_FAIL panels, Laplace d=16", "failing Laplace panels", "min over both sessions")
macro("lapFailHi", max(lap_fail), "{:.4f}", S1, "candidate_allowance of CANDIDATE_FAIL panels, Laplace d=16", "failing Laplace panels", "max over both sessions")
cpu = [c["cpu_wall_max"] for c in gate1 + c2]
macro("cpuWallMax", max(cpu), "{:.2f}", S1, "cpu_wall_max over gate cells, both sessions", "largest CPU/wall ratio", "max over both sessions")

# controls of run 1
ctl = {json.load(open(f))["params"]["control"]: json.load(open(f))["values"]
       for f in glob.glob(str(TRACK / S1 / "control=*.json"))}
alg = sorted(glob.glob(str(TRACK / S1 / "control=algebra*.json")))[0]
gau = sorted(glob.glob(str(TRACK / S1 / "control=gaussian*.json")))[0]
macro("ctlClosed", ctl["algebra"]["closed_vs_quad"], "sci", f"{S1}/{Path(alg).name}", "closed_vs_quad", "closed forms vs quadrature")
macro("ctlFrozen", ctl["algebra"]["frozen_vs_quad"], "sci", f"{S1}/{Path(alg).name}", "frozen_vs_quad", "frozen code vs quadrature")
macro("ctlCF", ctl["gaussian"]["cf_vs_arccos"], "sci", f"{S1}/{Path(gau).name}", "cf_vs_arccos", "CF reference on the Gaussian law")

# (law, d) accuracy grid at n=32, dense
grid = {}
for c in c1:
    if c["geometry"] == "dense" and c["n"] == 32 and c["d"] in (16, 64, 128):
        grid.setdefault((c["law"], c["d"]), []).append(c)
kg_err = [c["gaussian_allowance"] for (law, d), cs in grid.items() for c in cs
          if (law, d) in (("gamma1", 16), ("gamma1", 64), ("laplace", 16))]
macro("kgErrLo", min(kg_err), "{:.3f}", S1, "gaussian_allowance, dense n=32, Gamma(1) d16/d64 and Laplace d16", "K_G error range", "min over 30 panels")
macro("kgErrHi", max(kg_err), "{:.3f}", S1, "gaussian_allowance, dense n=32, Gamma(1) d16/d64 and Laplace d16", "K_G error range", "max over 30 panels")
g16 = [c["candidate_allowance"] for c in grid[("gamma1", 16)]]
macro("gSixteenLo", min(g16), "{:.3f}", S1, "candidate_allowance, dense Gamma(1) d=16 n=32", "Gamma d=16 failure", "min over 10 panels")
macro("gSixteenHi", max(g16), "{:.3f}", S1, "candidate_allowance, dense Gamma(1) d=16 n=32", "Gamma d=16 failure", "max over 10 panels")

# PSD: min eigenvalue / ||R|| over every accuracy and gate panel of run 1
mins = [c["candidate_mineig"] for c in c1]
neg = sum(m < -1e-10 for m in mins)
macro("psdPanels", len(mins), "{:d}", S1, "count of run-1 panels", "panels checked for PSD", "count")
macro("psdNeg", neg, "{:d}", S1, "candidate_mineig < -1e-10", "panels with a negative eigenvalue", "count")
macro("psdMin", min(mins), "sci", S1, "candidate_mineig (min)", "smallest normalised eigenvalue", "min over run 1")

fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
for i, law in enumerate(LAWS):
    for j, d in enumerate((16, 64, 128)):
        cs = grid[(law, d)]
        ax[i].scatter([j - 0.12] * len(cs), [c["gaussian_allowance"] for c in cs], s=8, marker="o",
                      color="#999999", label="$K_G$" if j == 0 else None)
        ax[i].scatter([j + 0.12] * len(cs), [c["candidate_allowance"] for c in cs], s=8, marker="s",
                      color="#0072B2", label=r"$\widehat K$" if j == 0 else None)
    ax[i].axhspan(1e-5, TARGET, color="#009E73", alpha=0.12, lw=0)
    ax[i].axhline(TARGET, color="#009E73", lw=0.8)
    ax[i].set_yscale("log")
    ax[i].set_xticks([0, 1, 2], ["16", "64", "128"])
    ax[i].set_title(LAWS[law], fontsize=9)
    ax[i].set_xlabel("$d$", fontsize=9)
    ax[i].tick_params(labelsize=8)
ax[0].set_ylabel("allowance", fontsize=9)
ax[0].set_ylim(1e-5, 0.3)
ax[0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(PAPER / "figures/fig_grid.pdf", metadata={"CreationDate": None})
plt.close(fig)

# H2 slope (from ANALYSIS.json) and descriptive slopes for the other laws
h2 = an["H2"]["all_panels"]
macro("hTwoSlope", h2["slope"], "{:.2f}", ANALYSIS, "H2.all_panels.slope", "remainder slope")
macro("hTwoLo", h2["lower95"], "{:.2f}", ANALYSIS, "H2.all_panels.lower95", "slope CI lower")
macro("hTwoHi", h2["upper95"], "{:.2f}", ANALYSIS, "H2.all_panels.upper95", "slope CI upper")
macro("hTwoN", h2["n"], "{:d}", ANALYSIS, "H2.all_panels.n", "panels in the slope fit")


def slope_fit(rows):
    """OLS slope of log error on log d with its HC3 (heteroscedasticity-robust) 95% upper bound."""
    x, y = np.log([r[0] for r in rows]), np.log([r[1] for r in rows])
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    inv = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, inv, X)
    V = inv @ (X.T * (e / (1 - h)) ** 2) @ X @ inv
    from scipy import stats
    return float(beta[1]), float(beta[1] + stats.t.ppf(0.975, len(x) - 2) * np.sqrt(V[1, 1]))


# post hoc robustness of the H2 slope: the same panels as analyze.py (ref_ok, error > 0)
h2_pts = [(c["d"], c["candidate_error"]) for c in c1 if c["law"] == "gamma1" and c["geometry"] == "dense"
          and c["n"] == 32 and c["ref_ok"] and c["candidate_error"] > 0]
if len(h2_pts) != h2["n"] or abs(slope_fit(h2_pts)[0] - h2["slope"]) > 1e-9:
    raise SystemExit("post hoc H2 refit does not reproduce ANALYSIS.json")
macro("hTwoHcUp", slope_fit(h2_pts)[1], "{:.2f}", S1, "HC3 95% upper bound of the H2 OLS slope",
      "robust slope upper bound", "post hoc; HC3 standard error, same 50 panels as H2")
no16 = [p for p in h2_pts if p[0] != 16]
macro("hTwoNoSixteen", slope_fit(no16)[0], "{:.2f}", S1, "H2 OLS slope without d=16",
      "slope without d=16", "post hoc; 40 panels at d=32..256")
macro("hTwoNoSixteenUp", slope_fit(no16)[1], "{:.2f}", S1, "HC3 95% upper bound, H2 slope without d=16",
      "robust upper bound without d=16", "post hoc; HC3, 40 panels")

sweep = [c for c in c1 if c["law"] == "gamma1" and c["geometry"] == "dense" and c["n"] == 32]
desc = {}
for law in LAWS:
    pts = [(c["d"], c["candidate_error"]) for (lw, d), cs in grid.items() if lw == law for c in cs]
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    desc[law] = float(np.polyfit(x, y, 1)[0])
for law in ("gamma4", "laplace", "uniform"):
    macro(f"slope{law.capitalize().replace('4', 'Four')}", desc[law], "{:.2f}", S1,
          f"OLS slope of log candidate_error on log d, dense {law} n=32, d=16/64/128",
          "descriptive slope", "computed here; three d values, not registered")

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ds = np.array([c["d"] for c in sweep])
es = np.array([c["candidate_error"] for c in sweep])
ax.scatter(ds, es, s=8, color="#0072B2")
xx = np.array([16, 256])
b = np.polyfit(np.log(ds), np.log(es), 1)
ax.plot(xx, np.exp(b[1]) * xx ** b[0], color="#0072B2", lw=1, label=f"OLS slope {b[0]:.2f}")
ax.plot(xx, np.exp(b[1]) * 16 ** b[0] * (xx / 16) ** -1.5, color="#D55E00", lw=1, ls="--", label="slope $-3/2$")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("$d$", fontsize=9)
ax.set_ylabel(r"relative error of $\widehat K$", fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(PAPER / "figures/fig_slope.pdf", metadata={"CreationDate": None})
plt.close(fig)

# competitor cost ladders, session 1, primary cells
lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r" & & \multicolumn{5}{c}{Median seconds} & CF & \multicolumn{2}{c}{Panels $>0.003$} \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){9-10}",
         r"Cell & $n$ & $\widehat K$ & Sobol & QCV & Cond. & CF & nodes & O1 & O2 \\",
         r"\midrule"]
for law, d in (("gamma1", 64), ("laplace", 16)):
    for n in (32, 128):
        sel = [c for c in gate1 if key(c) == (law, d, n)]

        def med(f):
            v = [c[f] for c in sel if f in c and not math.isnan(c[f])]
            return f"{np.median(v):.3g}" if v else "--"
        abl = f"{sum(c['third_allowance'] > TARGET for c in sel)} & {sum(c['fourth_allowance'] > TARGET for c in sel)}"
        DERIVED[f"tab_cost.{law}_d{d}_n{n}"] = {f"median_{f}": float(np.median([c[f] for c in sel if f in c and not math.isnan(c[f])]))
                                               for f in ("candidate_seconds", "sobol_seconds", "qcv_seconds", "conditional_seconds", "cf_seconds", "cf_level")
                                               if any(f in c and not math.isnan(c[f]) for c in sel)}
        DERIVED[f"tab_cost.{law}_d{d}_n{n}"].update(order1_fails=sum(c['third_allowance'] > TARGET for c in sel),
                                                   order2_fails=sum(c['fourth_allowance'] > TARGET for c in sel), source=S1)
        lines.append(f"{LAWS[law]}, {d} & {n} & {med('candidate_seconds')} & {med('sobol_seconds')} & "
                     f"{med('qcv_seconds')} & {med('conditional_seconds')} & {med('cf_seconds')} & "
                     f"{med('cf_level')} & {abl} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(PAPER / "tables/tab_cost.tex").write_text("\n".join(lines) + "\n")

# H5
h5 = an["H5"]
cf = h5["confusion"]
macro("hFiveAuc", h5["auc"], "{:.3f}", ANALYSIS, "H5.auc", "H5 AUC")
macro("hFiveTau", h5["tau"], "{:.3f}", ANALYSIS, "H5.tau", "H5 threshold")
macro("hFiveOver", h5["predicted_accurate_over_0006"], "{:d}", ANALYSIS, "H5.predicted_accurate_over_0006", "H5 false accepts above 0.006")
macro("hFivePrec", cf["tp"] / (cf["tp"] + cf["fp"]), "{:.2f}", ANALYSIS, "H5.confusion tp/(tp+fp)", "H5 precision", "tp/(tp+fp)")
macro("hFiveRec", cf["tp"] / (cf["tp"] + cf["fn"]), "{:.2f}", ANALYSIS, "H5.confusion tp/(tp+fn)", "H5 recall", "tp/(tp+fn)")
macro("hFivePanels", h5["panels"], "{:d}", ANALYSIS, "H5.panels", "H5 test panels")
macro("hFiveRefLim", h5["reference_limited"], "{:d}", ANALYSIS, "H5.reference_limited", "reference-limited sparse panels")

# post-hoc accuracy-threshold sensitivity (accuracy only, primary cells, session 1)
for t, nm in ((0.002, "Two"), (0.005, "Five")):
    for law, d in (("gamma1", 64), ("laplace", 16)):
        sel = [c for c in gate1 if (c["law"], c["d"]) == (law, d)]
        macro(f"sens{nm}{law.capitalize().replace('1', 'One')}", sum(c["candidate_allowance"] <= t for c in sel), "{:d}", S1,
              f"candidate_allowance <= {t}, gate {law} d={d}, n=32 and 128", "post-hoc threshold sensitivity",
              "count of 20 panels; post hoc, accuracy only")

# ---------------------------------------------------------------- run 2 (digits)
h6 = {c["law"]: c for c in cells(H6)}
spec = {c["law"]: c for c in cells(SPECTRUM)}
lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Law & Kernel & Allowance & $\lambda_{\min}/\lambda$ & Test MSE & Rel.\ diff.\ (\%) & Accuracy & Seconds \\",
         r"\midrule"]
for law in ("gamma1", "laplace"):
    v, sp = h6[law], spec[law]
    for nm, lab in (("ref", "CF reference"), ("candidate", r"$\widehat K$"), ("gaussian", "$K_G$"),
                    ("third", r"order 1 ($+\kappa_3$)"), ("fourth", r"order 2 ($+\kappa_4$)")):
        allow = "--" if nm == "ref" else f"{v[nm + '_allowance']:.4f}"
        mse = v["mse_ref"] if nm == "ref" else v[nm + "_mse"]
        relv = "--" if nm == "ref" else f"{100 * v[nm + '_mse_rel']:.2f}"
        if nm != "ref" and v[nm + "_mse_rel"] <= 0.01:
            relv = f"\\textbf{{{relv}}}"
        acc = f"{v[nm + '_test_acc']:.3f}" if nm + "_test_acc" in v else "--"
        secs = {"candidate": v["candidate_seconds"], "gaussian": v["gaussian_seconds"]}.get(nm)
        secs = f"{secs:.2f}" if secs else "--"
        eig = None if nm == "ref" else sp[nm + "_mineig_over_lambda"]
        eigs = "--" if eig is None else (f"\\ensuremath{{{eig:.1f}}}" if eig < 0 else f"{eig:.1f}")
        lines.append(f"{LAWS[law] if nm == 'ref' else ''} & {lab} & {allow} & {eigs} & {mse:.5f} & {relv} & {acc} & {secs} \\\\")
        DERIVED[f"tab_digits.{law}.{nm}"] = dict(allowance=None if nm == "ref" else v[nm + "_allowance"], test_mse=mse,
                                                train_mineig_over_lambda=eig, eig_source=sp["_file"],
                                                rel_diff_pct=None if nm == "ref" else 100 * v[nm + "_mse_rel"],
                                                test_acc=v.get(nm + "_test_acc"), seconds={"candidate": v["candidate_seconds"], "gaussian": v["gaussian_seconds"]}.get(nm),
                                                source=v["_file"])
    lines.append(r"\midrule" if law == "gamma1" else r"\bottomrule")
lines.append(r"\end{tabular}")
(PAPER / "tables/tab_digits.tex").write_text("\n".join(lines) + "\n")
for law, nm in (("gamma1", "Gamma"), ("laplace", "Laplace")):
    v, f = h6[law], h6[law]["_file"]
    macro(f"hSix{nm}KG", 100 * v["gaussian_mse_rel"], "{:.2f}", f, "gaussian_mse_rel x100", "K_G test-MSE gap (%)")
    macro(f"hSix{nm}Cand", 100 * v["candidate_mse_rel"], "{:.2f}", f, "candidate_mse_rel x100", "candidate test-MSE gap (%)")
    macro(f"hSix{nm}Ratio", v["time_ratio"], "{:.0f}", f, "time_ratio", "candidate / K_G time")
    macro(f"hSix{nm}KGAllow", v["gaussian_allowance"], "{:.3f}", f, "gaussian_allowance", "K_G kernel error on digits")
    macro(f"hSix{nm}CandAllow", v["candidate_allowance"], "{:.4f}", f, "candidate_allowance", "K-hat kernel error on digits")
macro("hSixThird", 100 * h6["gamma1"]["third_mse_rel"], "{:.0f}", h6["gamma1"]["_file"], "third_mse_rel x100", "order-1 MSE inflation (%)")
macro("hSixFourth", 100 * h6["gamma1"]["fourth_mse_rel"], "{:.1f}", h6["gamma1"]["_file"], "fourth_mse_rel x100", "order-2 MSE gap (%)")
sp = spec["gamma1"]
macro("hSixThirdEig", sp["third_mineig_over_lambda"], "{:.1f}", sp["_file"], "third_mineig_over_lambda",
      "order-1 smallest training eigenvalue over the ridge", "post hoc; smallest eigenvalue of K_train / lambda")

(PAPER / "numbers.tex").write_text("% Generated by scripts/make_tex.py from frozen result files. Do not edit.\n"
                                   + "\n".join(MACROS) + "\n")
ledger = dict(version=1, manuscript="paper/main.tex", generated="2026-10-26", macros=CLAIMS)
(PAPER / "numbers-ledger.yaml").write_text(yaml.safe_dump(ledger, sort_keys=False, allow_unicode=True))
out = TRACK / "experiments/paper-derived"
out.mkdir(exist_ok=True)
(out / "DERIVED.json").write_text(json.dumps(dict(producer="paper/scripts/make_tex.py",
                                                  note="derived from the frozen run folders; regenerate, never edit",
                                                  values=DERIVED), indent=1, sort_keys=True, default=float) + "\n")
print(f"{len(MACROS)} macros, tables 3, figures 2")
