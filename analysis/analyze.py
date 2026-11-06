#!/usr/bin/env python3
"""SkillBridge KZ — "Same Answer, Different Text?" analysis. One command builds every table and the main figure
from the saved raw judge outputs in runs/.

  python analysis/analyze.py --primary <model_id> [<model_id> ...] [--runs runs] [--out analysis/output]

Pre-specified (see protocol/OSF_PREREGISTRATION_DRAFT.md):
  * unit = answer_id; per-answer score = mean final_score_1to5 over valid repeats (status=ok)
  * primary: condition A, KK-nat − KK-orig on the 45 main-question answers, per primary model;
    Wilcoxon signed-rank (two-sided, zero_method="wilcox"), Holm–Bonferroni across primary models;
    95% percentile bootstrap CI, 10,000 resamples of answer_id, seed 20261107; Cohen's d_z
  * gap share = (Δ_orig−RU − Δ_nat−RU)/Δ_orig−RU on 43 answers (meaning check KK-nat↔RU passed),
    reported only if the CI of Δ_orig−RU excludes 0
  * A′ reduction = (gap_A − gap_A′)/gap_A for the KK-orig−RU gap, with CI, same rule
  * parse_failed rate per answer_version with denominators; test–retest exact agreement and QWK
Everything else is labelled exploratory in the outputs.
"""
import argparse, csv, glob, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SEED, B = 20261107, 10_000
VERS = ("ru", "kk", "kk_nat")
LABEL = {"ru": "RU", "kk": "KK-orig", "kk_nat": "KK-nat"}


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def load_meta():
    q = {x["question_id"]: x for x in rd(ROOT / "protocol/questions_skillbridge_v1_2.csv")}
    items = {x["answer_id"]: x for x in rd(ROOT / "data/items_v1_2.csv") if x["answer_version"] == "ru"}
    main = sorted(a for a, x in items.items() if q[x["question_id"]]["include_in_main"] == "1")
    mc = {x["answer_id"]: x for x in rd(ROOT / "reviews/kk_nat/MEANING_CHECK_v1.csv")}
    gap_set = [a for a in main if mc[a]["pass_nat_vs_ru"] == "YES"]
    edits = {x["answer_id"]: x for x in rd(ROOT / "linguistic/edit_stats_per_answer.csv")} \
        if (ROOT / "linguistic/edit_stats_per_answer.csv").exists() else {}
    return items, main, gap_set, edits


def load_runs(run_dir, include_mock):
    last = {}  # an api_error record can be followed by a re-sent record for the same key: keep the last one
    for f in sorted(glob.glob(str(Path(run_dir) / "run_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["provider"] == "mock" and not include_mock:
                continue
            last[(r["judge_model"], r["rubric_condition"], r["answer_id"], r["answer_version"], r["repeat"])] = r
    return list(last.values())


def boot(fn, arrays, rng):
    n = len(arrays[0])
    idx = rng.integers(0, n, size=(B, n))
    out = np.array([fn(*[a[i] for a in arrays]) for i in idx])
    out = out[np.isfinite(out)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) else (math.nan, math.nan)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def qwk(a, b, k=5):
    a, b = np.asarray(a, int) - 1, np.asarray(b, int) - 1
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[x, y] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    den = (w * e).sum()
    return float(1 - (w * o).sum() / den) if den else math.nan


def fmt(x, d=2):
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", nargs="+", required=True, help="judge_model IDs that are confirmatory")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "analysis/output"))
    ap.add_argument("--include-mock", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items, main_set, gap_set, edits = load_meta()
    recs = load_runs(a.runs, a.include_mock)
    if not recs:
        raise SystemExit(f"No run files in {a.runs}")

    # ---- per-answer mean scores and data-quality tables
    scores = defaultdict(list)          # (model, cond, aid, ver) -> [final scores]
    reps = defaultdict(dict)            # (model, cond, aid, ver) -> {repeat: score}
    elig = defaultdict(lambda: [0, 0])  # (model, cond, ver) -> [n, n_failed]
    versions_seen = defaultdict(set)
    for r in recs:
        m, c, aid, v = r["judge_model"], r["rubric_condition"], r["answer_id"], r["answer_version"]
        elig[(m, c, v)][0] += 1
        versions_seen[m].add(r.get("returned_model_version"))
        if r["status"] == "ok":
            scores[(m, c, aid, v)].append(r["final_score_1to5"])
            reps[(m, c, aid, v)][r["repeat"]] = r["final_score_1to5"]
        else:
            elig[(m, c, v)][1] += 1
    models = sorted({k[0] for k in scores})
    mean = {k: float(np.mean(v)) for k, v in scores.items()}

    with open(out / "table_parse_failed.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "condition", "version", "n_eligible", "n_parse_failed", "rate"])
        for (m, c, v), (n, k) in sorted(elig.items()):
            w.writerow([m, c, LABEL.get(v, v), n, k, round(k / n, 4)])

    rng = np.random.default_rng(SEED)
    rows, figdata, lines = [], [], []
    prim_p = {}

    def vec(m, c, ids, v):
        return np.array([mean.get((m, c, i, v), np.nan) for i in ids])

    for m in models:
        for c in ("A", "A_prime"):
            o, n_, r_ = (vec(m, c, main_set, v) for v in ("kk", "kk_nat", "ru"))
            ok = ~np.isnan(o) & ~np.isnan(n_)
            if ok.sum() < 2:
                continue
            d = n_[ok] - o[ok]
            ci = boot(lambda x: x.mean(), [d], rng)
            sd = d.std(ddof=1)
            try:
                p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                p = math.nan
            if c == "A" and m in a.primary:
                prim_p[m] = p
            # gaps vs RU on the 43-answer set
            og, ng, rg = (vec(m, c, gap_set, v) for v in ("kk", "kk_nat", "ru"))
            okg = ~np.isnan(og) & ~np.isnan(ng) & ~np.isnan(rg)
            g_o, g_n = og[okg] - rg[okg], ng[okg] - rg[okg]
            ci_go = boot(lambda x: x.mean(), [g_o], rng)
            ci_gn = boot(lambda x: x.mean(), [g_n], rng)
            share, ci_sh = math.nan, (math.nan, math.nan)
            if not (ci_go[0] <= 0 <= ci_go[1]) and g_o.mean() != 0:
                share = (g_o.mean() - g_n.mean()) / g_o.mean()
                ci_sh = boot(lambda x, y: (x.mean() - y.mean()) / x.mean() if x.mean() != 0 else np.nan, [g_o, g_n], rng)
            rows.append({"model": m, "confirmatory": "yes" if (m in a.primary and c == "A") else "no (exploratory)",
                         "condition": c, "n_pairs": int(ok.sum()), "mean_KKnat_minus_KKorig": d.mean(),
                         "ci_lo": ci[0], "ci_hi": ci[1], "cohen_dz": d.mean() / sd if sd > 0 else math.nan,
                         "wilcoxon_p": p, "holm_p": None,
                         "n_gap_set": int(okg.sum()), "gap_KKorig_RU": g_o.mean(), "gap_KKorig_RU_lo": ci_go[0],
                         "gap_KKorig_RU_hi": ci_go[1], "gap_KKnat_RU": g_n.mean(), "gap_KKnat_RU_lo": ci_gn[0],
                         "gap_KKnat_RU_hi": ci_gn[1], "gap_share_closed": share, "gap_share_lo": ci_sh[0],
                         "gap_share_hi": ci_sh[1]})
            if c == "A":
                figdata.append((m, g_o.mean(), ci_go, g_n.mean(), ci_gn))

    prim = [m for m in a.primary if m in prim_p]
    for m, ph in zip(prim, holm([prim_p[m] for m in prim])):
        for r in rows:
            if r["model"] == m and r["condition"] == "A":
                r["holm_p"] = ph
    with open(out / "table_main.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # ---- A′ reduction of the KK-orig−RU gap (and KK-nat−RU), per model
    red = []
    for m in models:
        for v in ("kk", "kk_nat"):
            gA = vec(m, "A", gap_set, v) - vec(m, "A", gap_set, "ru")
            gP = vec(m, "A_prime", gap_set, v) - vec(m, "A_prime", gap_set, "ru")
            ok = ~np.isnan(gA) & ~np.isnan(gP)
            if ok.sum() < 2:
                continue
            gA, gP = gA[ok], gP[ok]
            ciA = boot(lambda x: x.mean(), [gA], rng)
            r_, ci_r = math.nan, (math.nan, math.nan)
            if not (ciA[0] <= 0 <= ciA[1]) and gA.mean() != 0:
                r_ = (gA.mean() - gP.mean()) / gA.mean()
                ci_r = boot(lambda x, y: (x.mean() - y.mean()) / x.mean() if x.mean() != 0 else np.nan, [gA, gP], rng)
            red.append([m, LABEL[v] + "−RU", int(ok.sum()), round(gA.mean(), 4), round(gP.mean(), 4),
                        round(r_, 4) if not math.isnan(r_) else "NA (gap CI includes 0)", round(ci_r[0], 4), round(ci_r[1], 4)])
    with open(out / "table_a_prime_reduction.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "gap", "n", "gap_A", "gap_A_prime", "reduction_share", "ci_lo", "ci_hi"])
        w.writerows(red)

    # ---- test–retest (repeat 1 vs 2), all versions
    tr = []
    for m in models:
        for v in VERS:
            p1, p2 = [], []
            for (mm, c, aid, vv), d in reps.items():
                if mm == m and vv == v and 1 in d and 2 in d:
                    p1.append(d[1]); p2.append(d[2])
            if p1:
                tr.append([m, LABEL[v], len(p1), round(float(np.mean(np.array(p1) == np.array(p2))), 4), round(qwk(p1, p2), 4)])
    with open(out / "table_test_retest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "version", "n_pairs", "exact_agreement_r1_r2", "qwk_r1_r2"]); w.writerows(tr)

    # ---- exploratory: shift ~ share of words changed + length change + natural_spoken (condition A)
    reg = []
    if edits:
        for m in models:
            ids = [i for i in main_set if (m, "A", i, "kk") in mean and (m, "A", i, "kk_nat") in mean]
            if len(ids) < 6:
                continue
            y = np.array([mean[(m, "A", i, "kk_nat")] - mean[(m, "A", i, "kk")] for i in ids])
            X = np.column_stack([np.ones(len(ids)),
                                 [float(edits[i]["share_orig_words_changed"]) for i in ids],
                                 [float(edits[i]["length_change_pct"]) / 100 for i in ids],
                                 [1.0 if items[i]["style_label"] == "natural_spoken" else 0.0 for i in ids]])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            reg.append([m, len(ids)] + [round(float(b), 4) for b in beta])
    with open(out / "table_exploratory_regression.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "n", "intercept", "b_share_words_changed", "b_length_change", "b_natural_spoken"])
        w.writerows(reg)

    # ---- main figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.7 * len(figdata)))
        for k, (m, go, cio, gn, cin) in enumerate(figdata):
            ax.errorbar(go, k + 0.12, xerr=[[go - cio[0]], [cio[1] - go]], fmt="o", color="#B4474F", capsize=3,
                        label="KK-orig − RU" if k == 0 else None)
            ax.errorbar(gn, k - 0.12, xerr=[[gn - cin[0]], [cin[1] - gn]], fmt="s", color="#2F6F8F", capsize=3,
                        label="KK-nat − RU" if k == 0 else None)
        ax.axvline(0, color="#555", lw=1)
        ax.set_yticks(range(len(figdata))); ax.set_yticklabels([f[0] for f in figdata])
        ax.set_xlabel(f"Mean score gap vs RU (condition A, n={len(gap_set)} answers, 95% bootstrap CI)")
        ax.legend(frameon=False, loc="best"); fig.tight_layout()
        fig.savefig(out / "fig_main_gap.png", dpi=200); fig.savefig(out / "fig_main_gap.pdf")
    except Exception as e:  # figure is optional; tables are the record
        print("figure skipped:", e)

    # ---- human-readable summary
    lines += ["# Results (generated by analysis/analyze.py — do not edit by hand)", "",
              f"Run files: {a.runs}; records: {len(recs)}; models: {', '.join(models)}",
              f"Returned model versions: " + "; ".join(f"{m}: {sorted(map(str, versions_seen[m]))}" for m in models),
              f"Main set: {len(main_set)} answers; gap-share set: {len(gap_set)} answers.", "",
              "## Primary: KK-nat − KK-orig (condition A)", "",
              "| Model | n | Δ (95% CI) | d_z | Wilcoxon p | Holm p |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["condition"] == "A":
            lines.append(f"| {r['model']}{'' if r['confirmatory']=='yes' else ' (exploratory)'} | {r['n_pairs']} | "
                         f"{fmt(r['mean_KKnat_minus_KKorig'])} ({fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}) | {fmt(r['cohen_dz'])} | "
                         f"{fmt(r['wilcoxon_p'],3)} | {fmt(r['holm_p'],3)} |")
    lines += ["", "## Gaps vs RU and share of gap closed", "",
              "| Model | Cond | KK-orig−RU (CI) | KK-nat−RU (CI) | Share closed (CI) |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['condition']} | {fmt(r['gap_KKorig_RU'])} ({fmt(r['gap_KKorig_RU_lo'])}, {fmt(r['gap_KKorig_RU_hi'])}) | "
                     f"{fmt(r['gap_KKnat_RU'])} ({fmt(r['gap_KKnat_RU_lo'])}, {fmt(r['gap_KKnat_RU_hi'])}) | "
                     f"{fmt(r['gap_share_closed'])} ({fmt(r['gap_share_lo'])}, {fmt(r['gap_share_hi'])}) |")
    lines += ["", "Tables: table_main.csv, table_a_prime_reduction.csv, table_parse_failed.csv, table_test_retest.csv, "
              "table_exploratory_regression.csv. Figure: fig_main_gap.png/pdf."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
