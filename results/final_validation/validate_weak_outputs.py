#!/usr/bin/env python3
"""Independent checks of saved weak-form outcomes; no simulations or fits."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd


def validate(root, check, close):
    out = root / "results/revision_weak_comparison"
    protocol = json.loads((root / "revision_weak_protocol.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    raw = pd.read_csv(out / "raw_results.csv")
    keys = ["system", "sparse_factor", "noise", "seed"]
    armkeys = keys + ["method"]
    expected = {
        (name, factor, noise, seed, method)
        for name, cfg in protocol["systems"].items()
        for factor, noise, seed, method in itertools.product(
            cfg["factors"], protocol["noise"], protocol["seeds"], cfg["methods"])
    }
    observed = set(raw[armkeys].itertuples(index=False, name=None))
    check("Weak: frozen complete method coverage", observed == expected and len(raw) == len(expected) == 300)
    check("Weak: all runs successful", raw.status.eq("ok").all())
    check("Weak: all arms share identical noisy observations", raw.groupby(keys).observation_sha256.nunique().eq(1).all())
    check("Weak: protocol preserved in manifest", protocol == manifest["protocol"])
    check("Weak: protocol and execution counts", protocol["records"] == manifest["n_tasks"] == 60
          and protocol["method_rows"] == manifest["n_method_rows"] == manifest["n_expected_method_rows"] == 300
          and manifest["n_failed"] == 0 and manifest["sources_unchanged"])
    for filename, expected_hash in manifest["source_sha256"].items():
        check("Weak source hash: " + filename, hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected_hash)
    for filename, expected_hash in manifest["output_sha256"].items():
        check("Weak output hash: " + filename, hashlib.sha256((out / filename).read_bytes()).hexdigest() == expected_hash)
    for name, cfg in protocol["systems"].items():
        d = raw[raw.system == name]
        check("Weak physical sampling: " + name, close(d.obs_dt, cfg["base_dt"] * d.sparse_factor)
              and close(d.observed_t_end, cfg["horizon"])
              and close(d.n_observed, np.rint(cfg["horizon"] / d.obs_dt) + 1))
        check("Weak sample counts: " + name, close(d.n_state_samples, (d.n_observed - 1) * d.upsample + 1))
        check("Weak fixed test rows: " + name, d.weak_rows.eq(cfg["n_time_centers"] * cfg.get("n_space_centers", 1)).all())
        check("Weak library size: " + name, d.library_columns.eq(10 if name == "vanderpol_mu2" else 8).all())
        threshold_grid = protocol["ode_thresholds" if name == "vanderpol_mu2" else "pde_thresholds"]
        check("Weak allowed threshold grid: " + name, all(any(close(x, t) for t in threshold_grid) for x in d.threshold))
        for method, arm in d.groupby("method"):
            dense = method in ("edmd_weak", "linear_weak")
            check("Weak common preprocessing: " + name + "/" + method,
                  arm.upsample.eq(5 if dense else 1).all()
                  and arm.pod_rank.eq(cfg.get("pod_rank", 0) if method in ("edmd_weak", "pod_weak", "pod_tv_weak") else 0).all())
    for metric in ("support_f1", "coef_error", "practical_score", "state_error", "weak_truth_residual", "weak_fit_residual"):
        check("Weak finite nonnegative " + metric, np.isfinite(raw[metric]).all() and raw[metric].ge(0).all())
    check("Weak F1 bounds", raw.support_f1.between(0, 1).all())
    check("Weak score definition", close(raw.practical_score, raw.support_f1 / (1 + raw.coef_error)))

    coef = pd.read_csv(out / "coefficients.csv")
    check("Weak unique stored coefficients", not coef.duplicated(armkeys + ["feature", "equation"]).any())
    groups = coef.groupby(armkeys)
    check("Weak coefficient arm coverage", set(groups.groups) == expected)
    for row in raw.to_dict("records"):
        key = tuple(row[k] for k in armkeys)
        c = groups.get_group(key)
        active, true = np.abs(c.coef.to_numpy()) > 1e-10, np.abs(c.true_coef.to_numpy()) > 1e-10
        tp = np.sum(active & true); fp = np.sum(active & ~true); fn = np.sum(~active & true)
        f1 = 2 * tp / (2 * tp + fp + fn)
        error = np.linalg.norm(c.coef - c.true_coef) / np.linalg.norm(c.true_coef)
        check("Weak coefficient-derived metrics " + str(key), close(f1, row["support_f1"])
              and close(error, row["coef_error"]) and int(np.array_equal(active, true)) == row["exact_support"])
        check("Weak coefficient vector length " + str(key), len(c) == (20 if row["system"] == "vanderpol_mu2" else 8))

    summary_specs = {
        "support_f1_mean": ("support_f1", "mean"), "support_f1_se": ("support_f1", "sem"),
        "coef_error_median": ("coef_error", "median"), "coef_error_mean": ("coef_error", "mean"),
        "coef_error_se": ("coef_error", "sem"), "practical_score_mean": ("practical_score", "mean"),
        "practical_score_se": ("practical_score", "sem"), "exact_support_rate": ("exact_support", "mean"),
        "state_error_median": ("state_error", "median"), "weak_truth_residual_median": ("weak_truth_residual", "median"),
    }
    for filename, groupkeys in [("summary_by_condition.csv", ["system", "method", "sparse_factor", "noise"]),
                                ("summary_all_conditions.csv", ["system", "method"])]:
        summary = pd.read_csv(out / filename)
        group = raw.groupby(groupkeys)
        check("Weak summary coverage " + filename, len(summary) == len(group) and not summary.duplicated(groupkeys).any())
        for row in summary.to_dict("records"):
            key = tuple(row[k] for k in groupkeys); d = group.get_group(key)
            for column, (metric, statistic) in summary_specs.items():
                if column in row:
                    check("Weak " + filename + " " + str(key) + " " + column, close(row[column], getattr(d[metric], statistic)()))
            for column in ("n_planned", "n_ok"):
                if column in row: check("Weak " + filename + " " + str(key) + " " + column, row[column] == len(d))
    seedmeans = pd.read_csv(out / "pooled_seed_means.csv")
    byseed = raw.groupby(["system", "method", "seed"])
    check("Weak seed-mean coverage", len(seedmeans) == len(byseed) == 50)
    for row in seedmeans.to_dict("records"):
        key = tuple(row[k] for k in ["system", "method", "seed"]); d = byseed.get_group(key)
        check("Weak seed aggregation " + str(key), len(d) == 6 and all(close(row[m], d[m].mean()) for m in ("support_f1", "practical_score", "coef_error")))
    seedsummary = pd.read_csv(out / "pooled_seed_summary.csv")
    seedgroups = seedmeans.groupby(["system", "method"])
    for row in seedsummary.to_dict("records"):
        key = (row["system"], row["method"]); d = seedgroups.get_group(key)
        check("Weak pooled independent-seed SE " + str(key), len(d) == 5 and all(
            close(row[m + "_mean"], d[m].mean()) and close(row[m + "_se"], d[m].sem())
            for m in ("support_f1", "practical_score", "coef_error")))
    pairs = pd.read_csv(out / "paired_differences.csv")
    record = raw.set_index(armkeys)
    check("Weak paired coverage", len(pairs) == 240 and not pairs.duplicated(keys + ["control"]).any())
    for row in pairs.to_dict("records"):
        key = tuple(row[k] for k in keys)
        e = record.loc[key + ("edmd_weak",)]; c = record.loc[key + (row["control"],)]
        delta = e.practical_score - c.practical_score
        check("Weak paired contrasts " + str(key) + "/" + row["control"],
              close(row["delta_f1"], e.support_f1 - c.support_f1)
              and close(row["delta_error"], e.coef_error - c.coef_error)
              and close(row["delta_score"], delta)
              and row["edmd_score_win"] == int(delta > 1e-12) and row["edmd_score_tie"] == int(abs(delta) <= 1e-12))
    for row in pd.read_csv(out / "paired_summary.csv").to_dict("records"):
        d = pairs[(pairs.system == row["system"]) & (pairs.control == row["control"])]
        check("Weak paired summary " + row["system"] + "/" + row["control"],
              row["n_pairs"] == len(d) == 30 and close(row["mean_delta_f1"], d.delta_f1.mean())
              and close(row["median_delta_error"], d.delta_error.median()) and close(row["mean_delta_score"], d.delta_score.mean())
              and row["edmd_score_wins"] == d.edmd_score_win.sum() and row["score_ties"] == d.edmd_score_tie.sum())

    cv = pd.read_csv(out / "tv_cv.csv")
    tv = raw[raw.method.isin(["tv_weak", "pod_tv_weak"])]
    check("Weak TV complete candidate coverage", len(cv) == 2430 and len(tv) == 90
          and cv.groupby(armkeys).size().eq(27).all() and not cv.duplicated(armkeys + ["lam", "fold"]).any())
    check("Weak TV candidate convergence", cv.solver_converged.eq(True).all()
          and (cv.solver_max_relative_gap <= 1e-5 + 1e-12).all() and np.isfinite(cv.cv_mse).all())
    check("Weak TV final convergence", tv.tv_final_converged.eq(True).all() and (tv.tv_final_max_relative_gap <= 1e-5 + 1e-12).all())
    cvgroups = cv.groupby(armkeys)
    for row in tv.to_dict("records"):
        key = tuple(row[k] for k in armkeys); d = cvgroups.get_group(key)
        means = d.groupby("lam").cv_mse.mean().sort_index()
        check("Weak TV observation-only selected candidate " + str(key), close(row["tv_lambda"], means.idxmin())
              and close(row["tv_cv_mse"], means.min()) and d.fold.nunique() == 3 and d.lam.nunique() == 9)
    library_validation = json.loads((out / "validation.json").read_text())
    check("Weak library identity and clean-resolution checks", len(library_validation["checks"]) == 19 and all(c["passed"] for c in library_validation["checks"]))
    return {"rows": len(raw), "successful_rows": int(raw.status.eq("ok").sum()),
            "datasets": len(raw.groupby(keys)), "tv_cv_fits": len(cv),
            "raw_sha256": hashlib.sha256((out / "raw_results.csv").read_bytes()).hexdigest()}


if __name__ == "__main__":
    checks = []
    def check(name, condition):
        checks.append({"check": name, "passed": bool(condition)})
    def close(a, b):
        return bool(np.allclose(a, b, rtol=1e-9, atol=1e-11, equal_nan=True))
    root = Path(__file__).resolve().parents[2]
    metadata = validate(root, check, close)
    report = {"description": "Independent saved-data weak-form validation; no experiments rerun",
              "metadata": metadata, "checks_count": len(checks),
              "passed_count": sum(c["passed"] for c in checks),
              "errors": [c for c in checks if not c["passed"]], "checks": checks}
    (Path(__file__).parent / "weak_validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("metadata", "checks_count", "passed_count", "errors")}, indent=2))
    raise SystemExit(bool(report["errors"]))
