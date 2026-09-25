"""Read-only checks of revised TV/weak summaries, tables and provenance."""
import ast
import hashlib
import json
import re
from functools import lru_cache
import numpy as np
import pandas as pd


@lru_cache(maxsize=None)
def _historical_path_map(package):
    """Exact relocation map; original preservation records remain immutable."""
    audit = json.loads((package / "code_repository/results/final_validation/document_validation.json").read_text())
    return audit.get("packaging_revision_v8", {}).get("historical_path_map", {})


def historical_package_path(package, filename):
    """Locate a historical file after the packaging-only directory cleanup."""
    return package / _historical_path_map(package).get(filename, filename)


def validate(root, package, check, close):
    def fingerprint(base, records, prefix):
        for filename, expected in records.items():
            p = base / filename
            check(prefix + filename, p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == expected)

    documents = json.loads((root / "results/final_validation/document_validation.json").read_text())
    check("Final document reference/text/log audit", documents["passed"])
    fingerprint(package, {r["file"]: r["sha256"] for r in documents["documents"]}, "Final audited document: ")
    source = documents["source_reference_audit"]
    fingerprint(package, {source["file"]: source["sha256"]}, "Final audited manuscript source: ")
    fingerprint(package, documents["compiled_source_sha256"], "Final compiled source: ")

    tvout = root / "results/revision_tv_high_noise"
    source_audit = json.loads((root / "results/final_validation/tv_reporting_change_audit.json").read_text())
    tvsource = root / "revision_tv_comparison.py"
    check("TV reporting-only source audit current hash", hashlib.sha256(tvsource.read_bytes()).hexdigest() == source_audit["current_source_sha256"])
    tree = ast.parse(tvsource.read_text())
    for name, expected in source_audit["unchanged_numeric_function_ast_sha256"].items():
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        check("TV numerical function unchanged: " + name, hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == expected)
    tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name not in ["main", "summarize"]]
    check("TV numerical/configuration module unchanged", hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == source_audit["unchanged_module_excluding_main_and_summarize_ast_sha256"])
    for study in ("revision_tv_high_noise", "revision_nonoracle_tv"):
        historical = json.loads((root / "results" / study / "run_manifest.json").read_text())
        historical_hashes = historical.get("final_package_source_sha256", historical["source_sha256"])
        check("TV source audit linked to preserved execution: " + study, historical_hashes["revision_tv_comparison.py"] == source_audit["executed_source_sha256"])
    tv = pd.read_csv(tvout / "raw_results.csv")
    tv["band"] = np.where(tv.noise < .2, "5–10%", "20–50%")
    manifest = json.loads((tvout / "artifact_generation_manifest.json").read_text())
    fingerprint(tvout, manifest["preserved_execution_sha256"], "TV preserved execution: ")
    fingerprint(root, manifest["reporting_source_sha256"], "TV current reporting source: ")
    fingerprint(root, manifest["optional_evidence_input_sha256"], "TV optional input: ")
    fingerprint(tvout, manifest["output_sha256"], "TV generated output: ")
    presentation = json.loads((root / "results/revision1_reporting/presentation_generation_manifest.json").read_text())
    fingerprint(root, presentation["protected_execution_sha256"], "Presentation protected execution: ")
    fingerprint(root, presentation["reporting_source_sha256"], "Presentation current generator: ")
    fingerprint(package, presentation["manuscript_asset_sha256"], "Presentation manuscript asset: ")

    specs = {"support_f1_mean": ("support_f1", "mean"), "coef_error_median": ("coef_error", "median"),
             "coef_error_mean": ("coef_error", "mean"), "coef_error_max": ("coef_error", "max"),
             "derivative_error_median": ("derivative_error", "median"), "state_error_median": ("state_error", "median"),
             "exact_support_count": ("exact_support", "sum"), "exact_support_rate": ("exact_support", "mean"),
             "practical_score_mean": ("practical_score", "mean")}
    for filename, keys in [("summary_by_noise_band.csv", ["system", "method", "band"]),
                           ("summary_by_noise.csv", ["system", "method", "noise"]),
                           ("summary_by_noise_spacing.csv", ["system", "method", "noise", "sparse_factor"])]:
        summary = pd.read_csv(tvout / filename); groups = tv.groupby(keys)
        check("TV updated summary complete: " + filename, len(summary) == len(groups) and not summary.duplicated(keys).any())
        for row in summary.to_dict("records"):
            key = tuple(row[k] for k in keys); d = groups.get_group(key)
            for col, (metric, operation) in specs.items():
                check("TV updated summary " + filename + str(key) + " " + col, close(row[col], getattr(d[metric], operation)()))
            check("TV updated summary denominator " + filename + str(key), row["n_ok"] == len(d) and row["n_seeds"] == d.seed.nunique() == 5)
            seed = d.groupby("seed")[["support_f1", "coef_error", "practical_score"]].mean()
            for metric in seed:
                check("TV updated summary seed SE " + filename + str(key) + " " + metric, close(row[metric + "_se"], seed[metric].sem()))

    pairs = pd.read_csv(tvout / "paired_record_comparisons.csv")
    pairkeys = ["system", "sparse_factor", "noise", "seed"]
    records = tv.set_index(pairkeys + ["method"])
    check("TV updated paired-record coverage", len(pairs) == 520 and not pairs.duplicated(pairkeys + ["comparator"]).any())
    for row in pairs.to_dict("records"):
        key = tuple(row[k] for k in pairkeys)
        method = "edmd_poly3" if row["system"] in ("lorenz63", "vanderpol_mu2") else "pod_edmd_rbf"
        e, c = records.loc[key + (method,)], records.loc[key + (row["comparator"],)]
        check("TV paired underlying records " + str(key) + "/" + row["comparator"], all(
            close(row[m + "_edmd"], e[m]) and close(row[m + "_other"], c[m]) and close(row[m + "_delta"], e[m] - c[m])
            for m in ("support_f1", "coef_error", "practical_score")))
    pairs["band"] = np.where(pairs.noise < .2, "5–10%", "20–50%")
    for filename, keys in [("paired_by_noise.csv", ["system", "noise", "comparator"]),
                           ("paired_by_noise_band.csv", ["system", "band", "comparator"]),
                           ("paired_by_noise_spacing.csv", ["system", "noise", "sparse_factor", "comparator"])]:
        summary = pd.read_csv(tvout / filename); groups = pairs.groupby(keys)
        check("TV updated paired-summary coverage " + filename, len(summary) == len(groups) and not summary.duplicated(keys).any())
        for row in summary.to_dict("records"):
            key = tuple(row[k] for k in keys); d = groups.get_group(key)
            check("TV paired denominator " + filename + str(key), row["n_pairs"] == len(d) and row["n_seeds"] == d.seed.nunique() == 5)
            for metric in ("support_f1", "coef_error", "practical_score"):
                delta = d[metric + "_delta"]; seed = d.groupby("seed")[metric + "_delta"].mean()
                sign = -1 if metric == "coef_error" else 1
                check("TV paired summary " + filename + str(key) + " " + metric,
                      close(row[metric + "_delta_mean"], delta.mean()) and close(row[metric + "_delta_seed_se"], seed.sem())
                      and row[metric + "_edmd_wins"] == int((sign * delta > 1e-12).sum())
                      and row[metric + "_ties"] == int((delta.abs() <= 1e-12).sum())
                      and row[metric + "_seed_mean_wins"] == int((sign * seed > 1e-12).sum()))

    # Compare every displayed decimal in the new two-band TV table to raw data.
    systems = {"Lorenz--63": "lorenz63", "Van der Pol": "vanderpol_mu2", "Burgers": "burgers",
               "Fisher--KPP": "fisher_kpp", "Advection--diffusion": "advection_diffusion"}
    methods = {"Raw FD": "baseline", "Polynomial EDMD": "edmd_poly3", "POD-EDMD-RBF": "pod_edmd_rbf", "TV": "tv_diff", "POD+TV": "pod_tv_diff"}
    current = None; table_rows = 0
    for line in (package / "latex_source/tables/revision_tv_high_noise.tex").read_text().splitlines():
        for display, name in systems.items():
            if "textit{" + display + "}" in line: current = name
        label = line.split("&")[0].strip()
        if label not in methods: continue
        expected = []
        for band in ["5–10%", "20–50%"]:
            d = tv[(tv.system == current) & (tv.method == methods[label]) & (tv.band == band)]
            seed = d.groupby("seed")[["support_f1", "practical_score"]].mean()
            expected.extend([d.support_f1.mean(), seed.support_f1.sem(), d.coef_error.median(), d.practical_score.mean(), seed.practical_score.sem()])
        actual = re.findall(r"\d+\.\d{3}", line)
        check("TV displayed table row " + str(current) + "/" + label, actual == [f"{x:.3f}" for x in expected])
        table_rows += 1
    check("TV displayed table complete", table_rows == 18)

    # Numerically confirm every additional point singled out in the revised text.
    vdp = tv[(tv.system == "vanderpol_mu2") & (tv.noise == .2)]
    e = vdp[(vdp.method == "edmd_poly3") & (vdp.sparse_factor == 16)].set_index("seed")
    t = vdp[(vdp.method == "tv_diff") & (vdp.sparse_factor == 16)].set_index("seed")
    check("TV prose: 20% fine VdP all five paired error/Score wins", len(e) == len(t) == 5
          and (e.coef_error < t.coef_error).all() and (e.practical_score > t.practical_score).all())
    check("TV prose: 20% fine VdP median errors", f"{e.coef_error.median():.3f}" == "0.112" and f"{t.coef_error.median():.3f}" == "0.167")
    check("TV prose: 20% coarse VdP Score", [f"{vdp[(vdp.method == m) & (vdp.sparse_factor == 64)].practical_score.mean():.3f}" for m in ["tv_diff", "edmd_poly3"]] == ["0.669", "0.567"])
    lorenz = tv[tv.system == "lorenz63"].groupby(["noise", "method"]).practical_score.mean().unstack()
    check("TV prose: EDMD above TV Lorenz score at every noise", (lorenz.edmd_poly3 > lorenz.tv_diff).all())
    high = tv[tv.noise >= .2]
    check("TV prose: no exact high-noise Lorenz/PDE support", high[high.system != "vanderpol_mu2"].exact_support.eq(0).all())
    check("TV prose: high-noise VdP exact support percentages", close(high[high.system == "vanderpol_mu2"].groupby("method").exact_support.mean().reindex(["baseline", "edmd_poly3", "tv_diff"]), [.4, .25, .6]))

    weakout = root / "results/revision_weak_comparison"
    raw = pd.read_csv(weakout / "raw_results.csv")
    weaktable = (package / "latex_source/tables/revision_weak_comparison.tex").read_text()
    order = [("vanderpol_mu2", m) for m in ["raw_weak", "linear_weak", "edmd_weak", "tv_weak"]]
    order += [("burgers", m) for m in ["raw_weak", "linear_weak", "edmd_weak", "tv_weak", "pod_weak", "pod_tv_weak"]]
    rows = [line for line in weaktable.splitlines() if re.search(r"\d+/30", line)]
    check("Weak displayed table complete", len(rows) == len(order) == 10)
    for line, (system, method) in zip(rows, order):
        d = raw[(raw.system == system) & (raw.method == method)]
        seed = d.groupby("seed")[["support_f1", "practical_score"]].mean()
        expected = [d.support_f1.mean(), seed.support_f1.sem(), d.coef_error.median(), d.practical_score.mean(), seed.practical_score.sem()]
        check("Weak displayed table " + system + "/" + method, re.findall(r"\d+\.\d{3}", line) == [f"{x:.3f}" for x in expected]
              and f"{int(d.exact_support.sum())}/{len(d)}" in line)
    weak_reporting = json.loads((weakout / "reporting_manifest.json").read_text())
    fingerprint(root, weak_reporting["source_sha256"], "Weak reporting source: ")
    weak_manifest = json.loads((weakout / "table_manifest.json").read_text())
    check("Weak table provenance", hashlib.sha256((root / "make_revision_weak_table.py").read_bytes()).hexdigest() == weak_manifest["source_sha256"]
          and hashlib.sha256((weakout / "raw_results.csv").read_bytes()).hexdigest() == weak_manifest["input_sha256"]
          and hashlib.sha256((package / weak_manifest["output"]).read_bytes()).hexdigest() == weak_manifest["output_sha256"])

    v = raw[raw.system == "vanderpol_mu2"]
    for noise, factor, values in [(.05, 64, {"edmd_weak": "0.029", "raw_weak": "0.051", "tv_weak": "0.051", "linear_weak": "0.138"}),
                                  (.1, 16, {"edmd_weak": "0.023", "raw_weak": "0.045", "tv_weak": "0.043"})]:
        d = v[(v.noise == noise) & (v.sparse_factor == factor)]
        check("Weak prose: favorable condition errors " + str((noise, factor)), all(f"{d[d.method == method].coef_error.median():.3f}" == expected for method, expected in values.items()))
    d = v[(v.noise == .05) & (v.sparse_factor == 64)]
    check("Weak prose: favorable support counts", list(d.groupby("method").exact_support.sum().reindex(["edmd_weak", "raw_weak", "tv_weak", "linear_weak"])) == [5, 4, 4, 3])
    check("Weak prose: pooled mean error caution", [f"{v[v.method == m].coef_error.mean():.3f}" for m in ["edmd_weak", "raw_weak", "tv_weak"]] == ["0.317", "0.228", "0.231"])
