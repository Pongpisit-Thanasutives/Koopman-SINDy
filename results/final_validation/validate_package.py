#!/usr/bin/env python3
"""Independent read-only validation of the delivered numerical evidence.

Historical full-package audit of the submitted manuscript snapshot.
Run from the frozen complete research package, not a standalone code checkout.
This script does not rerun experiments; its hashes refer to the submitted files.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from validate_weak_outputs import validate as validate_weak
from validate_v2_reporting import validate as validate_v2_reporting, historical_package_path
from validate_presentation_v4 import validate as validate_presentation_v4
from validate_editorial_v5 import validate as validate_editorial_v5

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT.parent
OUT = Path(__file__).resolve().parent
required_package_files = [
    PACKAGE / "latex_source/koopman_sindy_revised.tex",
    PACKAGE / "submission/koopman_sindy_revised.pdf",
    PACKAGE / "submission/response_to_reviewers.pdf",
    PACKAGE / "local_checking/response/response_to_reviewers.tex",
]
if ROOT.name != "code_repository" or any(not path.is_file() for path in required_package_files):
    raise SystemExit(
        "This historical validator requires the frozen, complete submitted research "
        "package, including its manuscript, response, and original source hashes. "
        "It does not validate a standalone GitHub checkout. See README.md and "
        "results/final_validation/README.md for the standalone checks."
    )
checks = []
errors = []
datasets = {}


def check(name, condition, detail=None):
    item = {"check": name, "passed": bool(condition)}
    if detail is not None:
        item["detail"] = detail
    checks.append(item)
    if not condition:
        errors.append(item)


def close(a, b):
    return bool(np.allclose(a, b, rtol=1e-9, atol=1e-11, equal_nan=True))


def read(rel, n=None, key_extra=()):
    p = ROOT / rel
    d = pd.read_csv(p)
    datasets[rel] = {"rows": len(d), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    if n is not None:
        check(f"{rel}: record count", len(d) == n, {"expected": n, "actual": len(d)})
    keys = [k for k in ["system", "method", "sparse_factor", "noise", "seed", *key_extra] if k in d]
    check(f"{rel}: unique cases", not d.duplicated(keys).any())
    good = d[d.status == "ok"] if "status" in d else d
    datasets[rel]["valid_records"] = len(good)
    if "status" in d:
        datasets[rel]["status_counts"] = d.status.value_counts().to_dict()
    for col in ["support_f1", "coef_error", "practical_score"]:
        if col in good:
            check(f"{rel}: finite {col}", np.isfinite(good[col]).all())
    if "support_f1" in good:
        check(f"{rel}: F1 in [0,1]", good.support_f1.between(0, 1).all())
    if "coef_error" in good:
        check(f"{rel}: nonnegative coefficient error", (good.coef_error >= 0).all())
    if "practical_score" in good:
        check(f"{rel}: score definition", close(good.practical_score, good.support_f1/(1+good.coef_error)))
    return d


SUMMARY_STATS = {
    "support_f1_mean": ("support_f1", "mean"), "mean_support_f1": ("support_f1", "mean"),
    "support_f1_std": ("support_f1", "std"), "support_f1_median": ("support_f1", "median"),
    "coef_error_mean": ("coef_error", "mean"), "coef_error_median": ("coef_error", "median"),
    "median_coef_error": ("coef_error", "median"),
    "practical_score_mean": ("practical_score", "mean"), "mean_practical_score": ("practical_score", "mean"),
    "practical_score_median": ("practical_score", "median"),
    "exact_support_rate": ("exact_support", "mean"), "derivative_error_median": ("derivative_error", "median"),
    "exact_k_rate": ("exact_k", "mean"),
    "selected_support_median": ("support_size", "median"),
}


def summary_matches(raw, rel, keys):
    s = pd.read_csv(ROOT / rel)
    good = raw[raw.status == "ok"] if "status" in raw else raw.copy()
    if "support_size" in good:
        good = good.copy()
        good["exact_k"] = (good.support_size == good.true_support_size).astype(float)
    grouping = good.groupby(keys, dropna=False)
    for row in s.to_dict("records"):
        key = tuple(row[k] for k in keys)
        group = grouping.get_group(key)
        for col, (metric, operation) in SUMMARY_STATS.items():
            if col in row and metric in group:
                actual = getattr(group[metric], operation)()
                check(f"{rel}: {key} {col}", close(row[col], actual))
        for col in ("n_ok", "n_cases", "n"):
            if col in row:
                check(f"{rel}: {key} {col}", row[col] == len(group))


ode = read("results/ode_publication/ode_raw_results.csv", 1200)
pde = read("results/pde_publication/pde_raw_results.csv", 960)
ade = read("results/advection_diffusion_benchmark/advection_diffusion_raw.csv", 180)
qr = read("results/qr_sensitivity/qr_sensitivity_raw.csv", 545, ("q", "pod_rank"))
classical_ode = read("results/classical_interpolation_publication/ode/ode_raw_results.csv", 1600)
classical_pde = read("results/classical_interpolation_publication/pde/pde_raw_results.csv", 1920)
check("Final Appendix C POD-CV coverage",len(classical_pde[classical_pde.method=="pod_edmd_rbf"])==480)
cv=read("results/appendix_c_corrected_pod/pde_raw_results.csv",480)
strategy = read("results/upsampling_strategy_publication/upsampling_strategy_raw.csv", 600, ("strategy",))
front = read("results/fisher_kpp_front_sensitivity/fisher_kpp_front_raw.csv", 10)
nonoracle = read("results/model_selection_publication/model_selection_selected_by_equation.csv", 40, ("equation",))
tv = read("results/revision_tv_high_noise/raw_results.csv", 720)
nonoracle_tv = read("results/revision_nonoracle_tv/selected_by_equation.csv",70,("equation",))
nonoracle_tv_methods=read("results/revision_nonoracle_tv/method_records.csv",55)

for raw, prefix, stem in [(ode,"ode_publication","ode"),(pde,"pde_publication","pde"),
                         (cv,"appendix_c_corrected_pod","pde"),
                         (classical_ode,"classical_interpolation_publication/ode","ode"),
                         (classical_pde,"classical_interpolation_publication/pde","pde")]:
    for suffix, keys in [("summary",["system","method","sparse_factor","noise"]),
                         ("by_system",["system","method"])]:
        summary_matches(raw, f"results/{prefix}/{stem}_{suffix}.csv", keys)
summary_matches(ade,"results/advection_diffusion_benchmark/advection_diffusion_summary.csv",["method"])
summary_matches(front,"results/fisher_kpp_front_sensitivity/fisher_kpp_front_summary.csv",["method"])
summary_matches(strategy,"results/upsampling_strategy_publication/upsampling_strategy_summary.csv",["setting","method","strategy"])
summary_matches(qr,"results/qr_sensitivity/qr_sensitivity_summary.csv",["system","method","q","pod_rank"])
summary_matches(nonoracle,"results/model_selection_publication/model_selection_summary.csv",["system","equation","method"])
summary_matches(nonoracle_tv,"results/revision_nonoracle_tv/summary.csv",["system","equation","method"])
shared=nonoracle.merge(nonoracle_tv,on=["system","method","equation","sparse_factor","noise","seed"],suffixes=("_original","_extension"),validate="one_to_one")
check("Nonoracle TV: shared-arm coverage",len(shared)==len(nonoracle)==40)
for metric in ["threshold","support_size","support_f1","coef_error","rss","bic","ebic"]:
    check("Nonoracle TV: shared-arm "+metric,close(shared[metric+"_original"],shared[metric+"_extension"]))
summary_matches(tv,"results/revision_tv_high_noise/summary_all_noise.csv",["system","method"])
summary_matches(tv,"results/revision_tv_high_noise/summary_by_noise.csv",["system","method","noise"])
summary_matches(tv[tv.noise>=.2],"results/revision_tv_high_noise/summary_high_noise.csv",["system","method"])
classical=pd.concat([classical_ode.assign(setting="ODE"),classical_pde.assign(setting="PDE")],ignore_index=True)
summary_matches(classical,"results/classical_interpolation_publication/classical_interpolation_summary.csv",["setting","method"])
summary_matches(classical,"results/classical_interpolation_publication/classical_interpolation_by_system.csv",["setting","system","method"])

# Check the shared comparison records and copied figures directly.
case = ["system","method","sparse_factor","noise","seed"]
for label, left, right in [("Appendix C ODE EDMD", ode[ode.method=="edmd_poly3"], classical_ode[classical_ode.method=="edmd_poly3"]),
                           ("Appendix C POD CV",cv,classical_pde[classical_pde.method=="pod_edmd_rbf"])]:
    merged = left.merge(right,on=case,suffixes=("_left","_right"),validate="one_to_one")
    check(label+": identical key coverage", len(merged)==len(left)==len(right))
    for metric in ["support_f1","coef_error","practical_score"]:
        check(label+": "+metric, close(merged[metric+"_left"],merged[metric+"_right"]))

main = pd.concat([ode,pde,ade],ignore_index=True)
for label,current,filename in [("ODE",ode,"ode_publication/ode_raw_results.csv"),
                               ("PDE",pde,"pde_publication/pde_raw_results.csv"),
                               ("ADE",ade,"advection_diffusion_benchmark/advection_diffusion_raw.csv")]:
    archived=ROOT/"results/pre_complex_propagation_archive"/filename
    if archived.exists():
        old=pd.read_csv(archived)
        old=old[old.method.isin(["baseline","optdmd"])]
        now=current[current.method.isin(["baseline","optdmd"])]
        m=old.merge(now,on=case,suffixes=("_old","_new"),validate="one_to_one")
        check(label+": unaffected comparator coverage",len(m)==len(old)==len(now))
        for metric in ["support_f1","coef_error","practical_score"]:
            check(label+": retained comparator "+metric,close(m[metric+"_old"],m[metric+"_new"]))
stats = pd.read_csv(ROOT/"results/revision1_reporting/main_table_seed_cluster_se.csv")
for row in stats.to_dict("records"):
    full = main[(main.system==row["system"])&(main.method==row["method"])]
    d=full[full.status=="ok"]
    check(f"Main table {row['system']}/{row['method']}: counts",len(d)==row["n_valid"] and len(full)==row["n_total"])
    check(f"Main table {row['system']}/{row['method']}: median",close(d.coef_error.median(),row["coef_error_median"]))
    for metric,col in [("support_f1","f1"),("practical_score","score")]:
        check(f"Main table {row['system']}/{row['method']}: {col} mean",close(d[metric].mean(),row[col+"_mean"]))
        seed=d.groupby("seed")[metric].agg(["sum","count"])
        # Intercept-only CR1 sandwich; accommodates the two unbalanced optDMD rows.
        cluster_scores=seed["sum"]-seed["count"]*d[metric].mean()
        se=np.sqrt(len(seed)/(len(seed)-1)*np.dot(cluster_scores,cluster_scores))/len(d)
        check(f"Main table {row['system']}/{row['method']}: {col} seed SE",close(se,row[col+"_se"]))

wins=pd.read_csv(ROOT/"results/revision1_reporting/paired_win_rates_by_system.csv")
for row in wins.to_dict("records"):
    d=main[(main.system==row["system"]) & (main.status=="ok")]
    baseline=d[d.method=="baseline"]
    method=d[d.method==row["method"]]
    m=method.merge(baseline,on=["system","noise","sparse_factor","seed"],suffixes=("_m","_b"),validate="one_to_one")
    check(f"Paired wins {row['system']}/{row['method']}: denominator",len(m)==row["paired_n"])
    for metric,prefix,larger in [("coef_error","coefficient",False),("practical_score","score",True)]:
        winning=(m[metric+"_m"]>m[metric+"_b"]) if larger else (m[metric+"_m"]<m[metric+"_b"])
        check(f"Paired wins {row['system']}/{row['method']}: {prefix}",winning.sum()==row[prefix+"_wins"] and close(winning.mean()*100,row[prefix+"_win_percent"]))

# Recalculate high-noise metrics from the actual stored coefficient vectors.
for prefix,stem,raw in [("ode_publication","ode",ode),("pde_publication","pde",pde),
                        ("advection_diffusion_benchmark","advection_diffusion",ade),
                        ("classical_interpolation_publication/ode","ode",classical_ode),
                        ("classical_interpolation_publication/pde","pde",classical_pde),
                        ("appendix_c_corrected_pod","pde",cv)]:
    example_path=ROOT/"results"/prefix/(stem+"_coefficients_examples.csv")
    if not example_path.exists():continue
    example=pd.read_csv(example_path)
    records=raw.set_index(case)
    for key,c in example.groupby(case):
        row=records.loc[key]
        predicted=np.abs(c.coef.to_numpy())>1e-10
        true=np.abs(c.true_coef.to_numpy())>1e-10
        tp=np.sum(predicted&true);fp=np.sum(predicted&~true);fn=np.sum(~predicted&true)
        f1=2*tp/(2*tp+fp+fn)
        err=np.linalg.norm(c.coef-c.true_coef)/np.linalg.norm(c.true_coef)
        check(f"Stored coefficient example {prefix} {key}",close(f1,row.support_f1) and close(err,row.coef_error))
coef=pd.read_csv(ROOT/"results/revision_tv_high_noise/coefficients.csv")
groups=coef.groupby(case)
for row in tv[tv.status=="ok"].to_dict("records"):
    key=tuple(row[k] for k in case)
    c=groups.get_group(key)
    actual=np.abs(c.coef.to_numpy())>1e-10
    truth=np.abs(c.true_coef.to_numpy())>1e-10
    tp=np.sum(actual&truth); fp=np.sum(actual&~truth); fn=np.sum(~actual&truth)
    f1=2*tp/(2*tp+fp+fn)
    err=np.linalg.norm(c.coef-c.true_coef)/np.linalg.norm(c.true_coef)
    check(f"TV coefficients {key}",close(f1,row["support_f1"]) and close(err,row["coef_error"]) and np.array_equal(actual,truth)==bool(row["exact_support"]))

nt_coef=pd.read_csv(ROOT/"results/revision_nonoracle_tv/selected_coefficients.csv")
nt_keys=case+["equation"]
nt_groups=nt_coef.groupby(nt_keys)
for row in nonoracle_tv.to_dict("records"):
    key=tuple(row[k] for k in nt_keys)
    c=nt_groups.get_group(key)
    actual=np.abs(c.coefficient.to_numpy())>1e-10
    truth=np.abs(c.true_coefficient.to_numpy())>1e-10
    tp=np.sum(actual&truth);fp=np.sum(actual&~truth);fn=np.sum(~actual&truth)
    f1=2*tp/(2*tp+fp+fn)
    err=np.linalg.norm(c.coefficient-c.true_coefficient)/np.linalg.norm(c.true_coefficient)
    check(f"Nonoracle TV coefficients {key}",close(f1,row["support_f1"]) and close(err,row["coef_error"]) and np.array_equal(actual,truth)==bool(row["exact_support"]))
    n=row["n_regression_rows"];k=row["support_size"];p=row["n_library_terms"]
    import math
    bic=n*np.log(max(row["rss"],1e-12)/n)+k*np.log(n)
    ebic=bic+math.log(math.comb(p,k))
    check(f"Nonoracle TV EBIC {key}",close(bic,row["bic"]) and close(ebic,row["ebic"]))
for row in pd.read_csv(ROOT/"results/revision_nonoracle_tv/summary.csv").to_dict("records"):
    d=nonoracle_tv[(nonoracle_tv.system==row["system"])&(nonoracle_tv.equation==row["equation"])&(nonoracle_tv.method==row["method"])]
    check(f"Nonoracle TV support counts {row['system']}/{row['equation']}/{row['method']}",
          len(d)==row["n_valid"]==row["n_expected"] and d.exact_support.sum()==row["exact_support_count"]
          and d.exact_k.sum()==row["exact_k_count"] and d.support_size.min()==row["selected_k_min"]
          and d.support_size.max()==row["selected_k_max"] and close(d.support_size.median(),row["selected_k_median"]))

for filename in ["appendix_q_sensitivity_ode.png","appendix_qr_sensitivity_pde.png","model_selection_pareto_equationwise.png"]:
    a=ROOT/"figures"/filename;b=historical_package_path(PACKAGE,"latex_source/figures/"+filename)
    check("Figure copy "+filename,a.read_bytes()==b.read_bytes())
for filename in ["tv_high_noise_coefficient_error.pdf","tv_high_noise_support_f1.pdf","tv_noise_coefficient_score.pdf"]:
    a=ROOT/"results/revision_tv_high_noise"/filename;b=historical_package_path(PACKAGE,"latex_source/figures/"+filename)
    check("Figure copy "+filename,a.read_bytes()==b.read_bytes())
tv_reporting_manifest=json.loads((ROOT/"results/revision_tv_high_noise/artifact_generation_manifest.json").read_text())
for study in ["revision_tv_high_noise","revision_nonoracle_tv"]:
    manifest=json.loads((ROOT/"results"/study/"run_manifest.json").read_text())
    for filename,expected in manifest.get("final_package_source_sha256",manifest.get("source_sha256",{})).items():
        current=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
        # The original execution manifests remain immutable. Only the shared
        # TV reporting wrapper changed after execution; the new presentation
        # manifest records that source separately and protects the raw outputs.
        if filename == "revision_tv_comparison.py" and current != expected:
            expected_reporting=tv_reporting_manifest["reporting_source_sha256"][filename]
            check(study+": current reporting-wrapper hash "+filename,current==expected_reporting,
                  {"execution_source_sha256":expected,"current_reporting_source_sha256":current,
                   "note":"Reporting delegation/default output path changed; numerical routines unchanged."})
        elif filename == "revision_nonoracle_tv.py" and current != expected:
            presentation=json.loads((OUT/"presentation_preservation.json").read_text())
            record=presentation["reporting_only_changes"][filename]
            check(study+": current table-rendering source hash "+filename,
                  expected==record["baseline_source_sha256"] and current==record["current_source_sha256"],
                  {"execution_source_sha256":expected,"current_reporting_source_sha256":current,
                   "note":"Only build_table formatting changed; all numerical code and configuration are unchanged."})
        else:
            check(study+": source hash "+filename,current==expected)
complex_manifest=json.loads((ROOT/"results/complex_revision_manifest.json").read_text())
for filename,expected in complex_manifest["final_package_source_sha256"].items():
    check("Complex reruns: source hash "+filename,hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected)
for filename,record in complex_manifest["outputs"].items():
    check("Complex reruns: output hash "+filename,hashlib.sha256((ROOT/"results"/filename).read_bytes()).hexdigest()==record["sha256"])
reporting_manifest=json.loads((ROOT/"results/revision1_reporting/reporting_manifest.json").read_text())
for filename,expected in reporting_manifest["sha256"].items():
    check("Reporting: input hash "+filename,hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected)

# Resolve all current manuscript inputs and graphics without inspecting legacy files.
seen=set()
def dependencies(path):
    if path in seen:return
    seen.add(path)
    text=path.read_text()
    for name in re.findall(r"\\input\{([^}]+)\}",text):
        p=PACKAGE/"latex_source"/name
        if not p.suffix:p=p.with_suffix(".tex")
        check("Manuscript input "+name,p.exists())
        if p.exists():dependencies(p)
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",text):
        p=PACKAGE/"latex_source"/name
        check("Manuscript figure "+name,p.exists())
dependencies(PACKAGE/"latex_source/koopman_sindy_revised.tex")

datasets["results/revision_weak_comparison/raw_results.csv"]=validate_weak(ROOT,check,close)
validate_v2_reporting(ROOT,PACKAGE,check,close)
validate_presentation_v4(ROOT,PACKAGE,check)
validate_editorial_v5(ROOT,PACKAGE,check)

report={"description":"Independent saved-data validation; no experiments rerun", "datasets":datasets,
        "checks_count":len(checks),"passed_count":sum(c["passed"] for c in checks),"errors":errors,"checks":checks}
(OUT/"validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ["checks_count","passed_count","errors"]},indent=2))
raise SystemExit(bool(errors))
