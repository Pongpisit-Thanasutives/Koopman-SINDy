#!/usr/bin/env python3
"""Regenerate affected EDMD results and preserve unaffected comparison records.

Run from this directory with one BLAS thread:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python rerun_complex_revision.py
The TV revision has its own runner; this script does not touch its outputs.
"""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import koopman_sindy_ode_benchmark as ode
import koopman_sindy_pde_benchmark as pde
from koopman_sindy_model_selection_experiment import PDE_THRESHOLDS

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';WORK=RESULTS/'complex_reruns';ARCHIVE=RESULTS/'pre_complex_propagation_archive'
STAGES=['main_ode','main_pde','advection','appendix_a','appendix_c','model_selection','fisher_front','strategy','finalize']

def archive(relative):
    source=RESULTS/relative;target=ARCHIVE/relative
    if source.exists() and not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

def run_script(script,*arguments):
    command=[sys.executable,script,*map(str,arguments)]
    print('Running:', ' '.join(command),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)

def merged_results(relative,new,methods):
    archive(relative);old=pd.read_csv(ARCHIVE/relative)
    merged=pd.concat([old[~old.method.isin(methods)],new],ignore_index=True)
    merged.to_csv(RESULTS/relative,index=False)
    return merged

def merge_coefficients(relative,source,methods):
    archive(relative)
    old=pd.read_csv(ARCHIVE/relative) if (ARCHIVE/relative).exists() else pd.DataFrame()
    new=pd.read_csv(source) if Path(source).exists() else pd.DataFrame()
    kept=old[~old.method.isin(methods)] if not old.empty else old
    if not kept.empty or not new.empty:pd.concat([kept,new],ignore_index=True).to_csv(RESULTS/relative,index=False)

def summarize_core(raw,directory,prefix,module):
    directory=Path(directory);ok=raw[raw.status=='ok']
    summary=ok.groupby(['system','method','method_label','sparse_factor','noise'],as_index=False).agg(
        support_f1_mean=('support_f1','mean'),support_f1_std=('support_f1','std'),coef_error_median=('coef_error','median'),
        coef_error_mean=('coef_error','mean'),practical_score_mean=('practical_score','mean'),n_ok=('status','size'))
    summary.to_csv(directory/f'{prefix}_summary.csv',index=False)
    summary.sort_values(['system','sparse_factor','noise','practical_score_mean'],ascending=[True,True,True,False]).groupby(['system','sparse_factor','noise'],as_index=False).head(1).to_csv(directory/f'{prefix}_best_by_case.csv',index=False)
    module.make_tables(raw,str(directory),prefix);module.plot_results(raw,str(directory))

def existing_or_run(args,stage,prefix,count,command):
    f=WORK/stage/f'{prefix}_raw_results.csv'
    if args.reuse_completed and f.exists() and len(pd.read_csv(f))==count:
        print(f'Reusing completed corrected run: {stage} ({count} records)',flush=True)
    else:run_script(*command)
    data=pd.read_csv(f);assert len(data)==count,(stage,len(data),count)
    return data

def main_ode(args):
    new=existing_or_run(args,'main_ode','ode',800,['koopman_sindy_ode_benchmark.py','--preset','publication','--methods','edmd_poly3,edmd_rbf','--outdir',WORK/'main_ode','--no-progress'])
    raw=merged_results(Path('ode_publication/ode_raw_results.csv'),new,['edmd_poly3','edmd_rbf'])
    merge_coefficients(Path('ode_publication/ode_coefficients_examples.csv'),WORK/'main_ode/ode_coefficients_examples.csv',['edmd_poly3','edmd_rbf'])
    summarize_core(raw,RESULTS/'ode_publication','ode',ode)
    # Appendix C uses exactly the same polynomial cases and can reuse them.
    poly=new[new.method=='edmd_poly3'].copy()
    cr=merged_results(Path('classical_interpolation_publication/ode/ode_raw_results.csv'),poly,['edmd_poly3'])
    ex=pd.read_csv(WORK/'main_ode/ode_coefficients_examples.csv');ex[ex.method=='edmd_poly3'].to_csv(WORK/'ode_classical_coefficients.csv',index=False)
    merge_coefficients(Path('classical_interpolation_publication/ode/ode_coefficients_examples.csv'),WORK/'ode_classical_coefficients.csv',['edmd_poly3'])
    summarize_core(cr,RESULTS/'classical_interpolation_publication/ode','ode',ode)

def main_pde(args):
    new=existing_or_run(args,'main_pde','pde',320,['koopman_sindy_pde_benchmark.py','--preset','publication','--systems','burgers,fisher_kpp','--methods','pod_edmd_rbf','--rank-mode','system','--burgers-rank',8,'--fisher-rank',5,'--outdir',WORK/'main_pde','--no-progress'])
    raw=merged_results(Path('pde_publication/pde_raw_results.csv'),new,['pod_edmd_rbf'])
    merge_coefficients(Path('pde_publication/pde_coefficients_examples.csv'),WORK/'main_pde/pde_coefficients_examples.csv',['pod_edmd_rbf'])
    summarize_core(raw,RESULTS/'pde_publication','pde',pde)

def advection(args):
    from koopman_sindy_advection_diffusion_benchmark import advection_diffusion_system
    system=advection_diffusion_system();t,U=pde.integrate_pde(system,.01);k=pde.periodic_wavenumbers(48,2*np.pi)
    _,names=pde.pde_library(U[:2],k);truth=system.true_coefficients(names)
    thresholds=np.array([0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1]);rows=[];coeff=[]
    for sparse in [4,8,16]:
        ix=np.arange(0,len(t),sparse)
        if ix[-1]!=len(t)-1:ix=np.r_[ix,len(t)-1]
        for noise in [.01,.03,.05,.1]:
            for seed in range(5):
                rng=np.random.default_rng(seed+1000*sparse+100000*int(noise*1000));Y=pde.add_noise(U[ix],noise,rng);tn=pde.make_tnew(t[ix],5)
                # The original ADE loop ran optDMD before RBF and consumed this
                # random row subset. Replay that draw to preserve center choices.
                nrows=(len(tn)-2)*48
                if nrows>10000:rng.choice(nrows,size=10000,replace=False)
                fit=pde.pod_edmd_reconstruct(Y,t[ix],tn,4,'rbf',rng,12)
                result=pde.fit_pdefind_oracle(fit,tn,k,truth,thresholds,rng,max_rows=10000)
                rows.append(dict(system=system.name,method='pod_edmd_rbf',method_label='POD-EDMD-RBF',sparse_factor=sparse,noise=noise,seed=seed,upsample=5,pod_rank=4,support_f1=result['f1'],coef_error=result['coef_error'],practical_score=result['score'],threshold=result['threshold'],status='ok'))
                if seed==0:
                    for name,c,true in zip(names,result['xi'],truth):coeff.append(dict(system=system.name,method='pod_edmd_rbf',sparse_factor=sparse,noise=noise,seed=seed,feature=name,coef=c,true_coef=true))
    target=RESULTS/'advection_diffusion_benchmark';raw=merged_results(Path('advection_diffusion_benchmark/advection_diffusion_raw.csv'),pd.DataFrame(rows),['pod_edmd_rbf'])
    pd.DataFrame(coeff).to_csv(target/'advection_diffusion_coefficients_examples.csv',index=False)
    raw.groupby(['method','method_label'],as_index=False).agg(support_f1_mean=('support_f1','mean'),coef_error_median=('coef_error','median'),practical_score_mean=('practical_score','mean')).to_csv(target/'advection_diffusion_summary.csv',index=False)
    for key in ['noise','sparse_factor']:raw.groupby([key,'method_label']).coef_error.median().unstack().to_csv(target/f'advection_diffusion_coeferr_by_{key}.csv')
    base=raw[raw.method=='baseline'];wins=[]
    for method in ['optdmd','pod_edmd_rbf']:
        d=base.merge(raw[raw.method==method],on=['sparse_factor','noise','seed'],suffixes=('_base','_method'))
        wins.append(dict(method=method,method_label=d.method_label_method.iloc[0],matched_cases=len(d),coef_error_win_rate=float((d.coef_error_method<d.coef_error_base).mean()),practical_score_win_rate=float((d.practical_score_method>d.practical_score_base).mean())))
    pd.DataFrame(wins).to_csv(target/'advection_diffusion_winrate_vs_baseline.csv',index=False)

def appendix_a(args):
    archive(Path('qr_sensitivity/qr_sensitivity_raw.csv'))
    run_script('koopman_sindy_qr_sensitivity.py','--preset','appendix','--outdir',RESULTS/'qr_sensitivity')

def appendix_c(args):
    new=existing_or_run(args,'appendix_c_pod','pde',480,['koopman_sindy_pde_benchmark.py','--preset','publication','--systems','burgers,fisher_kpp,advection_diffusion','--methods','pod_edmd_rbf','--rank-mode','cv','--rank-grid','1,2,3,4,6,8','--rank-cv-folds',3,'--outdir',WORK/'appendix_c_pod','--no-progress'])
    shutil.copytree(WORK/'appendix_c_pod',RESULTS/'appendix_c_corrected_pod',dirs_exist_ok=True)
    raw=merged_results(Path('classical_interpolation_publication/pde/pde_raw_results.csv'),new,['pod_edmd_rbf'])
    merge_coefficients(Path('classical_interpolation_publication/pde/pde_coefficients_examples.csv'),WORK/'appendix_c_pod/pde_coefficients_examples.csv',['pod_edmd_rbf'])
    summarize_core(raw,RESULTS/'classical_interpolation_publication/pde','pde',pde)
    run_script('classical_interpolation_baselines.py','--skip-run','--outdir',RESULTS/'classical_interpolation_publication')

def model_selection(args):
    assert importlib.metadata.version('kneed')=='0.8.6','Install kneed==0.8.6 for the specified selector.'
    for name in ['model_selection_candidates.csv','model_selection_selected_by_equation.csv','model_selection_pareto_front.csv']:archive(Path('model_selection_publication')/name)
    run_script('koopman_sindy_model_selection_experiment.py','--preset','quick','--seeds','0,1,2,3,4','--noise','.01','--ode-sparse-factor',8,'--pde-sparse-factor',4,'--ode-systems','vanderpol_mu2','--pde-systems','burgers,fisher_kpp','--fisher-ic','front','--fisher-rank',2,'--max-rows',5000,'--outdir',RESULTS/'model_selection_publication','--no-progress')

def fisher_front(args):
    relative=Path('fisher_kpp_front_sensitivity/fisher_kpp_front_raw.csv');archive(relative);old=pd.read_csv(ARCHIVE/relative)
    sy=pde.fisher_kpp_system(48,ic_variant='front');t,U=pde.integrate_pde(sy,.01);ix=np.arange(0,len(t),8)
    if ix[-1]!=len(t)-1:ix=np.r_[ix,len(t)-1]
    k=pde.periodic_wavenumbers(48,2*np.pi);_,names=pde.pde_library(U[:2],k);truth=sy.true_coefficients(names);rows=[];coefs=[]
    for seed in range(5):
        rng=np.random.default_rng(seed+1008000);Y=pde.add_noise(U[ix],.01,rng)
        b=pde.fit_pdefind_oracle(Y,t[ix],k,truth,PDE_THRESHOLDS,rng,max_rows=5000)
        expected=old[(old.method=='baseline')&(old.seed==seed)].iloc[0]
        assert np.isclose(b['coef_error'],expected.coef_error,rtol=1e-10,atol=1e-12),'front provenance mismatch'
        tn=pde.make_tnew(t[ix],5);fit=pde.pod_edmd_reconstruct(Y,t[ix],tn,2,'rbf',rng,20)
        z=pde.fit_pdefind_oracle(fit,tn,k,truth,PDE_THRESHOLDS,rng,max_rows=5000)
        rows.append(dict(system='fisher_kpp_front',method='pod_edmd_rbf',method_label='POD-EDMD-RBF',seed=seed,sparse_factor=8,noise=.01,pod_rank=2,support_f1=z['f1'],coef_error=z['coef_error'],practical_score=z['score'],threshold=z['threshold']))
        for name,c,true in zip(names,z['xi'],truth):coefs.append(dict(seed=seed,feature=name,coef=c,true_coef=true))
    raw=merged_results(relative,pd.DataFrame(rows),['pod_edmd_rbf']);target=RESULTS/'fisher_kpp_front_sensitivity'
    raw.groupby(['method','method_label'],as_index=False).agg(support_f1_mean=('support_f1','mean'),coef_error_median=('coef_error','median'),practical_score_mean=('practical_score','mean')).to_csv(target/'fisher_kpp_front_summary.csv',index=False)
    pd.DataFrame(coefs).to_csv(target/'fisher_kpp_front_coefficients.csv',index=False)

def retain_strategy_baselines():
    target=RESULTS/'upsampling_strategy_publication'
    raw=pd.read_csv(target/'upsampling_strategy_raw.csv')
    raw.to_csv(WORK/'strategy_complete_reexecution.csv',index=False)
    old=pd.read_csv(ARCHIVE/'upsampling_strategy_publication/upsampling_strategy_raw.csv')
    key=['setting','system','method','strategy','seed','sparse_factor','noise']
    drift=old[old.strategy=='baseline'].merge(raw[raw.strategy=='baseline'],on=key,suffixes=('_archived','_reexecuted'))
    drift['coefficient_error_drift']=drift.coef_error_reexecuted-drift.coef_error_archived
    drift.to_csv(WORK/'strategy_baseline_reexecution_comparison.csv',index=False)
    assert (drift.support_f1_reexecuted==drift.support_f1_archived).all()
    raw=pd.concat([raw[raw.strategy!='baseline'],old[old.strategy=='baseline']],ignore_index=True)
    raw.to_csv(target/'upsampling_strategy_raw.csv',index=False)
    ok=raw[raw.status=='ok'];summary=ok.groupby(['setting','method','strategy'],as_index=False).agg(mean_support_f1=('support_f1','mean'),median_coef_error=('coef_error','median'),mean_practical_score=('practical_score','mean'),n_ok=('status','size'))
    order={'baseline':0,'local_reset':1,'global_rollout':2,'residual_corrected':3,'two_sided':4}
    summary['strategy_order']=summary.strategy.map(order);summary=summary.sort_values(['setting','method','strategy_order']).drop(columns='strategy_order')
    summary.to_csv(target/'upsampling_strategy_summary.csv',index=False)
    bysys=ok.groupby(['setting','system','method','strategy'],as_index=False).agg(mean_support_f1=('support_f1','mean'),median_coef_error=('coef_error','median'),mean_practical_score=('practical_score','mean'))
    bysys.to_csv(target/'upsampling_strategy_by_system.csv',index=False)
    (target/'upsampling_strategy_tables.md').write_text('# Complex-propagation strategy ablation\n\nArchived unaffected baselines are retained; paired assisted strategies use identical noisy observations and RBF centers.\n\n'+summary.to_markdown(index=False,floatfmt='.3f')+'\n\n'+bysys.to_markdown(index=False,floatfmt='.3f')+'\n')


def strategy(args):
    archive(Path('upsampling_strategy_publication/upsampling_strategy_raw.csv'))
    run_script('dmd_upsampling_strategy_ablation.py','--outdir',RESULTS/'upsampling_strategy_publication','--seeds',','.join(map(str,range(20))),'--noise','.10','--ode-sparse-factor',64,'--pde-sparse-factor',32,'--nx',64,'--rank',8,'--rbf-centers',40,'--no-progress')
    retain_strategy_baselines()

def finalize(args):
    record_paths=['ode_publication/ode_raw_results.csv','pde_publication/pde_raw_results.csv','advection_diffusion_benchmark/advection_diffusion_raw.csv','qr_sensitivity/qr_sensitivity_raw.csv','classical_interpolation_publication/ode/ode_raw_results.csv','classical_interpolation_publication/pde/pde_raw_results.csv','model_selection_publication/model_selection_selected_by_equation.csv','fisher_kpp_front_sensitivity/fisher_kpp_front_raw.csv','upsampling_strategy_publication/upsampling_strategy_raw.csv']
    manifest={'python':platform.python_version(),'stages':STAGES,'seed_policy':'Original main observation/RBF seeds retained; ADE replays original prior optDMD row draw; Appendix A uses paired stable seeds; strategy uses shared post-noise RNG state across strategies.','final_package_source_sha256':{},'outputs':{},'packages':{}}
    for fn in ['koopman_propagation.py','koopman_sindy_ode_benchmark.py','koopman_sindy_pde_benchmark.py','dmd_upsampling_strategy_ablation.py','rerun_complex_revision.py']:
        manifest['final_package_source_sha256'][fn]=hashlib.sha256((ROOT/fn).read_bytes()).hexdigest()
    for name in ['numpy','scipy','pandas','matplotlib','kneed','tabulate']:
        manifest['packages'][name]=importlib.metadata.version(name)
    for relative in record_paths:
        f=RESULTS/relative;d=pd.read_csv(f);entry={'records':len(d),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
        if 'status' in d:entry['status_counts']=d.status.value_counts().to_dict()
        manifest['outputs'][relative]=entry
        # All baseline records must retain their original noisy-data outcomes.
        oldpath=ARCHIVE/relative
        if oldpath.exists() and 'method' in d:
            old=pd.read_csv(oldpath);key=[c for c in ['setting','system','equation','method','strategy','sparse_factor','noise','seed','q','pod_rank'] if c in d and c in old]
            oldb=old[old.method=='baseline'];newb=d[d.method=='baseline']
            # The strategy CSV stores baselines in its strategy column.
            if 'strategy' in d:oldb=old[old.strategy=='baseline'];newb=d[d.strategy=='baseline']
            metrics=[c for c in ['support_f1','coef_error','practical_score'] if c in d and c in old]
            if len(oldb) and len(oldb)==len(newb):
                x=oldb.sort_values(key)[metrics].to_numpy();y=newb.sort_values(key)[metrics].to_numpy()
                assert np.allclose(x,y,rtol=1e-8,atol=1e-8,equal_nan=True),(relative,'baseline mismatch')
                entry['baseline_records_matching_to_tolerance']=len(oldb)
                entry['maximum_absolute_baseline_metric_drift']=float(np.nanmax(np.abs(x-y)))
                entry['baseline_comparison_tolerance']={'rtol':1e-8,'atol':1e-8}
    strategy_provenance=WORK/'strategy_execution_source_provenance.json'
    if strategy_provenance.exists():manifest['strategy_execution_source_provenance']=json.loads(strategy_provenance.read_text())
    drift=pd.read_csv(WORK/'strategy_baseline_reexecution_comparison.csv')
    manifest['strategy_archived_baseline_maximum_reexecution_coefficient_error_drift']=float(drift.coefficient_error_drift.abs().max())
    provenance=WORK/'isolated_execution_source_provenance.json'
    if provenance.exists():manifest['isolated_execution_source_provenance']=json.loads(provenance.read_text())
    (RESULTS/'complex_revision_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['outputs'],indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['all',*STAGES],default='all');ap.add_argument('--reuse-completed',action='store_true',help='Reuse completed isolated corrected main/C runs from this same source version.')
    args=ap.parse_args();os.chdir(ROOT);WORK.mkdir(parents=True,exist_ok=True);ARCHIVE.mkdir(parents=True,exist_ok=True)
    for stage in STAGES if args.stage=='all' else [args.stage]:
        print(f'=== {stage} ===',flush=True);globals()[stage](args)

if __name__=='__main__':main()
