#!/usr/bin/env python3
"""Reporting only for the archived TV comparison; never fits or reruns a model.

Run from any directory:
  python /path/to/code_repository/revision_tv_reporting.py
The execution manifest and raw results remain byte-for-byte unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SYSTEMS = {'lorenz63':'Lorenz–63','vanderpol_mu2':'Van der Pol','burgers':'Burgers',
           'fisher_kpp':'Fisher–KPP','advection_diffusion':'Advection–diffusion'}
LABELS = {'baseline':'Raw FD','edmd_poly3':'Polynomial EDMD','pod_edmd_rbf':'POD-EDMD-RBF',
          'tv_diff':'TV','pod_tv_diff':'POD+TV'}
METRICS = ['support_f1','coef_error','practical_score']
COLORS = {'baseline':'#555555','edmd_poly3':'#1764a0','pod_edmd_rbf':'#1764a0',
          'tv_diff':'#b96011','pod_tv_diff':'#38834b'}
MARKERS = {'baseline':'o','edmd_poly3':'s','pod_edmd_rbf':'s','tv_diff':'^','pod_tv_diff':'D'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def aggregate(raw, keys):
    """Record-level means/medians; SE of seed means on each fixed regime grid."""
    summary = raw.groupby(keys,as_index=False).agg(
        support_f1_mean=('support_f1','mean'),coef_error_median=('coef_error','median'),
        coef_error_mean=('coef_error','mean'),coef_error_max=('coef_error','max'),
        derivative_error_median=('derivative_error','median'),state_error_median=('state_error','median'),
        exact_support_count=('exact_support','sum'),exact_support_rate=('exact_support','mean'),
        practical_score_mean=('practical_score','mean'),n_ok=('status','size'),n_seeds=('seed','nunique'))
    seedmeans=raw.groupby(keys+['seed'],as_index=False)[METRICS].mean()
    ses=seedmeans.groupby(keys)[METRICS].sem().add_suffix('_se').reset_index()
    return summary.merge(ses,on=keys,validate='one_to_one')

def paired_summary(raw, groups):
    """Strict matched EDMD-minus-comparator differences; ties are not wins."""
    keys=['system','sparse_factor','noise','seed']
    assisted=raw[raw.method.isin(['edmd_poly3','pod_edmd_rbf'])]
    rows=[]
    for comparator in ['baseline','tv_diff','pod_tv_diff']:
        paired=assisted.merge(raw[raw.method==comparator],on=keys,suffixes=('_edmd','_other'),validate='one_to_one')
        # band is duplicated after the exact-record join.
        paired['band']=paired['band_edmd']
        for key,frame in paired.groupby(groups,sort=False):
            key=(key,) if not isinstance(key,tuple) else key
            row={**dict(zip(groups,key)),'comparator':comparator,'n_pairs':len(frame),'n_seeds':frame.seed.nunique()}
            for metric in METRICS:
                delta=frame[metric+'_edmd']-frame[metric+'_other']
                win=delta < -1e-12 if metric=='coef_error' else delta > 1e-12
                tie=np.abs(delta)<=1e-12
                means=pd.DataFrame({'seed':frame.seed,'delta':delta}).groupby('seed').delta.mean()
                row.update({metric+'_delta_mean':delta.mean(),metric+'_delta_seed_se':means.sem(),
                            metric+'_edmd_wins':int(win.sum()),metric+'_ties':int(tie.sum()),
                            metric+'_seed_mean_wins':int(((means < -1e-12) if metric=='coef_error' else (means > 1e-12)).sum())})
            rows.append(row)
    return pd.DataFrame(rows)

def record_comparisons(raw):
    keys=['system','sparse_factor','noise','seed']
    assisted=raw[raw.method.isin(['edmd_poly3','pod_edmd_rbf'])]
    frames=[]
    for comparator in ['baseline','tv_diff','pod_tv_diff']:
        pairs=assisted.merge(raw[raw.method==comparator],on=keys,suffixes=('_edmd','_other'),validate='one_to_one')
        pairs['comparator']=comparator
        for metric in METRICS:
            pairs[metric+'_delta']=pairs[metric+'_edmd']-pairs[metric+'_other']
        columns=keys+['comparator']+[m+suffix for m in METRICS for suffix in ['_edmd','_other','_delta']]
        frames.append(pairs[columns])
    return pd.concat(frames,ignore_index=True)

def nonoracle_evidence(outdir):
    source=ROOT/'results/revision_nonoracle_tv/selected_by_equation.csv'
    if not source.exists():return {}
    raw=pd.read_csv(source);raw['practical_score']=raw.support_f1/(1+raw.coef_error)
    keys=['system','equation','method']
    summary=raw.groupby(keys,as_index=False).agg(n=('seed','size'),support_f1_mean=('support_f1','mean'),
        coef_error_median=('coef_error','median'),practical_score_mean=('practical_score','mean'))
    exact=raw.assign(exact=np.isclose(raw.support_f1,1,rtol=0,atol=1e-12)).groupby(keys).exact.sum().rename('exact_support_count').reset_index()
    summary=summary.merge(exact,on=keys)
    summary.to_csv(outdir/'nonoracle_tv_evidence.csv',index=False)
    rows=[]
    index=['system','equation','seed']
    assisted=raw[raw.method.isin(['edmd_poly3','pod_edmd_rbf'])]
    for method in ['baseline','tv_diff','pod_tv_diff']:
        paired=assisted.merge(raw[raw.method==method],on=index,suffixes=('_edmd','_other'),validate='one_to_one')
        for (system,equation),g in paired.groupby(['system','equation']):
            row=dict(system=system,equation=equation,comparator=method,n_pairs=len(g))
            for metric in METRICS:
                delta=g[metric+'_edmd']-g[metric+'_other']
                row[metric+'_edmd_wins']=int(((delta < -1e-12) if metric=='coef_error' else (delta > 1e-12)).sum())
                row[metric+'_ties']=int((abs(delta)<=1e-12).sum())
                row[metric+'_delta_mean']=delta.mean()
                row[metric+'_delta_seed_se']=delta.sem()
            rows.append(row)
    pd.DataFrame(rows).to_csv(outdir/'nonoracle_tv_paired_evidence.csv',index=False)
    return {str(source.relative_to(ROOT)):digest(source)}

def figures(summary,outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    plt.rcParams.update({'font.family':'serif','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
                         'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42})
    x=np.arange(4)
    fig,axes=plt.subplots(5,2,figsize=(5.65,7.25),layout='constrained')
    for row,(system,title) in enumerate(SYSTEMS.items()):
        block=summary[summary.system==system]
        for method in ['baseline','edmd_poly3','pod_edmd_rbf','tv_diff','pod_tv_diff']:
            d=block[block.method==method].sort_values('noise')
            if d.empty:continue
            style=dict(color=COLORS[method],marker=MARKERS[method],ms=3,lw=1)
            axes[row,0].plot(x,d.coef_error_median,**style)
            axes[row,1].errorbar(x,d.practical_score_mean,yerr=d.practical_score_se,capsize=2,**style)
        for col,ax in enumerate(axes[row]):
            ax.set_xticks(x,[5,10,20,50]);ax.grid(alpha=.2)
            ax.axvline(1.5,color='#bbbbbb',linestyle=':',lw=.7,zorder=0)
            ax.set_title(f'({chr(97+2*row+col)}) {title}',loc='left',pad=2)
            if row==4:ax.set_xlabel('Relative noise (%)')
        axes[row,0].set_yscale('log')
        ticks={'lorenz63':[.9,1.,1.2],'vanderpol_mu2':[.02,.1,1.],
               'burgers':[.3,.5,1.],'fisher_kpp':[.03,.1,1.],'advection_diffusion':[.05,.1,1.]}
        axes[row,0].yaxis.set_major_locator(FixedLocator(ticks[system]))
        axes[row,0].yaxis.set_minor_locator(NullLocator())
        axes[row,0].yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        axes[row,0].set_ylabel('Median error')
        axes[row,1].set_ylim(0,1.04);axes[row,1].set_yticks([0,.5,1]);axes[row,1].set_ylabel('Mean Score')
    legend=[('baseline','Raw FD'),('edmd_poly3','EDMD / POD-EDMD'),('tv_diff','TV'),('pod_tv_diff','POD+TV')]
    handles=[Line2D([],[],color=COLORS[m],marker=MARKERS[m],ms=3,lw=1,label=l) for m,l in legend]
    fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False,columnspacing=1.2,handletextpad=.5)
    for ext in ['pdf','png']:fig.savefig(outdir/f'tv_noise_coefficient_score.{ext}',dpi=240)
    plt.close(fig)
    # Retain individual-metric companion assets and their historical filenames.
    for metric,filename,ylabel in [('coef_error_median','tv_high_noise_coefficient_error','Median coefficient error'),
                                   ('support_f1_mean','tv_high_noise_support_f1','Mean support F1')]:
        fig,axes=plt.subplots(2,3,figsize=(7.5,4.5),layout='constrained')
        for ax,(system,title) in zip(axes.flat,SYSTEMS.items()):
            for method,d in summary[summary.system==system].groupby('method',sort=False):
                d=d.sort_values('noise');kw=dict(color=COLORS[method],marker=MARKERS[method],ms=3,lw=1)
                if metric=='support_f1_mean':ax.errorbar(x,d[metric],yerr=d.support_f1_se,capsize=2,**kw)
                else:ax.plot(x,d[metric],**kw)
            ax.set_title(title);ax.set_xticks(x,[5,10,20,50]);ax.set_xlabel('Relative noise (%)');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
            if metric=='coef_error_median':ax.set_yscale('log')
            else:ax.set_ylim(0,1.04)
        axes.flat[-1].axis('off');axes.flat[-1].legend(handles=handles,loc='center',frameon=False)
        for ext in ['pdf','png']:fig.savefig(outdir/f'{filename}.{ext}',dpi=200)
        plt.close(fig)

def summarize(outdir):
    outdir=Path(outdir).resolve()
    preserved=['raw_results.csv','coefficients.csv','tv_cv_diagnostics.csv','run_manifest.json']
    before={n:digest(outdir/n) for n in preserved if (outdir/n).exists()}
    raw=pd.read_csv(outdir/'raw_results.csv')
    keys=['system','method','sparse_factor','noise','seed']
    assert not raw.duplicated(keys).any()
    ok=raw[(raw.status=='ok') & np.isfinite(raw[METRICS]).all(axis=1)].copy()
    assert len(ok)==len(raw), 'Review incomplete records before producing the complete-grid table.'
    ok['band']=np.where(ok.noise<=.1,'5–10%','20–50%')
    base=['system','method','method_label']
    summary=aggregate(ok,base+['noise']);summary.to_csv(outdir/'summary_by_noise.csv',index=False)
    aggregate(ok,base+['noise','sparse_factor']).to_csv(outdir/'summary_by_noise_spacing.csv',index=False)
    aggregate(ok,base).to_csv(outdir/'summary_all_noise.csv',index=False)
    bands=aggregate(ok,base+['band']);bands.to_csv(outdir/'summary_by_noise_band.csv',index=False)
    bands[bands.band=='20–50%'].drop(columns='band').to_csv(outdir/'summary_high_noise.csv',index=False)
    for grouping,name in [(['system','noise'],'paired_by_noise'),(['system','noise','sparse_factor'],'paired_by_noise_spacing'),
                           (['system','band'],'paired_by_noise_band')]:
        paired_summary(ok,grouping).to_csv(outdir/(name+'.csv'),index=False)
    ok[ok.method.str.contains('tv_diff')].groupby(['system','method'],as_index=False).agg(
        n=('status','size'),lower_edge=('tv_grid_lower','sum'),upper_edge=('tv_grid_upper','sum'),
        lambda_min=('tv_lambda','min'),lambda_max=('tv_lambda','max')).to_csv(outdir/'tv_tuning_summary.csv',index=False)
    record_comparisons(ok).to_csv(outdir/'paired_record_comparisons.csv',index=False)
    optional_inputs=nonoracle_evidence(outdir)
    figures(summary,outdir)
    raw[raw.status!='ok'].to_csv(outdir/'failed_records.csv',index=False)
    (outdir/'README_RESULTS.md').write_text('''# TV comparison: complete saved regime grid\n\nReporting can be regenerated without numerical experiments, from any working directory:\n\n```sh\npython /path/to/code_repository/revision_tv_reporting.py\n```\n\nThe archived experiment has 720 successful method records: five systems, four noise levels (5%,10%,20%,50%), two sampling spacings per system, five seeds, three ODE arms and four PDE arms. Every arm sees the same noisy observations for each matched record. All downstream STLSQ thresholds use the original oracle diagnostic rule.\n\n- `raw_results.csv`, `coefficients.csv`, `tv_cv_diagnostics.csv`, and `run_manifest.json` are execution artifacts and are not rewritten by reporting.\n- `summary_by_noise.csv`: ten records per system/method/noise.\n- `summary_by_noise_band.csv`: all records separated into 5–10% and20–50% bands; twenty records per system/method/band.\n- `summary_by_noise_spacing.csv`: five seeds for each individual sampling/noise condition.\n- Means are record-level means. SEs first average the fixed noise/sampling conditions within each seed and then take the standard deviation of five seed averages divided by sqrt(5). Coefficient-error medians are taken over records.\n- `paired_by_noise*.csv`: strict matched EDMD-minus-comparator mean differences, seed SEs, wins, ties, sample counts, and seed-mean direction counts. Smaller error is better; larger F1/Score is better. Ties are not wins. Exact-support counts are in the summary CSVs.\n- `tv_noise_coefficient_score.pdf/png`: each system's median coefficient error (left, log scale) and mean Score with seed SE (right), displaying all four noise levels; dotted dividers separate the two table bands. EDMD denotes polynomial EDMD for ODEs and POD-EDMD-RBF for PDEs.\n- `tv_high_noise_support_f1.pdf/png`: companion support-recovery figure.\n- `artifact_generation_manifest.json` identifies reporting code, input hashes, output hashes, and generation environment separately from the preserved execution provenance.\n\nPOD+TV and POD-EDMD-RBF use the same fixed rank. TV penalties use noisy holdouts only, with POD refitted in each training fold. The complete results establish conditional performance differences, not universal high-noise robustness or complete PDE recovery.\n''')
    after={n:digest(outdir/n) for n in before}
    assert before==after, 'Reporting changed an execution artifact.'
    outputs=[p for p in outdir.iterdir() if p.suffix in {'.csv','.pdf','.png','.md'} and p.name not in preserved]
    manifest=dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        purpose='Reporting-only regeneration; no fits or experimental reruns',
        command=' '.join(sys.argv),portable_command='python revision_tv_reporting.py',
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        preserved_execution_sha256=before,optional_evidence_input_sha256=optional_inputs,
        reporting_source_sha256={n:digest(ROOT/n) for n in ['revision_tv_reporting.py','revision_tv_comparison.py']},
        output_sha256={p.name:digest(p) for p in sorted(outputs)})
    (outdir/'artifact_generation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Regenerated TV reports from {len(ok)} successful saved records; execution artifacts unchanged.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=ROOT/'results/revision_tv_high_noise')
    summarize(parser.parse_args().outdir)
