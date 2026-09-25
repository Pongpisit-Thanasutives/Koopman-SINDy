#!/usr/bin/env python3
"""Synchronize manuscript assets after the corrected experiments and summaries.

Run without flags after regenerating all summaries to write generated_assets/;
use --latex-dir to select another destination. The old standalone
--merge-classical operation is deliberately disabled because it targets the
pre-propagation-correction Appendix C checkpoint. rerun_complex_revision.py now
updates the complete corrected source of truth.
"""
from pathlib import Path
import argparse
import re
import shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
LATEX=ROOT/'generated_assets'


def strategy_table(latex_dir=LATEX):
    """Regenerate every reported interpolation strategy from final raw results."""
    raw = pd.read_csv(ROOT/'results/upsampling_strategy_publication/upsampling_strategy_raw.csv')
    metrics = ['support_f1','coef_error','practical_score']
    ok = raw[(raw.status=='ok') & np.isfinite(raw[metrics]).all(axis=1)]
    rows = []
    for (setting,method,strategy), frame in ok.groupby(['setting','method','strategy'],sort=False):
        rows.append(dict(setting=setting,method=method,strategy=strategy,n_valid=len(frame),
            f1_mean=frame.support_f1.mean(),coef_error_median=frame.coef_error.median(),
            score_mean=frame.practical_score.mean()))
    summary=pd.DataFrame(rows)
    summary.to_csv(ROOT/'results/revision1_reporting/strategy_report_summary.csv',index=False)
    labels={'baseline':'Raw FD','local_reset':'Local reset','global_rollout':'Global rollout',
            'residual_corrected':'Residual correction','two_sided':'Two-sided'}
    methods={'edmd_poly3':'EDMD-polynomial','edmd_rbf':'EDMD-RBF','pod_edmd_rbf':'POD-EDMD-RBF'}
    order=list(labels)
    caption=(r'Stress-condition comparison of interpolation strategies, including the raw baseline. '
             r'The ODE rows pool Lorenz--63 and Van der Pol; the PDE rows pool Burgers and Fisher--KPP. '
             r'The setting uses 10\% noise, sparse factor 64 for ODEs and 32 for PDEs, $q=5$, and 20 seeds. '
             r'$N$ counts valid records. F1 and Score are means; coefficient error is the median. '
             r'Boldface marks the best displayed value among assisted strategies in each block')
    lines=[r'\begin{table}[!htbp]',r'\centering',r'\setlength{\tabcolsep}{4pt}',
           r'\TBL{\caption{'+caption+r'\label{tab:upsampling_strategy_app}}}',
           r'{\begin{tabular}{@{}llrccc@{}}\toprule',
           r'\TCH{Setting/method} & \TCH{Strategy} & \TCH{$N$} & \TCH{F1} & \TCH{Coeff. err.} & \TCH{Score} \\ \midrule']
    for bi,(setting,method) in enumerate([('ODE','edmd_poly3'),('ODE','edmd_rbf'),('PDE','pod_edmd_rbf')]):
        block=summary[(summary.setting==setting)&(summary.method==method)].set_index('strategy').reindex(order)
        assert not block.n_valid.isna().any(),f'Missing final strategy records for {method}'
        assisted=block.drop(index='baseline')
        for j,(strategy,row) in enumerate(block.iterrows()):
            heading=rf'\multirow{{5}}{{*}}{{\makecell[l]{{{setting}\\{methods[method]}}}}}' if j==0 else ''
            cells=[]
            for col,maximize in [('f1_mean',True),('coef_error_median',False),('score_mean',True)]:
                val=float(row[col]);best=float(assisted[col].max() if maximize else assisted[col].min())
                cell=f'{val:.3f}'
                if strategy!='baseline' and round(val,3)==round(best,3):cell=r'\textbf{'+cell+'}'
                cells.append(cell)
            lines.append(heading+f' & {labels[strategy]} & {int(row.n_valid)} & '+' & '.join(cells)+r' \\')
        if bi<2:lines.append(r'\midrule')
    lines.extend([r'\botrule\end{tabular}}',r'\end{table}'])
    (latex_dir/'tables/revision_strategy_ablation.tex').write_text('\n'.join(lines)+'\n')


def model_selection_table(latex_dir=LATEX):
    """Keep support-size selection distinct from exact support recovery."""
    selected=pd.read_csv(ROOT/'results/model_selection_publication/model_selection_selected_by_equation.csv')
    keys=['setting','system','equation','method','seed']
    assert not selected.duplicated(keys).any()
    rows=[]
    for (setting,system,equation,method),frame in selected.groupby(keys[:-1],sort=False):
        assert np.isfinite(frame[['support_size','support_f1','coef_error']]).all().all()
        assert frame.true_support_size.nunique()==1
        rows.append(dict(setting=setting,system=system,equation=equation,method=method,
            n=len(frame),true_k=int(frame.true_support_size.iloc[0]),
            selected_k_min=int(frame.support_size.min()),selected_k_max=int(frame.support_size.max()),
            selected_k_median=float(frame.support_size.median()),
            exact_k_count=int((frame.support_size==frame.true_support_size).sum()),
            exact_support_count=int(np.isclose(frame.support_f1,1,rtol=0,atol=1e-12).sum()),
            f1_mean=frame.support_f1.mean(),coef_error_median=frame.coef_error.median()))
    summary=pd.DataFrame(rows)
    summary.to_csv(ROOT/'results/revision1_reporting/model_selection_report_summary.csv',index=False)
    labels={'baseline':'Raw FD','edmd_poly3':'EDMD-polynomial','pod_edmd_rbf':'POD-EDMD-RBF'}
    groups=[('vanderpol_mu2','dx/dt',r'Van der Pol $\dot{x}$'),
            ('vanderpol_mu2','dy/dt',r'Van der Pol $\dot{y}$'),
            ('burgers','u_t',r'Burgers $u_t$'),('fisher_kpp','u_t',r'Fisher--KPP $u_t$')]
    caption=(r'Non-oracle equation-wise EBIC/Pareto selection on five seeds per method. '
        r'Ground-truth support and coefficients are used only after selection for evaluation. '
        r'The experiment uses 1\% noise, sparse factor eight for Van der Pol and four for the PDEs, $q=5$, '
        r'PDE $n_x=48$ and base $\Delta t=0.01$, POD ranks six for Burgers and two for front-type Fisher--KPP, '
        r'and at most 5000 PDE regression rows. Selected $k$ is the observed range, or a single value if constant. '
        r'Exact support counts correct active sets; a correct support size alone need not recover the correct terms')
    lines=[r'\begin{table}[H]',r'\centering',r'\setlength{\tabcolsep}{4pt}',
        r'\TBL{\caption{'+caption+r'\label{tab:model_selection}}}',
        r'{\begin{tabular}{@{}llrccc@{}}\toprule',
        r'\TCH{System/target} & \TCH{Method} & \TCH{$N$} & \TCH{True $k$} & \TCH{Selected $k$} & \TCH{Exact support} \\ \midrule']
    for gi,(system,equation,title) in enumerate(groups):
        block=summary[(summary.system==system)&(summary.equation==equation)].sort_values('method')
        assert len(block)==2
        for ri,row in enumerate(block.itertuples()):
            interval=str(row.selected_k_min) if row.selected_k_min==row.selected_k_max else f'{row.selected_k_min}--{row.selected_k_max}'
            heading=rf'\multirow{{2}}{{*}}{{{title}}}' if ri==0 else ''
            lines.append(heading+f' & {labels[row.method]} & {row.n} & {row.true_k} & {interval} & {row.exact_support_count}/{row.n}'+r' \\')
        if gi<3:lines.append(r'\midrule')
    lines.extend([r'\botrule\end{tabular}}',r'\end{table}'])
    (latex_dir/'tables/revision_model_selection.tex').write_text('\n'.join(lines)+'\n')

def tv_comparison_table(latex_dir=LATEX):
    """Show the complete saved TV grid in two fixed noise bands, with Score."""
    data=pd.read_csv(ROOT/'results/revision_tv_high_noise/summary_by_noise_band.csv')
    systems={'lorenz63':'Lorenz--63','vanderpol_mu2':'Van der Pol','burgers':'Burgers',
             'fisher_kpp':'Fisher--KPP','advection_diffusion':'Advection--diffusion'}
    labels={'baseline':'Raw FD','edmd_poly3':'Polynomial EDMD','pod_edmd_rbf':'POD-EDMD-RBF','tv_diff':'TV','pod_tv_diff':'POD+TV'}
    bands=['5–10%','20–50%']
    caption=(r'Comparison with TV differentiation over the full added noise grid. '
             r'Each method has $N=20$ records per band (two noise levels, two sparse factors, five seeds). '
             r'F1 and Score are means with standard errors in parentheses; each standard error uses the five seed averages, '
             r'averaging the four noise/sampling conditions within each seed. Error is the median coefficient error. '
             r'Boldface marks the best displayed value within each system and band. '
             r'All downstream STLSQ thresholds use the same oracle rule')
    lines=[r'\begin{table}[!htbp]',r'\centering',r'\setlength{\tabcolsep}{2.2pt}',
        r'\TBL{\caption{'+caption+r'\label{tab:tv_high_noise}}}',
        r'{\begin{tabular}{@{}lcccccc@{}}\toprule',
        r' & \multicolumn{3}{c}{\TCH{5--10\% noise}} & \multicolumn{3}{c}{\TCH{20--50\% noise}} \\',
        r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
        r'\TCH{Method} & \TCH{F1 (SE)} & \TCH{Error} & \TCH{Score (SE)} & \TCH{F1 (SE)} & \TCH{Error} & \TCH{Score (SE)} \\ \midrule']
    for k,(system,title) in enumerate(systems.items()):
        if k:lines.append(r'\midrule')
        lines.append(r'\multicolumn{7}{@{}l}{\textit{'+title+r'}} \\')
        methods=['baseline','edmd_poly3' if k<2 else 'pod_edmd_rbf','tv_diff']+(['pod_tv_diff'] if k>=2 else [])
        for method in methods:
            cells=[]
            for band in bands:
                block=data[(data.system==system)&(data.band==band)].set_index('method')
                assert (block.n_ok==20).all() and (block.n_seeds==5).all()
                row=block.loc[method]
                for metric,maximize in [('support_f1',True),('coef_error',False),('practical_score',True)]:
                    col=metric+('_median' if metric=='coef_error' else '_mean')
                    val=float(row[col]);best=float(block[col].max() if maximize else block[col].min())
                    cell=f'{val:.3f}'
                    if metric!='coef_error':cell+=rf'\,({row[metric+"_se"]:.3f})'
                    if round(val,3)==round(best,3):cell=r'\mathbf{'+cell+'}'
                    cells.append('$'+cell+'$')
            lines.append(labels[method]+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\botrule\end{tabular}}',r'\end{table}'])
    (latex_dir/'tables/revision_tv_high_noise.tex').write_text('\n'.join(lines)+'\n')


def merge_classical():
    raise SystemExit("Legacy --merge-classical is disabled: it would copy a pre-correction checkpoint. "
                     "Run python rerun_complex_revision.py, regenerate the classical summaries, "
                     "then run python sync_revision_outputs.py without this flag.")

def normalize_captions(text):
    # The DCE class supplies terminal punctuation for captions.
    pattern=r'\\caption\{'
    cursor=0
    while True:
        m=re.search(pattern,text[cursor:])
        if not m:break
        start=cursor+m.end();depth=1;end=start
        while depth and end<len(text):
            if text[end]=='{' and text[end-1]!='\\':depth+=1
            elif text[end]=='}' and text[end-1]!='\\':depth-=1
            end+=1
        content=text[start:end-1]
        content=re.sub(r'\.\s*(\\label\{[^}]+\})?$',lambda m:m.group(1) or '',content)
        text=text[:start]+content+text[end-1:];cursor=start+len(content)+1
    return text

def sync(latex_dir=LATEX):
    latex_dir=Path(latex_dir).expanduser().resolve()
    (ROOT/'results/revision1_reporting').mkdir(parents=True,exist_ok=True)
    for name in ['appendices','figures','tables']:
        (latex_dir/name).mkdir(parents=True,exist_ok=True)
    def copy_figure(source):
        destination=latex_dir/'figures'/source.name
        if source.resolve()!=destination:
            shutil.copy2(source,destination)
    for filename,folder in [('appendix_a_qr_sensitivity.tex','qr_sensitivity'),
                            ('appendix_c_classical_interpolation.tex','classical_interpolation_publication')]:
        text=(ROOT/'results'/folder/filename).read_text()
        text=text.replace('\\clearpage\n','',1)
        text=text.replace(r'\begin{figure}[H]',r'\begin{figure}[!htbp]')
        text=text.replace(r'\begin{table}[H]',r'\begin{table}[!htbp]')
        (latex_dir/'appendices'/filename).write_text(normalize_captions(text))
    for name in ['appendix_q_sensitivity_ode.png','appendix_qr_sensitivity_pde.png',
                 'dataset_examples.png','advection_diffusion_dataset.png']:
        copy_figure(ROOT/'figures'/name)
    # Preserve an explicitly regenerated illustration if already present.
    if not (latex_dir/'figures/revision_interpolated_data.pdf').exists():
        copy_figure(ROOT/'figures/revision_interpolated_data.pdf')
    pareto='model_selection_pareto_equationwise.png'
    copy_figure(ROOT/'results/model_selection_publication'/pareto)
    shutil.copy2(ROOT/'results/model_selection_publication'/pareto,ROOT/'figures'/pareto)
    for name in ['tv_high_noise_coefficient_error.pdf','tv_high_noise_support_f1.pdf','tv_noise_coefficient_score.pdf']:
        copy_figure(ROOT/'results/revision_tv_high_noise'/name)
    tv_comparison_table(latex_dir)
    for name in ['revision_ode_regime_figure.tex','revision_pde_regime_figure.tex','revision_interpolated_data_figure.tex']:
        f=latex_dir/'tables'/name
        if f.exists():f.write_text(normalize_captions(f.read_text()))
    strategy_table(latex_dir)
    model_selection_table(latex_dir)
    # Reformat the saved matched non-oracle summary without invoking experiments.
    from revision_nonoracle_tv import build_table
    build_table(pd.read_csv(ROOT/'results/revision_nonoracle_tv/summary.csv'),latex_dir/'tables')
    print('Synchronized corrected appendices, strategy/non-oracle/TV tables and figures.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merge-classical',action='store_true')
    p.add_argument('--latex-dir',type=Path,default=LATEX,
                   help='Asset destination; relative paths use the invoking working directory.')
    a=p.parse_args()
    if a.merge_classical:merge_classical()
    else:sync(a.latex_dir)
