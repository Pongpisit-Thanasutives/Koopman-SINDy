#!/usr/bin/env python3
"""Generate the pooled Appendix D table from all frozen weak-study records."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import pandas as pd


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=root/'results/revision_weak_comparison')
    parser.add_argument('--output',type=Path,default=root/'generated_assets/tables/revision_weak_comparison.tex',
                        help='Table destination; relative paths use the invoking working directory.')
    args=parser.parse_args()
    args.results=args.results.expanduser().resolve()
    args.output=args.output.expanduser().resolve()
    raw=pd.read_csv(args.results/'raw_results.csv')
    assert len(raw)==300 and raw.status.eq('ok').all()
    assert not raw.duplicated(['system','method','sparse_factor','noise','seed']).any()
    methods=['raw_weak','linear_weak','edmd_weak','tv_weak','pod_weak','pod_tv_weak']
    labels={'raw_weak':'Raw','linear_weak':'Linear ($q=5$)','edmd_weak':'Polynomial EDMD',
            'tv_weak':'TV states','pod_weak':'POD only','pod_tv_weak':'POD+TV states'}
    lines=[r'\begin{table}[!htbp]',r'\centering',r'\setlength{\tabcolsep}{5pt}',
           r'\TBL{\caption{Targeted integral weak-form SINDy comparison. Each preprocessing method is followed by the same weak fit. Each row includes all 30 records (three noise levels, two observation spacings, five seeds). Mean support $F_1$ and mean Score are shown with seed-level standard errors, computed after averaging the six conditions within each seed; coefficient error $E$ is the median across records. Exact denotes complete support recovery. All arms use oracle sparsity selection and the same weak library and physical test windows; the conservative PDE library differs in one nuisance term from the main strong-form study. Boldface marks the best displayed value within each system\label{tab:weak_comparison}}}',
           r'{\begin{tabular}{@{}lcccc@{}}\toprule',
           r'\TCH{Preprocessing} & \TCH{$F_1$} & \TCH{Median $E$} & \TCH{Score} & \TCH{Exact} \\',r'\midrule']
    for sysidx,system in enumerate(['vanderpol_mu2','burgers']):
        group=raw[raw.system==system]
        title='Van der Pol' if system=='vanderpol_mu2' else 'Burgers'
        lines.append(r'\multicolumn{5}{@{}l}{\textit{'+title+r'}} \\')
        best_f1=group.groupby('method').support_f1.mean().max()
        best_error=group.groupby('method').coef_error.median().min()
        best_score=group.groupby('method').practical_score.mean().max()
        best_exact=group.groupby('method').exact_support.sum().max()
        for i,method in enumerate(m for m in methods if m in set(group.method)):
            r=group[group.method==method];assert len(r)==30
            seed=r.groupby('seed')[['support_f1','practical_score']].mean()
            f1=r.support_f1.mean();fse=seed.support_f1.sem()
            score=r.practical_score.mean();sse=seed.practical_score.sem()
            label=labels[method]
            if system=='burgers' and method=='edmd_weak':label='POD-EDMD-RBF'
            cells=[]
            for value,se,best in [(f1,fse,best_f1),(r.coef_error.median(),None,best_error),(score,sse,best_score)]:
                cell=f'{value:.3f}'+(rf'\pm {se:.3f}' if se is not None else '')
                bold=f'{value:.3f}'==f'{best:.3f}'
                if bold:cell=r'\mathbf{'+cell+'}'
                cells.append('$'+cell+'$' if se is not None or bold else cell)
            exact=f'{int(r.exact_support.sum())}/30'
            if r.exact_support.sum()==best_exact:exact=r'$\mathbf{'+exact+'}$'
            lines.append(label+' & '+' & '.join(cells+[exact])+r' \\')
        if sysidx==0:lines.append(r'\midrule')
    lines += [r'\botrule\end{tabular}}',r'\end{table}','']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(lines))
    meta=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              input_sha256=hashlib.sha256((args.results/'raw_results.csv').read_bytes()).hexdigest(),
              output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
              output=os.path.relpath(args.output,root.parent),
              output_path_base='repository parent; ../ components permit external destinations')
    (args.results/'table_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
