#!/usr/bin/env python3
"""CPU-only analysis and paper artifacts for the fixed, uploaded revision cohort."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from hide.evaluate import evaluate, pcc
from hide.export_results import inspect_run, verify
from hide.parts import identity, part_path
from hide.provenance import file_sha256
from scripts.answer_subset import selected_tasks

NAMES = {'llama3-8b': 'Llama-3-8B', 'gemma-2-9b': 'Gemma-2-9B',
         'llama3-3b': 'Llama-3.2-3B', 'gemma-2-27b': 'Gemma-2-27B'}
MODELS = ['llama3-8b', 'gemma-2-9b']
DATASETS = ['SQuAD', 'race', 'nq_open', 'triviaqa']
DS = {'SQuAD': 'SQuAD', 'race': 'RACE', 'nq_open': 'NQ', 'triviaqa': 'TriviaQA'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/answer-2000-restored'))
    p.add_argument('--bundle', type=Path, default=Path('results/answer-2000-complete-01'))
    p.add_argument('--output', type=Path, default=Path('results/answer-2000-analysis-01'))
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    verify(args.bundle)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    tables = Path('paper/files/tables'); tables.mkdir(exist_ok=True)
    figures = Path('output/pdf'); figures.mkdir(parents=True, exist_ok=True)
    spec = json.loads((args.root/'subsets/answer-2000.json').read_text())
    plan = json.loads((args.root/'plan.json').read_text())
    assert spec['tasks'] == selected_tasks(plan)
    assert spec['parent_plan_sha256'] == plan['sha256']
    groups = defaultdict(list)
    for task in spec['tasks']:
        groups[(task['model'], task['dataset'])].append(task)
    frames, sources, results, diagnostics, controls = {}, [], [], [], []
    cohort_by_dataset = {}
    for model in MODELS:
        for ds in DATASETS:
            rows, meta_identity = [], None
            for task in sorted(groups[model, ds], key=lambda t: t['start']):
                path = part_path(args.root, task)
                report = inspect_run(path)
                assert report['complete'], report['issues']
                meta = json.loads(path.with_suffix('.manifest.json').read_text())
                assert meta['arguments']['answer_boundary'] == 'first-line'
                assert meta['arguments']['seed'] == 42
                if meta_identity is None:
                    meta_identity = identity(meta)
                else:
                    assert meta_identity == identity(meta), 'Mixed within-cohort protocol'
                # json.loads preserves binary64 precision; avoid pandas JSON float rounding.
                rr = [json.loads(line) for line in path.read_text().splitlines()]
                assert list(map(str, meta['selected_ids'])) == [str(r['id']) for r in rr]
                rows += rr
                sources.append(dict(path=str(path.relative_to(args.root)), sha256=file_sha256(path)))
            assert len(rows) == len({str(r['id']) for r in rows}) == 2000
            assert all(r['status'] == 'ok' and r['answer_boundary'] == 'first-line' for r in rows)
            cohort = [(str(r['id']), r['prompt'], r['answer']) for r in rows]
            if ds in cohort_by_dataset:
                assert cohort_by_dataset[ds] == cohort, 'Models have different evaluation examples'
            cohort_by_dataset[ds] = cohort
            frame = pd.DataFrame(rows)
            frames[model, ds] = frame
            res, excluded = evaluate(frame, bootstrap=args.bootstrap)
            assert all(e['excluded'] == 0 for e in excluded)
            res['model'], res['dataset'] = model, ds
            results.append(res)
            n = frame.n_eff.to_numpy(float)
            count = np.where(n >= 1, (n-1)/np.maximum(n, 1)**2, 0)
            controls.append(dict(model=model, dataset=ds,
                                 pcc_hide_count=pcc(frame.HIDE_score.to_numpy(), count)))
            diagnostics.append(dict(model=model, dataset=ds, n=2000,
                n_correct=int(frame.is_correct.sum()), n_cap=int(frame.hit_generation_cap.sum()),
                n_empty=int(frame.empty_answer.sum()), n_no_state=int(frame.no_output_state.sum()),
                boundary_token_suffix_nonempty=int(frame.boundary_token_suffix.str.strip().ne('').sum()),
                excluded=excluded, mean_output_tokens=float(frame.output_length.mean())))
            print(model, ds, 'complete; positives', int(frame.is_correct.sum()), flush=True)
    metrics = pd.concat(results, ignore_index=True)
    metrics.to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(controls).to_csv(out/'count_control.csv', index=False)
    (out/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    shutil.copyfile(args.root/'subsets/answer-2000.json', out/'subset.json')
    shutil.copyfile(__file__, out/'analysis_source.py')
    (out/'analysis_manifest.json').write_text(json.dumps(dict(
        bundle=str(args.bundle), bundle_index_sha256=file_sha256(args.bundle/'INDEX.json'),
        bootstrap=args.bootstrap, seed=42, precision='json.loads; no score rounding before ranking',
        selection=spec['selection'], sources=sources,
        timing_source='results/optimus-review-analysis-01/timing/timing_summary.csv',
        timing_source_sha256=file_sha256(Path('results/optimus-review-analysis-01/timing/timing_summary.csv')),
        evaluator_sha256=file_sha256(Path('hide/evaluate.py'))), indent=2)+'\n')

    def get(model, ds, method, target='sentence_similarity'):
        return metrics[(metrics.model == model) & (metrics.dataset == ds) &
                       (metrics.method == method) & (metrics.target == target)].iloc[0]

    for target, suffix, symbol in [('sentence_similarity','s','s'), ('rouge_l','r','r')]:
        lines = [r'\begin{table*}[t]', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
            r'\begin{tabular}{llr rr rr rr}', r'\toprule',
            r'& & & \multicolumn{2}{c}{HIDE} & \multicolumn{2}{c}{Attention $\Omega$} & \multicolumn{2}{c}{Norm $\widetilde\Delta_{\rm in}$} \\',
            rf'Model & Dataset & $N_+$ & AUC$_{symbol}$ & PCC$_{symbol}$ & AUC$_{symbol}$ & PCC$_{symbol}$ & AUC$_{symbol}$ & PCC$_{symbol}$ \\', r'\midrule']
        for model in MODELS:
            for ds in DATASETS:
                rows = [get(model, ds, method, target) for method in ['HIDE_score','Omega','Delta_in']]
                vals = [f'{100*r[key]:.2f}' for r in rows for key in ['auc','pcc_continuous']]
                lines.append(' & '.join([NAMES[model], DS[ds], str(int(rows[0].n_correct)), *vals])+r' \\')
            if model == MODELS[0]: lines.append(r'\midrule')
        desc = ('sentence similarity greater than 0.9' if target == 'sentence_similarity' else 'ROUGE-L greater than 0.5')
        lines += [r'\bottomrule', r'\end{tabular}',
            r'\caption{Matched detector comparison under first-answer-line stopping: 2,000 fixed seed-42 examples per model/dataset. '
            +f'$N_+$ counts answers with {desc}. '+r'AUC and PCC are multiplied by 100; PCC uses the continuous reference metric. '
            r'All scores use the same examples and labels; higher raw scores predict correctness. '
            r'No direction is selected using test performance. These results are separate from the historical full-cohort tables.}',
            rf'\label{{tab:paired_{suffix}}}', r'\end{table*}']
        (tables/f'paired_{suffix}.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{table*}[t]', r'\centering\small', r'\begin{tabular}{llrrr}',r'\toprule',
        r'Model & Dataset & HIDE $-$ attention & 95\% CI & HIDE $-$ norm \\',r'\midrule']
    for model in MODELS:
        for ds in DATASETS:
            r = get(model, ds, 'Omega'); norm = get(model, ds, 'Delta_in')
            lines.append(f'{NAMES[model]} & {DS[ds]} & {100*r.hide_minus_baseline_auc:+.2f} & '
                         f'[{100*r.hide_minus_baseline_auc_lo:+.2f}, {100*r.hide_minus_baseline_auc_hi:+.2f}] & '
                         f'{100*norm.hide_minus_baseline_auc:+.2f}'+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Paired AUC$_s$ differences in percentage points on the corrected 2,000-example cohorts. '
              r'Intervals use 2,000 paired example bootstrap draws (seed 42), are pointwise, and are not adjusted for multiple comparisons.}',
              r'\label{tab:paired_differences}\end{table*}']
    (tables/'paired_differences.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{llrrr}',r'\toprule',
             r'Model & Dataset & HIDE AUC$_s$ & Count-control AUC$_s$ & PCC(HIDE, control) \\',r'\midrule']
    for model in MODELS:
        for ds in DATASETS:
            r = get(model, ds, 'constant_kernel_control'); h = get(model, ds, 'HIDE_score')
            lines.append(f'{NAMES[model]} & {DS[ds]} & {100*h.auc:.2f} & {100*r.auc:.2f} & {r.pcc_with_hide:.6f}'+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Selected-token-count diagnostic on the same corrected cohorts. '
              r'The control is $(n_{\rm eff}-1)/n_{\rm eff}^2$ for $n_{\rm eff}\geq1$, and zero otherwise. '
              r'AUC is multiplied by 100; PCC is unscaled. This is a control for realized selected-token count, not raw output length.}',
              r'\label{tab:count_control}\end{table*}']
    (tables/'count_control.tex').write_text('\n'.join(lines)+'\n')

    timing_path = Path('results/optimus-review-analysis-01/timing/timing_summary.csv')
    timing = pd.read_csv(timing_path)
    shutil.copyfile(timing_path, out/'timing_summary.csv')
    lines = [r'\begin{table*}[t]',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{llrrrrl}',r'\toprule',
             r'Model & Dataset & Base (s) & HIDE total (s) & Overhead (s) & Overhead (\%) & 95\% CI (s) \\',r'\midrule']
    for model in ['llama3-3b','llama3-8b','gemma-2-9b','gemma-2-27b']:
        for ds in ['SQuAD','nq_open']:
            r = timing[(timing.model == model) & (timing.dataset == ds)].iloc[0]
            lines.append(f'{NAMES[model]} & {DS[ds]} & {r.base_s:.4f} & {r.total_s:.4f} & {r.overhead_s:.4f} & '
                         f'{r.overhead_pct:.2f} & [{r.overhead_s_lo:.4f}, {r.overhead_s_hi:.4f}]'+r' \\')
    lines += [r'\bottomrule\end{tabular}', r'\caption{Paired timing under the earlier stopping protocol on an A100 80GB PCIe: '
        r'200 queries per model/dataset, 10 warm-up pairs, and three measured pairs per query. '
        r'Total includes generation, hidden-state capture, keyword extraction, and HIDE scoring; model loading and correctness evaluation are excluded. '
        r'Overhead is total minus base; percentages are ratios of means. All differences are computed before rounding. '
        r'Intervals use 2,000 query bootstrap draws after averaging repeats; no observations are pruned.}',
        r'\label{tab:timing_revision}\end{table*}']
    (tables/'timing_revision.tex').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':10,'axes.linewidth':.8,'axes.grid':False,'pdf.fonttype':42,'ps.fonttype':42,
        'figure.facecolor':'white','axes.facecolor':'white'})
    correlations = []
    for model in MODELS:
        fig, axs = plt.subplots(2,2,figsize=(7.2,6.1))
        for j, ds in enumerate(['SQuAD','nq_open']):
            f = frames[model,ds]; wrong = f.is_correct == 0
            qcbw = wrong & (f.q_g_similarity >= f.loc[wrong,'q_g_similarity'].quantile(.75)) if ds == 'nq_open' else np.zeros(len(f),bool)
            for i, metric in enumerate(['Omega','Delta_in']):
                ax = axs[i,j]
                for mask, label, color, marker, size in [
                    (wrong & ~qcbw,'Incorrect','#377e7f','s',9),
                    (~wrong,'Correct','#9d2933','o',9), (qcbw,'QCBW','#b37b2d','*',25)]:
                    ax.scatter(f.loc[mask,'HIDE_score'],f.loc[mask,metric],s=size,c=color,
                               marker=marker,alpha=.48,label=label,linewidths=.15)
                r = pcc(f.HIDE_score.to_numpy(),f[metric].to_numpy())
                ax.set_xlabel('HIDE score')
                ax.set_ylabel(r'Attention mass ($\Omega$)' if i == 0 else r'Input-state norm ($\widetilde\Delta_{\rm in}$)')
                ax.set_title(f'({"abcd"[i*2+j]}) {DS[ds]}; PCC = {r:.3f}',fontsize=10)
                ax.tick_params(labelsize=9)
                if i == 0 and j == 1: ax.legend(fontsize=8,loc='best',framealpha=.9)
                correlations.append(dict(model=model,dataset=ds,metric=metric,pcc=r,n=2000,n_qcbw=int(np.sum(qcbw))))
        fig.subplots_adjust(left=.11,right=.98,bottom=.10,top=.94,wspace=.34,hspace=.43)
        fig.savefig(figures/f'{model}_Mechanistic_Flow_updated.pdf',bbox_inches='tight')
        fig.savefig(out/f'{model}_mechanistic.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
    pd.DataFrame(correlations).to_csv(out/'mechanistic_correlations.csv',index=False)
    fig, axs = plt.subplots(1,2,figsize=(10.5,4.2))
    for j, ds in enumerate(['SQuAD','nq_open']):
        group = timing[timing.dataset == ds].sort_values('parameter_count')
        ax = axs[j]
        for (_, r), marker in zip(group.iterrows(),['o','s','^','D']):
            ax.errorbar(r.parameter_count/1e9,r.overhead_s,
                        yerr=[[r.overhead_s-r.overhead_s_lo],[r.overhead_s_hi-r.overhead_s]],
                        fmt=marker,markersize=6,color='#377e7f' if j == 0 else '#9d2933',capsize=3,
                        label=f'{NAMES[r.model]} ($d={int(r.hidden_size)}$)')
        ax.set(xlabel='Model parameters (billions)',ylabel='Mean total incremental overhead (s)',
               title=f'({"ab"[j]}) {DS[ds]}')
        ax.set_xlim(0,30);ax.set_ylim(0,.5);ax.legend(fontsize=8,loc='upper right')
    fig.tight_layout(w_pad=2)
    fig.savefig(figures/'Scalability_Analysis_updated.pdf',bbox_inches='tight')
    fig.savefig(out/'scaling.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Extra timing data for interpreting cross-model sequence-length differences.
    timing[['model','dataset','mean_input_length','mean_output_length','max_output_length']].to_csv(out/'timing_lengths.csv',index=False)
    lines = [r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{llrr}',r'\toprule',
             r'Model & Dataset & Mean input tokens & Mean generated tokens \\',r'\midrule']
    for model in ['llama3-3b','llama3-8b','gemma-2-9b','gemma-2-27b']:
        for ds in ['SQuAD','nq_open']:
            r = timing[(timing.model == model) & (timing.dataset == ds)].iloc[0]
            lines.append(f'{NAMES[model]} & {DS[ds]} & {r.mean_input_length:.2f} & {r.mean_output_length:.2f}'+r' \\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{Mean token lengths for the timing queries in Table~\ref{tab:timing_revision}, under the earlier stopping protocol (maximum 256 generated tokens). Means include all queries and measured repeats; repeats share deterministic token sequences. These length differences limit a parameter-only interpretation of the scaling comparison.}',
        r'\label{tab:timing_lengths}\end{table*}']
    (tables/'timing_lengths.tex').write_text('\n'.join(lines)+'\n')
    for path in figures.glob('*updated.pdf'):
        shutil.copyfile(path, Path('paper/files/figures')/path.name)
    hashes = {str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'SHA256.json'}
    (out/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(out,flush=True)


if __name__ == '__main__':
    main()
