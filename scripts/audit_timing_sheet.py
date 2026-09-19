#!/usr/bin/env python3
"""Recompute the benchmark timing comparison from the supplied aggregate CSV."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/timing-sheet-audit'
SOURCE = OUT / 'source.csv'
DATASETS = ['RACE', 'SQuAD', 'NQ', 'TriviaQA']
HIDE = 'unbiased_HSIC_16_key'
BASELINES = ['LN-Entropy', 'Lexical Similarity', 'Eigenscore']


def main():
    groups = {}
    for row in list(csv.reader(SOURCE.open()))[2:]:
        if row[0]:
            model = row[0]
            groups[model] = {}
        groups[model][row[1]] = [float(x) for x in row[2:6]]
    assert len(groups) == 6 and all(len(g) == 8 for g in groups.values())
    averages = {m: {k: mean(v) for k, v in g.items()} for m, g in groups.items()}
    reductions = {b: mean(100*(1-g[HIDE]/g[b]) for g in averages.values()) for b in BASELINES}
    rows = []
    for model, g in groups.items():
        for j, ds in enumerate(DATASETS):
            base, total = g['Single Gen'][j], g[HIDE][j]
            rows.append(dict(model=model, dataset=ds, base_s=base, total_s=total,
                             overhead_s=total-base, overhead_pct=100*(total-base)/base))
    with (OUT/'overheads.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_hide_label=HIDE, dataset_order=DATASETS, averages=averages,
        mean_model_relative_reduction_pct=reductions,
        mean_over_all_three_baselines_pct=mean(reductions.values()),
        formula='For each model/method, average four dataset latencies equally; average 100*(1-HIDE/baseline) equally over six models.',
        caveat='Aggregate timing cells only; no raw repetitions or confidence intervals supplied. The HIDE row label contains 16_key; this file alone does not establish actual token-budget metadata.')
    (OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':11,'pdf.fonttype':42,'axes.linewidth':.8})
    order = ['Llama3-3b','Llama3-3b-instruct','Llama3-8B','Llama3-8B-instruct','Gemma2-9B','Gemma2-9B-Instruct']
    methods = ['Perplexity','Energy',*BASELINES,HIDE]
    colors = ['#c5c8cb','#858b91','#b37b2d','#9d2933','#596e8a','#377e7f']
    fig, ax = plt.subplots(figsize=(11.5,4.5))
    x = np.arange(6); width=.125
    for j,(method,color) in enumerate(zip(methods,colors)):
        ax.bar(x+(j-2.5)*width,[averages[m][method] for m in order],width,
               label='HIDE' if method==HIDE else method,color=color,edgecolor='white',linewidth=.35)
    ax.set_xticks(x, ['Llama-3.2-3B','Llama-3.2-3B\nInstruct','Llama-3-8B','Llama-3-8B\nInstruct','Gemma-2-9B','Gemma-2-9B\nInstruct'])
    ax.set_ylabel('Mean latency per example (s)'); ax.set_ylim(0,6.1)
    ax.legend(ncol=3,loc='upper right',frameon=False,fontsize=10)
    fig.tight_layout()
    target=ROOT/'output/pdf/computation_time_plot_updated.pdf'
    fig.savefig(target,bbox_inches='tight');plt.close(fig)
    shutil.copyfile(target,ROOT/'paper/files/figures'/target.name)
    print(json.dumps(reductions,indent=2))


if __name__ == '__main__':
    main()
