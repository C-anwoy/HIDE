"""Summarize measured timings without outcome-based pruning; bootstrap over queries."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hide.evaluate import load_frame


def summarize(df, bootstrap=2000, seed=42):
    columns = ['base_s', 'capture_s', 'score_s', 'total_s', 'overhead_s']
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    good = np.isfinite(df[columns]).all(axis=1)
    good &= (df[['base_s', 'capture_s', 'score_s', 'total_s']] > 0).all(axis=1)
    if 'status' in df:
        good &= df.status == 'ok'
    errors = df.loc[~good, ['id']].to_dict('records')
    df = df[good].copy()
    if df.duplicated(['id', 'repeat']).any():
        raise ValueError('Duplicate query/repeat pair')
    if df.empty:
        raise ValueError('No valid timing rows')
    if not np.allclose(df.total_s - df.base_s, df.overhead_s):
        raise ValueError('Inconsistent total/base/overhead arithmetic')
    by_query = df.groupby('id')[columns].mean()
    rng = np.random.default_rng(seed)
    means = by_query.mean().to_dict()
    means.update(n_queries=len(by_query), n_measurements=len(df), excluded=len(errors),
                 overhead_pct=100 * means['overhead_s']/means['base_s'],
                 mean_per_query_overhead_pct=float((100*by_query.overhead_s/by_query.base_s).mean()),
                 p50_total_s=float(df.total_s.median()), p95_total_s=float(df.total_s.quantile(.95)),
                 capture_overhead_s=means['capture_s']-means['base_s'],
                 bookkeeping_s=means['total_s']-means['capture_s']-means['score_s'])
    draws = []
    for _ in range(bootstrap):
        sub = by_query.iloc[rng.integers(0, len(by_query), len(by_query))].mean()
        draws.append([sub.total_s, sub.overhead_s, 100*sub.overhead_s/sub.base_s])
    if draws:
        bounds = np.quantile(draws, [.025,.975], axis=0)
        for j, key in enumerate(['total_s', 'overhead_s', 'overhead_pct']):
            means[key+'_lo'], means[key+'_hi'] = map(float, bounds[:,j])
    return means, errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, exclusions, frames = [], [], []
    for file in args.inputs:
        df = load_frame(file)
        frames.append(df)
        runtime_path = Path(file).with_suffix('.runtime.json')
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        for (model,dataset), group in df.groupby(['model','dataset']):
            result, invalid = summarize(group.copy())
            result.update(parameter_count=runtime.get('parameter_count'),
                          hidden_size=runtime.get('hidden_size'), num_hidden_layers=runtime.get('num_hidden_layers'))
            for column in ['input_length', 'output_length', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes']:
                if column in group:
                    result['mean_'+column] = float(group[column].mean())
                    result['max_'+column] = float(group[column].max())
            records.append(dict(model=model, dataset=dataset, source=file, **result))
            exclusions.append(dict(source=file, invalid=invalid))
    pd.DataFrame(records).to_csv(out/'timing_summary.csv',index=False)
    (out/'exclusions.json').write_text(json.dumps(exclusions,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for df in frames:
        df = df[df.status == 'ok']
        for (model,dataset), g in df.groupby(['model','dataset']):
            axs[0].scatter(g.n_eff,g.score_s,s=8,alpha=.35,label=f'{model}/{dataset}')
            axs[1].scatter(g.base_s,100*g.overhead_s/g.base_s,s=8,alpha=.35)
    axs[0].set(xlabel='Realized selected-token count',ylabel='Keyword selection + HSIC time (s)')
    axs[1].set(xlabel='Base generation time (s)',ylabel='Total incremental overhead (%)')
    axs[0].legend(fontsize=6)
    fig.savefig(out/'timing.png',dpi=200)
    fig.savefig(out/'timing.pdf')
    plt.close(fig)
    scaling = pd.DataFrame(records).dropna(subset=['parameter_count'])
    if not scaling.empty:
        datasets = list(scaling.dataset.unique())
        fig, axs = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 4), squeeze=False, layout='constrained')
        for j, dataset in enumerate(datasets):
            group = scaling[scaling.dataset == dataset].sort_values('parameter_count')
            ax = axs[0,j]
            for _, row in group.iterrows():
                yerr = np.array([[max(0,row.overhead_s-row.overhead_s_lo)],
                                 [max(0,row.overhead_s_hi-row.overhead_s)]])
                ax.errorbar(row.parameter_count/1e9, row.overhead_s, yerr=yerr, fmt='o', capsize=3,
                            label=f'{row.model} (d={int(row.hidden_size)})')
            ax.set(xlabel='Actual parameter count (billions)', ylabel='Mean total HIDE overhead (s)', title=dataset)
            ax.legend(fontsize=7)
        fig.savefig(out/'scaling.png', dpi=200)
        fig.savefig(out/'scaling.pdf')
        plt.close(fig)
    print(out/'timing_summary.csv')


if __name__ == '__main__':
    main()
