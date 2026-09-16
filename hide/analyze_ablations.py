"""Point estimates for the prespecified score sweeps; explicit per-variant exclusions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hide.evaluate import auc, pcc


def summarize(rows):
    expanded = []
    for row in rows:
        for variant in row.get('ablations', []):
            expanded.append(dict(model=row['model'], dataset=row['dataset'], id=str(row['id']),
                target=row.get('sentence_similarity'), primary_score=row.get('HIDE_score'),
                primary_status=row.get('status'), **variant))
    if not expanded:
        raise ValueError('No saved ablation scores')
    frame = pd.DataFrame(expanded)
    output = []
    exclusions = []
    for (model, dataset, name), group in frame.groupby(['model', 'dataset', 'name'], sort=False):
        if group.id.duplicated().any():
            raise ValueError(f'Duplicate ablation IDs: {model}/{dataset}/{name}')
        numeric = group[['score', 'target', 'primary_score']].apply(pd.to_numeric, errors='coerce')
        good = np.isfinite(numeric).all(axis=1) & (group.status == 'ok') & (group.primary_status == 'ok')
        selected = numeric[good]
        y = (selected.target.to_numpy() > .9).astype(int)
        s = selected.score.to_numpy()
        score_auc = auc(y, s) if len(y) else np.nan
        primary_auc = auc(y, selected.primary_score.to_numpy()) if len(y) else np.nan
        output.append(dict(model=model, dataset=dataset, variant=name, family=group.iloc[0]['family'],
            parameter=group.iloc[0]['parameter'], n=len(y), n_correct=int(y.sum()), total=len(group),
            excluded=int((~good).sum()), auc=score_auc, pcc_continuous=pcc(s, selected.target.to_numpy()) if len(y) else np.nan,
            primary_auc_same_cohort=primary_auc, primary_minus_variant_auc=primary_auc-score_auc))
        exclusions.append(dict(model=model, dataset=dataset, variant=name,
            records=group.loc[~good, ['id', 'status']].to_dict('records')))
    return pd.DataFrame(output), exclusions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    rows = [json.loads(line) for file in args.inputs for line in Path(file).read_text().splitlines()]
    metrics, exclusions = summarize(rows)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out/'metrics.csv', index=False)
    (out/'exclusions.json').write_text(json.dumps(exclusions, indent=2)+'\n')
    (out/'README.md').write_text('# Saved-score ablations\n\nPoint estimates; no confidence intervals or tuning on these results. '
        'Each variant uses all its valid examples; exclusions are listed. Primary AUC is also recomputed on '
        'that same cohort. Final HF hidden state includes final normalization. SVD uses exact thin SVD; '
        'biased estimators report both denominator conventions. Undefined scores are never replaced by zero.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (model, dataset, family), group in metrics.groupby(['model', 'dataset', 'family']):
        fig, ax = plt.subplots(figsize=(max(6, min(12, len(group)*.4)), 4), layout='constrained')
        ax.plot(range(len(group)), group.auc, marker='o')
        ax.set_xticks(range(len(group)), group.parameter.astype(str), rotation=60 if len(group)>8 else 30)
        ax.set(ylabel='AUC_s (correctness positive)', xlabel=family, title=f'{model} / {dataset}')
        fig.savefig(out/f'{model}_{dataset}_{family}.png', dpi=180)
        fig.savefig(out/f'{model}_{dataset}_{family}.pdf')
        plt.close(fig)


if __name__ == '__main__': main()
