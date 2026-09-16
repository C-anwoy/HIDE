"""Paired, unfiltered detection evaluation with bootstrap intervals and explicit PCC targets."""
import argparse
import json
from pathlib import Path
import re
import string

import numpy as np
import pandas as pd


def normalize_text(text):
    text = ''.join(c for c in str(text).lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def auc(y, score):
    # Mann-Whitney AUC with average ranks for ties, without an sklearn dependency.
    positive = y == 1
    n1 = positive.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(score, kind='stable')
    ordered = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1]
    ends = np.r_[starts[1:], len(y)]
    ranks = np.repeat((starts + 1 + ends) / 2, ends - starts)
    return float((ranks[positive[order]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pcc(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 else np.nan


def interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, 0
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return float(lo), float(hi), len(finite)


def load_frame(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return pd.read_json(path, lines=True, dtype={'id': str})
    return pd.read_csv(path, dtype={'id': str})


def evaluate(frame, bootstrap=2000, seed=42, include_controls=True):
    if frame['id'].duplicated().any() or frame['id'].isna().any():
        raise ValueError('IDs must be nonmissing and unique within each model/dataset')
    targets = {}
    for column, threshold in [('sentence_similarity', 0.9), ('rouge_l', 0.5), ('exact_match', 0.5)]:
        if column in frame:
            continuous = pd.to_numeric(frame[column], errors='coerce').to_numpy(float)
            targets[column] = (continuous > threshold, continuous)
    if not targets and 'is_correct' in frame:
        values = pd.to_numeric(frame['is_correct'], errors='coerce').to_numpy(float)
        if not set(values[np.isfinite(values)]).issubset({0, 1}):
            raise ValueError('is_correct must be 0 or 1')
        targets['binary_only'] = (values == 1, values)
    if not targets:
        raise ValueError('No correctness targets found')
    scores = {s: pd.to_numeric(frame[s], errors='coerce').to_numpy(float)
              for s in ['HIDE_score', 'Omega', 'Delta_in']}
    scores['negative_Delta_in'] = -scores['Delta_in']
    for name in ['negative_mnll', 'negative_energy', 'negative_ln_entropy', 'lexical_similarity',
                 'negative_eigenscore_paper', 'negative_eigenscore_cov_log10']:
        if name in frame:
            scores[name] = pd.to_numeric(frame[name], errors='coerce').to_numpy(float)
    if include_controls and 'n_eff' in frame:
        n = pd.to_numeric(frame['n_eff'], errors='coerce').to_numpy(float)
        scores['constant_kernel_control'] = np.where(n >= 1, (n - 1) / np.maximum(n, 1)**2, 0)
    if include_controls and 'output_length' in frame:
        scores['negative_output_length'] = -pd.to_numeric(frame['output_length'], errors='coerce').to_numpy(float)
    results = []
    diagnostics = []
    for target, (y, continuous) in targets.items():
        valid = np.isfinite(continuous)
        for s in scores.values():
            valid &= np.isfinite(s)
        if 'status' in frame:
            valid &= (frame['status'].to_numpy() == 'ok')
        diagnostics.append({'target': target, 'total': len(frame), 'retained': int(valid.sum()),
                            'excluded': int((~valid).sum()),
                            'excluded_ids': frame.loc[~valid, 'id'].tolist()})
        if not valid.any():
            continue
        y = y[valid].astype(int)
        c = continuous[valid]
        ss = {name: s[valid] for name, s in scores.items()}
        boot = {name: [] for name in ss}
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap):
            idx = rng.integers(0, len(y), len(y))  # Identical resamples for every method.
            ya, ca = y[idx], c[idx]
            hide_auc = auc(ya, ss['HIDE_score'][idx])
            for name, s in ss.items():
                bs = s[idx]
                baseline_auc = auc(ya, bs)
                boot[name].append([baseline_auc, pcc(bs, ca), pcc(bs, ya),
                                   hide_auc - baseline_auc, pcc(ss['HIDE_score'][idx], bs)])
        for name, s in ss.items():
            values = [auc(y, s), pcc(s, c), pcc(s, y), auc(y, ss['HIDE_score']) - auc(y, s),
                      pcc(ss['HIDE_score'], s)]
            row = {'target': target, 'method': name, 'n': len(y), 'n_correct': int(y.sum())}
            for j, metric in enumerate(['auc', 'pcc_continuous', 'pcc_binary', 'hide_minus_baseline_auc', 'pcc_with_hide']):
                row[metric] = values[j]
                if bootstrap:
                    lo, hi, count = interval(np.array(boot[name])[:, j])
                    row.update({metric+'_lo': lo, metric+'_hi': hi, metric+'_valid_bootstraps': count})
            results.append(row)
    return pd.DataFrame(results), diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-eff-bin', nargs=2, type=int, metavar=('MIN', 'MAX'))
    args = p.parse_args()
    if args.bootstrap < 0:
        p.error('--bootstrap must be nonnegative')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_results, exclusions = [], []
    for file in args.inputs:
        df = load_frame(file)
        if 'model' not in df:
            df['model'] = Path(file).stem
        if 'dataset' not in df:
            df['dataset'] = 'UNSPECIFIED'
        if args.n_eff_bin:
            if 'n_eff' not in df:
                raise ValueError('Length stratification requires n_eff')
            df = df[df.n_eff.between(*args.n_eff_bin)]
        for (model, dataset), group in df.groupby(['model', 'dataset'], sort=False):
            result, diag = evaluate(group, args.bootstrap, args.seed)
            result['model'], result['dataset'] = model, dataset
            all_results.append(result)
            exclusions.append({'source': file, 'model': model, 'dataset': dataset,
                               'n_eff_bin': args.n_eff_bin, 'details': diag})
    if not all_results:
        if args.n_eff_bin:
            pd.DataFrame(columns=['model', 'dataset', 'target', 'method', 'n', 'auc']).to_csv(
                output / 'metrics.csv', index=False)
            (output / 'exclusions.json').write_text('[]\n')
            (output / 'results.md').write_text('# No eligible examples\n\nNo examples in the requested n_eff range.\n')
            print(output / 'results.md')
            return
        raise ValueError('No eligible rows')
    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(output / 'metrics.csv', index=False)
    (output / 'exclusions.json').write_text(json.dumps(exclusions, indent=2))
    lines = ['# Paired detection results', '',
             'AUC: correctness is positive. PCC continuous: original similarity/overlap target; '
             'PCC binary: thresholded correctness. Values are on a 0–1 scale, PCC on −1–1.', '',
             'Both update-norm directions are reported explicitly; neither is chosen using test labels. '
             'Intervals are pointwise paired percentile bootstrap intervals, not multiplicity-adjusted.', '',
             '| Model | Dataset | Target | Method | N | AUC (95% CI) | PCC continuous | HIDE − baseline AUC (95% CI) |',
             '|---|---|---|---|---:|---|---:|---|']
    def fmt(row, name):
        v = row[name]
        if not np.isfinite(v):
            return 'undefined'
        if name+'_lo' in row:
            return f'{v:.4f} [{row[name+"_lo"]:.4f}, {row[name+"_hi"]:.4f}]'
        return f'{v:.4f}'
    for _, row in results.iterrows():
        lines.append(f'| {row.model} | {row.dataset} | {row.target} | {row.method} | {row.n} | '
                     f'{fmt(row,"auc")} | {row.pcc_continuous:.4f} | {fmt(row,"hide_minus_baseline_auc")} |')
    (output / 'results.md').write_text('\n'.join(lines)+'\n')
    print(output / 'results.md')


if __name__ == '__main__':
    main()
