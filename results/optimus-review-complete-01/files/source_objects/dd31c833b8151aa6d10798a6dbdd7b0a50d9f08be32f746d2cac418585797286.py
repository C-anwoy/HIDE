"""Plot every valid paired record; QCBW is the top similarity quartile among wrong answers."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from hide.evaluate import load_frame, pcc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load_frame(f) for f in args.inputs],ignore_index=True)
    correlations = []
    for model, group in df.groupby('model'):
        datasets = list(group.dataset.unique())
        fig, axs = plt.subplots(2,len(datasets),figsize=(4*len(datasets),7),squeeze=False,layout='constrained')
        for j, ds in enumerate(datasets):
            data = group[group.dataset == ds].copy()
            if data.id.duplicated().any():
                raise ValueError('Duplicate IDs')
            good = np.isfinite(data[['HIDE_score','Omega','Delta_in','is_correct']]).all(axis=1)
            if 'status' in data:
                good &= data.status == 'ok'
            data = data[good]
            wrong = data.is_correct == 0
            qcbw = pd.Series(False,index=data.index)
            if ds in ['nq_open','triviaqa'] and 'q_g_similarity' in data and wrong.any():
                threshold = data.loc[wrong,'q_g_similarity'].quantile(.75)
                qcbw = wrong & (data.q_g_similarity >= threshold)
            for k, metric in enumerate(['Omega','Delta_in']):
                ax = axs[k,j]
                for mask, label, color, marker in [
                    (wrong & ~qcbw,'Incorrect','#35788b','s'),
                    (~wrong,'Correct','#b33c3c','o'), (qcbw,'QCBW','#c18d2e','*')]:
                    ax.scatter(data.loc[mask,'HIDE_score'],data.loc[mask,metric],label=label,
                               color=color,marker=marker,s=18,alpha=.5)
                r = pcc(data.HIDE_score.to_numpy(),data[metric].to_numpy())
                ax.set(xlabel=f'HIDE score\n{ds}; n={len(data)}, r={r:.3f}',
                       ylabel='Attention mass to prompt' if k == 0 else 'Unprojected update-norm proxy')
                if k == 0 and j == 0:
                    ax.legend(fontsize=8)
                correlations.append({'model':model,'dataset':ds,'metric':metric,'n':len(data),'pcc':r,
                                     'n_qcbw':int(qcbw.sum()),'excluded':int((~good).sum())})
        fig.savefig(out/f'{model}_mechanistic.png',dpi=250)
        fig.savefig(out/f'{model}_mechanistic.pdf')
        plt.close(fig)
    pd.DataFrame(correlations).to_csv(out/'correlations.csv',index=False)


if __name__ == '__main__':
    main()
