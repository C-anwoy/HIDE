"""Add Appendix B reference panels and magenta markup to generated tables."""
from pathlib import Path
import re


def format_revision_tables(root=Path('.')):
    paper = root/'paper'
    benchmark = {}
    for line in (paper/'files/results_table.tex').read_text().splitlines():
        found = re.search(r'\\multicolumn\{21\}\{c\}\{\\cellcolor\[HTML\]\{C0C0C0\}([^}]+)', line)
        if found:
            model = found.group(1)
        if line.startswith(r'\textbf{\method} &'):
            values = [float(re.search(r'-?\d+(?:\.\d+)?', c).group()) for c in line.split('&')[1:]]
            assert len(values) == 20
            benchmark[model] = values
    counts = {'SQuAD':5928, 'RACE':3498, 'NQ':3610, 'TriviaQA':9960}
    offsets = {'RACE':0, 'SQuAD':4, 'NQ':8, 'TriviaQA':12}
    for p in (paper/'files/tables').glob('*.tex'):
        text = p.read_text()
        if r'\revisionr3{' in text:
            continue  # Already formatted; do not nest revision commands.
        if p.stem in {'paired_s', 'paired_r'}:
            suffix = p.stem[-1]; auc, pcc = (0,2) if suffix == 's' else (1,3)
            lines = [r'\revisionr3{\textbf{Panel A: HIDE full-benchmark reference (Appendix B)}}\par\smallskip',
                     r'\begin{tabular}{llrrr}', r'\toprule',
                     f'Model & Dataset & $N$ & AUC$_{suffix}$ & PCC$_{suffix}$'+r' \\', r'\midrule']
            for model in ['Llama-3-8B', 'Gemma-2-9B']:
                for ds in ['SQuAD','RACE','NQ','TriviaQA']:
                    v=benchmark[model]; i=offsets[ds]
                    lines.append(f'{model} & {ds} & {counts[ds]} & {v[i+auc]:.2f} & {v[i+pcc]:.2f}'+r' \\')
            lines += [r'\bottomrule\end{tabular}',r'\par\medskip',
                      r'\revisionr3{\textbf{Panel B: Matched first-answer-line comparison}}\par\smallskip']
            text=text.replace(r'\begin{tabular}{llr rr rr rr}', '\n'.join(lines)+'\n'+r'\begin{tabular}{llr rr rr rr}')
            text=text.replace('Matched detector comparison under first-answer-line stopping: 2,000 fixed seed-42 examples per model/dataset.',
                r'Panel A reproduces HIDE values from Table~\ref{tab:results_exp} of Appendix B without alteration ($N$: full-benchmark sample size). Panel B evaluates all three scores on the same 2,000 fixed seed-42 examples per model/dataset under first-answer-line stopping; $N_+$ refers only to Panel B. Because cohorts and stopping differ, Panel A is a reference, not a matched comparator for Panel B.')
            text=text.replace('All scores use the same examples and labels;', 'Within Panel B, all scores use the same examples and labels;')
            text=text.replace(' These results are separate from the historical full-cohort tables.', '')
        text=text.replace('earlier stopping protocol', 'benchmark stopping protocol')
        text=re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}',lambda m:r'\revisionr3{'+m.group()+'}',text,flags=re.S)
        # Generated captions occupy one line; nested references/braces are retained.
        text=re.sub(r'\\caption\{(.*)\}',lambda m:r'\caption{\revisionr3{'+m.group(1)+'}}',text)
        p.write_text(text)


if __name__ == '__main__':
    format_revision_tables()
