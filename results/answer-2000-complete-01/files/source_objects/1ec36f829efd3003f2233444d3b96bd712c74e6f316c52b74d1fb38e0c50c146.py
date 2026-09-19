"""Record completeness, generation-cap counts and actual per-example wall time."""
import argparse
import csv
import json
from pathlib import Path
from hide.export_results import inspect_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='outputs/final')
    args=p.parse_args()
    root=Path(args.source)
    rows=[]
    for path in sorted(root.rglob('*.jsonl')):
        row=inspect_run(path)
        row['source']=str(path.relative_to(root))
        row['mean_example_wall_s']=row.get('sum_example_wall_s',0)/max(1,row.get('observed',0))
        rows.append(row)
    out=root/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    (out/'run_overview.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=['source','model','dataset','mode','expected','successful','n_errors','complete',
            'no_output_state','hit_generation_cap','ablation_errors','ablation_undefined','mean_example_wall_s']
    with (out/'run_overview.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f'{r["source"]}: {r.get("successful",0)}/{r.get("expected","?")}; '
              f'complete={r["complete"]}; mean example wall={r["mean_example_wall_s"]:.3f}s')
    print(out/'run_overview.csv')


if __name__=='__main__': main()
