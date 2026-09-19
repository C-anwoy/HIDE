"""Prepare the original loader's SQuAD file without changing its preprocessing."""
import argparse
from pathlib import Path
import urllib.request
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    args=p.parse_args()
    out=Path(args.data_root)/'datasets'/'dev-v2.0.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        with urllib.request.urlopen('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json') as response:
            raw=response.read()
        data=json.loads(raw)
        if 'data' not in data:
            raise ValueError('Unexpected SQuAD data schema')
        out.write_bytes(raw)
    data=json.loads(out.read_text())
    count=sum(not q['is_impossible'] for d in data['data'] for p in d['paragraphs'] for q in p['qas'])
    if count != 5928:
        raise ValueError(f'Expected 5928 answerable questions; got {count}')
    print(out, 'answerable questions:', count)
    print('NQ, TriviaQA and RACE are prepared by their original loaders at first use.')


if __name__=='__main__':
    main()
