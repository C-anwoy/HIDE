"""Reproducible paired detection, ablations and synchronized timing; run with python -m hide.runner."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', default='custom')
    p.add_argument('--mode', choices=['detection', 'timing'], default='detection')
    p.add_argument('--model-path', required=True)
    p.add_argument('--model-name', required=True)
    p.add_argument('--dataset', choices=['SQuAD', 'nq_open', 'triviaqa', 'race'], required=True)
    p.add_argument('--data-root', required=True, help='Contains datasets/ (same layout as original repo)')
    p.add_argument('--keyword-model', required=True)
    p.add_argument('--judge-model', help='Required for detection; original nli-roberta-large checkpoint')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--keyword-device', default='cpu')
    p.add_argument('--judge-device', default='cpu')
    p.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='bfloat16')
    p.add_argument('--attention-backend', choices=['eager', 'sdpa'], default='eager')
    p.add_argument('--samples', type=int, default=0, help='0 (default) means the entire dataset')
    p.add_argument('--start', type=int, default=0, help='Start position in the seeded selected cohort')
    p.add_argument('--stop', type=int, help='Exclusive end position; omitted means cohort end')
    p.add_argument('--time-budget-seconds', type=float, default=0, help='Operational soft limit, checked between examples; 0 disables')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--layer', type=int, help='HF hidden_states index; default num_hidden_layers // 2')
    p.add_argument('--keywords', type=int, default=20)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=3, help='Timing repeats on each query')
    p.add_argument('--output', required=True, help='New JSONL, append/resume only with matching manifest')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--decoding', choices=['greedy', 'temperature', 'nucleus'], default='greedy')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--ablations', action='store_true', help='Score all prespecified variants on the same states')
    p.add_argument('--multipass-samples', type=int, default=0, help='0 disables; >=2 enables the five-baseline comparison')
    p.add_argument('--hash-weights', action='store_true', help='Hash full model/judge/keyword weights for strict provenance')
    return p


def sync(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def timed(fn, device):
    sync(device)
    t = time.perf_counter()
    result = fn()
    sync(device)
    return result, time.perf_counter() - t


def proxies(hidden_states, attentions, prompt_length, layer):
    """Mean over heads/tokens; HF attention[layer-1] updates hidden_states[layer].

    Use generated-token positions available in cached generation (exclude final token).
    Delta is the unprojected proxy, NOT W_O sum(alpha W_V h).
    """
    import torch
    if len(hidden_states) == 1:
        return 0.0, 0.0  # Explicit no-observed-output-state fallback, same as HIDE.
    h_input = hidden_states[0][layer - 1][0, :prompt_length].float()
    rows = []
    for step in attentions[1:]:
        a = step[layer - 1]
        if a is None:
            raise ValueError('Attention weights unavailable; use eager attention')
        rows.append(a[0, :, -1, :prompt_length].float().mean(dim=0))
    a = torch.stack(rows)
    return a.sum(-1).mean().item(), (a @ h_input).norm(dim=-1).mean().item()


def labels(text, answer, question, judge, rouge):
    from hide.evaluate import normalize_text
    from sentence_transformers import util
    emb = judge.encode([text, answer, question], convert_to_tensor=True)
    ss = util.cos_sim(emb[0], emb[1]).item()
    rl = rouge.score(target=answer, prediction=text)['rougeL'].fmeasure
    return {'sentence_similarity': ss, 'rouge_l': rl,
            'exact_match': int(normalize_text(text) == normalize_text(answer)),
            'is_correct': int(ss > 0.9), 'q_g_similarity': util.cos_sim(emb[0], emb[2]).item()}


def main():
    from hide.provenance import run_lock
    args = parser().parse_args()
    from hide.execution import RunControl, RunPaused
    with RunControl(args.time_budget_seconds) as control, run_lock(args.output):
        try:
            run(args, control)
        except RunPaused as exc:
            print(f'PAUSED: {exc}; saved rows are resumable', flush=True)
            raise SystemExit(75)


def run(args, control=None):
    from hide.provenance import atomic_json, source_files, checkpoint_identity, example_seed
    if args.mode == 'detection' and (not args.judge_model or args.attention_backend != 'eager'):
        raise SystemExit('Detection requires --judge-model and --attention-backend eager')
    if args.samples < 0 or args.keywords < 1 or args.max_new_tokens < 1 or args.repeats < 1 or args.warmup < 0:
        raise SystemExit('Invalid sample, token, repeat or warmup count')
    if not 0 < args.top_p <= 1 or args.temperature <= 0:
        raise ValueError('temperature must be positive and top-p must be in (0, 1]')
    if args.multipass_samples < 0 or args.multipass_samples == 1:
        raise ValueError('multipass-samples must be 0 or >=2')
    if args.multipass_samples and args.decoding != 'greedy':
        raise ValueError('The comparison study evaluates a greedy target answer')
    if args.mode == 'timing' and (args.decoding != 'greedy' or args.ablations or args.multipass_samples):
        raise ValueError('Timing uses greedy decoding and the default HIDE score only')
    if args.start < 0 or (args.stop is not None and args.stop <= args.start):
        raise ValueError('Require 0 <= start < stop')
    os.environ['HIDE_DATA_ROOT'] = str(Path(args.data_root).resolve())
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from sentence_transformers import SentenceTransformer
    from keybert import KeyBERT
    from hide.core import get_unbiased_hsic_score_keybert
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset_module = importlib.import_module('hide.datasets.' + args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    from hide.provenance import data_lock
    with data_lock(args.data_root, args.dataset):
        dataset = dataset_module.get_dataset(tokenizer)
    from hide.export_results import DATASET_COUNTS
    if len(dataset) != DATASET_COUNTS[args.dataset]:
        raise ValueError(f'Dataset has {len(dataset)} examples; paper requires {DATASET_COUNTS[args.dataset]}. '
                         'Audit the dataset cache; do not trim to force a match.')
    dataset_fingerprint = getattr(dataset, '_fingerprint', None)
    full_count = len(dataset)
    dataset = dataset.shuffle(seed=args.seed)
    if args.samples:
        dataset = dataset.select(range(min(args.samples, len(dataset))))
    if not len(dataset):
        raise ValueError('Empty dataset')
    parent_ids = [str(x) for x in dataset['id']]
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError('Duplicate parent dataset IDs; audit before partitioning')
    parent_hash = hashlib.sha256()
    for example in dataset:
        parent_hash.update(json.dumps([str(example['id']), example['prompt'], example['answer']],
                                      ensure_ascii=False).encode())
    stop = len(dataset) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(dataset):
        raise ValueError('Partition outside the selected cohort')
    partition = None
    if args.start or args.stop is not None:
        partition = dict(start=args.start, stop=stop, parent_selected_ids=parent_ids,
                         parent_cohort_sha256=parent_hash.hexdigest())
        dataset = dataset.select(range(args.start, stop))
    selected_ids = [str(x) for x in dataset['id']]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError('Duplicate dataset IDs; audit dataset provenance before proceeding')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix('.manifest.json')
    config = vars(args).copy()
    config.pop('resume')
    config.pop('time_budget_seconds')
    checkpoint_config = AutoConfig.from_pretrained(args.model_path)
    cohort_hash = hashlib.sha256()
    for example in dataset:
        cohort_hash.update(json.dumps([str(example['id']), example['prompt'], example['answer']],
                                     ensure_ascii=False).encode())
    metadata = {'arguments': config, 'selected_ids': selected_ids,
                'cohort_sha256': cohort_hash.hexdigest(),
                'checkpoint_config': json.loads(checkpoint_config.to_json_string()),
                'python': platform.python_version(), 'torch': torch.__version__,
                'transformers': transformers.__version__, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else args.device,
                'protocol': 'hide-v2; aligned-cached; first-reference; seeded-per-example; no filtering or rebalancing',
                'schema_version': 2, 'source_sha256': source_files(),
                'dataset_fingerprint': dataset_fingerprint, 'full_dataset_count': full_count,
                'checkpoint_identity': {name: checkpoint_identity(path, args.hash_weights)
                    for name, path in [('generator', args.model_path), ('keyword', args.keyword_model),
                                       ('judge', args.judge_model)] if path}}
    if partition is not None:
        metadata['partition'] = partition
    if args.ablations:
        from hide.ablations import variant_names
        metadata['expected_ablation_names'] = variant_names(checkpoint_config.num_hidden_layers)
    metadata['packages'] = {}
    for package in ['numpy', 'pandas', 'datasets', 'keybert', 'sentence-transformers',
                    'sentencepiece', 'scikit-learn', 'rouge-score', 'accelerate']:
        try:
            metadata['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata['packages'][package] = 'not installed'
    completed = set()
    if out.exists():
        if not args.resume:
            raise FileExistsError(f'{out} exists. Choose a new filename or --resume')
        old = json.loads(manifest_path.read_text())
        if old != metadata:
            raise ValueError('Resume manifest differs; use a new output path')
        for line in out.read_text().splitlines():
            row = json.loads(line)  # A truncated line is an error, never silently skipped.
            if row.get('status') != 'ok':
                raise ValueError('Output contains an error record; fix the cause and use a fresh output path')
            key = (str(row['id']), row.get('repeat', 0))
            if key in completed:
                raise ValueError(f'Duplicate stored record: {key}')
            completed.add(key)
        expected = {(i, rep) for i in selected_ids for rep in range(args.repeats if args.mode == 'timing' else 1)}
        if not completed.issubset(expected):
            raise ValueError('Stored records do not belong to this manifest')
        if completed == expected:
            print(f'Already complete: {out}', flush=True)
            return
    else:
        atomic_json(manifest_path, metadata)
    # Preserve the exact code used by this run, including uncommitted changes.
    snapshot_dir = out.with_suffix('.sources')
    snapshot_dir.mkdir(exist_ok=True)
    for source, digest in metadata['source_sha256'].items():
        path = Path(source)
        destination = snapshot_dir / (path.parent.name + '__' + path.name)
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError('Stored source snapshot differs from this run')
        destination.write_bytes(path.read_bytes())
    if control:
        control.check()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attention_backend,
        low_cpu_mem_usage=True
    ).to(args.device).eval()
    layer = args.layer if args.layer is not None else model.config.num_hidden_layers // 2
    if not 1 <= layer < model.config.num_hidden_layers:
        raise ValueError('Select an internal layer in [1, num_hidden_layers-1]')
    kw = KeyBERT(model=SentenceTransformer(args.keyword_model, device=args.keyword_device))
    judge = None
    rouge = None
    if args.mode == 'detection':
        from rouge_score.rouge_scorer import RougeScorer
        judge = SentenceTransformer(args.judge_model, device=args.judge_device)
        rouge = RougeScorer(['rougeL'], use_stemmer=True)
    gen_config = dataset_module._generate_config(tokenizer)
    gen_config.update(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                      do_sample=args.decoding != 'greedy', num_beams=1, use_cache=True,
                      top_k=0, top_p=args.top_p if args.decoding == 'nucleus' else 1.0,
                      temperature=args.temperature if args.decoding == 'temperature' else 1.0)
    if args.decoding == 'greedy':
        gen_config.pop('temperature', None)
        gen_config.pop('top_p', None)
        gen_config.pop('top_k', None)
    gen_config = GenerationConfig(**gen_config)
    runtime = {'hostname': platform.node(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'gpu_uuid': str(getattr(torch.cuda.get_device_properties(args.device), 'uuid', ''))
                   if str(args.device).startswith('cuda') else None, 'cpu_count': os.cpu_count(), 'platform': platform.platform(),
               'float32_matmul_precision': torch.get_float32_matmul_precision(),
               'keyword_device': args.keyword_device, 'judge_device': args.judge_device if judge else None,
               'layer': layer, 'hidden_size': model.config.hidden_size,
               'low_cpu_mem_usage': True,
               'num_hidden_layers': model.config.num_hidden_layers,
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'generation_config': gen_config.to_dict(),
               'gpu_memory_bytes': torch.cuda.get_device_properties(args.device).total_memory
                   if str(args.device).startswith('cuda') else None}
    from datetime import datetime, timezone
    import uuid
    execution_id = uuid.uuid4().hex
    runtime['execution_id'] = execution_id
    runtime['started_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(out.with_suffix('.runtime.json'), runtime)
    atomic_json(out.with_suffix('.executions') / (execution_id+'.json'), runtime)
    print(json.dumps({'model': args.model_name, 'layer': layer, 'hidden_size': model.config.hidden_size,
                      'samples': len(dataset), 'backend': args.attention_backend}), flush=True)

    def generate(ids, mask, hidden=False, attention=False):
        return model.generate(ids, attention_mask=mask, generation_config=gen_config,
                              return_dict_in_generate=True, output_hidden_states=hidden,
                              output_attentions=attention, output_scores=False,
                              output_logits=args.mode == 'detection')

    def hide_score(result, ids):
        return get_unbiased_hsic_score_keybert(result.hidden_states, tokenizer, ids[0],
                result.sequences[0, ids.shape[1]:], keywords=args.keywords, layer=layer,
                kernel='rbf', kw_model=kw)

    def base_pipeline(ids, mask):
        result = generate(ids, mask)
        return result.sequences[0].detach().cpu().tolist()

    def hide_pipeline(ids, mask):
        result, capture_s = timed(lambda: generate(ids, mask, hidden=True), args.device)
        scores, score_s = timed(lambda: hide_score(result, ids), args.device)
        return result.sequences[0].detach().cpu().tolist(), scores, capture_s, score_s

    with torch.inference_mode(), out.open('a') as stream:
        if args.mode == 'timing':
            for j in range(args.warmup):
                if control:
                    control.check()
                ex = dataset[j % len(dataset)]
                ids = ex['input_ids'].unsqueeze(0).to(args.device)
                mask = ex['attention_mask'].unsqueeze(0).to(args.device)
                base_pipeline(ids, mask)
                hide_pipeline(ids, mask)
        for i, ex in enumerate(dataset):
            example_id = str(ex['id'])
            ids = ex['input_ids'].unsqueeze(0).to(args.device)
            mask = ex['attention_mask'].unsqueeze(0).to(args.device)
            repeats = args.repeats if args.mode == 'timing' else 1
            for repeat in range(repeats):
                if (example_id, repeat) in completed:
                    continue
                if control:
                    control.check()
                run_seed = example_seed(args.seed, args.dataset, example_id, repeat)
                random.seed(run_seed)
                torch.manual_seed(run_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(run_seed)
                row = {'schema_version': 2, 'execution_id': execution_id, 'example_seed': run_seed, 'decoding': args.decoding,
                       'id': example_id, 'repeat': repeat, 'model': args.model_name,
                       'dataset': args.dataset, 'layer': layer, 'hidden_size': model.config.hidden_size,
                       'input_length': ids.shape[1], 'status': 'ok'}
                if str(args.device).startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats(args.device)
                example_start = time.perf_counter()
                result = None
                try:
                    if args.mode == 'detection':
                        result = generate(ids, mask, hidden=True, attention=True)
                        gen_ids = result.sequences[0, ids.shape[1]:]
                        sliding = getattr(model.config, 'sliding_window', None)
                        if model.config.model_type == 'gemma2' and (layer - 1) % 2 == 0 and sliding:
                            if ids.shape[1] + len(gen_ids) - 2 >= sliding - 1:
                                raise ValueError('Selected local-attention layer crossed sliding-window boundary; '
                                                 'prompt-key indexing needs a window-aware implementation')
                        from hide.baselines import token_statistics
                        row.update(token_statistics(result.logits, gen_ids))
                        score, ik, ok, it, ot = hide_score(result, ids)
                        omega, delta = proxies(result.hidden_states, result.attentions, ids.shape[1], layer)
                        row.update(HIDE_score=score, Omega=omega, Delta_in=delta,
                                   input_ids=ids[0].cpu().tolist(),
                                   input_keywords=ik, output_keywords=ok,
                                   input_tokens_topk=it, output_tokens_topk=ot,
                                   n_eff=len(ot) if isinstance(ot, list) else 0,
                                   generated_ids=gen_ids.cpu().tolist(), output_length=len(gen_ids),
                                   generated_text=tokenizer.decode(gen_ids, skip_special_tokens=True),
                                   question=ex['question'], answer=ex['answer'],
                                   prompt=ex['prompt'], no_output_state=len(gen_ids) <= 1,
                                   hit_generation_cap=len(gen_ids) == args.max_new_tokens)
                        if args.ablations:
                            from hide.ablations import score_variants
                            row['ablations'] = score_variants(result.hidden_states, tokenizer, ids[0], gen_ids,
                                                             layer, kw, args.keywords)
                        del result
                        result = None
                        row.update(labels(row['generated_text'], ex['answer'], ex['question'], judge, rouge))
                        if args.multipass_samples:
                            from hide.baselines import sampled_baselines
                            row.update(sampled_baselines(model, tokenizer, ids, mask, gen_config, layer,
                                                         args.multipass_samples, rouge))
                        # Preserve original first-reference evaluation. Aliases are a separate sensitivity analysis.
                        if 'additional_answers' in ex:
                            row['additional_answers'] = ex['additional_answers']
                    else:
                        # Alternate order to reduce systematic warm-cache / thermal drift.
                        order = ['base', 'hide'] if (args.start + i + repeat) % 2 == 0 else ['hide', 'base']
                        for variant in order:
                            if variant == 'base':
                                base_ids, base_s = timed(lambda: base_pipeline(ids, mask), args.device)
                            else:
                                (hide_ids, scores, capture_s, score_s), total_s = timed(
                                    lambda: hide_pipeline(ids, mask), args.device)
                        if base_ids != hide_ids:
                            raise ValueError('Generation differs with hidden-state capture; paired latency invalid')
                        row.update(base_s=base_s, capture_s=capture_s, score_s=score_s,
                                   total_s=total_s, overhead_s=total_s-base_s,
                                   HIDE_score=scores[0],
                                   input_ids=ids[0].cpu().tolist(), generated_ids=base_ids[ids.shape[1]:],
                                   generated_text=tokenizer.decode(base_ids[ids.shape[1]:], skip_special_tokens=True),
                                   prompt=ex['prompt'], question=ex['question'], answer=ex['answer'],
                                   input_keywords=scores[1], output_keywords=scores[2],
                                   input_tokens_topk=scores[3], output_tokens_topk=scores[4],
                                   n_eff=len(scores[4]) if isinstance(scores[4], list) else 0,
                                   no_output_state=len(base_ids)-ids.shape[1] <= 1,
                                   hit_generation_cap=len(base_ids)-ids.shape[1] == args.max_new_tokens,
                                   output_length=len(base_ids)-ids.shape[1])
                    for key in ['HIDE_score', 'Omega', 'Delta_in', 'sentence_similarity', 'rouge_l']:
                        if key in row and not math.isfinite(row[key]):
                            raise ValueError(f'Non-finite {key}')
                except Exception as exc:
                    row['status'] = 'error'
                    row['error'] = f'{type(exc).__name__}: {exc}'
                    for key, value in list(row.items()):
                        if isinstance(value, float) and not math.isfinite(value):
                            row[key] = None
                    if result is not None:
                        del result
                        result = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                row['example_wall_s'] = time.perf_counter() - example_start
                if str(args.device).startswith('cuda'):
                    row['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
                    row['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved(args.device)
                stream.write(json.dumps(row, allow_nan=False, default=str) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                print(f'{i+1}/{len(dataset)} {example_id} repeat={repeat} {row["status"]}', flush=True)
                if row['status'] == 'error':
                    # Fail fast; fix the cause, then rerun to a new output. Errors cannot become hallucination zeros.
                    raise RuntimeError(row['error'])


if __name__ == '__main__':
    main()
