"""Reuse local checkpoints; download missing exact revisions into a user-owned cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from hide.provenance import atomic_json, checkpoint_identity, data_lock


def catalog():
    return json.loads((Path(__file__).parent/'config'/'checkpoints.json').read_text())


def cache_root():
    return Path(os.environ.get('HIDE_CHECKPOINT_CACHE', Path.cwd()/'checkpoints')).expanduser().resolve()


def validate(path, spec):
    """Validate inference structure; a local folder name is not a verified Hub revision."""
    path=Path(path)
    model_root=path
    if spec['kind']=='sentence-transformer':
        modules_path=path/'modules.json'
        if not modules_path.is_file():
            raise ValueError(f'{path}: missing sentence-transformer modules.json; do not substitute a plain encoder')
        modules=json.loads(modules_path.read_text())
        transformers=[m for m in modules if m.get('type','').endswith('.Transformer')]
        if len(transformers)!=1:
            raise ValueError(f'{path}: expected one sentence-transformer Transformer module')
        for module in modules:
            relative=PurePosixPath(module.get('path',''))
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe module path')
            # Normalize has no parameters/config and is commonly an absent empty directory on the Hub.
            if module.get('type','').endswith('.Normalize'):
                continue
            if not (path/relative).is_dir():
                raise ValueError(f'{path}: missing module directory {relative}')
            if module.get('type','').endswith('.Pooling') and not (path/relative/'config.json').is_file():
                raise ValueError(f'{path}: missing pooling configuration')
        model_root=path/transformers[0].get('path','')
    config_path=model_root/'config.json'
    if not config_path.is_file():
        raise ValueError(f'{path}: missing config.json')
    config=json.loads(config_path.read_text())
    if spec.get('model_type') and config.get('model_type')!=spec['model_type']:
        raise ValueError(f'{path}: model_type does not match {spec["repo_id"]}')
    if config.get('quantization_config'):
        raise ValueError(f'{path}: quantized checkpoints do not match the BF16 experiment protocol')
    if not any((model_root/name).is_file() for name in ('tokenizer.json','tokenizer.model','vocab.json','vocab.txt')):
        raise ValueError(f'{path}: missing tokenizer vocabulary')
    weights=[]
    for index in ('model.safetensors.index.json','pytorch_model.bin.index.json'):
        if (model_root/index).is_file():
            mapping=json.loads((model_root/index).read_text()).get('weight_map',{})
            if not mapping:
                raise ValueError(f'{path}: empty weight index')
            for name in set(mapping.values()):
                rel=PurePosixPath(name)
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('Unsafe weight path')
                weights.append(model_root/rel)
            break
    else:
        weights=[model_root/name for name in ('model.safetensors','pytorch_model.bin') if (model_root/name).is_file()]
    if not weights or any(not p.is_file() or p.stat().st_size==0 for p in weights):
        raise ValueError(f'{path}: missing/empty weight file or shard; repair this local checkpoint explicitly')


def prepare(name, local_root=None, allow_download=None):
    spec=catalog()[name]
    local_root=Path(local_root or os.environ.get('HIDE_MODEL_ROOT','checkpoints')).expanduser().resolve()
    local=local_root/spec['directory']
    cache=cache_root()
    enabled=os.environ.get('HIDE_DOWNLOAD_MISSING','1')=='1' if allow_download is None else allow_download
    # A supplied local directory is authoritative; never replace or edit it silently.
    if local.exists():
        validate(local,spec)
        print(f'Using local checkpoint: {local}',flush=True)
        identity=checkpoint_identity(local,hash_weights=True)
        return local,dict(name=name,origin='local',path=str(local),revision=None,
                          expected_repository=spec['repo_id'],identity=identity)
    target=cache/'downloads'/spec['repo_id'].replace('/','--')/spec['revision']
    ready=target.parent/(spec['revision']+'.ready.json')
    lock_name=hashlib.sha256(spec['repo_id'].encode()).hexdigest()
    with data_lock(cache,lock_name):
        if not ready.is_file():
            if not enabled:
                raise FileNotFoundError(f'{local} is absent and downloads are disabled')
            print(f'Downloading {spec["repo_id"]}@{spec["revision"]} to {target}',flush=True)
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id=spec['repo_id'],revision=spec['revision'],
                                  local_dir=str(target),allow_patterns=spec['files'],max_workers=4)
            except Exception as exc:
                raise RuntimeError(f'Could not download {spec["repo_id"]}. For gated models, accept access on its '
                                   'Hugging Face page and authenticate with huggingface-cli login in this environment. '
                                   'Then rerun; cached download progress is retained.') from exc
            validate(target,spec)
            # Hash once before marking ready. Cached hashes are validated against current file stats on reuse.
            identity=checkpoint_identity(target,hash_weights=True)
            atomic_json(ready,dict(repo_id=spec['repo_id'],revision=spec['revision']))
        else:
            saved=json.loads(ready.read_text())
            if saved!={'repo_id':spec['repo_id'],'revision':spec['revision']}:
                raise ValueError('Cached revision record differs from the configured checkpoint')
            validate(target,spec)
            identity=checkpoint_identity(target,hash_weights=True)
    return target,dict(name=name,origin='download-cache',path=str(target),repo_id=spec['repo_id'],
                       revision=spec['revision'],identity=identity)


def prepare_arguments(args):
    """Resolve only the paper's named runs; custom direct-runner checkpoints stay explicit."""
    if args.model_name not in catalog():
        return None
    declarations=[('generator','model_path',args.model_name),('keyword','keyword_model','keyword')]
    if args.mode=='detection':
        declarations.append(('judge','judge_model','judge'))
    results={}
    for role,field,name in declarations:
        requested=Path(getattr(args,field)).expanduser()
        # Launchers supply MODEL_ROOT/DIRECTORY; preserve any explicitly supplied custom directory.
        spec=catalog()[name]
        if requested.name!=spec['directory']:
            validate(requested,spec)
            identity=checkpoint_identity(requested,hash_weights=True)
            location=requested.resolve()
            record=dict(name=name,origin='explicit-local',path=str(location),revision=None,identity=identity)
        else:
            location,record=prepare(name,local_root=requested.parent)
        setattr(args,field,str(location))
        results[role]=record
    atomic_json(Path(args.output).with_suffix('.checkpoints.json'),results)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=list(catalog()),default=list(catalog()))
    p.add_argument('--local-only',action='store_true',help='Do not download anything missing')
    args=p.parse_args()
    records=[]
    for name in args.models:
        _,record=prepare(name,allow_download=False if args.local_only else None)
        records.append(record)
    summary=cache_root()/'checkpoint_inventory.json'
    # Keep reports from previously prepared models as well.
    with data_lock(cache_root(),'inventory'):
        existing=json.loads(summary.read_text()) if summary.exists() else {}
        existing.update({r['name']:r for r in records})
        atomic_json(summary,existing)
    print(f'Checkpoint inventory: {summary}',flush=True)


if __name__=='__main__':
    main()
