"""Prespecified score ablations computed from one cached generation per example."""
import math
import torch
from hide.core import extract_keyword_representation, unbiased_HSIC
from hide.kernels import KERNEL_FUNCTIONS


def centered_hsic(k, l, denominator='n2'):
    n = len(k)
    if denominator == 'nminus1' and n <= 1:
        raise ValueError('(n-1)^2 estimator is undefined for n=1')
    h = torch.eye(n, device=k.device, dtype=k.dtype) - torch.ones_like(k)/n
    return torch.trace((h @ k @ h) @ (h @ l @ h)) / (n*n if denominator == 'n2' else (n-1)**2)


class KeywordCache:
    def __init__(self, model):
        self.model, self.cache = model, {}

    def extract_keywords(self, text, **kwargs):
        key = (text, kwargs['top_n'])
        if key not in self.cache:
            self.cache[key] = self.model.extract_keywords(text, **kwargs)
        return self.cache[key]


def score_variants(hidden, tokenizer, input_ids, output_ids, layer, keyword_model, budget=20):
    kw = KeywordCache(keyword_model)
    variants = []
    selected = {}
    no_state = len(hidden) <= 1

    def pair(index):
        return hidden[0][index][0].float(), torch.cat([s[index][0] for s in hidden[1:]]).float()

    def keyword_pair(index, k):
        key = index, k
        if key not in selected:
            x, y = pair(index)
            selected[key] = extract_keyword_representation(x, y, tokenizer, input_ids, output_ids[:-1],
                                                           k=k, kw_model=kw)[:2]
        return selected[key]

    def add(name, family, parameter, fn):
        item = dict(name=name, family=family, parameter=parameter)
        if no_state:
            # Match HIDE's declared no-state fallback, except mathematically undefined comparators.
            item.update(score=None, status='undefined', reason='No forwarded output state', n_eff=0)
        else:
            try:
                value, n = fn()
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError('Non-finite score')
                item.update(score=value, n_eff=n, status='ok')
            except (ValueError, RuntimeError) as exc:
                status = 'undefined' if 'undefined for n=1' in str(exc) else 'error'
                item.update(score=None, status=status, reason=str(exc))
        variants.append(item)

    def score(index=layer, k=budget, kernel='rbf', gamma=None, estimator='adapted'):
        x, y = keyword_pair(index, k)
        kernel_fn = KERNEL_FUNCTIONS[kernel]
        options = {'gamma': gamma} if gamma is not None else {}
        a, b = kernel_fn(x, **options), kernel_fn(y, **options)
        value = unbiased_HSIC(a, b) if estimator == 'adapted' else centered_hsic(a, b, estimator)
        return value, len(x)

    for k in [5, 10, 15, 20, 25, 30]:
        add(f'budget_{k}', 'budget', k, lambda k=k: score(k=k))
    # 1..L includes the final normalized HF hidden state; never call it a raw decoder-block write.
    for index in range(1, len(hidden[0])):
        add(f'layer_{index}', 'layer', index, lambda index=index: score(index=index))
    for kernel in KERNEL_FUNCTIONS:
        add(f'kernel_{kernel}', 'kernel', kernel, lambda kernel=kernel: score(kernel=kernel))
    for gamma in [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]:
        add(f'gamma_{gamma:g}', 'gamma', gamma, lambda gamma=gamma: score(gamma=gamma))
    for estimator in ['adapted', 'n2', 'nminus1']:
        add(f'estimator_{estimator}', 'estimator', estimator, lambda estimator=estimator: score(estimator=estimator))

    def cosine():
        x, y = keyword_pair(layer, budget)
        denominator = x.norm() * y.norm()
        if denominator == 0:
            raise ValueError('Cosine undefined for zero norm')
        return (x.flatten() @ y.flatten())/denominator, len(x)
    add('flattened_cosine', 'geometry', 'cosine', cosine)

    def svd_score(estimator):
        x, y = pair(layer)
        k = min(budget, len(x), len(y), x.shape[1])
        def project(a):
            _, s, vh = torch.linalg.svd(a, full_matrices=False)
            return s[:k, None] * vh[:k]
        a, b = KERNEL_FUNCTIONS['rbf'](project(x)), KERNEL_FUNCTIONS['rbf'](project(y))
        value = unbiased_HSIC(a, b) if estimator == 'adapted' else centered_hsic(a, b)
        return value, k
    for estimator in ['adapted', 'n2']:
        add(f'svd_{estimator}', 'selection', f'svd_{estimator}', lambda estimator=estimator: svd_score(estimator))
    return variants


def variant_names(num_layers):
    return ([f'budget_{k}' for k in [5,10,15,20,25,30]]
            + [f'layer_{i}' for i in range(1,num_layers+1)]
            + [f'kernel_{k}' for k in KERNEL_FUNCTIONS]
            + [f'gamma_{g:g}' for g in [1e-9,1e-7,1e-5,1e-3,1e-1,1.0]]
            + [f'estimator_{e}' for e in ['adapted','n2','nminus1']]
            + ['flattened_cosine','svd_adapted','svd_n2'])
