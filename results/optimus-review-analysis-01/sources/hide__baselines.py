"""Five training-free comparators with explicit probability/covariance conventions."""
import math
import torch


def token_statistics(logits, token_ids):
    """Use unprocessed LM logits and the token actually generated, including the stop token."""
    if len(logits) != len(token_ids) or not len(token_ids):
        raise ValueError('Logits and generated token IDs must align and be nonempty')
    log_probs, log_normalizers = [], []
    for scores, token in zip(logits, token_ids):
        scores = scores[0].float()
        z = torch.logsumexp(scores, dim=-1)
        log_probs.append(float(scores[int(token)] - z))
        log_normalizers.append(float(z))
    mnll = -sum(log_probs) / len(log_probs)
    energy = -sum(log_normalizers) / len(log_normalizers)  # T=1; average over generated positions.
    if not math.isfinite(mnll) or not math.isfinite(energy):
        raise ValueError('Non-finite logit baseline')
    return dict(token_log_probs=log_probs, token_log_normalizers=log_normalizers,
                mnll=mnll, energy=energy, negative_mnll=-mnll, negative_energy=-energy)


def eigen_statistics(embeddings, alpha=1e-3):
    """Paper Sigma=Z^T C_d Z; also expose the original sample-covariance convention."""
    x = torch.stack(embeddings).double()  # N x d
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError('EigenScore requires at least two vectors with width >=2')
    centered = x - x.mean(dim=1, keepdim=True)
    gram = centered @ centered.T
    eye = torch.eye(len(x), dtype=x.dtype, device=x.device)
    eig = torch.linalg.eigvalsh(gram + alpha * eye)
    cov_eig = torch.linalg.eigvalsh(gram / (x.shape[1] - 1) + alpha * eye)
    if torch.any(eig <= 0) or torch.any(cov_eig <= 0):
        raise ValueError('Regularized covariance is not positive definite')
    return dict(negative_eigenscore_paper=-float(eig.log().mean()),
                negative_eigenscore_cov_log10=-float(cov_eig.log10().mean()),
                eigen_centered_gram=gram.cpu().tolist(), eigen_width=x.shape[1], eigen_alpha=alpha,
                eigen_representation='last forwarded generated token; HF hidden_states[layer]')


def sampled_baselines(model, tokenizer, ids, mask, generation_config, layer, count, rouge):
    """Sequential accuracy sampling. This function is NOT a batched serving latency benchmark."""
    from copy import deepcopy
    cfg = deepcopy(generation_config)
    cfg.do_sample, cfg.temperature, cfg.top_p, cfg.top_k = True, 0.5, 0.99, 10
    samples, embeddings = [], []
    for _ in range(count):
        result = model.generate(ids, attention_mask=mask, generation_config=cfg,
                                return_dict_in_generate=True, output_hidden_states=True,
                                output_logits=True, output_scores=False, output_attentions=False)
        tokens = result.sequences[0, ids.shape[1]:]
        stats = token_statistics(result.logits, tokens)
        samples.append(dict(generated_ids=tokens.cpu().tolist(),
                            generated_text=tokenizer.decode(tokens, skip_special_tokens=True),
                            hit_generation_cap=len(tokens) == cfg.max_new_tokens,
                            no_output_state=len(result.hidden_states) <= 1, **stats))
        if len(result.hidden_states) > 1:
            embeddings.append(result.hidden_states[-1][layer][0, -1].detach().clone())
        del result
    lexical = [rouge.score(target=a['generated_text'], prediction=b['generated_text'])['rougeL'].fmeasure
               for i, a in enumerate(samples) for b in samples[i+1:]]
    values = dict(multipass_samples=samples, negative_ln_entropy=-sum(s['mnll'] for s in samples)/count,
                  lexical_similarity=sum(lexical)/len(lexical),
                  multipass_config=dict(count=count, temperature=.5, top_p=.99, top_k=10,
                                        probability='unprocessed LM logits', execution='sequential accuracy sampling'))
    if len(embeddings) == count:
        values.update(eigen_statistics(embeddings))
    else:
        values.update(negative_eigenscore_paper=None, negative_eigenscore_cov_log10=None,
                      eigen_undefined_reason='At least one sample has no forwarded generated-token state')
    return values
