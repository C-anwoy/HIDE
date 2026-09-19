"""HIDE score: original token ordering, duplicates, FP32 kernels and formula retained.

Only model construction is injected; the reference implementation is archived for parity testing.
The final generated token has no cached state and is excluded, as in that implementation.
"""

import torch
from sklearn.feature_extraction.text import CountVectorizer
from hide.kernels import KERNEL_FUNCTIONS

def extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens, k=20, kw_model=None):
    if kw_model is None:
        raise ValueError('Pass a resident KeyBERT instance explicitly')

    input_text = tokenizer.decode(input_tokens, skip_special_tokens=True)
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    vectorizer = CountVectorizer(
                    ngram_range=(1,1),
                    stop_words=None,
                    token_pattern=r"(?u)\b\w+\b",
                    min_df=1,
                    vocabulary=None,
                    lowercase=False,
                    max_features=None
                )
    try:
        input_keywords = kw_model.extract_keywords(input_text, keyphrase_ngram_range=(1, 1),
                                             top_n=k, use_mmr=True, diversity=1, vectorizer=vectorizer)
    except Exception as exc:
        if (isinstance(exc, ValueError) and str(exc).startswith('empty vocabulary;')
                and not vectorizer.build_analyzer()(input_text)):
            input_keywords = []  # Reach the existing token fallback for symbol-only text.
        else:
            raise RuntimeError(f'Input keyword extraction failed: {type(exc).__name__}: {exc}') from exc
    try:
        output_keywords = kw_model.extract_keywords(output_text, keyphrase_ngram_range=(1, 1),
                                              top_n=k, use_mmr=True, diversity=1, vectorizer=vectorizer)
    except Exception as exc:
        if (isinstance(exc, ValueError) and str(exc).startswith('empty vocabulary;')
                and not vectorizer.build_analyzer()(output_text)):
            output_keywords = []  # Reach the existing token fallback for symbol-only text.
        else:
            raise RuntimeError(f'Output keyword extraction failed: {type(exc).__name__}: {exc}') from exc

    def find_all_occurrences(keyword, token_sequence, tokenizer):
        variations = [keyword, " " + keyword]

        indices = []
        for variant in variations:
            keyword_tokens = tokenizer.encode(variant, add_special_tokens=False)

            if len(keyword_tokens) <= len(token_sequence):
                for i in range(len(token_sequence) - len(keyword_tokens) + 1):
                    if token_sequence[i:i+len(keyword_tokens)] == keyword_tokens:
                        indices.extend(range(i, i+len(keyword_tokens)))

        return indices

    input_indices = []
    input_tokens = input_tokens.tolist()
    output_tokens = output_tokens.tolist()
    for keyword, _ in input_keywords:
        keyword_indices = find_all_occurrences(keyword, input_tokens, tokenizer)
        if keyword_indices:
            input_indices.extend(keyword_indices)

    output_indices = []
    for keyword, _ in output_keywords:
        keyword_indices = find_all_occurrences(keyword, output_tokens, tokenizer)
        if keyword_indices:
            output_indices.extend(keyword_indices)


    if not input_indices:
        print("No input indices found")
        input_indices = list(range(min(k, len(input_tokens))))
    if not output_indices:
        print("No output indices found")
        output_indices = list(range(min(k, len(output_tokens))))

    k = min(k,len(input_indices),len(output_indices))
    if len(input_indices) > k:
        input_indices = input_indices[:k]
    if len(output_indices) > k:
        output_indices = output_indices[:k]

    selected_input = [input_tokens[i] for i in input_indices]
    selected_output = [output_tokens[i] for i in output_indices]
    input_tokens_topk = [tokenizer.decode([token]) for token in selected_input]
    output_tokens_topk = [tokenizer.decode([token]) for token in selected_output]

    X_keywords = X[input_indices, :]
    Y_keywords = Y[output_indices, :]

    return X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def unbiased_HSIC(K_X, K_Y):
    tK = K_X - torch.diag(torch.diag(K_X))
    tL = K_Y - torch.diag(torch.diag(K_Y))

    N = K_X.shape[0]

    hsic = (
        torch.trace(tK @ tL)
        + (torch.sum(tK) * torch.sum(tL) / N**2 )
        - (2 * torch.sum(tK @ tL) / N)
    )

    return hsic / N**2

def compute_unbiased_hsic_with_keywords(X, Y, tokenizer, input_tokens, output_tokens, kernel_func_X, kernel_func_Y, k=20, **kwargs):
    """
    Compute HSIC using keywords extraction with KeyBERT instead of SVD.

    Args:
        X -> torch tensor: First matrix (input hidden states).
        Y -> torch tensor: Second matrix (output hidden states).
        tokenizer: The tokenizer used by the model.
        input_ids: Raw input ids.
        output_ids: Raw output ids.
        kernel_func_X: Kernel function for X (callable).
        kernel_func_Y: Kernel function for Y (callable).
        k: Number of top keywords to extract.
        **kwargs: Additional arguments for kernel functions.

    Returns:
        hsic: Hilbert-Schmidt Independence Criterion score.
    """
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens, k=k, kw_model=kwargs.pop('kw_model'))

    K_X = kernel_func_X(X_keywords, **kwargs)
    K_Y = kernel_func_Y(Y_keywords, **kwargs)



    hsic = unbiased_HSIC(K_X, K_Y)

    return hsic, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_unbiased_hsic_score_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, kernel = 'rbf', **kwargs):

    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)

    if len(selected_states) == 1:
        return 0, 0, 0, 0, 0
    Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    if len(output_tokens) - 1 != len(Y):
        raise ValueError('Cached output tokens and hidden states are not aligned')
    Y = Y.to(torch.float32)

    kernel_X = KERNEL_FUNCTIONS[kernel]
    kernel_Y = KERNEL_FUNCTIONS[kernel]
    hsic_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_unbiased_hsic_with_keywords(
        X, Y, tokenizer, input_tokens, output_tokens[:-1],
        kernel_X, kernel_Y, k=keywords, **kwargs
    )

    return float(hsic_score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk
