import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import MinCovDet
from rouge_score import rouge_scorer
from sentence_transformers import util
import heapq
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
from .kernels import KERNEL_FUNCTIONS
from keybert import KeyBERT
import _settings
import os
from sklearn.feature_extraction.text import CountVectorizer
from .dcor import *
from .probe_utils import *
from pathlib import Path

rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_hsic(X, Y, kernel_func_X, kernel_func_Y, svd=False, k=None, **kwargs):
    """
    Compute HSIC using the formal definition with optional truncated SVD.
    Args:
        X -> torch tensor: First matrix (N1 x d). 
        Y -> torch tensor: Second matrix (N2 x d). 
        kernel_func_X: Kernel function for X (callable).
        kernel_func_Y: Kernel function for Y (callable).
        svd: If True, use truncated SVD.
        k: Number of top eigenvectors to keep in truncated SVD.
        **kwargs: Additional arguments for kernel functions.
    Returns:
        hsic: Hilbert-Schmidt Independence Criterion score.
    """
    if svd is True and k is not None:
        
        # print(f"Before SVD: {X.shape}")
        # print(f"Before SVD: {Y.shape}")
        if X.shape[0] < k or Y.shape[0] < k:
            k = min(X.shape[0], Y.shape[0])
        # Apply truncated SVD to X and Y matrices
        U_X, S_X, V_X = torch.svd_lowrank(X, q=k)
        U_Y, S_Y, V_Y = torch.svd_lowrank(Y, q=k)
        #reduced matrices
        X_r = torch.mm(torch.diag(S_X[:k]), V_X[:, :k].T).to(X.device)
        Y_r = torch.mm(torch.diag(S_Y[:k]), V_Y[:, :k].T).to(X.device)
        
        # print(f"After SVD: {X_r.shape}")
        # print(f"After SVD: {Y_r.shape}")
        # print(X_r)
        # print(Y_r)
        # Compute kernel matrices on reduced matrices
        K_X = kernel_func_X(X_r, **kwargs)
        K_Y = kernel_func_Y(Y_r, **kwargs)
    
    elif svd is False:
        # Compute kernel matrices
        K_X = kernel_func_X(X, **kwargs)
        K_Y = kernel_func_Y(Y, **kwargs)
    
    n = K_X.shape[0]
    # print(f"n: {n}")
    
    # Center the kernel matrices
    H = torch.eye(n, device=X.device, dtype=K_X.dtype) - (1 / n) * torch.ones((n, n), device=X.device, dtype=K_X.dtype)
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H

    # Compute HSIC
    hsic = torch.trace(K_X_centered @ K_Y_centered) / (n ** 2)
    return hsic

# hidden_states shape (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
def get_hsic_score_mid(hidden_states):
    selected_layer = int(len(hidden_states[0])/2)
    # selected_layer = -2
    # print(f"generation length {len(hidden_states)}")
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states] # (num_tokens, num_seq, num_input_tokens/1, embedding_size)

    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)
    # print(X.dtype)
    # print(X.shape)
    Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    Y = Y.to(torch.float32)
    # print(Y.dtype)
    # print(Y.shape)
    # sample_result = {}

    # for kernel in KERNEL_FUNCTIONS:
        # Select kernel functions
    kernel_X = KERNEL_FUNCTIONS['rbf']
    kernel_Y = KERNEL_FUNCTIONS['rbf']

    hsic_score_svd = compute_hsic(X, Y, kernel_X, kernel_Y, svd=True, k=20)

    # sample_result[kernel] = float(hsic_score_svd)
    
    return float(hsic_score_svd)

def extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens, k=20):
    # Initialize KeyBERT
    kw_model = KeyBERT(model=os.path.join(_settings.MODEL_PATH, 'all-MiniLM-L6-v2'))
    
    # Decoding text from input ids
    input_text = tokenizer.decode(input_tokens, skip_special_tokens=True)
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    # print("output_text",output_text)
    # print('input text', input_text)
    # print('hello')
    # Extract keywords from input and output text
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
    except:
        input_keywords=[]
    try:
        output_keywords = kw_model.extract_keywords(output_text, keyphrase_ngram_range=(1, 1), 
                                              top_n=k, use_mmr=True, diversity=1, vectorizer=vectorizer)
    except:
        output_keywords=[]
    # print(input_keywords, output_keywords)
    # print("output_text",output_text)
    # Get tokenized input and output
    # input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
    # output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
    
    # Function to find all occurrences of a token sequence in a larger token sequence
    def find_all_occurrences(keyword, token_sequence, tokenizer):
        variations = [keyword, " " + keyword]

        indices = []
        for variant in variations:
            keyword_tokens = tokenizer.encode(variant, add_special_tokens=False)
            # print('hello')
            # print(variant)
            # print([(token, tokenizer.decode([token])) for token in keyword_tokens])
            # print(keyword_tokens)
            # print(token_sequence)
            
            if len(keyword_tokens) <= len(token_sequence):
                for i in range(len(token_sequence) - len(keyword_tokens) + 1):
                    # print(token_sequence[i:i+len(keyword_tokens)])
                    # print(keyword_tokens)
                    # print(token_sequence[i:i+len(keyword_tokens)], keyword_tokens)
                    if token_sequence[i:i+len(keyword_tokens)] == keyword_tokens:
                        # Add indices for all subtokens
                        # print("FOUND")
                        indices.extend(range(i, i+len(keyword_tokens)))
                        # break
                        # print("indices", indices)
        
        return indices
    
    # Get indices for input keywords
    input_indices = []
    # print([(token, tokenizer.decode([token])) for token in output_tokens.tolist()])
    # print([(token, tokenizer.decode([token])) for token in input_tokens.tolist()])
    input_tokens = input_tokens.tolist()
    output_tokens = output_tokens.tolist()  
    for keyword, _ in input_keywords:
        keyword_indices = find_all_occurrences(keyword, input_tokens, tokenizer)
        if keyword_indices:
            input_indices.extend(keyword_indices)
    
    # Get indices for output keywords
    output_indices = []
    for keyword, _ in output_keywords:
        keyword_indices = find_all_occurrences(keyword, output_tokens, tokenizer)
        if keyword_indices:
            output_indices.extend(keyword_indices)
    
    # Remove duplicates and sort
    # input_indices = sort(list(set(input_indices)))
    # output_indices = sort(list(set(output_indices)))
    
    # Ensure we have at least one index
    if not input_indices:
        print("No input indices found")
        input_indices = list(range(min(k, len(input_tokens))))
    if not output_indices:
        print("No output indices found")
        output_indices = list(range(min(k, len(output_tokens))))

    # print(input_indices, output_indices)
    k = min(k,len(input_indices),len(output_indices))
    # Limit to top k if we have more indices
    if len(input_indices) > k:
        input_indices = input_indices[:k]
    if len(output_indices) > k:
        output_indices = output_indices[:k]
    
    selected_input = [input_tokens[i] for i in input_indices]
    selected_output = [output_tokens[i] for i in output_indices]
    input_tokens_topk = [tokenizer.decode([token]) for token in selected_input]
    output_tokens_topk = [tokenizer.decode([token]) for token in selected_output]
    # print(input_tokens_topk, output_tokens_topk)

    # Extract vectors using indices
    X_keywords = X[input_indices, :]
    Y_keywords = Y[output_indices, :]

    return X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def extract_keyword_representation_duplication(X, Y, tokenizer, input_tokens, output_tokens, k=20):
    # Initialize KeyBERT
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    
    # Decoding text from input ids
    input_text = tokenizer.decode(input_tokens, skip_special_tokens=True)
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    # print("output_text",output_text)
    # print('input text', input_text)
    # print('hello')
    # Extract keywords from input and output text
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
    except:
        input_keywords=[]
    try:
        output_keywords = kw_model.extract_keywords(output_text, keyphrase_ngram_range=(1, 1), 
                                              top_n=k, use_mmr=True, diversity=1, vectorizer=vectorizer)
    except:
        output_keywords=[]
    # print(input_keywords, output_keywords)
    # print("output_text",output_text)
    # Get tokenized input and output
    # input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
    # output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
    
    # Function to find all occurrences of a token sequence in a larger token sequence
    def find_all_occurrences(keyword, token_sequence, tokenizer):
        variations = [keyword, " " + keyword]

        indices = []
        for variant in variations:
            keyword_tokens = tokenizer.encode(variant, add_special_tokens=False)
            # print('hello')
            # print(variant)
            # print([(token, tokenizer.decode([token])) for token in keyword_tokens])
            # print(keyword_tokens)
            # print(token_sequence)
            
            if len(keyword_tokens) <= len(token_sequence):
                for i in range(len(token_sequence) - len(keyword_tokens) + 1):
                    # print(token_sequence[i:i+len(keyword_tokens)])
                    # print(keyword_tokens)
                    # print(token_sequence[i:i+len(keyword_tokens)], keyword_tokens)
                    if token_sequence[i:i+len(keyword_tokens)] == keyword_tokens:
                        # Add indices for all subtokens
                        # print("FOUND")
                        indices.extend(range(i, i+len(keyword_tokens)))
                        # break
                        # print("indices", indices)
        
        return indices
    
    # Get indices for input keywords
    input_indices = []
    # print([(token, tokenizer.decode([token])) for token in output_tokens.tolist()])
    # print([(token, tokenizer.decode([token])) for token in input_tokens.tolist()])
    input_tokens = input_tokens.tolist()
    output_tokens = output_tokens.tolist()  
    for keyword, _ in input_keywords:
        keyword_indices = find_all_occurrences(keyword, input_tokens, tokenizer)
        if keyword_indices:
            input_indices.extend(keyword_indices)
    
    # Get indices for output keywords
    output_indices = []
    for keyword, _ in output_keywords:
        keyword_indices = find_all_occurrences(keyword, output_tokens, tokenizer)
        if keyword_indices:
            output_indices.extend(keyword_indices)
    
    # Remove duplicates and sort
    # input_indices = sort(list(set(input_indices)))
    # output_indices = sort(list(set(output_indices)))
    
    # Ensure we have at least one index
    if not input_indices:
        print("No input indices found")
        input_indices = list(range(min(k, len(input_tokens))))
    if not output_indices:
        print("No output indices found")
        output_indices = list(range(min(k, len(output_tokens))))

    # Only limit k based on input length
    k = min(k, len(input_indices))

    # Limit input indices to k
    input_indices = input_indices[:k]

    if len(output_indices) < k:
        # Duplicate output indices to match k by cycling through available indices
        output_indices = [output_indices[i % len(output_indices)] for i in range(k)]
    else:
        # If we have enough or more output indices, just take the first k
        output_indices = output_indices[:k]
    
    selected_input = [input_tokens[i] for i in input_indices]
    selected_output = [output_tokens[i] for i in output_indices]
    input_tokens_topk = [tokenizer.decode([token]) for token in selected_input]
    output_tokens_topk = [tokenizer.decode([token]) for token in selected_output]
    # print(input_tokens_topk, output_tokens_topk)

    # Extract vectors using indices
    X_keywords = X[input_indices, :]
    Y_keywords = Y[output_indices, :]

    return X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

##using KEYBERT for keywords
def compute_hsic_with_keywords(X, Y, tokenizer, input_tokens, output_tokens, kernel_func_X, kernel_func_Y, k=20, **kwargs):
    
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens, k=k)
    
    # Compute kernel matrices on keyword matrices
    K_X = kernel_func_X(X_keywords, **kwargs)
    K_Y = kernel_func_Y(Y_keywords, **kwargs)
    
    n = K_X.shape[0]
    # print(f"n: {n}")
    
    # Center the kernel matrices
    H = torch.eye(n, device=X.device, dtype=K_X.dtype) - (1 / n) * torch.ones((n, n), device=X.device, dtype=K_X.dtype)
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H

    # Compute HSIC
    hsic = torch.trace(K_X_centered @ K_Y_centered) / (n ** 2)
    return hsic, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_hsic_score_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, kernel='rbf', **kwargs):
    
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)
    # print(f"X.shape {X.shape}")

    Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    Y = Y.to(torch.float32)
    # print(f"Y.shape {Y.shape}")
    # print(f"length of input tokens : {len(input_tokens)}")
    # print(f"length of output tokens : {len(output_tokens)}")
    
    kernel_X = KERNEL_FUNCTIONS[kernel]
    kernel_Y = KERNEL_FUNCTIONS[kernel]
    # print(tokenizer.decode(output_tokens, skip_special_tokens=False))
    hsic_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_hsic_with_keywords(
        X, Y, tokenizer, input_tokens, output_tokens[:-1], 
        kernel_X, kernel_Y, k=keywords, **kwargs
    )
    
    return float(hsic_score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

##using KEYBERT for keywords
def compute_cka_with_keywords(X, Y, tokenizer, input_tokens, output_tokens, kernel_func_X, kernel_func_Y, k=20, **kwargs):
    
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens, k=k)
    
    # Compute kernel matrices on keyword matrices
    K_X = kernel_func_X(X_keywords, **kwargs)
    K_Y = kernel_func_Y(Y_keywords, **kwargs)
    
    n = K_X.shape[0]
    # print(f"n: {n}")
    
    # Center the kernel matrices
    H = torch.eye(n, device=X.device, dtype=K_X.dtype) - (1 / n) * torch.ones((n, n), device=X.device, dtype=K_X.dtype)
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H

    # Convert to higher precision immediately before any calculations
    K_X_centered = K_X_centered.to(torch.float64)
    K_Y_centered = K_Y_centered.to(torch.float64)

    # Compute HSIC``
    hsic_x_y = torch.trace(K_X_centered @ K_Y_centered) 
    hsic_x_x = torch.trace(K_X_centered @ K_X_centered) 
    hsic_y_y = torch.trace(K_Y_centered @ K_Y_centered) 
    # print(hsic_x_y, hsic_x_x, hsic_y_y)
    cka_score = hsic_x_y/torch.sqrt(hsic_x_x * hsic_y_y)

    return cka_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_cka_score_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, kernel='rbf', **kwargs):
    
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)
    # print(f"X.shape {X.shape}")

    Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    Y = Y.to(torch.float32)
    # print(f"Y.shape {Y.shape}")
    # print(f"length of input tokens : {len(input_tokens)}")
    # print(f"length of output tokens : {len(output_tokens)}")
    
    kernel_X = KERNEL_FUNCTIONS[kernel]
    kernel_Y = KERNEL_FUNCTIONS[kernel]
    # print(tokenizer.decode(output_tokens, skip_special_tokens=False))
    cka_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_cka_with_keywords(
        X, Y, tokenizer, input_tokens, output_tokens[:-1], 
        kernel_X, kernel_Y, k=keywords, **kwargs
    )
    
    return float(cka_score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

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

##using KEYBERT for keywords
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
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens, k=k)
    
    # Compute kernel matrices on keyword matrices
    K_X = kernel_func_X(X_keywords, **kwargs)
    K_Y = kernel_func_Y(Y_keywords, **kwargs)
    
    # n = K_X.shape[0]
    # # print(f"n: {n}")
    
    # # Center the kernel matrices
    # H = torch.eye(n, device=X.device, dtype=K_X.dtype) - (1 / n) * torch.ones((n, n), device=X.device, dtype=K_X.dtype)
    # K_X_centered = H @ K_X @ H
    # K_Y_centered = H @ K_Y @ H

    # # Compute HSIC
    # hsic = torch.trace(K_X_centered @ K_Y_centered) / (n ** 2)
    hsic = unbiased_HSIC(K_X, K_Y)
    
    return hsic, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def compute_unbiased_hsic_with_keywords_duplication(X, Y, tokenizer, input_tokens, output_tokens, kernel_func_X, kernel_func_Y, k=20, **kwargs):
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
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation_duplication(X, Y, tokenizer, input_tokens, output_tokens, k=k)
    
    # Compute kernel matrices on keyword matrices
    K_X = kernel_func_X(X_keywords, **kwargs)
    K_Y = kernel_func_Y(Y_keywords, **kwargs)
    
    # n = K_X.shape[0]
    # # print(f"n: {n}")
    
    # # Center the kernel matrices
    # H = torch.eye(n, device=X.device, dtype=K_X.dtype) - (1 / n) * torch.ones((n, n), device=X.device, dtype=K_X.dtype)
    # K_X_centered = H @ K_X @ H
    # K_Y_centered = H @ K_Y @ H

    # # Compute HSIC
    # hsic = torch.trace(K_X_centered @ K_Y_centered) / (n ** 2)
    hsic = unbiased_HSIC(K_X, K_Y)
    
    return hsic, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_unbiased_hsic_score_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, kernel = 'rbf', **kwargs):
    
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)
    # print(f"X.shape {X.shape}")

    try:
        Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    except:
        return 0, 0, 0, 0, 0
    Y = Y.to(torch.float32)
    # print(f"Y.shape {Y.shape}")
    # print(f"length of input tokens : {len(input_tokens)}")
    # print(f"length of output tokens : {len(output_tokens)}")
    
    kernel_X = KERNEL_FUNCTIONS[kernel]
    kernel_Y = KERNEL_FUNCTIONS[kernel]
    # print(tokenizer.decode(output_tokens, skip_special_tokens=False))
    hsic_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_unbiased_hsic_with_keywords(
        X, Y, tokenizer, input_tokens, output_tokens[:-1], 
        kernel_X, kernel_Y, k=keywords, **kwargs
    )
    
    return float(hsic_score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def l1_norm_distance(A, B):
    """
    Calculates the L1 norm (Manhattan distance) between two matrices A and B.
    """
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape.")
    return np.sum(np.abs(A - B))

def l2_norm_distance(A, B):
    """
    Calculates the L2 norm (Frobenius norm) distance between two matrices A and B.
    """
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape.")
    # np.linalg.norm with 'fro' (Frobenius) is the standard way
    return np.linalg.norm(A - B, 'fro')

def cosine_similarity(A, B):
    """
    Calculates the cosine similarity between two matrices A and B.
    The matrices are flattened into vectors before comparison.
    """
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape.")
    print(A.shape, B.shape)
    # Flatten the matrices into 1D vectors
    A_flat = A.ravel()
    B_flat = B.ravel()
    print(A_flat.shape, B_flat.shape)
    # Calculate cosine similarity
    dot_product = np.dot(A_flat, B_flat)
    norm_A = np.linalg.norm(A_flat)
    norm_B = np.linalg.norm(B_flat)
    
    # Avoid division by zero if one of the vectors is all zeros
    if norm_A == 0 or norm_B == 0:
        return 0.0
    
    return dot_product / (norm_A * norm_B)

def get_cosine_sim_score_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, **kwargs):
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]
    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)

    try:
        Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    except:
        return 0, 0, 0, 0, 0
    Y = Y.to(torch.float32)
    
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens[:-1], k=keywords)
    
    sim_score = cosine_similarity(X_keywords.cpu().numpy(), Y_keywords.cpu().numpy())
    print(f"sim_score : {sim_score}")
    return float(sim_score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_l1norm_sim_score_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, **kwargs):
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]
    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)

    try:
        Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    except:
        return 0, 0, 0, 0, 0
    Y = Y.to(torch.float32)
    
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens[:-1], k=keywords)
    
    sim_score = l1_norm_distance(X_keywords.cpu().numpy(), Y_keywords.cpu().numpy())
    print(f"sim_score : {sim_score}")
    return float(sim_score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk


def get_l2norm_sim_score_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, **kwargs):
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]
    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)

    try:
        Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    except:
        return 0, 0, 0, 0, 0
    Y = Y.to(torch.float32)
    
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens[:-1], k=keywords)
    
    sim_score = l2_norm_distance(X_keywords.cpu().numpy(), Y_keywords.cpu().numpy())
    print(f"sim_score : {sim_score}")
    return float(sim_score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk


def get_max_token_probe_score(hidden_states, model, device):
    probe_id = model_probe_config[model]
    probe_dir = Path(os.path.join(_settings.PROBE_FOLDER, probe_id))
    print(f"Loading probe from {probe_dir}")
    probe_head, probe_layer_idx = load_probe_head(probe_dir, device=device)
    print(f"Using probe at layer {probe_layer_idx}")
    selected_states = [token_tuple[probe_layer_idx] for token_tuple in hidden_states]
    
    try:
        Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    except:
        print(f"Error in concatenating hidden states for probe scoring.")
        return 0
    Y = Y.to(torch.float32)
    print(f"Y.shape {Y.shape}")

    probe_head = probe_head.to(torch.float32)
    probe_score = probe_head(Y) 
    # Convert raw score (logit) to probability (0 to 1)
    probe_probability = torch.sigmoid(probe_score)
    print(f"Probe probabilities shape: {probe_probability.shape}")
    final_score = torch.max(probe_probability)
    print(f"Max probe score: {final_score}")

    return float(final_score.item())


def get_unbiased_hsic_score_keybert_duplication(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, kernel='rbf', **kwargs):
    
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)
    # print(f"X.shape {X.shape}")

    try:
        Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    except:
        return 0, 0, 0, 0, 0
    Y = Y.to(torch.float32)
    # print(f"Y.shape {Y.shape}")
    # print(f"length of input tokens : {len(input_tokens)}")
    # print(f"length of output tokens : {len(output_tokens)}")
    
    kernel_X = KERNEL_FUNCTIONS[kernel]
    kernel_Y = KERNEL_FUNCTIONS[kernel]
    # print(tokenizer.decode(output_tokens, skip_special_tokens=False))
    hsic_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_unbiased_hsic_with_keywords_duplication(
        X, Y, tokenizer, input_tokens, output_tokens[:-1], 
        kernel_X, kernel_Y, k=keywords, **kwargs
    )
    
    return float(hsic_score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_unbiased_hsic_score_keybert_layerablation(hidden_states, tokenizer, input_tokens, output_tokens, keywords):
    
    hsic_scores = []
    for selected_layer in range(len(hidden_states[0])):
        # selected_layer = int(len(hidden_states[0])/2)
        selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

        X = selected_states[0][0,:,:]
        X = X.to(torch.float32)
        # print(f"X.shape {X.shape}")

        try:
            Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
        except:
            return 0, 0, 0, 0, 0
        Y = Y.to(torch.float32)
        # print(f"Y.shape {Y.shape}")
        # print(f"length of input tokens : {len(input_tokens)}")
        # print(f"length of output tokens : {len(output_tokens)}")
        
        kernel_X = KERNEL_FUNCTIONS['rbf']
        kernel_Y = KERNEL_FUNCTIONS['rbf']
        # print(tokenizer.decode(output_tokens, skip_special_tokens=False))
        hsic_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_unbiased_hsic_with_keywords(
            X, Y, tokenizer, input_tokens, output_tokens[:-1], 
            kernel_X, kernel_Y, k=keywords
        )
        hsic_scores.append(float(hsic_score))
        
    return hsic_scores, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_unbiased_hsic_score_keybert_keywordsablation(hidden_states, tokenizer, input_tokens, output_tokens):
    
    hsic_scores = []

    for k in range(5,101,5):
        selected_layer = int(len(hidden_states[0])/2)
        selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

        X = selected_states[0][0,:,:]
        X = X.to(torch.float32)
        # print(f"X.shape {X.shape}")

        try:
            Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
        except:
            return 0, 0, 0, 0, 0
        Y = Y.to(torch.float32)
        # print(f"Y.shape {Y.shape}")
        # print(f"length of input tokens : {len(input_tokens)}")
        # print(f"length of output tokens : {len(output_tokens)}")
        
        kernel_X = KERNEL_FUNCTIONS['rbf']
        kernel_Y = KERNEL_FUNCTIONS['rbf']
        # print(tokenizer.decode(output_tokens, skip_special_tokens=False))
        hsic_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_unbiased_hsic_with_keywords(
            X, Y, tokenizer, input_tokens, output_tokens[:-1], 
            kernel_X, kernel_Y, k=k
        )
        hsic_scores.append(float(hsic_score))
        
    return hsic_scores, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

##using KEYBERT for keywords
def compute_unbiased_cka_with_keywords(X, Y, tokenizer, input_tokens, output_tokens, kernel_func_X, kernel_func_Y, k=20, **kwargs):
    
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens, k=k)
    
    # Compute kernel matrices on keyword matrices
    K_X = kernel_func_X(X_keywords, **kwargs)
    K_Y = kernel_func_Y(Y_keywords, **kwargs)
    
    # n = K_X.shape[0]
    # # print(f"n: {n}")
    
    # # Center the kernel matrices
    # H = torch.eye(n, device=X.device, dtype=K_X.dtype) - (1 / n) * torch.ones((n, n), device=X.device, dtype=K_X.dtype)
    # K_X_centered = H @ K_X @ H
    # K_Y_centered = H @ K_Y @ H

    # # Convert to higher precision immediately before any calculations
    # K_X_centered = K_X_centered.to(torch.float64)
    # K_Y_centered = K_Y_centered.to(torch.float64)

    # Compute HSIC``
    hsic_x_y = unbiased_HSIC(K_X, K_Y)
    hsic_x_x = unbiased_HSIC(K_X, K_X)
    hsic_y_y = unbiased_HSIC(K_Y, K_Y)
    # print(hsic_x_y, hsic_x_x, hsic_y_y)
    cka_score = hsic_x_y/torch.sqrt(hsic_x_x * hsic_y_y)

    return cka_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_unbiased_cka_score_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, kernel='rbf', **kwargs):
    
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)
    # print(f"X.shape {X.shape}")

    Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    Y = Y.to(torch.float32)
    # print(f"Y.shape {Y.shape}")
    # print(f"length of input tokens : {len(input_tokens)}")
    # print(f"length of output tokens : {len(output_tokens)}")
    
    kernel_X = KERNEL_FUNCTIONS[kernel]
    kernel_Y = KERNEL_FUNCTIONS[kernel]
    # print(tokenizer.decode(output_tokens, skip_special_tokens=False))
    cka_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_unbiased_cka_with_keywords(
        X, Y, tokenizer, input_tokens, output_tokens[:-1], 
        kernel_X, kernel_Y, k=keywords, **kwargs
    )
    
    return float(cka_score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

##using KEYBERT for keywords
def compute_dcor_with_keywords(X, Y, tokenizer, input_tokens, output_tokens, kernel_func_X, kernel_func_Y, k=20, **kwargs):
    
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens, k=k)
    
    dcor_score = compute_dcor(X_keywords, Y_keywords)

    return dcor_score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_dcor_score_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, kernel='rbf', **kwargs):
    
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)
    # print(f"X.shape {X.shape}")

    Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    Y = Y.to(torch.float32)
    # print(f"Y.shape {Y.shape}")
    # print(f"length of input tokens : {len(input_tokens)}")
    # print(f"length of output tokens : {len(output_tokens)}")
    
    kernel_X = KERNEL_FUNCTIONS[kernel]
    kernel_Y = KERNEL_FUNCTIONS[kernel]
    # print(tokenizer.decode(output_tokens, skip_special_tokens=False))
    score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_dcor_with_keywords(
        X, Y, tokenizer, input_tokens, output_tokens[:-1], 
        kernel_X, kernel_Y, k=keywords, **kwargs
    )
    
    return float(score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

##using KEYBERT for keywords
def compute_unbiased_dcor_with_keywords(X, Y, tokenizer, input_tokens, output_tokens, kernel_func_X, kernel_func_Y, k=20, **kwargs):
    
    # Extract keyword representations
    X_keywords, Y_keywords, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = extract_keyword_representation(X, Y, tokenizer, input_tokens, output_tokens, k=k)
    
    score = compute_udcorr(X_keywords, Y_keywords)

    return score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def get_unbiased_dcor_score_mid_keybert(hidden_states, tokenizer, input_tokens, output_tokens, keywords, layer, kernel='rbf', **kwargs):
    
    selected_layer = layer
    selected_states = [token_tuple[selected_layer] for token_tuple in hidden_states]

    X = selected_states[0][0,:,:]
    X = X.to(torch.float32)
    # print(f"X.shape {X.shape}")

    Y = torch.cat(selected_states[1:], dim=0)[:,0,:]
    Y = Y.to(torch.float32)
    # print(f"Y.shape {Y.shape}")
    # print(f"length of input tokens : {len(input_tokens)}")
    # print(f"length of output tokens : {len(output_tokens)}")
    
    kernel_X = KERNEL_FUNCTIONS[kernel]
    kernel_Y = KERNEL_FUNCTIONS[kernel]
    # print(tokenizer.decode(output_tokens, skip_special_tokens=False))
    score, input_keywords, output_keywords, input_tokens_topk, output_tokens_topk = compute_unbiased_dcor_with_keywords(
        X, Y, tokenizer, input_tokens, output_tokens[:-1], 
        kernel_X, kernel_Y, k=keywords, **kwargs
    )
    
    return float(score), input_keywords, output_keywords, input_tokens_topk, output_tokens_topk

def getRouge(rouge, generations, answers):
    # results = rouge.compute(predictions=[generations], references=[answers], use_aggregator=False)
    results = rouge.score(target = answers, prediction = generations)
    RoughL = results["rougeL"].fmeasure  #fmeasure/recall/precision
    return RoughL


def getSentenceSimilarity(generations, answers, SenSimModel):
    gen_embeddings = SenSimModel.encode(generations)
    ans_embeddings = SenSimModel.encode(answers)
    similarity = util.cos_sim(gen_embeddings, ans_embeddings)
    return similarity.item()

def get_perplexity_score(scores):
    perplexity = 0.0
    for logits in scores:
        conf = torch.max(logits.softmax(1)).cpu().item()
        perplexity += np.log(conf)
    perplexity = -1.0 * perplexity/len(scores)
    return perplexity

def get_energy_score(scores):
    avg_energy = 0.0
    for logits in scores:
        energy = - torch.logsumexp(logits[0], dim=0, keepdim=False).item()
        avg_energy += energy
    avg_energy = avg_energy/len(scores)
    return avg_energy

def get_entropy_score(batch_scores, num_tokens):  
    Conf = []
    for logits in batch_scores: 
        conf, index = torch.max(logits.softmax(1), dim=1)
        Conf.append(conf.cpu().numpy())
    Conf = np.array(Conf) 
    Conf = Conf + 1e-6
    entropy = -1.0 * np.sum(np.log(Conf))/logits.shape[0]
    return entropy

def get_lenghthNormalized_entropy(batch_scores, num_tokens):  
    seq_entropy = np.zeros(len(num_tokens))
    for ind1, logits in enumerate(batch_scores): 
        for ind2, seq_logits in enumerate(logits):
            if ind1 < num_tokens[ind2]:
                conf, _ = torch.max(seq_logits.softmax(0), dim=0)
                seq_entropy[ind2] = seq_entropy[ind2] + np.log(conf.cpu().numpy())
    normalized_entropy = 0
    for ind, entropy in enumerate(seq_entropy):
        normalized_entropy += entropy/num_tokens[ind]
    normalized_entropy = -1.0* normalized_entropy/len(num_tokens)
    return normalized_entropy

def getLexicalSim(generated_texts):
    LexicalSim = 0
    for i in range(len(generated_texts)):
        for j in range(len(generated_texts)):
            if j<=i:
                continue
            LexicalSim += getRouge(rougeEvaluator, generated_texts[i], generated_texts[j])
    LexicalSim = LexicalSim/(len(generated_texts)*(len(generated_texts)-1)/2)
    return LexicalSim


def get_sent_scores_bertscore(best_generation, batch_generations):
    selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
    sent_scores_bertscore = selfcheck_bertscore.predict(
    sentences = best_generation, sampled_passages = batch_generations)
    return sent_scores_bertscore


def getAvgBertScore(bertscore, best_generated_text, generated_texts):
    sent_scores_bertscore = 0
    for item in generated_texts:
        sent_scores_bertscore += 0 
    sent_scores_bertscore = 1 - sent_scores_bertscore/len(generated_texts)
    return sent_scores_bertscore#.cpu().item()

def getEigenIndicatorOutput(generated_texts, SenSimModel):
    alpha = 1e-3
    _embeddings = []
    for ind in range(len(generated_texts)):
        embeddings = SenSimModel.encode(generated_texts[ind])
        _embeddings.append(embeddings)
    _embeddings = np.array(_embeddings)
    CovMatrix = np.cov(_embeddings)
    CovMatrix = CovMatrix + alpha*np.eye(CovMatrix.shape[0])
    u, s, vT = np.linalg.svd(CovMatrix)
    eigenIndicatorOutput = np.mean(np.log10(s))
    return eigenIndicatorOutput, s

def getEigenScoreOutput(generated_texts, SenSimModel):
    alpha = 1e-3
    _embeddings = []
    for ind in range(len(generated_texts)):
        embeddings = SenSimModel.encode(generated_texts[ind])
        _embeddings.append(embeddings)
    _embeddings = np.array(_embeddings)
    CovMatrix = np.cov(_embeddings)
    CovMatrix = CovMatrix + alpha*np.eye(CovMatrix.shape[0])
    u, s, vT = np.linalg.svd(CovMatrix)
    eigenIndicatorOutput = np.mean(np.log10(s))
    return eigenIndicatorOutput, s

def getEigenIndicator(hidden_states): #[num_tokens, 41, num_seq, [n/1], 5120]
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[0][-1].shape[2]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = hidden_state[selected_layer][:,0,:]
        last_embeddings += _last_embeddings
    last_embeddings/=(len(hidden_states)-1)
    last_embeddings = torch.squeeze(last_embeddings)
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+1.0*np.eye(CovMatrix.shape[0]))
    # eigenIndicator = np.log10(np.prod(s))
    eigenIndicator = np.log10(np.linalg.det(CovMatrix+alpha*np.eye(CovMatrix.shape[0])))
    return eigenIndicator, s

def getEigenIndicator_v0(hidden_states, num_tokens): 
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    # selected_layer = -1
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        last_embeddings[ind,:] = hidden_states[num_tokens[ind]-2][selected_layer][ind,0,:]
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float64)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s

def getEigenIndicator_v1(hidden_states, num_tokens): 
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        for ind1 in range(len(hidden_states)-1):
            if ind1 > num_tokens[ind]-1:
                continue
            last_embeddings[ind,:] += hidden_states[ind1+1][selected_layer][ind,0,:]
        last_embeddings[ind,:] = last_embeddings[ind,:]/(num_tokens[ind]-1)
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s


def getEigenScore(hidden_states, num_tokens): 
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        for ind1 in range(len(hidden_states)-1):
            if ind1 > num_tokens[ind]-1:
                continue
            last_embeddings[ind,:] += hidden_states[ind1+1][selected_layer][ind,0,:]
        last_embeddings[ind,:] = last_embeddings[ind,:]/(num_tokens[ind]-1)
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s

def getEigenIndicator_v2(hidden_states, num_tokens):
    alpha = 1e-3
    LayerEigens = []
    if len(hidden_states)<2:
        return 0, "None"
    for layer_ind in range(len(hidden_states[0])):
        last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
        for seq_ind in range(hidden_states[1][-1].shape[0]):
            for token_ind in range(len(hidden_states)-1):
                if token_ind > num_tokens[seq_ind]-1:
                    continue
                last_embeddings[seq_ind,:] += hidden_states[token_ind+1][layer_ind][seq_ind,0,:]
            last_embeddings[seq_ind,:] = last_embeddings[seq_ind,:]/(num_tokens[seq_ind]-1)
        CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
        u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
        eigenIndicator = np.mean(np.log10(s))
        LayerEigens.append(eigenIndicator)
    LayerEigens = np.array(LayerEigens)
    print("LayerEigens: ", LayerEigens)
    return np.mean(LayerEigens[20:-2]), s

def getEigenIndicator_v3(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
    alpha = 1e-3
    layer_ind_min = 10
    layer_ind_max = 35
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = torch.zeros(last_embeddings.shape).to("cuda")
        for k in range(len(hidden_state)):
            if k < layer_ind_min or k > layer_ind_max:
                continue
            _last_embeddings += hidden_state[k][:,0,:]
        last_embeddings += _last_embeddings/(layer_ind_max-layer_ind_min)
    last_embeddings/=(len(hidden_states)-1)
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s

def getEigenIndicator_v4(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
    alpha = 1e-3
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = hidden_state[-2][:,0,:]
        last_embeddings += _last_embeddings
    last_embeddings/=(len(hidden_states)-1)
    last_embeddings = torch.squeeze(last_embeddings)
    last_embeddings = last_embeddings[:,::40]
    CovMatrix = torch.cov(last_embeddings.transpose(0,1)).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s

def getEigenIndicator_v5(hidden_states, features): #[num_tokens, 41, num_seq, ?, 5120]
    alpha = 1e-3
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = hidden_state[-2][:,0,:]
        last_embeddings += _last_embeddings
    last_embeddings/=(len(hidden_states)-1)
    features = features[::40].cpu().numpy()
    last_embeddings = last_embeddings[:,::40].cpu().numpy()
    # last_embeddings = sample_selected(last_embeddings, features)
    Mean = np.mean(last_embeddings, axis=0)
    CovMatrix = np.cov(last_embeddings.transpose())
    print(CovMatrix)
    CovMatrix = CovMatrix + alpha*np.eye(CovMatrix.shape[0])
    pro = np.matmul(np.matmul((features-Mean).reshape(1,-1), np.linalg.inv(CovMatrix)), (features-Mean).reshape(-1,1))
    # pro = np.exp(-0.5*np.matmul(np.matmul((features-Mean).reshape(1,-1), np.linalg.inv(CovMatrix)), (features-Mean).reshape(-1,1)))
    # pro = pro[0][0]/np.sqrt(np.linalg.det(CovMatrix))
    u, s, vT = np.linalg.svd(CovMatrix)
    pro = -pro[0][0] - np.sum(np.log(s))
    return pro, "None"

def get_features(hidden_states):
    last_embeddings = torch.zeros(hidden_states[0][-1].shape[-1]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = hidden_state[-2][0,0,:]
        last_embeddings += _last_embeddings
    last_embeddings/=(len(hidden_states)-1)
    return last_embeddings


def sample_selected(last_embeddings, features):
    dist = []
    for k in range(last_embeddings.shape[0]):
        dist.append(np.linalg.norm(last_embeddings[k,:]-features)+1e-12*np.random.random())
    temp_dist = heapq.nsmallest(int(0.5*last_embeddings.shape[0]), dist)
    index = [dist.index(i) for i in temp_dist]
    last_embeddings = last_embeddings[index,:]
    print(index)
    print(last_embeddings.shape)
    return last_embeddings



def ParameterClip(model):
    ratio_high = 0.1
    ratio_low = 0.3
    lm_head_weight = np.load("./data/features/lm_head_weight.npy")
    weight_importance = np.load("./data/features/weight_importance.npy")
    k_high = int(ratio_high/100*weight_importance.shape[1])
    k_low = int(ratio_low/100*weight_importance.shape[1])
    for i in range(weight_importance.shape[0]):
        # value_max, ind_max = torch.topk(torch.tensor(weight_importance[i,:]), k_high)
        value_min, ind_min = torch.topk(torch.tensor(weight_importance[i,:]), k_low, largest=False)
        # weight_importance[i,:][weight_importance[i,:]>= value_max.numpy()[-1]] = 0
        weight_importance[i,:][weight_importance[i,:] <= value_min.numpy()[-1]] = -1000
        weight_importance[i,:][weight_importance[i,:] > value_min.numpy()[-1]] = 1
        weight_importance[i,:][weight_importance[i,:] ==-1000] = 0
    print(weight_importance)
    head_weights_op = weight_importance*lm_head_weight
    model.state_dict()["lm_head.weight"].copy_(torch.tensor(head_weights_op))
    return model


def ParameterClip_v1(model):
    # for name, param in model.named_parameters():
        # if name == "lm_head.weight":
        #     np.save("./data/features/lm_head_weight.npy", param.cpu().numpy())
    ratio_high = 0.0001
    ratio_low = 0.001
    lm_head_weight = np.load("./data/features/lm_head_weight.npy")
    weight_importance = np.load("./data/features/weight_importance1.npy")
    # k_high = int(ratio_high/100*weight_importance.shape[0]*weight_importance.shape[1])
    k_low = int(ratio_low/100*weight_importance.shape[0]*weight_importance.shape[1])
    # value_max, ind_max = torch.topk(torch.tensor(weight_importance.flatten()), k_high)
    value_min, ind_min = torch.topk(torch.tensor(weight_importance.flatten()), k_low, largest=False)
    # weight_importance[weight_importance >= value_max.numpy()[-1]] = 1000
    # weight_importance[weight_importance < value_max.numpy()[-1]] = 1
    # weight_importance[weight_importance == 1000] = 0
    weight_importance[weight_importance <= value_min.numpy()[-1]] = -1000
    weight_importance[weight_importance > value_min.numpy()[-1]] = 1
    weight_importance[weight_importance ==-1000] = 0
    print(weight_importance)
    head_weights_op = weight_importance*lm_head_weight
    model.state_dict()["lm_head.weight"].copy_(torch.tensor(head_weights_op))
    return model



def ParameterClip_v2(model):
    ratio_high = 0.1
    ratio_low = 1
    lm_head_weight = np.load("./data/features/lm_head_weight.npy")
    weight_importance = np.load("./data/features/weight_importance.npy")
    weight_importance = np.linalg.norm(weight_importance, axis=0)
    k_high = int(ratio_high/100*4096)
    k_low = int(ratio_low/100*4096)
    value_max, ind_max = torch.topk(torch.tensor(weight_importance), k_high)
    value_min, ind_min = torch.topk(torch.tensor(weight_importance), k_low, largest=False)
    print(ind_max)
    lm_head_weight[:, ind_min] = 0
    model.state_dict()["lm_head.weight"].copy_(torch.tensor(lm_head_weight))
    return model

    







