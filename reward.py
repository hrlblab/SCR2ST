# reward.py
import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
import math


# ============ 1) SC manifold coverage (based on cluster coverage, no radius) ============
def reward_sc_manifold(selected_embed: np.ndarray,
                       sc_centroids: np.ndarray,
                       n_clusters: int) -> float:
    """
    Use single-cell cluster coverage to measure how well ST sampling covers the SC manifold.
    No radius used, no distance threshold, only determines which cluster center each sample is closest to.

    Args:
        selected_embed : (k, D) selected ST embedding (PCA space)
        sc_centroids   : (C, D) single-cell cluster centers (from KMeans)
        n_clusters     : total number of clusters (e.g., 1000)

    Returns:
        coverage ratio in [0, 1]
    """
    if selected_embed is None or len(selected_embed) == 0:
        return 0.0

    # Find nearest cluster center for selected ST
    from scipy.spatial.distance import cdist
    d = cdist(selected_embed, sc_centroids)  # (k, C)
    nn_idx = np.argmin(d, axis=1)            # nearest cluster index

    # Count how many clusters are covered
    covered = len(np.unique(nn_idx)) / float(n_clusters)
    return float(covered)


# ========= 2) SC type diversity (GPU + cosine similarity) ========= #
import torch
from collections import Counter
import math
def reward_sc_type(selected_embed: np.ndarray,
                   sc_embed: np.ndarray,
                   sc_type: np.ndarray,
                   device=None,
                   chunk_size: int = 4000) -> float:
    """
    Chunked computation version (significantly reduces GPU memory)
    selected_embed: (k, D)
    sc_embed: (M, D)
    sc_type: (M,)
    """

    if selected_embed is None or len(selected_embed) == 0:
        return 0.0

    # ====== Init device ====== #
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== Convert selected to torch ====== #
    sel = torch.from_numpy(selected_embed).float().to(device)  # (k, D)
    sel = torch.nn.functional.normalize(sel, dim=1)

    M = sc_embed.shape[0]
    k = sel.shape[0]

    # k selected spots' final nearest single-cell index
    nn_sim = torch.full((k,), -1e9, device=device)  # max similarity for each spot
    nn_idx = torch.full((k,), -1, dtype=torch.long, device=device)

    # ====== Chunked computation ====== #
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)

        sc_chunk_np = sc_embed[start:end]               # (chunk, D)
        sc_chunk = torch.from_numpy(sc_chunk_np).float().to(device)
        sc_chunk = torch.nn.functional.normalize(sc_chunk, dim=1)

        # (k, D) @ (D, chunk) = (k, chunk)
        sim = torch.matmul(sel, sc_chunk.T)

        # Find nearest single-cell within chunk
        chunk_max_sim, chunk_idx = torch.max(sim, dim=1)

        # Global comparison: if chunk has higher similarity, update nn_idx
        better = chunk_max_sim > nn_sim
        nn_sim[better] = chunk_max_sim[better]
        nn_idx[better] = start + chunk_idx[better]

        del sc_chunk, sim, chunk_max_sim, chunk_idx
        torch.cuda.empty_cache()

    # ====== Get final nearest type ====== #
    nn_idx_np = nn_idx.cpu().numpy()
    mapped_types = sc_type[nn_idx_np]  # (k,)

    # ====== Compute entropy of type distribution ====== #
    cnt = Counter(mapped_types)
    total = sum(cnt.values())
    if total == 0:
        return 0.0

    probs = [c / total for c in cnt.values()]
    entropy = -sum(p * math.log(p + 1e-9) for p in probs)
    norm_entropy = entropy / (math.log(len(cnt) + 1e-9))

    return float(norm_entropy)


# ============ 3) Spatial uniformity (coverage + uniformity) ============
def reward_spatial(selected_coords: np.ndarray,
                   all_coords: np.ndarray) -> float:
    """
    Spatial reward: encourage sampled points to be both dispersed and cover most of the region.

    Args:
        selected_coords: (k, 2) currently selected ST coordinates
        all_coords:      (N, 2) all ST coordinates

    Returns:
        a scalar, higher value indicates better spatial distribution (dispersed + coverage)
    """
    if selected_coords is None or len(selected_coords) < 2:
        return 0.0

    # Distance among selected points (larger = more dispersed)
    d_within = np.mean(cdist(selected_coords, selected_coords))

    # Distance from all points to nearest selected point (smaller = better coverage)
    d_cover = np.mean(np.min(cdist(all_coords, selected_coords), axis=1))

    # Simple average combination
    return float((d_within + d_cover) / 2.0)
