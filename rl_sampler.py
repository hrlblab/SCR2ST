# rl_sampler.py
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

from reward import reward_sc_manifold, reward_sc_type, reward_spatial


class RLSampler(nn.Module):
    """
    REINFORCE-based RL sampler for Active ST,
    enhanced by external single-cell priors with clustering.

    - Policy network runs on GPU if available
    - PCA + KMeans + rewards remain on CPU (sklearn / numpy optimized)
    """

    def __init__(self,
                 embed_dim: int,
                 sc_embed: np.ndarray,
                 sc_type: np.ndarray,
                 total_ratio: float = 0.30,
                 per_round_ratio: float = 0.10,
                 lr: float = 1e-4,
                 w_sc: float = 10,
                 w_type: float = 10,
                 w_spatial: float = 0.1,
                 n_clusters: int = 1000,
                 device=None):

        super().__init__()

        # ===================== Device ===================== #
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[Sampler] Using device: {self.device}")

        # ===================== Policy network (GPU) ===================== #
        self.policy = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # ===================== RL controls ===================== #
        self.total_ratio = total_ratio
        self.per_round_ratio = per_round_ratio
        self.total_budget = None
        self.per_round_k = None

        # ===== New: internal exclusion mask =====
        self.excluded_mask = None

        # ===================== Single-cell priors ===================== #
        self.sc_type = np.asarray(sc_type)
        self.sc_embed_full = np.asarray(sc_embed)
        self.w_sc = w_sc
        self.w_type = w_type
        self.w_spatial = w_spatial

        # ===================== PCA + KMeans on SC (CPU) ===================== #
        print("[Sampler] Running PCA on SC embedding...")
        self.pca = PCA(n_components=50)
        t0 = time.time()
        self.sc_embed_pca = self.pca.fit_transform(sc_embed)
        print(f"[Sampler] PCA(SC) done in {time.time()-t0:.2f}s")

        print(f"[Sampler] Running MiniBatchKMeans ({n_clusters} clusters)...")
        t1 = time.time()
        kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                 batch_size=2048,
                                 max_iter=50).fit(self.sc_embed_pca)
        print(f"[Sampler] KMeans done in {time.time()-t1:.2f}s")

        self.sc_centroids = kmeans.cluster_centers_

        # ====== Estimate cluster radius ====== #
        print("[Sampler] Estimating cluster radius...")
        assigned = kmeans.predict(self.sc_embed_pca)
        dists = []
        for i in range(n_clusters):
            pts = self.sc_embed_pca[assigned == i]
            if len(pts) > 1:
                center = self.sc_centroids[i]
                d = np.linalg.norm(pts - center, axis=1)
                dists.extend(list(d))
        self.radius = float(np.percentile(dists, 80))
        print(f"[Sampler] PCA done. Clusters={n_clusters}, radius={self.radius:.4f}")

        # Cache ST PCA (only once)
        self.st_embed_pca = None


    # ===================== Sampling Config ===================== #
    def init_sampling_config(self, N: int):
        self.total_budget = int(N * self.total_ratio)
        self.per_round_k = int(N * self.per_round_ratio)

        # ===== Initialize internal exclusion mask =====
        self.excluded_mask = np.zeros(N, dtype=bool)

        print(f"[Sampler Config] Total={N} | Budget={self.total_budget} | Per round={self.per_round_k}")


    # ===================== Sample + Reward + Update ===================== #
    def sample_and_update(self, embeds: np.ndarray, coords: np.ndarray):

        if self.per_round_k is None:
            self.init_sampling_config(len(embeds))

        # ---------- ST PCA (only once) ---------- #
        if self.st_embed_pca is None:
            print("[Sampler] Projecting ST embeds into PCA space...")
            t0 = time.time()
            self.st_embed_pca = np.vstack([
                self.pca.transform(embeds[i:i+2000])
                for i in tqdm(range(0, len(embeds), 2000),
                              desc="PCA Projection", ncols=100)
            ])
            print(f"[Sampler] ST PCA done in {time.time()-t0:.2f}s")

        # ---------- Policy scoring (GPU) ---------- #
        print("[Sampler] Predicting policy scores...")
        scores = []
        for i in tqdm(range(0, len(embeds), 2000),
                      desc="Policy Scoring", ncols=100):
            x = torch.from_numpy(embeds[i:i+2000]).float().to(self.device)
            with torch.no_grad():
                scores.append(self.policy(x).squeeze(-1).cpu().numpy())
        scores = np.concatenate(scores)
        probs = torch.softmax(torch.tensor(scores), dim=0).numpy()

        # ======== Key modification: only sample from non-excluded indices ======== #
        candidates = np.where(~self.excluded_mask)[0]

        if len(candidates) == 0:
            print("[Sampler] WARNING: No more candidates to sample.")
            return np.array([], dtype=int), 0.0, dict(r_sc=0, r_type=0, r_sp=0)

        probs_cand = probs[candidates]
        probs_cand = probs_cand / probs_cand.sum()  # re-normalize

        real_k = min(self.per_round_k, len(candidates))
        chosen_local = np.random.choice(
            len(candidates),
            size=real_k,
            replace=False,
            p=probs_cand
        )
        idx = candidates[chosen_local]

        print(f"[Sampler] Sampled {len(idx)} new spots "
              f"(from {len(candidates)} candidates).")

        # Mark as selected
        self.excluded_mask[idx] = True

        # ---------- Rewards ---------- #
        sel_embed_orig = embeds[idx]
        sel_embed_pca  = self.st_embed_pca[idx]
        sel_coords     = coords[idx]

        r_sc = reward_sc_manifold(sel_embed_pca,
                          self.sc_centroids,
                          n_clusters=self.sc_centroids.shape[0])

        r_type = reward_sc_type(sel_embed_orig, self.sc_embed_full, self.sc_type)

        r_sp = reward_spatial(sel_coords, coords)

        reward = self.w_sc*r_sc + self.w_type*r_type + self.w_spatial*r_sp

        # ---------- Update policy (REINFORCE) ---------- #
        self._update_policy(sel_embed_orig, reward)

        return idx, float(reward), {
            "r_sc": float(r_sc),
            "r_type": float(r_type),
            "r_sp": float(r_sp)
        }


    # ===================== REINFORCE Update ===================== #
    def _update_policy(self, selected_embed: np.ndarray, reward: float):
        x_sel = torch.from_numpy(selected_embed).float().to(self.device)
        scores = self.policy(x_sel).squeeze(-1)
        loss = -torch.mean(scores) * reward
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
