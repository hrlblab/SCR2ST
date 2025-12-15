import torch
from torch import nn
import torch.nn.functional as F
import timm
from collections import Counter
from torchvision import models


def pcc_loss(pred, target, eps=1e-8):
    pred = pred.float()
    target = target.float()

    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    cov = (pred_centered * target_centered).mean(dim=0)
    pred_std = pred_centered.pow(2).mean(dim=0).sqrt() + eps
    target_std = target_centered.pow(2).mean(dim=0).sqrt() + eps

    pcc = cov / (pred_std * target_std + eps)

    # mean over all 300 genes
    return 1 - pcc.mean()


# ===================================================================
# Shared Heads
# ===================================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.gelu = nn.GELU()
        self.fc   = nn.Linear(proj_dim, proj_dim)
        self.drop = nn.Dropout(dropout)
        self.ln   = nn.LayerNorm(proj_dim)

    def forward(self, x):
        z = self.proj(x)
        x = self.gelu(z)
        x = self.fc(x)
        x = self.drop(x)
        x = x + z
        return self.ln(x)



class DenseRegressor(nn.Module):
    def __init__(self, num_outputs=300, hidden_dim=1024, pretrained=True):
        super().__init__()
        backbone = models.densenet121(pretrained=pretrained)
        self.features = backbone.features
        in_features = backbone.classifier.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_outputs)
        )

    def forward(self, x: torch.Tensor):
        feat_map = self.features(x)                                # [B, C, H', W']
        pooled   = nn.functional.adaptive_avg_pool2d(feat_map, 1)  # [B, C, 1, 1]
        features = pooled.view(pooled.size(0), -1)                 # [B, C]
        output   = self.classifier(features)                       # [B, 300]
        return features, output


class retrieval_module(nn.Module):
    def __init__(
        self,
        image_embed_dim=1024,
        spot_embed_dim=300,
        proj_dim=256,
        topk=50,
        num_type=10,
        lambda_kd=0,
        lambda_contrast=0.1,
        sim_threshold=0.1,
        temperature=0.5
    ):
        super().__init__()
        self.image_proj = ProjectionHead(image_embed_dim, proj_dim)
        self.spot_proj  = ProjectionHead(spot_embed_dim, proj_dim)

        # Hyperparameters
        self.topk          = topk
        self.num_type      = num_type
        self.lambda_kd     = lambda_kd
        self.lambda_contrast = lambda_contrast
        self.sim_threshold = sim_threshold
        self.temperature   = temperature

    def forward_contrastive(self, batch):
        feat = batch["feat"]
        spot = batch["reduced_expression"]
        img_emb  = self.image_proj(feat)
        spot_emb = self.spot_proj(spot)

        # Normalize + similarity + contrastive loss
        img_norm  = F.normalize(img_emb, dim=1)
        spot_norm = F.normalize(spot_emb, dim=1)
        logits = spot_norm @ img_norm.T / self.temperature
        labels = torch.arange(len(logits), device=logits.device)

        loss_i2s = F.cross_entropy(logits,     labels)
        loss_s2i = F.cross_entropy(logits.T,   labels)
        loss_con = (loss_i2s + loss_s2i)

        return loss_con * self.lambda_contrast

    def forward_retrieval(self, batch):
        """
        batch: {
          "image": [B,3,H,W],
          "expr":  [B,300],
          "all_expr":       [N,300],
          "all_cell_type":  list of length N
        }
        """
        feat      = batch["feature"]
        all_expr  = batch["all_expr"]       # [N, 300]
        all_types = batch["all_cell_type"]  # list of length N

        with torch.no_grad():
            img_p   = self.image_proj(feat)              # [B, emb_dim]
            expr_p  = self.spot_proj(all_expr)
            img_norm  = F.normalize(img_p,  dim=1)
            expr_norm = F.normalize(expr_p, dim=1)

            sim   = img_norm @ expr_norm.T               # [B, N]
            vals, idxs = sim.topk(self.topk, dim=1)      # [B, topk]

            retrieves, confs = [], []
            for i in range(feat.size(0)):
                sel_idxs = idxs[i].cpu().tolist()                      # length topk
                sel_types = [all_types[j] for j in sel_idxs]           # corresponding types

                # 1) Count top 10 most common cell types in topk
                top10_types = {t for t,_ in Counter(sel_types).most_common(self.num_type)}

                filt = [j for j in sel_idxs if all_types[j] in top10_types]
                if not filt:
                    filt = sel_idxs  # fallback: if all filtered out, use all topk

                retrieves.append(all_expr[filt].mean(dim=0))
                confs.append(vals[i].mean().item())  # confidence based on all topk

            pred_ret = torch.stack(retrieves, dim=0).to(feat.device)  # [B,300]
            conf     = torch.tensor(confs, device=feat.device)       # [B]

        mask = (conf > self.sim_threshold).float().unsqueeze(-1)      # [B,1]

        return mask, pred_ret
