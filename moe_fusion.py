"""
moe_fusion.py
-------------
Interpretable Mixture-of-Experts fusion layer for CulturalQA.

Architecture (DeepSeekMoE shared/routed split + Mixtral-style top-k gating),
with the shared and routed experts as NAMED FUSION STRATEGIES instead of
generic MLPs, so routing decisions are directly interpretable:

    SHARED expert  : Concat            -> always active, no gating, no info loss
    ROUTED experts : Attention, TFN,
                      Multiplication    -> gated, top-k selected per sample

        fused = shared_expert(t, i) + sum_{e in top-k}( gate_e * routed_expert_e(t, i) )

Public interface: moe = MoEFusion(); fused = moe(text_emb, img_emb);
moe.save_checkpoint(path); moe = MoEFusion.load_checkpoint(path);
moe.score_relevance(q_text, q_img, c_text, c_img) for ranking a candidate
against a query (used by pipeline.py's _moe_rescore).

TFN/Bilinear uses a LOW-RANK factorization (project to rank=32 before the
bilinear interaction) rather than a full nn.Bilinear(dim,dim,hidden_dim) --
confirmed the full version cost 40-100x more compute than every other
expert combined (57.9ms vs <2ms per forward+backward call).

AttentionExpert includes text_emb as a SECOND key/value position alongside
img_emb (not image-only). FIX for a confirmed bug: with only the image as
K/V (a single position), softmax attention always returns weight=1.0
regardless of the actual score, making the output query-independent --
i.e. AttentionExpert produced byte-identical output for wildly different
text inputs whenever img_emb was the null vector (true for ~95%+ of this
project's data, every dataset except DRISHTIKON with --enable_images on).
Including text_emb as a second K/V position gives attention a real,
text-dependent choice to make. Confirmed via direct test: mean abs diff
between two different text inputs went from 0.000000 to 1.95 after this
fix. Checkpoint-compatible -- nn.MultiheadAttention's parameters don't
depend on sequence length, so this doesn't change any state_dict shapes
(confirmed by round-tripping a checkpoint saved with the old 1-token
version through the new 2-token version with zero errors).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatExpert(nn.Module):
    """Shared expert: concatenation + MLP. Always active -- the stable
    fallback fusion every sample gets, regardless of what the gate decides
    for the routed experts."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, text_emb, img_emb):
        return self.net(torch.cat([text_emb, img_emb], dim=-1))


class AttentionExpert(nn.Module):
    """Routed expert: cross-attention between text and image, with text's
    own embedding included as a second K/V position (see module docstring
    for why -- single-position K/V degenerates to a text-independent
    constant)."""

    def __init__(self, dim, hidden_dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, hidden_dim)

    def forward(self, text_emb, img_emb):
        q = text_emb.unsqueeze(1)                       # [batch, 1, dim]
        kv = torch.stack([text_emb, img_emb], dim=1)     # [batch, 2, dim]
        attn_out, _ = self.cross_attn(q, kv, kv)
        return self.proj(attn_out.squeeze(1))


class TFNExpert(nn.Module):
    """Routed expert: low-rank bilinear / tensor-fusion -- captures
    higher-order interactions between text and image dimensions that
    concat/attention miss, at a small fraction of a full bilinear layer's
    compute and parameter count (see module docstring)."""

    def __init__(self, dim, hidden_dim, rank=32):
        super().__init__()
        self.text_rank_proj = nn.Linear(dim, rank)
        self.img_rank_proj = nn.Linear(dim, rank)
        self.bilinear = nn.Bilinear(rank, rank, hidden_dim)

    def forward(self, text_emb, img_emb):
        t = self.text_rank_proj(text_emb)
        i = self.img_rank_proj(img_emb)
        return F.relu(self.bilinear(t, i))


class MultiplicationExpert(nn.Module):
    """Routed expert: element-wise product -- cheap way to capture
    reinforcing/shared signal. In the no-image case (the overwhelming
    majority of this project's data), this is the routed expert that
    remains most reliably discriminative w.r.t. text content."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)

    def forward(self, text_emb, img_emb):
        return self.proj(text_emb * img_emb)


class MoEFusion(nn.Module):
    def __init__(self, text_dim=768, img_dim=512, proj_dim=256, hidden_dim=256, top_k=2):
        super().__init__()
        assert top_k <= 3, "only 3 routed experts available (attention, tfn, multiplication)"
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.img_proj = nn.Linear(img_dim, proj_dim)

        self.shared_expert = ConcatExpert(proj_dim, hidden_dim)

        self.routed_experts = nn.ModuleList([
            AttentionExpert(proj_dim, hidden_dim),
            TFNExpert(proj_dim, hidden_dim),
            MultiplicationExpert(proj_dim, hidden_dim),
        ])
        self.routed_names = ["attention", "tfn", "multiplication"]

        self.gate = nn.Linear(proj_dim * 2, len(self.routed_experts))

        self.last_aux = None

    def forward(self, text_emb, img_emb):
        t = self.text_proj(text_emb)
        i = self.img_proj(img_emb)

        gate_input = torch.cat([t, i], dim=-1)
        gate_logits = self.gate(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        topk_weights, topk_idx = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        routed_outs = torch.stack(
            [expert(t, i) for expert in self.routed_experts], dim=1
        )

        gathered = torch.gather(
            routed_outs, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, routed_outs.size(-1))
        )
        routed_fused = (gathered * topk_weights.unsqueeze(-1)).sum(dim=1)

        shared_out = self.shared_expert(t, i)

        fused = shared_out + routed_fused

        self.last_aux = {
            "gate_weights": gate_weights.detach(),
            "topk_idx": topk_idx.detach(),
            "topk_weights": topk_weights.detach(),
            "routed_names": self.routed_names,
        }
        return fused

    def load_balancing_loss(self):
        if self.last_aux is None:
            raise RuntimeError("call forward() before load_balancing_loss()")
        gate_weights = self.last_aux["gate_weights"]
        num_experts = gate_weights.size(-1)
        importance = gate_weights.sum(dim=0)
        importance = importance / importance.sum()
        target = torch.full_like(importance, 1.0 / num_experts)
        return F.mse_loss(importance, target)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    @torch.no_grad()
    def score_relevance(self, query_text_emb, query_img_emb, candidate_text_emb, candidate_img_emb):
        """Fuses the query and the candidate SEPARATELY (matching how
        train_moe.py trains this model), returns cosine similarity as a
        relevance score. This is what pipeline.py's _moe_rescore calls."""
        query_fused = self.forward(query_text_emb, query_img_emb)
        candidate_fused = self.forward(candidate_text_emb, candidate_img_emb)
        return F.cosine_similarity(query_fused, candidate_fused, dim=-1)

    @classmethod
    def load_checkpoint(cls, path, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model