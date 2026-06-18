"""CDMP prompt learner: K independent abnormal context-vector sets.

K=1 exactly reproduces AnomalyCLIP_PromptLearner (regression-test safe).
K>1 adds K-1 extra ctx_neg parameter sets; forward() returns (1+K) prompts
stacked as [normal, anom_0, ..., anom_{K-1}].
"""

import torch
import torch.nn as nn
from prompt_ensemble import AnomalyCLIP_PromptLearner


class CDMPPromptLearner(AnomalyCLIP_PromptLearner):
    def __init__(self, clip_model, design_details, cdmp_K=1):
        super().__init__(clip_model, design_details)
        self.cdmp_K = cdmp_K

        if cdmp_K > 1:
            # ctx_neg from parent is the first abnormal prototype (index 0).
            # Create K-1 additional independent ones.
            self.ctx_neg_extra = nn.ParameterList([
                nn.Parameter(torch.empty_like(self.ctx_neg))
                for _ in range(cdmp_K - 1)
            ])
            for p in self.ctx_neg_extra:
                nn.init.normal_(p, std=0.02)

    def _all_ctx_neg(self):
        """Returns list of K ctx_neg tensors, each (n_cls, anomaly_num, n_ctx, d)."""
        result = [self.ctx_neg]
        if self.cdmp_K > 1:
            result.extend(list(self.ctx_neg_extra))
        return result

    def forward(self, cls_id=None):
        if self.cdmp_K == 1:
            return super().forward(cls_id)

        # --- positive (normal) prompt — same as parent ---
        prompts_pos = torch.cat(
            [self.token_prefix_pos, self.ctx_pos, self.token_suffix_pos], dim=2
        )
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)  # (1, seq_len, d)

        # --- K negative (anomaly) prompts ---
        prompts_neg_list = []
        for ctx_neg_k in self._all_ctx_neg():
            p_neg = torch.cat(
                [self.token_prefix_neg, ctx_neg_k, self.token_suffix_neg], dim=2
            )
            _, _, ln, dn = p_neg.shape
            prompts_neg_list.append(p_neg.reshape(-1, ln, dn))  # (1, seq_len, d)
        prompts_neg = torch.cat(prompts_neg_list, dim=0)  # (K, seq_len, d)

        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)  # (1+K, seq_len, d)

        # tokenized_prompts — anomaly template is the same for all K prototypes
        _, _, dp = self.tokenized_prompts_pos.shape
        tok_pos = self.tokenized_prompts_pos.reshape(-1, dp)       # (1, 77)
        _, _, dn2 = self.tokenized_prompts_neg.shape
        tok_neg_one = self.tokenized_prompts_neg.reshape(-1, dn2)  # (1, 77)
        tok_neg = tok_neg_one.repeat(self.cdmp_K, 1)               # (K, 77)
        tokenized_prompts = torch.cat([tok_pos, tok_neg], dim=0)   # (1+K, 77)

        return prompts, tokenized_prompts, self.compound_prompts_text

    def anomaly_text_features(self, all_feats):
        """Given encode_text_learn output (1+K, d), return (K, d) anomaly features."""
        return all_feats[1:]  # (K, d)

    def normal_text_feature(self, all_feats):
        """Given encode_text_learn output (1+K, d), return (d,) normal feature."""
        return all_feats[0]
