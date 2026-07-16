import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def encode_text_with_prompt_ensemble(model, texts, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence)
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features



def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

# OAP: number of abnormal severity levels beyond the base pole (K=1 -> original binary case, bit-exact at init)
OAP_K = 4

class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        # OAP: non-registered reference to the (frozen) CLIP model, used internally to encode
        # the K+1 ordinal level prompts for self.last_level_feats / ordinal_score(). List-wrapped
        # so nn.Module does not register it as a submodule (would pull the whole frozen CLIP
        # backbone into this module's .parameters()/.state_dict() and break optimizer/checkpoint).
        self._clip_model_ref = [clip_model]
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]

        
        self.classnames = classnames

        self.state_normal_list = [
            "{}",
        ]

        self.state_anomaly_list = [
            "damaged {}",
        ]
        
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            #初始化text成bpd编码
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                #生成相应的text embedding
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            #这些是去除出来EOS 和 # CLS, EOS， 获得可学习的textual prompt
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if True:
                print("Initializing class-specific contexts")
                #这里是cls是类的个数，n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]


        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]
        print("normal prompt templates:")
        for prompt in prompts_pos:
            print(f"  {prompt}")
        print("anomaly prompt templates:")
        for prompt in prompts_neg:
            print(f"  {prompt}")
        print(f"learnable prompt length: normal={n_ctx_pos}, anomaly={n_ctx_neg}, compound={self.text_encoder_n_ctx}, depth={self.compound_prompts_depth}")

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
     
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        #生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        # OAP: the frozen "damaged" token used to sit at embedding_neg[:, :, 1+n_ctx_neg, :]
        # (verified: "damaged" is exactly 1 BPE token). Carve it out of the frozen suffix and
        # turn it into a learnable base severity token (self.base_abnormal_token below); keep
        # the REST of the suffix ("object.", EOT, padding) frozen exactly as before, one
        # position later.
        token_state_neg = embedding_neg[:, :, 1 + n_ctx_neg: 2 + n_ctx_neg, :]  # (n_cls, anormaly_num, 1, dim) "damaged" slot
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 2 + n_ctx_neg:, :])  # "object." + EOT + pad, frozen

        # OAP: learnable ordinal severity tokens, monotonic by construction.
        # level 1 == base_abnormal_token exactly (no delta) -> bit-exact vs baseline at OAP_K=1.
        # levels 2..K: base + cumsum(softplus(level_deltas)), so severity is non-decreasing.
        # level_deltas init small (std=0.01) -> all levels start clustered near the base token.
        self.base_abnormal_token = nn.Parameter(token_state_neg[:, :, 0, :].clone())  # (n_cls, anormaly_num, dim)
        self.level_deltas = nn.Parameter(torch.empty(max(OAP_K - 1, 0), self.n_cls, anormaly_num, ctx_dim, dtype=dtype))
        nn.init.normal_(self.level_deltas, std=0.01)

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # tokenized_prompts = torch.cat([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)



    def forward(self, cls_id =None):
        
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        # print(prefix_pos.shape, prefix_neg.shape)

        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        # OAP: build K ordinal abnormal severity tokens, monotonic by construction.
        # level_tokens[0]  == self.base_abnormal_token exactly (no delta added)
        # level_tokens[1:] == base + cumsum(softplus(level_deltas))  (non-decreasing severity)
        base_tok = self.base_abnormal_token  # (n_cls, anormaly_num, dim)
        if OAP_K > 1:
            deltas = F.softplus(self.level_deltas)                                    # (K-1, n_cls, anormaly_num, dim), >= 0
            cum = torch.cumsum(deltas, dim=0)                                          # (K-1, n_cls, anormaly_num, dim)
            level_tokens = torch.cat([base_tok.unsqueeze(0), base_tok.unsqueeze(0) + cum], dim=0)  # (K, n_cls, anormaly_num, dim)
        else:
            level_tokens = base_tok.unsqueeze(0)                                       # (1, n_cls, anormaly_num, dim)

        def _build_abnormal_prompt(state_tok):
            # state_tok: (n_cls, anormaly_num, dim) -> full prompt (n_cls, anormaly_num, 77, dim)
            return torch.cat(
                [
                    prefix_neg,                # (n_cls, anormaly_num, 1, dim)
                    ctx_neg,                   # (n_cls, anormaly_num, n_ctx, dim)
                    state_tok.unsqueeze(2),    # (n_cls, anormaly_num, 1, dim)  <- ordinal level token
                    suffix_neg,                # (n_cls, anormaly_num, *, dim)  "object." + EOT + pad, frozen
                ],
                dim=2,
            )

        # OAP: interface collapse (Step 3). train.py/test.py still call
        # model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text) themselves on
        # whatever is returned here, and expect exactly [2, D] out of that single encode call. The
        # text encoder is nonlinear, so we cannot return something that reproduces an arbitrary
        # MEAN-OF-POST-ENCODER-FEATURES via one downstream encode call -- so the collapsed abnormal
        # pole handed through this [2, D] interface is encoder(mean of the K PRE-ENCODER level
        # tokens), which is exact at OAP_K=1 and a smooth differentiable aggregate for OAP_K>1.
        # The TRUE per-level POST-encoder features (for ordinal_score) are computed separately
        # below into self.last_level_feats, encoding each level individually.
        mean_level_tok = level_tokens.mean(dim=0)              # (n_cls, anormaly_num, dim)
        prompts_neg = _build_abnormal_prompt(mean_level_tok)   # (n_cls, anormaly_num, 77, dim)
        _, _, l_emb, d_emb = prompts_neg.shape  # OAP: capture true embedding dims before l/d get reused below for seq_len
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)


        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)


        # OAP (Step 3/4): populate self.last_level_feats = [1+K, D] true per-level post-encoder
        # features (row 0 = normal, rows 1..K = ordinal abnormal levels), for the inference-only
        # ordinal_score() helper below. Not used by the training loss -> computed under no_grad.
        with torch.no_grad():
            level_prompts_flat = [prompts_pos] + [
                _build_abnormal_prompt(level_tokens[k]).reshape(-1, l_emb, d_emb) for k in range(level_tokens.shape[0])
            ]  # each item: (n_cls*num, 77, dim); n_cls==normal_num==anormaly_num==1 in this repo
            all_level_prompts = torch.cat(level_prompts_flat, dim=0)  # (1+K, 77, dim)
            all_tokenized = torch.cat(
                [tokenized_prompts_pos] + [tokenized_prompts_neg] * level_tokens.shape[0], dim=0
            )  # (1+K, 77)
            self.last_level_feats = self._clip_model_ref[0].encode_text_learn(
                all_level_prompts, all_tokenized, self.compound_prompts_text
            )  # (1+K, D)

        return prompts, tokenized_prompts, self.compound_prompts_text

    def ordinal_score(self, image_feats):
        # OAP (Step 4): inference-only helper, never called during training.
        # image_feats: (B, D), assumed L2-normalized like elsewhere in this pipeline.
        # Returns the expected ordinal severity level in [0, 1] (soft rank over 1+K levels).
        level_feats = self.last_level_feats                                  # (1+K, D)
        level_feats = level_feats / level_feats.norm(dim=-1, keepdim=True)    # (1+K, D)
        logit_scale = getattr(self._clip_model_ref[0], "logit_scale", None)
        temperature = float(logit_scale.exp()) if logit_scale is not None else 1.0 / 0.07
        sims = image_feats.float() @ level_feats.float().t()                  # (B, 1+K)
        probs = torch.softmax(sims * temperature, dim=-1)                     # (B, 1+K)
        levels = torch.arange(level_feats.shape[0], device=image_feats.device, dtype=probs.dtype)  # 0..K
        expected_level = (probs * levels).sum(dim=-1)                        # (B,)
        return expected_level / OAP_K                                        # normalized to [0, 1]
