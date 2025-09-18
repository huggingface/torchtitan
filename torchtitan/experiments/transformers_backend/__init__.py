# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from dataclasses import dataclass
from typing import Optional

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.components.tokenizer import build_hf_tokenizer

from torchtitan.models.llama3 import pipeline_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize_hf_transformers import parallelize_hf_transformers
from .model.hf_transformers_args import HFTransformerModelArgs, HFTransformerModel

from torchtitan.models.moe import MoEArgs


__all__ = [
    "HFTransformerModelArgs",
    "HFTransformerModel",
    "hf_transformers_configs",
]

@dataclass
class TitanModelArgs:
    """Arguments for the base TorchTitan model."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 128256
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0


@dataclass
class DeepSeekV3Args:
    """Arguments specific to DeepSeekV3 models."""
    moe_args: Optional[MoEArgs] = None
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    inter_dim: Optional[int] = None
    moe_inter_dim: Optional[int] = None
    n_dense_layers: Optional[int] = None
    n_expert_groups: Optional[int] = None
    n_limited_groups: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    original_seq_len: Optional[int] = None
    rope_factor: Optional[float] = None
    beta_fast: Optional[int] = None
    beta_slow: Optional[int] = None
    mscale: Optional[float] = None

# #TODO(3outeille): identify that if MoE model is used, we add a moe_args field

if os.environ.get("MODEL_TYPE") == "llama":
    print("Using llama model")
    flavors = {
        "debugmodel": HFTransformerModelArgs(
            titan_args=TitanModelArgs(
                max_seq_len=2048,
                dim=256,
                n_layers=6,
                n_heads=16,
                n_kv_heads=16,
                vocab_size=2000,
                rope_theta=500000
            ),
        ),
        "medium": HFTransformerModelArgs(
            titan_args=TitanModelArgs(
                dim=1024,
                n_layers=12,
            ),
        ),
        "full": HFTransformerModelArgs(
            titan_args=TitanModelArgs(),
        ),
    }
else:
    print("Using deepseek model")
    flavors = {
        "debugmodel": HFTransformerModelArgs(
            titan_args=TitanModelArgs(
                vocab_size=2000,
                dim=256,
                n_layers=3,
                n_heads=16,
                n_kv_heads=16,
            ),
            deepseek_v3_args=DeepSeekV3Args(
                inter_dim=1024,
                moe_inter_dim=256,
                n_dense_layers=1,
                n_group=2,
                topk_group=1,
                kv_lora_rank=16,
                q_lora_rank=0,
                qk_nope_head_dim=32,
                qk_rope_head_dim=16,
                v_head_dim=32,
                mscale=0.70,
                moe_args=MoEArgs(
                    num_experts=8,
                    num_shared_experts=2,
                    top_k=3,
                    score_func="softmax",
                    route_norm=True,
                    score_before_experts=False,
                ),
            )
        ),
    }

hf_train_spec = TrainSpec(
    name="hf_auto_model",
    model_cls=HFTransformerModel,
    model_args=flavors,
    parallelize_fn=parallelize_hf_transformers,
    pipelining_fn=pipeline_llama,
    build_optimizers_fn=build_optimizers,
    build_lr_schedulers_fn=build_lr_schedulers,
    build_dataloader_fn=build_hf_dataloader,
    build_tokenizer_fn=build_hf_tokenizer,
    build_loss_fn=build_cross_entropy_loss,
)

register_train_spec(hf_train_spec)