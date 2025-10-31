# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

import torch.nn as nn

from torchtitan.components.ft import FTManager
from torchtitan.models.moe import MoEArgs
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import (
    build_optimizers,
    build_optimizers_with_moe_load_balancing,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize_hf_transformers import parallelize_hf_transformers

from .infra.pipeline_hf import pipeline_hf_transformers
from .model.args import HFTransformerModelArgs
from .model.model import HFTransformerModel


__all__ = [
    "HFTransformerModelArgs",
    "HFTransformerModel",
]

@dataclass
class TitanDenseModelArgs:
    """Arguments for the base TorchTitan model."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"

@dataclass
class TitanMoeModelArgs:
    """Arguments specific to DeepSeekV3 models."""

    moe_args: MoEArgs | None = None
    n_group: int | None = None
    topk_group: int | None = None
    inter_dim: int | None = None
    moe_inter_dim: int | None = None
    n_dense_layers: int | None = None
    n_expert_groups: int | None = None
    n_limited_groups: int | None = None
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    original_seq_len: int | None = None
    rope_factor: float | None = None
    beta_fast: int | None = None
    beta_slow: int | None = None
    mscale: float | None = None
    partial_rotary_factor: float | None = None
    rope_interleave: bool = True

flavors = {
    "debugmodel": HFTransformerModelArgs(
        titan_dense_args=TitanDenseModelArgs(
            dim=256,
            n_layers=6,
            n_heads=16,
            n_kv_heads=16,
        ),
    ),
    "debugmodel_moe": HFTransformerModelArgs(
        titan_dense_args=TitanDenseModelArgs(
            dim=256,
            n_layers=3,
            n_heads=16,
            n_kv_heads=16,
        ),
        titan_moe_args=TitanMoeModelArgs(
            partial_rotary_factor=4.0,
            inter_dim=1024,
            moe_inter_dim=256,
            n_dense_layers=1,
            n_group=2,
            topk_group=1,
            kv_lora_rank=512,
            q_lora_rank=0,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            mscale=0.70,
            moe_args=MoEArgs(
                num_experts=8,
                num_shared_experts=2,
                top_k=3,
                score_func="softmax",
                route_norm=True,
                score_before_experts=False,
                load_balance_coeff=1e-3,
            ),
        ),
    ),
    "full": HFTransformerModelArgs(
        titan_dense_args=TitanDenseModelArgs(),
    ),
}

def build_optimizers_auto_detect_moe(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:

    # Check if any model part has MoE enabled
    has_moe = False
    for model_part in model_parts:
        if hasattr(model_part, "layers"):
            for layer in model_part.layers:
                if hasattr(layer, "moe_enabled") and layer.moe_enabled:
                    has_moe = True
                    break
        if has_moe:
            break
    
    if has_moe:
        # NOTE(3outeille): Monkey-patch temporarily for compatibility. Otherwise, I will need to copy optimizer.py just to loop over layer instead of layer.values().
        for model_part in model_parts:
            if hasattr(model_part, "layers") and not hasattr(model_part.layers, "values"):
                model_part.layers.values = lambda self=model_part.layers: iter(self)

    return_val = (build_optimizers_with_moe_load_balancing if has_moe else build_optimizers)(
        model_parts=model_parts,
        optimizer_config=optimizer_config,
        parallel_dims=parallel_dims,
        ft_manager=ft_manager,
    )
    return return_val

def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=HFTransformerModel,
        model_args=flavors,
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        build_optimizers_fn=build_optimizers_auto_detect_moe,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
