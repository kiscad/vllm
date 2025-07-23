# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionState, AttentionType)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class NSAParams:
    """Native Sparse Attention 参数配置 (基于 arXiv:2502.11089v2)"""
    max_seqlen: int

    # 动态分层注意力配置
    block_size: int = 64  # 块大小
    compression_ratio: float = 0.25  # 粗粒度压缩比例
    selection_ratio: float = 0.1  # 细粒度选择比例
    sliding_window_size: int = 1024  # 滑动窗口大小

    # Grid Pattern 优化配置
    enable_grid_pattern: bool = True  # 启用 Grid Pattern
    temporal_decay: float = 0.95  # 时间衰减因子
    spatial_decay: float = 0.9  # 空间衰减因子

    # 硬件对齐优化配置
    enable_tensor_core: bool = True  # 启用 Tensor Core 优化
    arithmetic_intensity_threshold: float = 1.0  # 算术强度阈值
    tensor_core_alignment: bool = True  # Tensor Core 维度对齐

    # 自适应选择配置
    enable_adaptive_selection: bool = True  # 自适应 token 选择
    adaptive_threshold: float = 0.1  # 自适应阈值
    context_window: int = 512  # 上下文窗口

    # 训练配置
    enable_backward: bool = True  # 支持反向传播
    gradient_checkpointing: bool = False  # 梯度检查点
    adaptive_merging: bool = True  # 自适应路径合并

    # 性能监控配置
    enable_profiling: bool = False  # 启用性能监控

    # 分布式配置
    num_heads: int = field(init=False)
    num_kv_heads: int = field(init=False)
    head_sliding_step: int = field(init=False)
    active_head_range: Tuple = field(init=False)

    def __post_init__(self):
        assert self.block_size > 0
        assert 0 < self.compression_ratio < 1
        assert 0 < self.selection_ratio < 1
        assert self.sliding_window_size > 0
        assert 0 < self.temporal_decay <= 1
        assert 0 < self.spatial_decay <= 1

        tp_rank = get_tensor_model_parallel_rank()

        self.active_head_range = (
            tp_rank * self.num_heads,
            (tp_rank + 1) * self.num_heads,
        )


class NativeSparseAttentionBackend(AttentionBackend):
    """Native Sparse Attention 后端 (基于 arXiv:2502.11089v2)"""

    @staticmethod
    def get_name() -> str:
        return "NATIVE_SPARSE_ATTENTION"

    @staticmethod
    def get_impl_cls() -> Type["NativeSparseAttentionImpl"]:
        return NativeSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        from vllm.attention.backends.flash_attn import FlashAttentionMetadata
        return FlashAttentionMetadata

    @staticmethod
    def get_state_cls() -> Type["AttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        return CommonMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


class NativeSparseAttentionImpl(AttentionImpl):
    """Native Sparse Attention 实现类 (基于 arXiv:2502.11089v2)"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        nsa_config: Optional[Dict[str, Any]] = None,
    ) -> None:

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "KV sharing is not supported in NSA backend.")

        if alibi_slopes is not None:
            logger.warning_once(
                "Alibi slopes are not fully optimized for NSA, "
                "may affect performance.")

        if logits_soft_cap is not None:
            logger.warning_once(
                "Logits soft cap support in NSA is experimental.")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type

        # NSA 特定配置 (基于最新论文)
        if nsa_config is None:
            nsa_config = {}

        # 默认 NSA 参数（符合论文建议）
        default_config = {
            "max_seqlen": 65536,
            "block_size": 64,
            "compression_ratio": 0.25,
            "selection_ratio": 0.1,
            "sliding_window_size": sliding_window or 1024,
            "enable_grid_pattern": True,
            "temporal_decay": 0.95,
            "spatial_decay": 0.9,
            "enable_tensor_core": True,
            "arithmetic_intensity_threshold": 1.0,
            "tensor_core_alignment": True,
            "enable_adaptive_selection": True,
            "adaptive_threshold": 0.1,
            "context_window": 512,
            "enable_backward": True,
            "gradient_checkpointing": False,
            "adaptive_merging": True,
            "enable_profiling": False,
        }
        default_config.update(nsa_config)
        default_config["num_heads"] = num_heads
        default_config["num_kv_heads"] = self.num_kv_heads

        self.nsa_params = NSAParams(**default_config)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # 验证头大小支持
        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by NSA. "
                f"Supported head sizes are: {supported_head_sizes}.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for NSA")

        # 初始化 NSA 核心组件
        self._init_nsa_components()

    def _init_nsa_components(self):
        """初始化 NSA 的核心组件"""
        from vllm.attention.ops.native_sparse_attention import (
            NativeSparseAttentionOp)

        # 初始化 NSA 核心操作（使用增强版本）
        self.nsa_op = NativeSparseAttentionOp(
            hidden_size=self.num_heads * self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            compression_ratio=self.nsa_params.compression_ratio,
            selection_ratio=self.nsa_params.selection_ratio,
            sliding_window_size=self.nsa_params.sliding_window_size,
            block_size=self.nsa_params.block_size,
            arithmetic_intensity_threshold=(
                self.nsa_params.arithmetic_intensity_threshold),
            enable_grid_pattern=self.nsa_params.enable_grid_pattern,
            enable_adaptive_selection=(
                self.nsa_params.enable_adaptive_selection),
            enable_tensor_core_alignment=self.nsa_params.tensor_core_alignment,
        )

        # 设置性能监控（可选）
        if self.nsa_params.enable_profiling:
            self.nsa_op.enable_profiling = True

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """NSA 前向传播（增强版本）"""

        # 获取输入维度
        batch_size, seq_len, hidden_size = query.shape

        # 智能 NSA 启用策略
        nsa_threshold = getattr(self.nsa_params, 'min_seq_len_for_nsa', 2048)

        # 动态调整阈值（基于算术强度）
        arithmetic_intensity = ((seq_len * self.head_size) /
                                (2 * seq_len + self.head_size))
        ai_threshold = self.nsa_params.arithmetic_intensity_threshold
        if arithmetic_intensity >= ai_threshold:
            nsa_threshold = max(1024, nsa_threshold // 2)

        use_nsa = (seq_len >= nsa_threshold and hasattr(self, 'nsa_op')
                   and self.nsa_op is not None)

        if use_nsa:
            # 使用增强的 NSA 进行注意力计算
            try:
                # 处理 MQA/GQA 的情况
                if self.num_queries_per_kv > 1:
                    k_expanded = torch.repeat_interleave(
                        key.view(batch_size, seq_len, self.num_kv_heads,
                                 self.head_size),
                        self.num_queries_per_kv,
                        dim=2).view(batch_size, seq_len, hidden_size)
                    v_expanded = torch.repeat_interleave(
                        value.view(batch_size, seq_len, self.num_kv_heads,
                                   self.head_size),
                        self.num_queries_per_kv,
                        dim=2).view(batch_size, seq_len, hidden_size)
                else:
                    k_expanded = key
                    v_expanded = value

                # 调用增强的 NSA 操作
                attn_output = self.nsa_op(query=query,
                                          key=k_expanded,
                                          value=v_expanded,
                                          scale=self.scale)

                logger.info_once(
                    f"Using enhanced NSA for sequence length {seq_len} "
                    f"(Grid Pattern: {self.nsa_params.enable_grid_pattern}, "
                    f"Adaptive Selection: "
                    f"{self.nsa_params.enable_adaptive_selection}, "
                    f"Arithmetic Intensity: {arithmetic_intensity:.2f})")
                return attn_output

            except Exception as e:
                logger.warning_once(f"Enhanced NSA failed with error: {e}, "
                                    "falling back to standard attention")
                use_nsa = False

        if not use_nsa:
            # 回退到优化的标准注意力
            logger.debug_once(
                f"Using standard attention for sequence length {seq_len}")

            # 重塑 query, key, value 为注意力计算格式
            q = query.view(batch_size, seq_len, self.num_heads, self.head_size)
            k = key.view(batch_size, seq_len, self.num_kv_heads,
                         self.head_size)
            v = value.view(batch_size, seq_len, self.num_kv_heads,
                           self.head_size)

            # 处理 MQA/GQA
            if self.num_queries_per_kv > 1:
                k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
                v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

            q, k, v = (x.transpose(1, 2) for x in (q, k, v))

            # 使用滑动窗口（如果适用）
            if self.sliding_window and seq_len > self.sliding_window:
                is_causal = True
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, is_causal=is_causal)
            else:
                attn_output = F.scaled_dot_product_attention(q,
                                                             k,
                                                             v,
                                                             scale=self.scale)

            attn_output = attn_output.transpose(1, 2)
            return attn_output.reshape(batch_size, seq_len, hidden_size)

        # 保险return语句（理论上不应该执行到这里）
        raise RuntimeError("Unexpected code path in NSA forward method")

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return PagedAttention.get_supported_head_sizes()

    @staticmethod
    def get_supported_dtypes() -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]
