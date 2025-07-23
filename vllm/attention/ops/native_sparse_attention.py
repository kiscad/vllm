# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Native Sparse Attention 核心操作实现

基于论文 "Native Sparse Attention: Hardware-Aligned and Natively Trainable"
" Sparse Attention" (arXiv:2502.11089v2) 实现动态分层稀疏注意力机制，包含：
1. 动态 Token 压缩（粗粒度）with Grid Pattern
2. 自适应 Token 选择（细粒度）
3. 滑动窗口注意力
4. 算术强度平衡的硬件对齐优化
5. 分层结果合并
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DynamicTokenCompressor(nn.Module):
    """动态粗粒度 Token 压缩器
    
    基于论文中的 Grid Pattern，实现时间和空间局部性优化的压缩
    """

    def __init__(
        self,
        hidden_size: int,
        compression_ratio: float = 0.25,
        block_size: int = 64,
        grid_pattern_enabled: bool = True,
        arithmetic_intensity_threshold: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.compression_ratio = compression_ratio
        self.block_size = block_size
        self.compressed_size = max(1, int(block_size * compression_ratio))
        self.grid_pattern_enabled = grid_pattern_enabled
        self.arithmetic_intensity_threshold = arithmetic_intensity_threshold

        # 动态压缩投影 - 支持不同的压缩策略
        self.global_compression_proj = nn.Linear(hidden_size * block_size,
                                                 hidden_size *
                                                 self.compressed_size,
                                                 bias=False)

        # Grid Pattern 相关的投影
        if grid_pattern_enabled:
            self.grid_importance_head = nn.Linear(hidden_size, 1, bias=False)
            self.temporal_decay = nn.Parameter(torch.tensor(0.95))
            self.spatial_decay = nn.Parameter(torch.tensor(0.9))

        # 算术强度优化的权重
        self.arithmetic_intensity_weight = nn.Parameter(torch.ones(1))

        # 初始化权重
        nn.init.xavier_uniform_(self.global_compression_proj.weight)
        if grid_pattern_enabled:
            nn.init.xavier_uniform_(self.grid_importance_head.weight)

    def _apply_grid_pattern(self, kv_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """应用 Grid Pattern 进行智能压缩"""
        batch_size, seq_len, hidden_size_doubled = kv_tensor.shape

        # 分离 K 和 V
        k_tensor = kv_tensor[..., :hidden_size_doubled // 2]

        # 计算每个位置的重要性分数
        importance_scores = self.grid_importance_head(k_tensor).squeeze(
            -1)  # [B, L]

        # 应用时间衰减（最新的 tokens 更重要）
        time_weights = torch.pow(
            self.temporal_decay,
            torch.arange(seq_len,
                         0,
                         -1,
                         device=kv_tensor.device,
                         dtype=torch.float32))
        importance_scores = importance_scores * time_weights.unsqueeze(0)

        # Grid Pattern: 保持结构化稀疏模式
        if seq_len >= self.block_size:
            num_blocks = seq_len // self.block_size
            remaining = seq_len % self.block_size

            compressed_indices_list = []
            compressed_kv_list = []

            for block_idx in range(num_blocks):
                start_idx = block_idx * self.block_size
                end_idx = start_idx + self.block_size

                block_importance = importance_scores[:, start_idx:end_idx]
                block_kv = kv_tensor[:, start_idx:end_idx]

                # 在每个块内应用 Grid Pattern 选择
                selected_indices = torch.topk(block_importance,
                                              k=self.compressed_size,
                                              dim=-1).indices

                # 使用 gather 操作选择重要的 K,V pairs
                batch_indices = torch.arange(batch_size).unsqueeze(1).expand(
                    -1, self.compressed_size)
                selected_kv = block_kv[batch_indices, selected_indices]

                compressed_kv_list.append(selected_kv)
                compressed_indices_list.append(selected_indices + start_idx)

            # 处理剩余的 tokens
            if remaining > 0:
                remaining_importance = importance_scores[:, -remaining:]
                remaining_kv = kv_tensor[:, -remaining:]
                remaining_compressed_size = max(
                    1, int(remaining * self.compression_ratio))

                remaining_selected_indices = torch.topk(
                    remaining_importance,
                    k=min(remaining_compressed_size, remaining),
                    dim=-1).indices

                batch_indices = torch.arange(batch_size).unsqueeze(1).expand(
                    -1, remaining_selected_indices.shape[1])
                remaining_selected_kv = remaining_kv[
                    batch_indices, remaining_selected_indices]

                compressed_kv_list.append(remaining_selected_kv)
                compressed_indices_list.append(remaining_selected_indices +
                                               seq_len - remaining)

            compressed_kv = torch.cat(compressed_kv_list, dim=1)
            compressed_indices = torch.cat(compressed_indices_list, dim=1)
        else:
            # 序列太短，直接选择 top-k
            num_selected = max(1, int(seq_len * self.compression_ratio))
            selected_indices = torch.topk(importance_scores,
                                          k=num_selected,
                                          dim=-1).indices

            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(
                -1, num_selected)
            compressed_kv = kv_tensor[batch_indices, selected_indices]
            compressed_indices = selected_indices

        return compressed_kv, compressed_indices

    def forward(self, kv_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            kv_tensor: [batch_size, seq_len, hidden_size * 2] (K 和 V 拼接)
            
        Returns:
            compressed_kv: [batch_size, compressed_len, hidden_size * 2]
            compressed_indices: [batch_size, compressed_len]
        """
        if self.grid_pattern_enabled:
            return self._apply_grid_pattern(kv_tensor)
        else:
            # 传统的块压缩方法
            return self._apply_block_compression(kv_tensor)

    def _apply_block_compression(self,
                                 kv_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """传统的块压缩方法（回退）"""
        batch_size, seq_len, hidden_size_doubled = kv_tensor.shape

        # 计算需要padding的长度
        pad_len = (self.block_size -
                   seq_len % self.block_size) % self.block_size
        if pad_len > 0:
            padding = torch.zeros(batch_size,
                                  pad_len,
                                  hidden_size_doubled,
                                  device=kv_tensor.device,
                                  dtype=kv_tensor.dtype)
            kv_padded = torch.cat([kv_tensor, padding], dim=1)
        else:
            kv_padded = kv_tensor

        padded_len = kv_padded.shape[1]
        num_blocks = padded_len // self.block_size

        # 重塑为块格式
        kv_blocks = kv_padded.view(batch_size, num_blocks, self.block_size,
                                   hidden_size_doubled)
        kv_blocks_flat = kv_blocks.view(batch_size, num_blocks, -1)

        # 应用压缩投影
        compressed_blocks = self.global_compression_proj(kv_blocks_flat)
        compressed_blocks = compressed_blocks.view(batch_size, num_blocks,
                                                   self.compressed_size,
                                                   hidden_size_doubled)

        # 重塑回序列格式
        compressed_kv = compressed_blocks.view(batch_size, -1,
                                               hidden_size_doubled)

        # 生成压缩后的索引（均匀采样）
        indices_per_block = torch.arange(self.compressed_size,
                                         device=kv_tensor.device)
        compressed_indices = []

        for block_idx in range(num_blocks):
            block_start = block_idx * self.block_size
            block_indices = indices_per_block * (
                self.block_size // self.compressed_size) + block_start
            compressed_indices.append(block_indices)

        compressed_indices = torch.cat(compressed_indices).unsqueeze(0).expand(
            batch_size, -1)

        # 如果原序列被padding，需要调整索引
        if pad_len > 0:
            mask = compressed_indices < seq_len
            compressed_kv = compressed_kv[mask.unsqueeze(-1).expand_as(
                compressed_kv)]
            compressed_indices = compressed_indices[mask]
            # 重塑维度
            valid_len = int(mask.sum().item())
            compressed_kv = compressed_kv.view(batch_size, valid_len,
                                               hidden_size_doubled)
            compressed_indices = compressed_indices.view(batch_size, valid_len)

        return compressed_kv, compressed_indices


class AdaptiveTokenSelector(nn.Module):
    """自适应细粒度 Token 选择器
    
    基于论文中的动态选择策略，实现上下文感知的 token 选择
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        selection_ratio: float = 0.1,
        adaptive_threshold: float = 0.1,
        context_window: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.selection_ratio = selection_ratio
        self.adaptive_threshold = adaptive_threshold
        self.context_window = context_window

        # 多头重要性查询网络
        self.importance_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.importance_key = nn.Linear(hidden_size, hidden_size, bias=False)

        # 自适应选择阈值网络
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4), nn.ReLU(),
            nn.Linear(hidden_size // 4, 1), nn.Sigmoid())

        # 每个注意力头的重要性权重（可学习）
        self.head_importance_weights = nn.Parameter(
            torch.ones(num_heads) / num_heads)

        # 上下文感知的位置编码
        self.position_bias = nn.Parameter(torch.randn(context_window) * 0.02)

        # 初始化
        nn.init.xavier_uniform_(self.importance_query.weight)
        nn.init.xavier_uniform_(self.importance_key.weight)

    def forward(self, query: Tensor, key: Tensor,
                value: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            query: [batch_size, seq_len, hidden_size]
            key: [batch_size, seq_len, hidden_size] 
            value: [batch_size, seq_len, hidden_size]
            
        Returns:
            selected_key: [batch_size, selected_len, hidden_size]
            selected_value: [batch_size, selected_len, hidden_size]
            selection_indices: [batch_size, selected_len]
            selection_weights: [batch_size, selected_len]
        """
        batch_size, seq_len, hidden_size = query.shape

        # 计算重要性查询和键
        importance_q = self.importance_query(query)  # [B, L, H]
        importance_k = self.importance_key(key)  # [B, L, H]

        # 重塑为多头格式进行计算
        importance_q = importance_q.view(batch_size, seq_len, self.num_heads,
                                         self.head_size)
        importance_k = importance_k.view(batch_size, seq_len, self.num_heads,
                                         self.head_size)

        # 计算每个头的注意力分数 [B, H, L, L]
        attn_scores = torch.einsum('blhd,bkhd->bhlk', importance_q,
                                   importance_k)
        attn_scores = attn_scores / math.sqrt(self.head_size)

        # 应用因果 mask
        causal_mask = torch.triu(torch.ones(seq_len,
                                            seq_len,
                                            device=query.device),
                                 diagonal=1).bool()
        attn_scores.masked_fill_(
            causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # 添加位置偏置（如果序列在范围内）
        if seq_len <= self.context_window:
            pos_bias = self.position_bias[:seq_len].unsqueeze(0).unsqueeze(
                0).unsqueeze(0)
            attn_scores = attn_scores + pos_bias

        # 计算注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, L, L]

        # 计算每个位置对所有查询的平均重要性
        key_importance = torch.sum(attn_probs, dim=-2)  # [B, H, L]

        # 使用可学习权重聚合不同头的重要性
        aggregated_importance = torch.einsum('bhl,h->bl', key_importance,
                                             self.head_importance_weights)

        # 自适应选择阈值
        query_summary = torch.mean(query, dim=1)  # [B, H]
        adaptive_factor = self.adaptive_gate(query_summary)  # [B, 1]
        adaptive_selection_ratio = self.selection_ratio * (
            1 + adaptive_factor.squeeze(-1))

        # 为每个样本动态选择 tokens
        selected_keys = []
        selected_values = []
        selection_indices_list = []
        selection_weights_list = []

        for b in range(batch_size):
            sample_importance = aggregated_importance[b]  # [L]
            sample_selection_ratio = adaptive_selection_ratio[b].item()

            # 动态确定选择数量
            num_selected = max(
                1, min(seq_len, int(seq_len * sample_selection_ratio)))

            # 选择最重要的 tokens
            selected_values_tensor, selected_indices = torch.topk(
                sample_importance, k=num_selected, dim=0)

            # 获取选择的 K, V
            sample_selected_key = key[b, selected_indices]  # [selected_len, H]
            sample_selected_value = value[
                b, selected_indices]  # [selected_len, H]

            # 归一化重要性权重
            normalized_weights = F.softmax(selected_values_tensor, dim=0)

            selected_keys.append(sample_selected_key)
            selected_values.append(sample_selected_value)
            selection_indices_list.append(selected_indices)
            selection_weights_list.append(normalized_weights)

        # 对齐到相同长度（使用最大长度）
        max_selected = max(len(indices) for indices in selection_indices_list)

        aligned_keys = []
        aligned_values = []
        aligned_indices = []
        aligned_weights = []

        for b in range(batch_size):
            current_len = len(selection_indices_list[b])
            if current_len < max_selected:
                # Padding
                pad_len = max_selected - current_len
                padded_key = F.pad(selected_keys[b], (0, 0, 0, pad_len))
                padded_value = F.pad(selected_values[b], (0, 0, 0, pad_len))
                padded_indices = F.pad(selection_indices_list[b], (0, pad_len),
                                       value=seq_len - 1)
                padded_weights = F.pad(selection_weights_list[b], (0, pad_len))
            else:
                padded_key = selected_keys[b]
                padded_value = selected_values[b]
                padded_indices = selection_indices_list[b]
                padded_weights = selection_weights_list[b]

            aligned_keys.append(padded_key)
            aligned_values.append(padded_value)
            aligned_indices.append(padded_indices)
            aligned_weights.append(padded_weights)

        # 堆叠结果
        final_selected_keys = torch.stack(aligned_keys, dim=0)
        final_selected_values = torch.stack(aligned_values, dim=0)
        final_selection_indices = torch.stack(aligned_indices, dim=0)
        final_selection_weights = torch.stack(aligned_weights, dim=0)

        return (final_selected_keys, final_selected_values,
                final_selection_indices, final_selection_weights)


class ArithmeticIntensityOptimizedAttention(nn.Module):
    """算术强度优化的滑动窗口注意力
    
    实现论文中的硬件对齐优化
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        window_size: int = 1024,
        causal: bool = True,
        arithmetic_intensity_threshold: float = 1.0,
        tensor_core_alignment: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.window_size = window_size
        self.causal = causal
        self.arithmetic_intensity_threshold = arithmetic_intensity_threshold
        self.tensor_core_alignment = tensor_core_alignment
        self.scale = 1.0 / math.sqrt(head_size)

        # Tensor Core 对齐优化
        if tensor_core_alignment:
            # 确保维度是 8 的倍数（适合 Tensor Core）
            self.aligned_head_size = ((head_size + 7) // 8) * 8
            if self.aligned_head_size != head_size:
                self.head_size_proj = nn.Linear(head_size,
                                                self.aligned_head_size,
                                                bias=False)
                self.head_size_unproj = nn.Linear(self.aligned_head_size,
                                                  head_size,
                                                  bias=False)
        else:
            self.aligned_head_size = head_size
            self.head_size_proj = None
            self.head_size_unproj = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """
        Args:
            query: [batch_size, seq_len, num_heads, head_size]
            key: [batch_size, seq_len, num_heads, head_size]
            value: [batch_size, seq_len, num_heads, head_size]
            
        Returns:
            output: [batch_size, seq_len, num_heads, head_size]
        """
        batch_size, seq_len, num_heads, head_size = query.shape

        # Tensor Core 对齐处理
        if self.head_size_proj is not None:
            q = self.head_size_proj(query)
            k = self.head_size_proj(key)
            v = self.head_size_proj(value)
        else:
            q, k, v = query, key, value

        # 转置到 [batch_size, num_heads, seq_len, aligned_head_size]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 算术强度检查
        arithmetic_intensity = ((seq_len * self.aligned_head_size) /
                                (2 * seq_len + self.aligned_head_size))

        if arithmetic_intensity >= self.arithmetic_intensity_threshold:
            # 高算术强度：使用优化的实现
            output = self._optimized_sliding_window_attention(q, k, v)
        else:
            # 低算术强度：使用内存友好的实现
            output = self._memory_efficient_sliding_window_attention(q, k, v)

        # 转置回 [batch_size, seq_len, num_heads, aligned_head_size]
        output = output.transpose(1, 2)

        # Tensor Core 对齐后处理
        if self.head_size_unproj is not None:
            output = self.head_size_unproj(output)

        return output

    def _optimized_sliding_window_attention(self, q, k, v):
        """优化的滑动窗口注意力（高算术强度）"""
        batch_size, num_heads, seq_len, head_size = q.shape

        # 使用 PyTorch 的优化实现
        if hasattr(F, 'scaled_dot_product_attention'
                   ) and self.window_size >= seq_len:
            # 窗口大小足够，使用标准 SDPA
            causal_mask = self.causal
            output = F.scaled_dot_product_attention(q,
                                                    k,
                                                    v,
                                                    is_causal=causal_mask,
                                                    scale=self.scale)
        else:
            # 需要滑动窗口 mask
            output = self._manual_optimized_attention(q, k, v)

        return output

    def _memory_efficient_sliding_window_attention(self, q, k, v):
        """内存高效的滑动窗口注意力（低算术强度）"""
        batch_size, num_heads, seq_len, head_size = q.shape

        # 分块处理以减少内存使用
        chunk_size = min(512, self.window_size)
        output = torch.zeros_like(q)

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i]

            # 计算这个查询块的上下文范围
            start_ctx = max(0, i - self.window_size)
            end_ctx = min(seq_len, end_i +
                          self.window_size) if not self.causal else end_i

            k_ctx = k[:, :, start_ctx:end_ctx]
            v_ctx = v[:, :, start_ctx:end_ctx]

            # 计算注意力
            attn_scores = torch.matmul(q_chunk, k_ctx.transpose(
                -2, -1)) * self.scale

            # 应用 mask
            if self.causal or self.window_size < seq_len:
                mask = self._create_sliding_window_mask(q_len=end_i - i,
                                                        kv_len=end_ctx -
                                                        start_ctx,
                                                        q_start=i,
                                                        kv_start=start_ctx,
                                                        device=q.device)
                attn_scores = attn_scores + mask

            attn_probs = F.softmax(attn_scores, dim=-1)
            chunk_output = torch.matmul(attn_probs, v_ctx)

            output[:, :, i:end_i] = chunk_output

        return output

    def _create_sliding_window_mask(self, q_len, kv_len, q_start, kv_start,
                                    device):
        """创建滑动窗口mask"""
        mask = torch.zeros(q_len, kv_len, device=device)

        for qi in range(q_len):
            global_qi = qi + q_start
            for ki in range(kv_len):
                global_ki = ki + kv_start

                # 因果 mask
                if self.causal and global_ki > global_qi or abs(
                        global_qi - global_ki) > self.window_size:
                    mask[qi, ki] = float('-inf')

        return mask

    def _manual_optimized_attention(self, q, k, v):
        """手动优化的注意力计算"""
        batch_size, num_heads, seq_len, head_size = q.shape

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 创建滑动窗口 + 因果 mask
        mask = torch.full((seq_len, seq_len), float('-inf'), device=q.device)

        for i in range(seq_len):
            start_j = max(0, i - self.window_size)
            end_j = i + 1 if self.causal else min(seq_len, i +
                                                  self.window_size + 1)
            mask[i, start_j:end_j] = 0.0

        attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)

        # 计算注意力权重和输出
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)

        return output


class HierarchicalAttentionMerger(nn.Module):
    """增强的分层注意力结果合并器
    
    使用论文中的动态权重策略进行路径合并
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        learnable_weights: bool = True,
        adaptive_merging: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.learnable_weights = learnable_weights
        self.adaptive_merging = adaptive_merging

        if learnable_weights:
            # 全局路径权重
            self.global_path_weights = nn.Parameter(
                torch.tensor([0.4, 0.3, 0.3]))

            # 每个头的路径权重
            self.head_path_weights = nn.Parameter(torch.ones(num_heads, 3) / 3)
        else:
            self.register_buffer('global_path_weights',
                                 torch.tensor([0.4, 0.3, 0.3]))

        if adaptive_merging:
            # 自适应权重网络
            self.adaptive_weight_net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4), nn.ReLU(),
                nn.Linear(hidden_size // 4, 3), nn.Softmax(dim=-1))

            # 上下文感知的合并网络
            self.context_merger = nn.MultiheadAttention(embed_dim=hidden_size,
                                                        num_heads=min(
                                                            num_heads, 8),
                                                        dropout=0.0,
                                                        batch_first=True)

    def forward(
        self,
        compressed_output: Tensor,
        selected_output: Tensor,
        sliding_output: Tensor,
        compressed_indices: Tensor,
        selected_indices: Tensor,
        seq_len: int,
    ) -> Tensor:
        """
        Args:
            compressed_output: [batch_size, compressed_len, hidden_size]
            selected_output: [batch_size, selected_len, hidden_size] 
            sliding_output: [batch_size, seq_len, hidden_size]
            compressed_indices: [batch_size, compressed_len]
            selected_indices: [batch_size, selected_len]
            seq_len: 原始序列长度
            
        Returns:
            merged_output: [batch_size, seq_len, hidden_size]
        """
        batch_size = sliding_output.shape[0]

        # 1. 基础：滑动窗口输出
        merged_output = sliding_output.clone()

        if self.adaptive_merging:
            # 计算自适应权重
            sliding_summary = torch.mean(sliding_output, dim=1)  # [B, H]
            adaptive_weights = self.adaptive_weight_net(
                sliding_summary)  # [B, 3]
            path_weights = adaptive_weights
        else:
            path_weights = self.global_path_weights.unsqueeze(0).expand(
                batch_size, -1)

        # 2. 合并压缩路径的贡献
        if compressed_output.shape[1] > 0:
            for b in range(batch_size):
                valid_mask = compressed_indices[b] < seq_len
                if valid_mask.any():
                    valid_indices = compressed_indices[b][valid_mask].long()
                    valid_compressed = compressed_output[b][valid_mask]

                    # 加权合并
                    weight = path_weights[b, 0]  # 压缩路径权重
                    merged_output[b, valid_indices] = (
                        merged_output[b, valid_indices] * (1 - weight) +
                        valid_compressed * weight)

        # 3. 合并选择路径的贡献
        if selected_output.shape[1] > 0:
            for b in range(batch_size):
                valid_mask = selected_indices[b] < seq_len
                if valid_mask.any():
                    valid_indices = selected_indices[b][valid_mask].long()
                    valid_selected = selected_output[b][valid_mask]

                    # 加权合并
                    weight = path_weights[b, 1]  # 选择路径权重
                    merged_output[b, valid_indices] = (
                        merged_output[b, valid_indices] * (1 - weight) +
                        valid_selected * weight)

        # 4. 可选：上下文感知的后处理
        if self.adaptive_merging and hasattr(self, 'context_merger'):
            # 使用自注意力进行最终的上下文整合
            merged_output, _ = self.context_merger(merged_output,
                                                   merged_output,
                                                   merged_output)

        return merged_output


class NativeSparseAttentionOp(nn.Module):
    """增强的 Native Sparse Attention 完整操作
    
    基于最新论文的完整实现
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        compression_ratio: float = 0.25,
        selection_ratio: float = 0.1,
        sliding_window_size: int = 1024,
        block_size: int = 64,
        arithmetic_intensity_threshold: float = 1.0,
        enable_grid_pattern: bool = True,
        enable_adaptive_selection: bool = True,
        enable_tensor_core_alignment: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = hidden_size // num_heads

        # 初始化增强的组件
        self.token_compressor = DynamicTokenCompressor(
            hidden_size=hidden_size,
            compression_ratio=compression_ratio,
            block_size=block_size,
            grid_pattern_enabled=enable_grid_pattern,
            arithmetic_intensity_threshold=arithmetic_intensity_threshold,
        )

        self.token_selector = AdaptiveTokenSelector(
            hidden_size=hidden_size,
            num_heads=num_heads,
            selection_ratio=selection_ratio,
            adaptive_threshold=0.1 if enable_adaptive_selection else 1.0,
            context_window=sliding_window_size,
        )

        self.sliding_window_attn = ArithmeticIntensityOptimizedAttention(
            num_heads=num_heads,
            head_size=self.head_size,
            window_size=sliding_window_size,
            arithmetic_intensity_threshold=arithmetic_intensity_threshold,
            tensor_core_alignment=enable_tensor_core_alignment,
        )

        self.hierarchical_merger = HierarchicalAttentionMerger(
            hidden_size=hidden_size,
            num_heads=num_heads,
            learnable_weights=True,
            adaptive_merging=True,
        )

        # 性能监控
        self.enable_profiling = False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: Optional[float] = None,
    ) -> Tensor:
        """
        Args:
            query: [batch_size, seq_len, hidden_size]
            key: [batch_size, seq_len, hidden_size]
            value: [batch_size, seq_len, hidden_size]
            scale: attention scale factor
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = query.shape

        if scale is None:
            scale = 1.0 / math.sqrt(self.head_size)

        # 路径 1: 动态压缩注意力
        compressed_kv, compressed_indices = self.token_compressor(
            torch.cat([key, value], dim=-1))
        compressed_k = compressed_kv[..., :hidden_size]
        compressed_v = compressed_kv[..., hidden_size:]

        # 计算压缩注意力
        if compressed_k.shape[1] > 0:
            compressed_output = F.scaled_dot_product_attention(
                query.view(batch_size, seq_len, self.num_heads,
                           self.head_size).transpose(1, 2),
                compressed_k.view(batch_size, -1, self.num_heads,
                                  self.head_size).transpose(1, 2),
                compressed_v.view(batch_size, -1, self.num_heads,
                                  self.head_size).transpose(1, 2),
                scale=scale).transpose(1, 2).reshape(batch_size, seq_len,
                                                     hidden_size)
        else:
            compressed_output = torch.zeros_like(query)

        # 路径 2: 自适应选择注意力
        (selected_k, selected_v, selected_indices,
         selection_weights) = self.token_selector(query, key, value)

        if selected_k.shape[1] > 0:
            selected_output = F.scaled_dot_product_attention(
                query.view(batch_size, seq_len, self.num_heads,
                           self.head_size).transpose(1, 2),
                selected_k.view(batch_size, -1, self.num_heads,
                                self.head_size).transpose(1, 2),
                selected_v.view(batch_size, -1, self.num_heads,
                                self.head_size).transpose(1, 2),
                scale=scale).transpose(1, 2).reshape(batch_size, seq_len,
                                                     hidden_size)
        else:
            selected_output = torch.zeros_like(query)

        # 路径 3: 算术强度优化的滑动窗口注意力
        q_heads = query.view(batch_size, seq_len, self.num_heads,
                             self.head_size)
        k_heads = key.view(batch_size, seq_len, self.num_heads, self.head_size)
        v_heads = value.view(batch_size, seq_len, self.num_heads,
                             self.head_size)

        sliding_output = self.sliding_window_attn(q_heads, k_heads, v_heads)
        sliding_output = sliding_output.view(batch_size, seq_len, hidden_size)

        # 分层合并所有路径的结果
        final_output = self.hierarchical_merger(compressed_output,
                                                selected_output,
                                                sliding_output,
                                                compressed_indices,
                                                selected_indices, seq_len)

        return final_output
