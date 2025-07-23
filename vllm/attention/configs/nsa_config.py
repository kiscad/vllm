# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Native Sparse Attention (NSA) 配置示例

本文件包含了各种使用场景下的 NSA 配置示例，帮助用户根据具体需求
选择合适的 NSA 参数配置。
"""

from typing import Any, Dict

# 基础配置：适用于大多数长序列场景
DEFAULT_NSA_CONFIG: Dict[str, Any] = {
    "max_seqlen": 65536,
    "block_size": 64,
    "compression_ratio": 0.25,  # 压缩到原来的 25%
    "selection_ratio": 0.1,  # 选择 10% 的重要 tokens
    "sliding_window_size": 1024,  # 1K 滑动窗口
    "enable_tensor_core": True,
    "arithmetic_intensity_threshold": 1.0,
    "enable_backward": True,
    "gradient_checkpointing": False,
    "min_seq_len_for_nsa": 2048,  # 序列长度超过 2K 才启用 NSA
}

# 高压缩配置：适用于极长序列（如 1M+ tokens）
HIGH_COMPRESSION_NSA_CONFIG: Dict[str, Any] = {
    "max_seqlen": 1048576,  # 1M tokens
    "block_size": 128,
    "compression_ratio": 0.1,  # 更高的压缩比
    "selection_ratio": 0.05,  # 更少的选择比例
    "sliding_window_size": 2048,  # 更大的滑动窗口
    "enable_tensor_core": True,
    "arithmetic_intensity_threshold": 2.0,
    "enable_backward": True,
    "gradient_checkpointing": True,  # 启用梯度检查点节省内存
    "min_seq_len_for_nsa": 4096,
}

# 高精度配置：适用于需要保持更多细节的任务
HIGH_PRECISION_NSA_CONFIG: Dict[str, Any] = {
    "max_seqlen": 32768,
    "block_size": 32,  # 更小的块大小
    "compression_ratio": 0.5,  # 更低的压缩比
    "selection_ratio": 0.2,  # 更多的选择比例
    "sliding_window_size": 512,
    "enable_tensor_core": True,
    "arithmetic_intensity_threshold": 0.5,
    "enable_backward": True,
    "gradient_checkpointing": False,
    "min_seq_len_for_nsa": 1024,
}

# 快速推理配置：适用于推理阶段，优化速度
FAST_INFERENCE_NSA_CONFIG: Dict[str, Any] = {
    "max_seqlen": 32768,
    "block_size": 64,
    "compression_ratio": 0.2,
    "selection_ratio": 0.08,
    "sliding_window_size": 768,
    "enable_tensor_core": True,
    "arithmetic_intensity_threshold": 1.5,
    "enable_backward": False,  # 推理时不需要反向传播
    "gradient_checkpointing": False,
    "min_seq_len_for_nsa": 1536,
}

# 训练配置：适用于预训练或微调
TRAINING_NSA_CONFIG: Dict[str, Any] = {
    "max_seqlen": 65536,
    "block_size": 64,
    "compression_ratio": 0.3,
    "selection_ratio": 0.12,
    "sliding_window_size": 1024,
    "enable_tensor_core": True,
    "arithmetic_intensity_threshold": 1.0,
    "enable_backward": True,
    "gradient_checkpointing": True,  # 训练时启用梯度检查点
    "min_seq_len_for_nsa": 2048,
}

# 内存受限配置：适用于显存较小的设备
MEMORY_EFFICIENT_NSA_CONFIG: Dict[str, Any] = {
    "max_seqlen": 16384,
    "block_size": 32,
    "compression_ratio": 0.15,  # 更激进的压缩
    "selection_ratio": 0.06,
    "sliding_window_size": 512,
    "enable_tensor_core": True,
    "arithmetic_intensity_threshold": 2.0,
    "enable_backward": True,
    "gradient_checkpointing": True,
    "min_seq_len_for_nsa": 1024,
}

# 代码生成配置：适用于代码生成任务，平衡精度和效率
CODE_GENERATION_NSA_CONFIG: Dict[str, Any] = {
    "max_seqlen": 49152,  # 48K tokens
    "block_size": 64,
    "compression_ratio": 0.25,
    "selection_ratio": 0.15,  # 代码需要更多细节
    "sliding_window_size": 1536,  # 更大的滑动窗口保持局部结构
    "enable_tensor_core": True,
    "arithmetic_intensity_threshold": 1.0,
    "enable_backward": True,
    "gradient_checkpointing": False,
    "min_seq_len_for_nsa": 2048,
}

# 文档处理配置：适用于长文档理解任务
DOCUMENT_PROCESSING_NSA_CONFIG: Dict[str, Any] = {
    "max_seqlen": 131072,  # 128K tokens
    "block_size": 128,
    "compression_ratio": 0.2,
    "selection_ratio": 0.1,
    "sliding_window_size": 2048,  # 较大的滑动窗口保持段落连贯性
    "enable_tensor_core": True,
    "arithmetic_intensity_threshold": 1.2,
    "enable_backward": True,
    "gradient_checkpointing": True,
    "min_seq_len_for_nsa": 3072,
}


def get_nsa_config(config_name: str = "default") -> Dict[str, Any]:
    """
    根据配置名称获取对应的 NSA 配置
    
    Args:
        config_name: 配置名称，可选值包括：
            - "default": 默认配置
            - "high_compression": 高压缩配置
            - "high_precision": 高精度配置
            - "fast_inference": 快速推理配置
            - "training": 训练配置
            - "memory_efficient": 内存高效配置
            - "code_generation": 代码生成配置
            - "document_processing": 文档处理配置
    
    Returns:
        对应的 NSA 配置字典
    """
    configs = {
        "default": DEFAULT_NSA_CONFIG,
        "high_compression": HIGH_COMPRESSION_NSA_CONFIG,
        "high_precision": HIGH_PRECISION_NSA_CONFIG,
        "fast_inference": FAST_INFERENCE_NSA_CONFIG,
        "training": TRAINING_NSA_CONFIG,
        "memory_efficient": MEMORY_EFFICIENT_NSA_CONFIG,
        "code_generation": CODE_GENERATION_NSA_CONFIG,
        "document_processing": DOCUMENT_PROCESSING_NSA_CONFIG,
    }

    if config_name not in configs:
        raise ValueError(f"Unknown NSA config: {config_name}. "
                         f"Available configs: {list(configs.keys())}")

    return configs[config_name].copy()


def create_custom_nsa_config(base_config: str = "default",
                             **overrides) -> Dict[str, Any]:
    """
    基于基础配置创建自定义 NSA 配置
    
    Args:
        base_config: 基础配置名称
        **overrides: 要覆盖的配置参数
    
    Returns:
        自定义的 NSA 配置字典
        
    Example:
        custom_config = create_custom_nsa_config(
            base_config="default",
            compression_ratio=0.3,
            sliding_window_size=2048
        )
    """
    config = get_nsa_config(base_config)
    config.update(overrides)
    return config


def validate_nsa_config(config: Dict[str, Any]) -> bool:
    """
    验证 NSA 配置的有效性
    
    Args:
        config: NSA 配置字典
        
    Returns:
        配置是否有效
        
    Raises:
        ValueError: 配置无效时抛出异常
    """
    required_keys = {
        "max_seqlen", "block_size", "compression_ratio", "selection_ratio",
        "sliding_window_size"
    }

    # 检查必需的键
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    # 验证数值范围
    if not (0 < config["compression_ratio"] < 1):
        raise ValueError("compression_ratio must be between 0 and 1")

    if not (0 < config["selection_ratio"] < 1):
        raise ValueError("selection_ratio must be between 0 and 1")

    if config["block_size"] <= 0:
        raise ValueError("block_size must be positive")

    if config["sliding_window_size"] <= 0:
        raise ValueError("sliding_window_size must be positive")

    if config["max_seqlen"] <= 0:
        raise ValueError("max_seqlen must be positive")

    # 检查逻辑关系
    if config["sliding_window_size"] > config["max_seqlen"]:
        raise ValueError("sliding_window_size cannot exceed max_seqlen")

    return True
