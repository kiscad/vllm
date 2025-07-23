#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Native Sparse Attention (NSA) 在 vLLM 中的使用示例

本示例基于最新论文 "Native Sparse Attention: Hardware-Aligned and Natively"
" Trainable Sparse Attention" (arXiv:2502.11089v2) 展示如何在 vLLM 中配置和使用 "
"Native Sparse Attention，
包括不同场景下的配置选择和性能优化建议。
"""

import os
import time

# 设置环境变量启用 NSA
os.environ["VLLM_ATTENTION_BACKEND"] = "NATIVE_SPARSE_ATTENTION"
# 或者使用简写: os.environ["VLLM_USE_NSA"] = "1"

from vllm import LLM, SamplingParams
from vllm.attention.configs.nsa_config import (
    create_custom_nsa_config,
    get_nsa_config,
    validate_nsa_config,
)


def basic_nsa_usage():
    """基础 NSA 使用示例"""
    print("=== 基础 NSA 使用示例 ===")

    # 使用默认 NSA 配置（基于论文推荐参数）
    nsa_config = get_nsa_config("default")
    print(f"使用 NSA 配置: {nsa_config}")

    # 创建 LLM 实例，传入 NSA 配置
    llm = LLM(
        model="microsoft/DialoGPT-medium",  # 示例模型
        max_model_len=8192,  # 设置较长的上下文长度
        tensor_parallel_size=1,
        attention_backend="NATIVE_SPARSE_ATTENTION",
        # NSA配置将通过环境变量或配置文件传递
    )

    # 准备测试输入
    prompts = [
        "Explain the concept of machine learning in detail.",
        "Write a comprehensive guide on Python programming.",
        "Describe the process of natural language processing.",
    ]

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    # 执行推理
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    print(f"生成完成，耗时: {end_time - start_time:.2f} 秒")

    # 输出结果
    for output in outputs:
        print(f"输入: {output.prompt[:50]}...")
        print(f"输出: {output.outputs[0].text[:200]}...")
        print("-" * 50)


def advanced_nsa_configurations():
    """演示不同的 NSA 配置选项"""
    print("\n=== 高级 NSA 配置示例 ===")

    # 1. 高压缩配置（适用于极长序列）
    print("1. 高压缩配置测试")
    high_compression_config = get_nsa_config("high_compression")

    llm_high_compression = LLM(
        model="microsoft/DialoGPT-medium",
        max_model_len=32768,  # 更长的序列
        attention_backend="NATIVE_SPARSE_ATTENTION",
        **{"nsa_config": high_compression_config},
    )

    # 2. 高精度配置（保留更多细节）
    print("2. 高精度配置测试")
    high_precision_config = get_nsa_config("high_precision")

    llm_high_precision = LLM(
        model="microsoft/DialoGPT-medium",
        max_model_len=16384,
        attention_backend="NATIVE_SPARSE_ATTENTION",
        **{"nsa_config": high_precision_config},
    )

    # 3. 自定义配置
    print("3. 自定义配置测试")
    custom_config = create_custom_nsa_config(
        base_config="default",
        compression_ratio=0.3,  # 压缩比例
        selection_ratio=0.15,  # 选择比例
        sliding_window_size=2048,  # 滑动窗口大小
        enable_grid_pattern=True,  # 启用 Grid Pattern
        enable_adaptive_selection=True,  # 启用自适应选择
        arithmetic_intensity_threshold=1.5,  # 算术强度阈值
        min_seq_len_for_nsa=1024,  # NSA 启用阈值
    )

    # 验证配置
    try:
        validate_nsa_config(custom_config)
        print("✓ 自定义配置验证成功")
    except Exception as e:
        print(f"✗ 配置验证失败: {e}")
        return

    llm_custom = LLM(
        model="microsoft/DialoGPT-medium",
        max_model_len=16384,
        attention_backend="NATIVE_SPARSE_ATTENTION",
        **{"nsa_config": custom_config},
    )

    # 测试不同配置的性能
    test_prompt = (
        "Write a detailed analysis of the advantages and challenges of "
        "artificial intelligence in modern society. " * 10
    )  # 创建长输入

    configs_to_test = [
        ("High Compression", llm_high_compression, high_compression_config),
        ("High Precision", llm_high_precision, high_precision_config),
        ("Custom", llm_custom, custom_config),
    ]

    for config_name, llm_instance, config in configs_to_test:
        print(f"\n测试 {config_name} 配置:")
        print(f"  - 压缩比例: {config.get('compression_ratio', 'N/A')}")
        print(f"  - 选择比例: {config.get('selection_ratio', 'N/A')}")
        print(f"  - Grid Pattern: {config.get('enable_grid_pattern', 'N/A')}")
        print(f"  - 自适应选择: {config.get('enable_adaptive_selection', 'N/A')}")

        start_time = time.time()
        try:
            llm_instance.generate([test_prompt], SamplingParams(max_tokens=500))
            end_time = time.time()
            print(f"  ✓ 生成成功，耗时: {end_time - start_time:.2f} 秒")
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")


def environment_variable_usage():
    """演示通过环境变量配置 NSA"""
    print("\n=== 环境变量配置示例 ===")

    # 方法1: 直接指定后端
    os.environ["VLLM_ATTENTION_BACKEND"] = "NATIVE_SPARSE_ATTENTION"

    # 方法2: 使用简化配置
    os.environ["VLLM_USE_NSA"] = "1"
    os.environ["VLLM_NSA_COMPRESSION_RATIO"] = "0.25"
    os.environ["VLLM_NSA_SELECTION_RATIO"] = "0.1"
    os.environ["VLLM_NSA_WINDOW_SIZE"] = "1024"
    os.environ["VLLM_NSA_ENABLE_GRID_PATTERN"] = "1"
    os.environ["VLLM_NSA_ENABLE_ADAPTIVE_SELECTION"] = "1"

    print("环境变量设置:")
    nsa_env_vars = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("VLLM_NSA", "VLLM_USE_NSA", "VLLM_ATTENTION_BACKEND"))
    }
    for key, value in nsa_env_vars.items():
        print(f"  {key}={value}")

    # 创建LLM实例（会自动读取环境变量）
    LLM(
        model="microsoft/DialoGPT-medium",
        max_model_len=8192,
    )

    print("✓ 成功通过环境变量配置 NSA")


def performance_comparison():
    """NSA vs 标准注意力性能对比"""
    print("\n=== 性能对比测试 ===")

    test_prompts = [
        "Explain quantum computing and its potential applications in cryptography.",
        "Describe the evolution of programming languages from assembly to "
        "modern high-level languages.",
        "Analyze the impact of artificial intelligence on healthcare, "
        "education, and transportation sectors.",
    ]

    # 标准注意力
    print("1. 标准注意力测试")
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)  # 移除NSA设置

    llm_standard = LLM(
        model="microsoft/DialoGPT-medium",
        max_model_len=8192,
        tensor_parallel_size=1,
    )

    start_time = time.time()
    outputs_standard = llm_standard.generate(
        test_prompts, SamplingParams(max_tokens=800)
    )
    standard_time = time.time() - start_time

    print(f"标准注意力耗时: {standard_time:.2f} 秒")

    # NSA 注意力
    print("2. NSA 注意力测试")
    os.environ["VLLM_ATTENTION_BACKEND"] = "NATIVE_SPARSE_ATTENTION"

    nsa_config = get_nsa_config("fast_inference")  # 专为推理优化的配置

    llm_nsa = LLM(
        model="microsoft/DialoGPT-medium",
        max_model_len=8192,
        tensor_parallel_size=1,
        attention_backend="NATIVE_SPARSE_ATTENTION",
        **{"nsa_config": nsa_config},
    )

    start_time = time.time()
    outputs_nsa = llm_nsa.generate(test_prompts, SamplingParams(max_tokens=800))
    nsa_time = time.time() - start_time

    print(f"NSA 注意力耗时: {nsa_time:.2f} 秒")

    # 性能对比
    speedup = standard_time / nsa_time if nsa_time > 0 else float("inf")
    print(f"\n性能提升: {speedup:.2f}x")

    # 质量对比（简单的长度对比）
    standard_avg_len = sum(
        len(output.outputs[0].text) for output in outputs_standard
    ) / len(outputs_standard)
    nsa_avg_len = sum(len(output.outputs[0].text) for output in outputs_nsa) / len(
        outputs_nsa
    )

    print(f"标准注意力平均输出长度: {standard_avg_len:.1f}")
    print(f"NSA 注意力平均输出长度: {nsa_avg_len:.1f}")
    print(f"输出长度比率: {nsa_avg_len / standard_avg_len:.2f}")


def troubleshooting_guide():
    """故障排除指南"""
    print("\n=== 故障排除指南 ===")

    print("常见问题及解决方案:")
    print("1. 'NSA backend not found' 错误")
    print("   解决方案: 确保正确设置 VLLM_ATTENTION_BACKEND='NATIVE_SPARSE_ATTENTION'")

    print("\n2. 序列长度过短，NSA 未启用")
    print("   解决方案: 降低 min_seq_len_for_nsa 参数或增加输入序列长度")

    print("\n3. 内存不足")
    print("   解决方案: 增加 compression_ratio，减少 selection_ratio")

    print("\n4. 精度下降")
    print("   解决方案: 使用 'high_precision' 配置或减少 compression_ratio")

    print("\n5. 性能提升不明显")
    print(
        "   解决方案: 确保序列长度足够长（>2048），"
        "检查 arithmetic_intensity_threshold 设置"
    )

    # 配置诊断
    print("\n配置诊断工具:")
    try:
        from vllm.attention.configs.nsa_config import diagnose_nsa_config

        config = get_nsa_config("default")
        diagnosis = diagnose_nsa_config(config, sequence_length=4096, model_size="7B")

        print("诊断结果:")
        for key, value in diagnosis.items():
            print(f"  {key}: {value}")

    except ImportError:
        print("  诊断工具不可用，请检查安装")


def main():
    """主函数"""
    print("🚀 Native Sparse Attention (NSA) 使用示例")
    print("基于论文: arXiv:2502.11089v2")
    print("=" * 60)

    try:
        # 1. 基础使用
        basic_nsa_usage()

        # 2. 高级配置
        advanced_nsa_configurations()

        # 3. 环境变量配置
        environment_variable_usage()

        # 4. 性能对比
        performance_comparison()

        # 5. 故障排除
        troubleshooting_guide()

    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        print("请检查:")
        print("1. vLLM 是否正确安装")
        print("2. NSA 后端是否可用")
        print("3. 模型是否可访问")
        return

    print("\n✅ 所有示例执行完成！")
    print("\n📚 更多信息:")
    print("- NSA 论文: https://arxiv.org/abs/2502.11089")
    print("- vLLM 文档: https://docs.vllm.ai/")
    print("- GitHub 仓库: https://github.com/vllm-project/vllm")


if __name__ == "__main__":
    main()
