#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
金融教育模块入口文件
用于接收股票预测信息并生成金融教育内容
"""

import json
import sys
import os
import argparse
from typing import Dict, Any

# 使用包相对导入
from .modules import OpenAIClient
from .input_processor import InputProcessor
from .error_handler import ErrorHandler
from .factcheck import FinancialFactChecker

# 全局配置（路径相对于当前文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_TEMPLATE_PATH = os.path.join(BASE_DIR, "prompt_template.txt")
FINANCIAL_CONCEPTS_PATH = os.path.join(BASE_DIR, "financial_concepts.json")
DEFAULT_TEST_FILE = os.path.join(BASE_DIR, "inputnew.json")


def load_prompt_template() -> str:
    """
    加载prompt模板

    Returns:
        str: prompt模板内容
    """
    try:
        with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"加载prompt模板失败: {e}")
        sys.exit(1)


def load_financial_concepts() -> Dict[str, Any]:
    """
    加载金融概念映射表

    Returns:
        Dict: 金融概念数据
    """
    try:
        with open(FINANCIAL_CONCEPTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"加载金融概念映射表失败: {e}")
        sys.exit(1)


def load_prediction_data(file_path: str) -> Dict[str, Any]:
    """
    加载股票预测数据

    Args:
        file_path: 预测数据文件路径

    Returns:
        Dict: 预测数据
    """
    # 验证JSON文件格式
    is_valid_json, data, json_error = ErrorHandler.validate_json_file(file_path)
    if not is_valid_json:
        print(json_error)
        sys.exit(1)

    # 验证数据结构
    is_valid_structure, structure_message = ErrorHandler.validate_prediction_data_structure(
        data
    )
    if not is_valid_structure:
        print(structure_message)
        sys.exit(1)

    # 显示警告（如果有）
    if structure_message:
        print(f"\n警告: {structure_message}")

    return data


def format_prediction_for_prompt(prediction_data: Dict[str, Any]) -> str:
    """
    将预测数据格式化为prompt的一部分

    Args:
        prediction_data: 预测数据

    Returns:
        str: 格式化后的预测数据文本
    """
    return json.dumps(prediction_data, ensure_ascii=False, indent=2)


def filter_investment_advice(content: str) -> str:
    """
    过滤可能的投资建议

    Args:
        content: 生成的内容

    Returns:
        str: 过滤后的内容
    """
    # 简单过滤器；复杂情况可以用正则或NLP
    problematic_patterns = [
        "建议你现在购买",
        "建议你现在出售",
        "应该立即购入",
        "应该马上卖出",
        "现在是买入的好时机",
        "现在是卖出的好时机",
        "建议你持有",
        "建议你清仓",
        "你应该买入",
        "你应该卖出",
        "推荐购买",
        "推荐出售",
        "仓位应该",
        "仓位建议",
        "分配仓位",
        "应该配置",
    ]

    filtered_content = content
    replacement = "投资者可以考虑自己的风险承受能力和投资目标"
    for pattern in problematic_patterns:
        if pattern in filtered_content:
            filtered_content = filtered_content.replace(pattern, replacement)

    return filtered_content


def generate_financial_education(
    client: OpenAIClient,
    prediction_data: Dict[str, Any],
    prompt_template: str,
    concepts_data: Dict[str, Any],
    max_retries: int = 2,
    use_factcheck: bool = True,
) -> str:
    """
    生成金融教育内容

    Args:
        client: OpenAI客户端
        prediction_data: 预测数据（建议为已处理后的结构）
        prompt_template: 提示词模板
        concepts_data: 金融概念数据
        max_retries: 最大重试次数
        use_factcheck: 是否使用事实验证功能

    Returns:
        str: 生成的金融教育内容
    """
    try:
        # 初始化输入处理器
        processor = InputProcessor(FINANCIAL_CONCEPTS_PATH)

        # 处理输入数据（如果传入的是原始预测数据，这里会标准化结构）
        processed_data = processor.process_input(prediction_data)

        # 初始化并使用事实验证器（如果启用）
        factcheck_results = None
        knowledge_path = os.path.join(BASE_DIR, "Financial_knowledge.jsonl")
        if use_factcheck and os.path.exists(knowledge_path):
            try:
                fact_checker = FinancialFactChecker(knowledge_path)
                # 验证supporting_facts中的事实
                if "supporting_facts" in processed_data and isinstance(
                    processed_data["supporting_facts"], list
                ):
                    factcheck_results = fact_checker.verify_facts(
                        processed_data["supporting_facts"]
                    )
                    print(f"完成了 {len(factcheck_results)} 条事实的验证")

                # 获取与预测相关的金融知识
                relevant_knowledge = fact_checker.get_relevant_knowledge(processed_data)
                if relevant_knowledge:
                    print(f"找到 {len(relevant_knowledge)} 条相关金融知识")
                    processed_data["relevant_knowledge"] = relevant_knowledge
            except Exception as e:
                print(f"事实验证过程中出现异常: {e}")
                # 即使事实验证失败，也继续生成内容

        # 获取需要解释的金融概念
        concepts_to_explain = processor.get_concepts_to_explain(processed_data)
        concept_info = "\n\n优先解释的金融概念：\n"
        for concept in concepts_to_explain:
            concept_info += f"- {concept['name']}\n"

        # 安全提示与免责声明（prompt 级别）
        safety_instructions = (
            "\n\n重要提示：\n"
            "1. 请仅关注金融教育内容，不要讨论其他话题\n"
            "2. 不要提供具体的投资建议（如购买或出售特定股票、仓位分配等）\n"
            "3. 内容仅供教育参考，不构成投资建议"
        )

        # 构建完整的提示词
        full_prompt = (
            f"{prompt_template}{safety_instructions}\n\n"
            f"以下是股票预测数据:\n{format_prediction_for_prompt(processed_data)}"
            f"{concept_info}\n\n请根据上述数据生成金融教育内容。"
        )

        # API 调用重试机制
        retry_count = 0
        last_error = None

        while retry_count <= max_retries:
            try:
                result = client.call_llm(full_prompt)

                if result["success"]:
                    print(
                        f"生成成功！消耗tokens: {result['total_tokens']}, "
                        f"响应时间: {result['response_time']:.2f}秒"
                    )

                    # 过滤可能的投资建议
                    content = filter_investment_advice(result["content"])

                    # 添加免责声明（输出级别）
                    disclaimer = (
                        "\n\n---\n\n**Disclaimer**: The information provided is intended solely for educational purposes "
                        "and should not be interpreted as financial or investment advice. Any trading or investment activity "
                        "should be based on your independent research and consultation with licensed financial professionals."
                    )

                    return content + disclaimer
                else:
                    last_error = result["error"]
                    print(f"尝试 {retry_count + 1}/{max_retries + 1} 失败: {last_error}")
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = 2 ** retry_count  # 2, 4, 8 秒...
                        print(f"将在 {wait_time} 秒后重试...")
                        import time

                        time.sleep(wait_time)
            except Exception as e:
                last_error = str(e)
                print(f"尝试 {retry_count + 1}/{max_retries + 1} 出现异常: {last_error}")
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = 2 ** retry_count
                    print(f"将在 {wait_time} 秒后重试...")
                    import time

                    time.sleep(wait_time)

        # 所有重试失败
        print(f"\n错误: 生成内容失败。所有重试都未成功。\n最后错误: {last_error}")

        return f"""对不起，生成金融教育内容时出现错误。

请尝试以下操作：
- 检查您的网络连接
- 确认API密钥有效（scripts/tools/llm_config.py）
- 稍后再尝试运行程序

技术错误信息：{last_error}"""

    except Exception as e:
        print(f"\n处理错误: {str(e)}")
        return f"生成内容时发生内部错误: {str(e)}"


def generate_education_from_file(
    json_path: str,
    use_factcheck: bool = True,
) -> str:
    """
    供外部调用的封装函数：
    给定预测JSON文件路径，返回生成的教育内容文本。
    """
    prompt_template = load_prompt_template()
    concepts_data = load_financial_concepts()
    raw_prediction_data = load_prediction_data(json_path)

    client = OpenAIClient()
    return generate_financial_education(
        client,
        raw_prediction_data,
        prompt_template,
        concepts_data,
        use_factcheck=use_factcheck,
    )


def main():
    """主函数：命令行入口"""
    parser = argparse.ArgumentParser(description="金融教育内容生成器")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=DEFAULT_TEST_FILE,
        help=f"预测数据文件路径 (默认: {DEFAULT_TEST_FILE})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认: 输出到控制台)",
    )
    parser.add_argument(
        "-s",
        "--show-processed",
        action="store_true",
        help="显示处理后的数据结构",
    )
    parser.add_argument(
        "--factcheck",
        action="store_true",
        help="启用事实验证功能",
    )
    parser.add_argument(
        "--no-factcheck",
        action="store_true",
        help="禁用事实验证功能",
    )
    args = parser.parse_args()

    # 加载必要数据
    prompt_template = load_prompt_template()
    concepts_data = load_financial_concepts()
    raw_prediction_data = load_prediction_data(args.file)

    # 初始化输入处理器和 OpenAI 客户端
    processor = InputProcessor(FINANCIAL_CONCEPTS_PATH)
    client = OpenAIClient()

    # 处理输入数据
    processed_data = processor.process_input(raw_prediction_data)

    # 可选：显示处理后的数据
    if args.show_processed:
        print("\n===== 处理后的预测数据 =====\n")
        print(json.dumps(processed_data, ensure_ascii=False, indent=2))

        concepts_to_explain = processor.get_concepts_to_explain(processed_data)
        print("\n===== 需要解释的金融概念 =====\n")
        for concept in concepts_to_explain:
            print(f"- {concept['name']}")
            if "definition" in concept:
                print(f"  定义: {concept['definition']}")
            if "analogy" in concept:
                print(f"  类比: {concept['analogy']}")
        print()

    # 决定是否使用事实验证
    use_factcheck = True
    if args.no_factcheck:
        use_factcheck = False
    elif args.factcheck:
        use_factcheck = True

    # 生成金融教育内容
    education_content = generate_financial_education(
        client,
        processed_data,
        prompt_template,
        concepts_data,
        use_factcheck=use_factcheck,
    )

    # === Save education report automatically ===
    education_reports_dir = os.path.join(os.path.dirname(__file__), "..", "tools", "cache", "education_reports")
    os.makedirs(education_reports_dir, exist_ok=True)

    # Default filename is based on the insight JSON filename
    input_filename = os.path.basename(args.file)
    output_filename = input_filename.replace(".json", ".md")
    default_output_path = os.path.join(education_reports_dir, output_filename)

    # If user provided --output, override default
    save_path = args.output if args.output else default_output_path

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(education_content)
        print(f"📌 Education report saved: {save_path}")
    except Exception as e:
        print(f"❌ Failed to save education report: {e}")

    # Also print to console
    print("\n===== Education content preview =====\n")
    print(education_content[:800] + ("\n...\n" if len(education_content) > 800 else ""))



if __name__ == "__main__":
    main()
