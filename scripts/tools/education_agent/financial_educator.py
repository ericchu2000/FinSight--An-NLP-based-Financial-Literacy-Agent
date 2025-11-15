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
from typing import Dict, Any, Optional

# 导入百炼客户端
from modules import BailianClient
# 导入输入处理器
from input_processor import InputProcessor
# 导入错误处理器
from error_handler import ErrorHandler
# 导入事实验证器
from factcheck import FinancialFactChecker

# 全局配置
PROMPT_TEMPLATE_PATH = "prompt_template.txt"
FINANCIAL_CONCEPTS_PATH = "financial_concepts.json"
DEFAULT_TEST_FILE = "inputnew.json"

# 使用环境变量或默认值获取API密钥
API_KEY = os.environ.get("BAILIAN_API_KEY", "sk-00c0fc19147d468cbb23f1c5ed46f0fb")

def load_prompt_template() -> str:
    """
    加载prompt模板
    
    Returns:
        str: prompt模板内容
    """
    try:
        with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
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
        with open(FINANCIAL_CONCEPTS_PATH, 'r', encoding='utf-8') as f:
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
    is_valid_structure, structure_message = ErrorHandler.validate_prediction_data_structure(data)
    if not is_valid_structure:
        print(structure_message)
        sys.exit(1)
    
    # 显示警告（如果有）
    if structure_message:
        print(f"\n\u8b66告: {structure_message}")
    
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

def generate_financial_education(client: BailianClient, prediction_data: Dict[str, Any], 
                               prompt_template: str, concepts_data: Dict[str, Any],
                               max_retries: int = 2, use_factcheck: bool = True) -> str:
    """
    生成金融教育内容
    
    Args:
        client: 百炼客户端
        prediction_data: 预测数据
        prompt_template: 提示词模板
        concepts_data: 金融概念数据
        max_retries: 最大重试次数
        use_factcheck: 是否使用事实验证功能
        
    Returns:
        str: 生成的金融教育内容
    """
    # 初始化输入处理器
    try:
        processor = InputProcessor(FINANCIAL_CONCEPTS_PATH)
        
        # 处理输入数据
        processed_data = processor.process_input(prediction_data)
        
        # 初始化并使用事实验证器（如果启用）
        factcheck_results = None
        if use_factcheck and os.path.exists("Financial_knowledge.jsonl"):
            try:
                fact_checker = FinancialFactChecker("Financial_knowledge.jsonl")
                # 验证supporting_facts中的事实
                if "supporting_facts" in prediction_data and isinstance(prediction_data["supporting_facts"], list):
                    factcheck_results = fact_checker.verify_facts(prediction_data["supporting_facts"])
                    print(f"完成了 {len(factcheck_results)} 条事实的验证")
                    
                # 获取与预测相关的金融知识
                relevant_knowledge = fact_checker.get_relevant_knowledge(processed_data)
                if relevant_knowledge:
                    print(f"找到 {len(relevant_knowledge)} 条相关金融知识")
                    
                    # 将相关金融知识添加到处理后的数据中
                    processed_data["relevant_knowledge"] = relevant_knowledge
            except Exception as e:
                print(f"事实验证过程中出现异常: {e}")
                # 即使事实验证失败，也继续生成内容
        
        # 获取需要解释的金融概念
        concepts_to_explain = processor.get_concepts_to_explain(processed_data)
        concept_info = "\n\n优先解释的金融概念\uff1a\n"
        for concept in concepts_to_explain:
            concept_info += f"- {concept['name']}\n"
        
        # 增加免责声明和提示词安全限制
        safety_instructions = "\n\n重要提示\uff1a\n1. 请仅关注金融教育内容\uff0c不要讨论其他话题\n2. 不要提供具体的投资建议\uff08如购买或出售特定股票\u3001仓位分配等\uff09\n3. 内容仅供教育参考\uff0c不构成投资建议"
        
        # 构建完整的提示词
        full_prompt = f"{prompt_template}{safety_instructions}\n\n以下是股票预测数据:\n{format_prediction_for_prompt(processed_data)}{concept_info}\n\n请根据上述数据生成金融教育内容。"
        
        # 实现API调用重试机制
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # 调用百炼 API
                result = client.call_qwen_plus(full_prompt)
                
                if result["success"]:
                    print(f"生成成功！消耗tokens: {result['total_tokens']}, 响应时间: {result['response_time']:.2f}秒")
                    
                    # 过滤可能的投资建议
                    content = filter_investment_advice(result["content"])
                    
                    # 添加免责声明
                    disclaimer = "\n\n---\n\n**免责声明**: 本内容仅供教育目的，不构成投资建议。投资决策应基于个人研究和专业建议。"
                    
                    return content + disclaimer
                else:
                    last_error = result["error"]
                    print(f"尝试 {retry_count + 1}/{max_retries + 1} 失败: {last_error}")
                    retry_count += 1
                    # 在重试前等待一个递增的时间
                    if retry_count <= max_retries:
                        wait_time = 2 ** retry_count  # 指数退避策略: 2, 4, 8秒...
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
        
        # 所有重试失败后的错误处理
        print(f"\n错误: 生成内容失败。所有重试都未成功。\n最后错误: {last_error}")
        
        # 返回一个错误信息页面
        return f"""对不起，生成金融教育内容时出现错误。

请尝试以下操作：
- 检查您的网络连接
- 确认API密钥有效
- 稍后再尝试运行程序

技术错误信息：{last_error}"""
        
    except Exception as e:
        print(f"\n处理错误: {str(e)}")
        return f"生成内容时发生内部错误: {str(e)}"

def filter_investment_advice(content: str) -> str:
    """
    过滤可能的投资建议
    
    Args:
        content: 生成的内容
        
    Returns:
        str: 过滤后的内容
    """
    # 这里实现一个简单的过滤器，对于复杂情况可以使用正则表达式或NLP工具
    
    # 可能标志直接投资建议的短语
    problematic_patterns = [
        "建议你现在购买", "建议你现在出售", "应该立即购入", "应该马上卖出",
        "现在是买入的好时机", "现在是卖出的好时机", "建议你持有", "建议你清仓",
        "你应该买入", "你应该卖出", "推荐购买", "推荐出售",
        "仓位应该", "仓位建议", "分配仓位", "应该配置"
    ]
    
    filtered_content = content
    for pattern in problematic_patterns:
        # 用更中性的表述替换可能的投资建议
        if pattern in filtered_content:
            replacement = "投资者可以考虑自己的风险承受能力和投资目标"
            filtered_content = filtered_content.replace(pattern, replacement)
    
    return filtered_content

def main():
    """主函数"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='金融教育内容生成器')
    parser.add_argument('-f', '--file', type=str, default=DEFAULT_TEST_FILE,
                        help=f'预测数据文件路径 (默认: {DEFAULT_TEST_FILE})')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出文件路径 (默认: 输出到控制台)')
    parser.add_argument('-s', '--show-processed', action='store_true',
                        help='显示处理后的数据结构')
    parser.add_argument('--factcheck', action='store_true', 
                        help='启用事实验证功能')
    parser.add_argument('--no-factcheck', action='store_true',
                        help='禁用事实验证功能')
    args = parser.parse_args()
    
    # 加载必要数据
    prompt_template = load_prompt_template()
    concepts_data = load_financial_concepts()
    raw_prediction_data = load_prediction_data(args.file)
    
    # 初始化输入处理器和百炼客户端
    processor = InputProcessor(FINANCIAL_CONCEPTS_PATH)
    client = BailianClient(API_KEY)
    
    # 处理输入数据
    processed_data = processor.process_input(raw_prediction_data)
    
    # 显示处理后的数据（如果需要）
    if args.show_processed:
        print("\n===== 处理后的预测数据 =====\n")
        print(json.dumps(processed_data, ensure_ascii=False, indent=2))
        
        # 显示需要解释的概念
        concepts_to_explain = processor.get_concepts_to_explain(processed_data)
        print("\n===== 需要解释的金融概念 =====\n")
        for concept in concepts_to_explain:
            print(f"- {concept['name']}")
            if "definition" in concept:
                print(f"  定义: {concept['definition']}")
            if "analogy" in concept:
                print(f"  类比: {concept['analogy']}")
        print()
    
    # 决定是否使用事实验证功能
    use_factcheck = True
    if args.no_factcheck:
        use_factcheck = False
    elif args.factcheck:
        use_factcheck = True
        
    # 生成金融教育内容
    education_content = generate_financial_education(client, processed_data, prompt_template, concepts_data, use_factcheck=use_factcheck)
    
    # 输出结果
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(education_content)
            print(f"结果已保存到: {args.output}")
        except Exception as e:
            print(f"保存结果失败: {e}")
    else:
        print("\n===== 生成的金融教育内容 =====\n")
        print(education_content)

if __name__ == "__main__":
    main()
