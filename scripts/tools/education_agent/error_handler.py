#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
错误处理模块
提供各种错误检测、处理和用户友好提示功能
"""

import json
import os
import sys
from typing import Dict, Any, Optional, Tuple

class ErrorHandler:
    """错误处理类，提供各种错误检测和处理功能"""
    
    @staticmethod
    def validate_json_file(file_path: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        验证JSON文件的格式和内容
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            Tuple: (是否有效, 解析后的数据(如果有效), 错误信息(如果无效))
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return False, None, f"错误：文件 '{file_path}' 不存在。请检查文件路径是否正确。"
        
        # 检查是否有读取权限
        if not os.access(file_path, os.R_OK):
            return False, None, f"错误：没有权限读取文件 '{file_path}'。请检查文件权限。"
        
        # 尝试读取和解析JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    return True, data, ""
                except json.JSONDecodeError as e:
                    # 提供详细的解析错误信息
                    line_col = f"第 {e.lineno} 行, 第 {e.colno} 列"
                    error_context = ErrorHandler._get_error_context(file_path, e.lineno)
                    
                    error_message = f"""
JSON解析错误: {e.msg} 在 {line_col}
问题所在行: {error_context}

常见的JSON错误包括:
- 缺少逗号或多余的逗号
- 未闭合的引号、括号或大括号
- 属性名称没有使用双引号
- 使用了单引号而不是双引号
- 数组或对象末尾有多余的逗号

建议使用在线JSON验证工具如 jsonlint.com 检查您的JSON文件。
"""
                    return False, None, error_message
        except UnicodeDecodeError:
            # 处理编码问题
            return False, None, f"错误：文件 '{file_path}' 编码不是UTF-8。请确保文件使用UTF-8编码。"
        except Exception as e:
            # 处理其他可能的文件读取错误
            return False, None, f"错误：读取文件时出现未知错误: {str(e)}"
    
    @staticmethod
    def _get_error_context(file_path: str, error_line: int, context_lines: int = 1) -> str:
        """获取错误行及其上下文"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            start = max(0, error_line - context_lines - 1)
            end = min(len(lines), error_line + context_lines)
            
            context = ""
            for i in range(start, end):
                line_prefix = "> " if i == error_line - 1 else "  "
                context += f"{line_prefix}{i+1}: {lines[i].rstrip()}\n"
            
            return context.strip()
        except Exception:
            return "无法获取上下文"
    
    @staticmethod
    def validate_prediction_data_structure(data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证预测数据是否包含必要的字段和结构
        
        Args:
            data: 解析后的JSON数据
            
        Returns:
            Tuple: (是否有效, 错误信息(如果无效))
        """
        errors = []
        warnings = []
        
        # 检查必要字段 - 直接或通过InputProcessor可以识别的
        prediction_result_found = False
        
        # 检查标准格式的预测结果
        if "prediction" in data:
            if isinstance(data["prediction"], dict):
                if "result" in data["prediction"]:
                    prediction_result_found = True
            else:
                errors.append("'prediction' 字段应该是一个对象，而不是 " + type(data["prediction"]).__name__)
        
        # 检查直接在顶层的预测结果
        result_fields = ["result", "predict_result", "direction", "trend"]
        if any(field in data for field in result_fields):
            prediction_result_found = True
        
        if not prediction_result_found:
            errors.append("缺少预测结果。应包含 'prediction.result' 或顶层的 'result'/'predict_result'/'direction'/'trend' 中的一个")
        
        # 检查用户预测
        user_prediction_found = False
        if "user" in data and isinstance(data["user"], dict) and "prediction" in data["user"]:
            user_prediction_found = True
        
        # 检查直接在顶层的用户预测
        user_fields = ["user_prediction", "user_direction"]
        if any(field in data for field in user_fields):
            user_prediction_found = True
        
        if not user_prediction_found:
            errors.append("缺少用户预测。应包含 'user.prediction' 或顶层的 'user_prediction'/'user_direction' 中的一个")
        
        # 检查因素/指标
        factors_found = False
        
        if "factors" in data and isinstance(data["factors"], list) and len(data["factors"]) > 0:
            factors_found = True
        
        # 检查其他常见名称
        factor_fields = ["indicators", "analysis_factors", "reasons"]
        if any(field in data and isinstance(data[field], list) and len(data[field]) > 0 for field in factor_fields):
            factors_found = True
        
        # 检查常见的单独指标
        common_indicators = ["macd", "rsi", "kdj", "ma", "volume", "boll"]
        if any(indicator.lower() in map(str.lower, data.keys()) for indicator in common_indicators):
            factors_found = True
        
        if not factors_found:
            warnings.append("没有发现分析因素或指标。建议包含 'factors' 或 'indicators' 数组，或包含常见技术指标如 'MACD', 'RSI' 等")
        
        # 生成结果信息
        if errors:
            error_message = "JSON结构不完整，缺少以下必要字段:\n"
            for error in errors:
                error_message += f"- {error}\n"
            
            # 添加最小必要结构示例
            error_message += "\n最小必要JSON结构示例:\n"
            error_message += """
{
  "prediction": {
    "result": "上涨"  // 或 "下跌"
  },
  "user": {
    "prediction": "下跌"  // 或 "上涨"
  },
  "factors": [
    {
      "name": "指标名称",
      "value": "指标值"
    }
  ]
}

或者扁平结构:

{
  "result": "上涨",
  "user_prediction": "下跌",
  "indicators": [
    {
      "indicator": "MACD",
      "reading": "金叉形成"
    }
  ]
}
"""
            return False, error_message
        
        # 如果有警告但没有错误
        if warnings:
            warning_message = "警告(数据仍可使用):\n"
            for warning in warnings:
                warning_message += f"- {warning}\n"
            return True, warning_message
        
        return True, ""


# 测试代码
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python error_handler.py <预测数据JSON文件>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # 验证JSON文件格式
    is_valid_json, data, json_error = ErrorHandler.validate_json_file(file_path)
    if not is_valid_json:
        print(json_error)
        sys.exit(1)
    
    print(f"JSON文件格式有效: {file_path}\n")
    
    # 验证数据结构
    is_valid_structure, structure_message = ErrorHandler.validate_prediction_data_structure(data)
    if not is_valid_structure:
        print(structure_message)
        sys.exit(1)
    
    if structure_message:
        print(structure_message)
    else:
        print("数据结构验证通过，包含所有必要字段。")
