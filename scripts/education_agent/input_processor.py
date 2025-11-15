#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
股票预测数据输入解析器
处理多种格式的预测数据并识别关键金融指标
"""

import json
import os
from typing import Dict, Any, List, Tuple, Optional

class InputProcessor:
    """
    预测数据输入处理类
    """
    
    def __init__(self, financial_concepts_path: str = "financial_concepts.json"):
        """
        初始化输入处理器
        
        Args:
            financial_concepts_path: 金融概念数据文件路径
        """
        self.financial_concepts = self._load_financial_concepts(financial_concepts_path)
        # 创建概念名称映射表，用于快速查找
        self.concept_names = {concept["name"].lower(): concept for concept in self.financial_concepts.get("concepts", [])}
        # 增加常见别名映射
        self._add_concept_aliases()
        
    def _load_financial_concepts(self, path: str) -> Dict[str, Any]:
        """
        加载金融概念数据
        
        Args:
            path: 数据文件路径
            
        Returns:
            Dict: 金融概念数据
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告: 加载金融概念数据失败: {e}")
            return {"concepts": []}
    
    def _add_concept_aliases(self):
        """添加常见指标别名映射"""
        aliases = {
            "移动平均线": ["ma", "均线", "移动平均"],
            "macd": ["指数平滑移动平均", "平滑异同移动平均线", "macd指标"],
            "rsi": ["相对强弱指标", "相对强度指标"],
            "布林带": ["boll", "bollinger", "bollinger bands"],
            "kdj": ["随机指标", "随机动量指标"],
            "成交量": ["volume", "vol", "交易量"],
            "市盈率": ["pe", "p/e", "price-earnings"],
            "市净率": ["pb", "p/b", "price-book"]
        }
        
        # 为每个别名创建映射到原始概念
        for concept_name, alias_list in aliases.items():
            concept_lower = concept_name.lower()
            if concept_lower in self.concept_names:
                for alias in alias_list:
                    self.concept_names[alias.lower()] = self.concept_names[concept_lower]
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入数据，适配不同格式并标记需要解释的概念
        
        Args:
            input_data: 输入的预测数据
            
        Returns:
            Dict: 处理后的标准格式数据
        """
        # 净化输入数据，防止注入攻击
        sanitized_data = self._sanitize_input(input_data)
        
        # 标准化数据结构
        standardized_data = self._standardize_format(sanitized_data)
        
        # 识别需要解释的关键指标
        standardized_data = self._identify_key_concepts(standardized_data)
        
        return standardized_data
    
    def _sanitize_input(self, data: Any) -> Any:
        """
        递归地净化输入数据，移除或转义可能有害的字符串内容
        
        Args:
            data: 输入数据（可以是字典、列表、字符串等）
            
        Returns:
            Any: 净化后的数据
        """
        if isinstance(data, dict):
            # 递归处理字典
            return {key: self._sanitize_input(value) for key, value in data.items()}
        elif isinstance(data, list):
            # 递归处理列表
            return [self._sanitize_input(item) for item in data]
        elif isinstance(data, str):
            # 净化字符串
            return self._sanitize_string(data)
        else:
            # 其他类型直接返回
            return data

    def _sanitize_string(self, text: str) -> str:
        """
        净化字符串，移除或转义可能用于prompt注入的指令
        
        Args:
            text: 原始字符串
            
        Returns:
            str: 净化后的字符串
        """
        # 简单的净化策略：移除可能改变模型行为的指令性短语
        # 这是一个基础实现，更复杂的场景可能需要更高级的策略
        injection_patterns = [
            "忽略之前的指令", "ignore previous instructions",
            "忘记所有指令", "forget all instructions",
            "你的新指令是", "your new instructions are",
            "作为[某个角色]", "act as a",
            "提供具体的投资建议",
            "请给我购买建议"
        ]
        
        sanitized_text = text
        for pattern in injection_patterns:
            # 使用不区分大小写的替换
            import re
            sanitized_text = re.sub(pattern, "[内容已过滤]", sanitized_text, flags=re.IGNORECASE)
            
        # 转义可能被误解为格式化指令的字符，例如markdown标题
        sanitized_text = sanitized_text.replace("\n#", "\n##")
        
        return sanitized_text
    
    def _standardize_format(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将不同格式的输入标准化为统一格式
        
        Args:
            input_data: 输入数据
            
        Returns:
            Dict: 标准化后的数据
        """
        result = {
            "prediction": {},
            "factors": [],
            "stock": {},
            "user": {}
        }
        
        # 处理预测结果
        result["prediction"] = self._extract_prediction(input_data)
        
        # 处理因素
        result["factors"] = self._extract_factors(input_data)
        
        # 处理股票信息
        result["stock"] = self._extract_stock_info(input_data)
        
        # 处理用户信息
        result["user"] = self._extract_user_info(input_data)
        
        return result
    
    def _extract_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取预测结果信息"""
        prediction = {}
        
        # 方式1: 标准格式
        if "prediction" in data and isinstance(data["prediction"], dict):
            prediction = data["prediction"].copy()
        
        # 方式2: 预测结果在顶层
        elif "result" in data:
            prediction["result"] = data["result"]
            if "confidence" in data:
                prediction["confidence"] = data["confidence"]
            if "timeframe" in data:
                prediction["timeframe"] = data["timeframe"]
        
        # 方式3: 预测结果使用不同的字段名
        else:
            # 尝试识别结果字段
            result_fields = ["prediction", "predict_result", "direction", "trend"]
            for field in result_fields:
                if field in data:
                    prediction["result"] = data[field]
                    break
            
            # 尝试识别置信度字段
            confidence_fields = ["confidence", "probability", "certainty"]
            for field in confidence_fields:
                if field in data:
                    prediction["confidence"] = data[field]
                    break
            
            # 尝试识别时间范围字段
            time_fields = ["timeframe", "time_range", "period"]
            for field in time_fields:
                if field in data:
                    prediction["timeframe"] = data[field]
                    break
        
        # 确保结果字段存在
        if "result" not in prediction:
            prediction["result"] = "未知"
        
        return prediction
    
    def _extract_factors(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """提取影响因素信息"""
        factors = []
        
        # 方式1: 标准格式的因素列表
        if "factors" in data and isinstance(data["factors"], list):
            factors = data["factors"].copy()
        
        # 方式2: 使用不同的字段名
        elif any(key in data for key in ["indicators", "analysis_factors", "reasons"]):
            factors_key = next((k for k in ["indicators", "analysis_factors", "reasons"] if k in data), None)
            if factors_key and isinstance(data[factors_key], list):
                # 转换为标准格式
                for factor in data[factors_key]:
                    if isinstance(factor, dict):
                        std_factor = {}
                        
                        # 映射字段名
                        name_fields = ["name", "indicator", "factor"]
                        for field in name_fields:
                            if field in factor:
                                std_factor["name"] = factor[field]
                                break
                        
                        value_fields = ["value", "reading", "data"]
                        for field in value_fields:
                            if field in factor:
                                std_factor["value"] = factor[field]
                                break
                        
                        impact_fields = ["impact", "effect", "influence"]
                        for field in impact_fields:
                            if field in factor:
                                std_factor["impact"] = factor[field]
                                break
                        
                        weight_fields = ["weight", "importance"]
                        for field in weight_fields:
                            if field in factor:
                                std_factor["weight"] = factor[field]
                                break
                        
                        if "name" in std_factor:
                            factors.append(std_factor)
                    elif isinstance(factor, str):
                        # 简单字符串因素
                        factors.append({"name": factor, "value": ""})
        
        # 方式3: 每个因素作为单独的顶层字段
        else:
            # 检查常见的技术指标名称
            common_indicators = ["macd", "rsi", "kdj", "ma", "volume", "boll"]
            for indicator in common_indicators:
                if indicator in data:
                    factors.append({
                        "name": indicator.upper(),
                        "value": str(data[indicator])
                    })

        # 方式4: 处理来自 model_prediction 和 supporting_facts 的特定字段
        # 处理 supporting_facts
        if "supporting_facts" in data and isinstance(data["supporting_facts"], list):
            for fact_string in data["supporting_facts"]:
                if isinstance(fact_string, str):
                    # 尝试从字符串中解析出指标名称
                    parts = fact_string.replace(':', ' ').replace('>', ' ').replace('<', ' ').split()
                    factor_name = parts[0]  # 默认使用第一个词作为名称
                    for part in parts:
                        # 如果找到一个已知的概念或别名，则用它作为名称
                        if part.lower() in self.concept_names:
                            factor_name = self.concept_names[part.lower()]['name']
                            break
                    factors.append({"name": factor_name, "value": fact_string})

        # 使用映射处理 model_prediction 内部字段和其他顶层字段
        field_mappings = {
            ("model_prediction", "direction"): "模型预测方向",
            ("model_prediction", "confidence"): "模型预测置信度",
            ("model_prediction", "reason"): "模型预测原因",
            ("explanation_short",): "简短解释",
            ("teaching_note",): "教学笔记"
        }

        for key_path, target_name in field_mappings.items():
            value = data
            try:
                for key in key_path:
                    value = value[key]
                factors.append({"name": target_name, "value": str(value)})
            except (KeyError, TypeError):
                continue
        
        return factors
    
    def _extract_stock_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取股票信息"""
        stock_info = {}
        
        # 方式1: 标准格式
        if "stock" in data and isinstance(data["stock"], dict):
            stock_info = data["stock"].copy()
        
        # 方式2: 股票信息在顶层
        else:
            # 尝试识别股票名称
            name_fields = ["stock_name", "company", "name"]
            for field in name_fields:
                if field in data:
                    stock_info["name"] = data[field]
                    break
            
            # 尝试识别股票代码
            code_fields = ["stock_code", "code", "symbol", "ticker"]
            for field in code_fields:
                if field in data:
                    stock_info["code"] = data[field]
                    break
            
            # 尝试识别行业
            industry_fields = ["industry", "sector"]
            for field in industry_fields:
                if field in data:
                    stock_info["industry"] = data[field]
                    break
        
        return stock_info
    
    def _extract_user_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取用户信息"""
        user_info = {}
        
        # 方式1: 标准格式
        if "user" in data and isinstance(data["user"], dict):
            user_info = data["user"].copy()
        
        # 方式2: 用户信息在顶层
        else:
            # 尝试识别用户预测
            prediction_fields = ["user_prediction", "user_direction"]
            for field in prediction_fields:
                if field in data:
                    user_info["prediction"] = data[field]
                    break
            
            # 尝试识别用户经验级别
            experience_fields = ["user_experience", "experience", "level"]
            for field in experience_fields:
                if field in data:
                    user_info["experience"] = data[field]
                    break
        
        return user_info
    
    def _identify_key_concepts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        识别需要解释的关键金融概念
        
        Args:
            data: 标准化后的数据
            
        Returns:
            Dict: 添加了关键概念标记的数据
        """
        # 复制数据，避免修改原始数据
        result = data.copy()
        
        # 为因素添加解释标记
        if "factors" in result and isinstance(result["factors"], list):
            for factor in result["factors"]:
                if "name" in factor:
                    factor_name = factor["name"].lower()
                    
                    # 检查是否是已知的金融概念
                    if factor_name in self.concept_names:
                        # 添加需要解释的标记
                        factor["needs_explanation"] = True
                        
                        # 添加概念信息（定义和类比）
                        concept = self.concept_names[factor_name]
                        if "definition" in concept:
                            factor["concept_definition"] = concept["definition"]
                        if "analogy" in concept:
                            factor["concept_analogy"] = concept["analogy"]
                    else:
                        factor["needs_explanation"] = False
        
        return result
    
    def get_concepts_to_explain(self, data: Dict[str, Any], max_concepts: int = 3) -> List[Dict[str, Any]]:
        """
        获取需要优先解释的概念列表
        
        Args:
            data: 处理过的预测数据
            max_concepts: 最多返回的概念数量
            
        Returns:
            List: 需要解释的概念列表
        """
        concepts_to_explain = []
        
        # 从因素中收集需要解释的概念
        if "factors" in data and isinstance(data["factors"], list):
            # 先按权重（如果有）排序因素
            sorted_factors = sorted(
                data["factors"], 
                key=lambda x: float(x.get("weight", 0)) if "weight" in x and x["weight"] is not None else 0,
                reverse=True
            )
            
            # 收集需要解释的概念
            for factor in sorted_factors:
                if factor.get("needs_explanation", False):
                    concept_info = {
                        "name": factor["name"],
                        "context": factor.get("value", ""),
                    }
                    
                    # 添加定义和类比（如果有）
                    if "concept_definition" in factor:
                        concept_info["definition"] = factor["concept_definition"]
                    if "concept_analogy" in factor:
                        concept_info["analogy"] = factor["concept_analogy"]
                    
                    concepts_to_explain.append(concept_info)
                    
                    # 达到最大数量后停止
                    if len(concepts_to_explain) >= max_concepts:
                        break
        
        return concepts_to_explain

# 测试代码
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python input_processor.py <预测数据JSON文件>")
        sys.exit(1)
    
    # 加载测试数据
    try:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        sys.exit(1)
    
    # 处理数据
    processor = InputProcessor()
    processed_data = processor.process_input(test_data)
    
    # 获取需要解释的概念
    concepts_to_explain = processor.get_concepts_to_explain(processed_data)
    
    # 显示结果
    print("\n处理后的标准格式数据:")
    print(json.dumps(processed_data, ensure_ascii=False, indent=2))
    
    print("\n需要解释的概念:")
    for concept in concepts_to_explain:
        print(f"- {concept['name']}")
        if "definition" in concept:
            print(f"  定义: {concept['definition']}")
        if "analogy" in concept:
            print(f"  类比: {concept['analogy']}")
