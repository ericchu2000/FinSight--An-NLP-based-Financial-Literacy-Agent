#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
金融知识检索和事实验证模块
用于检索金融知识库并验证模型输出的事实准确性
"""

import json
import re
import os
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

class FinancialFactChecker:
    """金融知识检索和事实验证类"""
    
    def __init__(self, knowledge_path: str = "Financial_knowledge.jsonl"):
        """
        初始化金融知识检索器
        
        Args:
            knowledge_path: 知识库文件路径
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_path)
        # 创建索引用于快速检索
        self.title_index = self._build_title_index()
        self.content_index = self._build_content_index()
        
    def _load_knowledge_base(self, path: str) -> List[Dict[str, Any]]:
        """
        加载金融知识库
        
        Args:
            path: 知识库文件路径
            
        Returns:
            List[Dict]: 知识条目列表
        """
        knowledge_items = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            knowledge_items.append(item)
                        except json.JSONDecodeError:
                            print(f"警告: 跳过无效的JSON行: {line[:50]}...")
        except Exception as e:
            print(f"警告: 加载知识库失败: {e}")
        
        print(f"成功加载了 {len(knowledge_items)} 条金融知识")
        return knowledge_items
    
    def _build_title_index(self) -> Dict[str, List[int]]:
        """构建标题索引"""
        index = defaultdict(list)
        for i, item in enumerate(self.knowledge_base):
            if "title" in item:
                # 将标题分词并建立倒排索引
                words = self._tokenize(item["title"])
                for word in words:
                    index[word].append(i)
        return index
    
    def _build_content_index(self) -> Dict[str, List[int]]:
        """构建内容索引"""
        index = defaultdict(list)
        for i, item in enumerate(self.knowledge_base):
            # 索引定义和示例
            for field in ["definition", "example"]:
                if field in item:
                    words = self._tokenize(item[field])
                    for word in words:
                        index[word].append(i)
        return index
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词，提取关键词"""
        if not text:
            return []
        
        # 转为小写，移除标点符号，拆分单词
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # 移除停用词（简单版本）
        stop_words = {"the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "for", "with", "on", "at", "by", "is", "are"}
        return [token for token in tokens if token not in stop_words]
    
    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相关金融知识
        
        Args:
            query: 搜索查询
            top_k: 返回的最大结果数量
            
        Returns:
            List[Dict]: 相关知识条目
        """
        if not query:
            return []
        
        # 分词
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # 统计每个知识条目的匹配分数
        scores = defaultdict(float)
        
        # 标题匹配分数更高
        for token in query_tokens:
            for idx in self.title_index.get(token, []):
                scores[idx] += 2.0  # 标题匹配权重高
        
        # 内容匹配
        for token in query_tokens:
            for idx in self.content_index.get(token, []):
                scores[idx] += 1.0
        
        # 排序并返回前K个结果
        if not scores:
            return []
            
        top_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        results = [self.knowledge_base[idx] for idx in top_indices]
        
        return results
    
    def verify_facts(self, facts: List[str]) -> List[Dict[str, Any]]:
        """
        验证事实陈述
        
        Args:
            facts: 需要验证的事实陈述列表
            
        Returns:
            List[Dict]: 验证结果和相关知识
        """
        results = []
        for fact in facts:
            verification = self._verify_single_fact(fact)
            results.append(verification)
        return results
    
    def _verify_single_fact(self, fact: str) -> Dict[str, Any]:
        """验证单条事实"""
        # 搜索相关知识
        related_knowledge = self.search_knowledge(fact, top_k=3)
        
        # 构建验证结果
        result = {
            "fact": fact,
            "related_knowledge": related_knowledge,
            "has_related_info": len(related_knowledge) > 0
        }
        
        return result
    
    def get_relevant_knowledge(self, processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        获取与预测数据相关的金融知识
        
        Args:
            processed_data: 处理后的预测数据
            
        Returns:
            List[Dict]: 相关的知识条目
        """
        relevant_items = []
        
        # 从因素中提取查询关键词
        queries = []
        if "factors" in processed_data and isinstance(processed_data["factors"], list):
            for factor in processed_data["factors"]:
                if "name" in factor:
                    queries.append(factor["name"])
                if "value" in factor and isinstance(factor["value"], str):
                    queries.append(factor["value"])
        
        # 从预测中提取查询关键词
        if "prediction" in processed_data and isinstance(processed_data["prediction"], dict):
            if "result" in processed_data["prediction"]:
                queries.append(f"stock {processed_data['prediction']['result']}")
            if "reason" in processed_data["prediction"]:
                queries.append(processed_data["prediction"]["reason"])
        
        # 从supporting_facts中提取查询关键词
        if "supporting_facts" in processed_data and isinstance(processed_data["supporting_facts"], list):
            queries.extend(processed_data["supporting_facts"])
        
        # 搜索每个查询的相关知识
        unique_items = {}  # 使用字典去重
        for query in queries:
            items = self.search_knowledge(query, top_k=2)
            for item in items:
                if "id" in item:
                    unique_items[item["id"]] = item
        
        # 转换为列表
        relevant_items = list(unique_items.values())
        
        # 限制返回的条目数量，避免过多
        return relevant_items[:10]


# 测试代码
if __name__ == "__main__":
    import sys
    
    checker = FinancialFactChecker()
    
    # 测试知识搜索
    print("\n=== 测试知识搜索 ===")
    test_query = "P/E ratio valuation stocks"
    results = checker.search_knowledge(test_query)
    for i, result in enumerate(results):
        print(f"{i+1}. {result.get('title')}: {result.get('definition')[:100]}...")
    
    # 测试事实验证
    print("\n=== 测试事实验证 ===")
    test_facts = [
        "A higher P/E ratio indicates investors expect higher earnings growth.",
        "RSI above 70 is considered overbought territory.",
        "MACD crossing above signal line is a bullish indicator."
    ]
    verification_results = checker.verify_facts(test_facts)
    for i, result in enumerate(verification_results):
        print(f"\n事实 {i+1}: {result['fact']}")
        print(f"找到相关信息: {result['has_related_info']}")
        if result['has_related_info']:
            for j, knowledge in enumerate(result['related_knowledge']):
                print(f"  - {knowledge.get('title')}: {knowledge.get('definition')[:100]}...")
