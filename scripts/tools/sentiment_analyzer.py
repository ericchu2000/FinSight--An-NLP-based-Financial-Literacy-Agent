import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import re

# 尝试导入 SnowNLP
try:
    from snownlp import SnowNLP  # type: ignore
    SNOWNLP_AVAILABLE = True
except ImportError:
    SNOWNLP_AVAILABLE = False
    SnowNLP = None  # type: ignore

# 基础情感词典（手动定义）
SENTIMENT_DICT = {
    "positive": {
        "strong": [
            "涨", "上升", "增长", "增加", "提高", "上涨", "上扬", "升高",
            "创历史新高", "历史新高", "利好", "好", "优秀", "优良", "强势",
            "强劲", "稳健", "稳定", "安全", "收益", "盈利", "增盈", "增利",
            "突破", "突进", "创新高", "新高", "繁荣", "兴旺", "蓬勃", "向上",
            "看好", "看涨", "乐观", "信心", "利润", "增效", "增值", "升值",
            "改善", "改进", "优化", "完善", "加强", "强化", "提升", "升级",
            "成交", "成交额", "环比增加", "环比增", "放量", "量增",
            "平均上涨", "平均上升", "坚挺", "反弹", "企稳"
        ],
        "weak": [
            "小涨", "微涨", "略升", "稍增", "温和上升", "缓慢增长",
            "平稳", "稳中有升", "保持", "坚挺", "反弹", "企稳",
            "可观", "不错", "良好", "正面", "积极", "健康", "领先"
        ]
    },
    "negative": {
        "strong": [
            "跌", "下跌", "下降", "减少", "下滑", "衰退", "衰落", "亏损",
            "亏", "损失", "掉", "暴跌", "巨亏", "崩溃", "崩盘", "废墟",
            "利空", "坏", "差", "糟糕", "恶劣", "严峻", "困难", "困境",
            "衰弱", "虚弱", "脆弱", "危险", "风险", "危机", "警告", "告急",
            "看空", "看跌", "悲观", "困扰", "担忧", "忧虑", "不安", "失望",
            "打击", "打压", "压低", "跳水", "跳空", "断头", "砍", "割肉",
            "平均下跌", "平均下降"
        ],
        "weak": [
            "小跌", "微跌", "略降", "稍减", "温和下跌", "缓慢下滑",
            "震荡", "波动", "调整", "回调", "下行", "承压",
            "遇冷", "冷淡", "平淡", "乏力", "疲弱", "谨慎"
        ]
    },
    "negation": [
        "不", "没有", "无", "非", "并非", "绝非", "并不",
        "莫", "勿", "勿论", "毋庸", "未能", "未曾", "未有"
    ],
    "transition": [
        "但是", "但", "然而", "不过", "虽然", "尽管", "虽有", "即使"
    ]
}

# 股票市场特有词汇扩展（金融领域专用）
FINANCE_CUSTOM_WORDS = {
    "positive": {
        "strong": [
            "破位上行", "收盘上涨", "高开", "低开高走", "向上突破",
            "成交量放大", "成交活跃", "换手率上升", "龙头股", "涨停",
            "利好政策", "业绩向好", "融资净买入", "主力买入", "底部信号",
            "技术面强势", "形成底部", "突破压力位", "均线向上", "金叉"
        ],
        "weak": [
            "缓慢上涨", "低位盘整", "温和反弹", "成交温和", "量能温和",
            "盘整向上", "小幅上升"
        ]
    },
    "negative": {
        "strong": [
            "破位下行", "收盘下跌", "开盘跳水", "高开低走", "向下跌破",
            "成交量萎缩", "成交清淡", "换手率下降", "跌幅扩大", "跌停",
            "利空政策", "业绩变脸", "融资净卖出", "主力出货", "顶部信号",
            "技术面弱势", "形成顶部", "跌破支撑位", "均线向下", "死叉",
            "股权质押", "减持", "暴雷", "监管约谈"
        ],
        "weak": [
            "缓慢下跌", "高位盘整", "温和回调", "成交萎靡", "量能不足",
            "盘整向下", "小幅下降"
        ]
    }
}


class SentimentAnalyzer:
    """基于词典法和SnowNLP的中文股票新闻情绪分析器"""
    
    def __init__(self, use_snownlp: bool = False, add_finance_words: bool = True):
        """
        初始化情绪分析器
        
        Args:
            use_snownlp: 是否使用SnowNLP混合方案
            add_finance_words: 是否添加金融领域专用词汇扩展
        """
        self.use_snownlp = use_snownlp and SNOWNLP_AVAILABLE
        self.window_size = 5
        
        # 构建完整的词典
        self.sentiment_dict = SENTIMENT_DICT.copy()
        
        # 如果启用金融词汇扩展，则合并到基础词典
        if add_finance_words:
            for sentiment_type in ["positive", "negative"]:
                for strength in ["strong", "weak"]:
                    self.sentiment_dict[sentiment_type][strength].extend(
                        FINANCE_CUSTOM_WORDS[sentiment_type][strength]
                    )
        
        self._build_word_index()
    
    def _build_word_index(self):
        """构建高效的词汇查找索引"""
        self.word_index = {}
        
        for sentiment_type in ["positive", "negative"]:
            for strength in ["strong", "weak"]:
                for word in self.sentiment_dict[sentiment_type][strength]:
                    weight = 1.0 if strength == "strong" else 0.6
                    weight = weight if sentiment_type == "positive" else -weight
                    self.word_index[word] = (sentiment_type, strength, weight)
    
    def _tokenize(self, text: str) -> List[str]:
        """改进的中文分词方法"""
        text = re.sub(r'[^\u4e00-\u9fff\w\uff0c\u3002\uff01\uff1f\uff1b\uff1a\s%]', '', text)
        sentences = re.split(r'[\uff0c\u3002\uff01\uff1f\uff1b\uff1a]', text)
        
        words = []
        for sentence in sentences:
            remaining = sentence.strip()
            while remaining:
                matched = False
                for length in range(min(4, len(remaining)), 0, -1):
                    candidate = remaining[:length]
                    if candidate in self.word_index:
                        words.append(candidate)
                        remaining = remaining[length:]
                        matched = True
                        break
                    elif candidate in self.sentiment_dict.get("negation", []):
                        words.append(candidate)
                        remaining = remaining[length:]
                        matched = True
                        break
                
                if not matched:
                    if remaining[0].strip():
                        words.append(remaining[0])
                    remaining = remaining[1:]
        
        return [w for w in words if w.strip()]
    
    def _find_sentiment_words(self, text: str) -> Tuple[List[Dict], float]:
        """识别文本中的情感词汇"""
        words = self._tokenize(text)
        sentiment_words = []
        total_score = 0.0
        
        for i, word in enumerate(words):
            if word not in self.word_index:
                continue
                
            sentiment_type, strength, weight = self.word_index[word]
            
            # 检查是否受到否定词影响
            is_negated = False
            for j in range(max(0, i - self.window_size), i):
                if words[j] in self.sentiment_dict["negation"]:
                    is_negated = True
                    break
            
            if is_negated:
                weight = -weight
            
            sentiment_words.append({
                "word": word,
                "type": sentiment_type,
                "strength": strength,
                "weight": weight,
                "position": i,
                "negated": is_negated
            })
            total_score += weight
        
        return sentiment_words, total_score
    
    def _score_to_sentiment(self, score: float) -> Tuple[str, float]:
        """将得分转换为情绪标签和标准化评分"""
        normalized_score = (score / (abs(score) + 1) + 1) / 2 if score != 0 else 0.5
        
        if normalized_score > 0.6:
            return "positive", normalized_score
        elif normalized_score < 0.4:
            return "negative", normalized_score
        else:
            return "neutral", normalized_score
    
    def _calculate_confidence(
        self,
        matched_words_count: int,
        total_words_count: int,
        sentiment_score: float,
        consistency: float = 1.0
    ) -> float:
        """计算基于词典法的置信度"""
        match_ratio = min(
            matched_words_count / max(total_words_count, 1),
            1.0
        )
        intensity = min(abs(sentiment_score - 0.5) * 2, 1.0)
        confidence = (match_ratio * 0.4 + consistency * 0.4 + intensity * 0.2)
        return round(confidence, 2)
    
    def analyze_text(self, text: str, text_type: str = "content") -> Dict[str, Any]:
        """分析单个文本的情绪"""
        
        if self.use_snownlp:
            return self._analyze_text_with_snownlp(text, text_type)
        else:
            return self._analyze_text_with_dict(text, text_type)
    
    def _analyze_text_with_dict(self, text: str, text_type: str) -> Dict[str, Any]:
        """使用词典法分析文本"""
        words = self._tokenize(text)
        sentiment_words, total_score = self._find_sentiment_words(text)
        
        sentiment_label, normalized_score = self._score_to_sentiment(total_score)
        
        confidence = self._calculate_confidence(
            matched_words_count=len(sentiment_words),
            total_words_count=len(words),
            sentiment_score=normalized_score
        )
        
        return {
            "text_type": text_type,
            "sentiment_label": sentiment_label,
            "sentiment_score": round(normalized_score, 2),
            "confidence": confidence,
            "matched_words_count": len(sentiment_words),
            "total_words_count": len(words),
            "matched_words": [w["word"] for w in sentiment_words],
            "detailed_words": sentiment_words,
            "method": "dictionary_based"
        }
    
    def _analyze_text_with_snownlp(self, text: str, text_type: str) -> Dict[str, Any]:
        """使用SnowNLP + 自定义词典的混合方案分析"""
        
        if not SNOWNLP_AVAILABLE or SnowNLP is None:
            return self._analyze_text_with_dict(text, text_type)
        
        try:
            s = SnowNLP(text)  # type: ignore
            snownlp_sentiment = s.sentiments
        except Exception as e:
            print(f"SnowNLP分析出错: {e}，回退到词典法")
            return self._analyze_text_with_dict(text, text_type)
        
        # 再用自定义词典补充分析
        words = self._tokenize(text)
        sentiment_words, dict_total_score = self._find_sentiment_words(text)
        
        dict_label, dict_normalized_score = self._score_to_sentiment(dict_total_score)
        
        # SnowNLP 60%权重，自定义词典40%权重
        combined_score = snownlp_sentiment * 0.6 + dict_normalized_score * 0.4
        final_label, _ = self._score_to_sentiment(combined_score - 0.5)
        
        confidence = self._calculate_confidence(
            matched_words_count=len(sentiment_words),
            total_words_count=len(words),
            sentiment_score=combined_score
        )
        
        return {
            "text_type": text_type,
            "sentiment_label": final_label,
            "sentiment_score": round(combined_score, 2),
            "confidence": confidence,
            "matched_words_count": len(sentiment_words),
            "total_words_count": len(words),
            "matched_words": [w["word"] for w in sentiment_words],
            "detailed_words": sentiment_words,
            "method": "snownlp_hybrid",
            "snownlp_score": round(snownlp_sentiment, 2),
            "dict_score": round(dict_normalized_score, 2)
        }
    
    def analyze_news(self, news_item: Dict[str, str]) -> Dict[str, Any]:
        """分析单条新闻的情绪"""
        
        title = news_item.get("title", "")
        content = news_item.get("content", "")
        
        title_analysis = self.analyze_text(title, text_type="title") if title else None
        content_analysis = self.analyze_text(content, text_type="content") if content else None
        
        # 计算最终情绪（加权融合）
        if title_analysis and content_analysis:
            final_score = (
                title_analysis["sentiment_score"] * 0.6 +
                content_analysis["sentiment_score"] * 0.4
            )
            
            title_is_positive = title_analysis["sentiment_score"] > 0.5
            content_is_positive = content_analysis["sentiment_score"] > 0.5
            consistency = 1.0 if title_is_positive == content_is_positive else 0.6
            
            total_matched = (
                title_analysis["matched_words_count"] +
                content_analysis["matched_words_count"]
            )
            total_words = (
                title_analysis["total_words_count"] +
                content_analysis["total_words_count"]
            )
            
            final_confidence = self._calculate_confidence(
                matched_words_count=total_matched,
                total_words_count=total_words,
                sentiment_score=final_score,
                consistency=consistency
            )
        elif title_analysis:
            final_score = title_analysis["sentiment_score"]
            final_confidence = title_analysis["confidence"]
        elif content_analysis:
            final_score = content_analysis["sentiment_score"]
            final_confidence = content_analysis["confidence"]
        else:
            final_score = 0.5
            final_confidence = 0.0
        
        final_label, _ = self._score_to_sentiment(final_score - 0.5)
        
        return {
            "title": title,
            "content": content[:100] + "..." if len(content) > 100 else content,
            "publish_time": news_item.get("publish_time", ""),
            "source": news_item.get("source", ""),
            "title_analysis": title_analysis,
            "content_analysis": content_analysis,
            "final_sentiment": {
                "label": final_label,
                "score": round(final_score, 2),
                "confidence": final_confidence
            }
        }
    
    def analyze_news_list(self, news_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析新闻列表并生成汇总"""
        
        analyses = []
        sentiment_scores = []
        confidences = []
        
        for news_item in news_list:
            analysis = self.analyze_news(news_item)
            analyses.append(analysis)
            sentiment_scores.append(analysis["final_sentiment"]["score"])
            confidences.append(analysis["final_sentiment"]["confidence"])
        
        avg_sentiment = round(sum(sentiment_scores) / len(sentiment_scores), 2) if sentiment_scores else 0.5
        avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
        overall_label, _ = self._score_to_sentiment(avg_sentiment - 0.5)
        
        time_series_sentiment = self._calculate_time_series_sentiment(analyses)
        
        return {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "total_news_count": len(analyses),
            "overall_sentiment": {
                "label": overall_label,
                "score": avg_sentiment,
                "confidence": avg_confidence
            },
            "sentiment_distribution": self._get_sentiment_distribution(analyses),
            "time_series_sentiment": time_series_sentiment,
            "detailed_analyses": analyses
        }
    
    def _calculate_time_series_sentiment(self, analyses: List[Dict]) -> Dict[str, Optional[float]]:
        """计算不同时间段的情绪趋势"""
        time_groups = {
            "recent_7d": [],
            "recent_30d": [],
            "all_time": []
        }
        
        now = datetime.now()
        
        for analysis in analyses:
            try:
                publish_time_str = analysis.get("publish_time", "")
                if not publish_time_str:
                    continue
                
                publish_time = datetime.strptime(publish_time_str, "%Y-%m-%d %H:%M:%S")
                days_diff = (now - publish_time).days
                
                score = analysis["final_sentiment"]["score"]
                
                time_groups["all_time"].append(score)
                if days_diff <= 7:
                    time_groups["recent_7d"].append(score)
                if days_diff <= 30:
                    time_groups["recent_30d"].append(score)
            except Exception:
                continue
        
        result = {}
        for key, scores in time_groups.items():
            if scores:
                result[key] = round(sum(scores) / len(scores), 2)
            else:
                result[key] = None
        
        return result
    
    def _get_sentiment_distribution(self, analyses: List[Dict]) -> Dict[str, int]:
        """计算情绪分布"""
        distribution = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
        
        for analysis in analyses:
            label = analysis["final_sentiment"]["label"]
            if label in distribution:
                distribution[label] += 1
        
        return distribution


def load_news_from_json(file_path: str) -> Dict[str, Any]:
    """从JSON文件加载新闻数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_analysis_result(result: Dict[str, Any], output_path: str):
    """保存分析结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存到: {output_path}")


def main():
    """主函数：加载新闻数据并进行情绪分析"""
    
    input_file = r"cache/news/stock_news/000300/000300_news_2025-11-07.json"
    output_dir = r"cache/sentiment_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在加载新闻数据: {input_file}")
    news_data = load_news_from_json(input_file)
    
    # 创建分析器 - 使用SnowNLP + 自定义词典组合方案
    use_snownlp = SNOWNLP_AVAILABLE
    print(f"\n使用分析方案: {'SnowNLP + 自定义词典混合方案' if use_snownlp else '词典法'}")
    print("添加金融领域专用词汇扩展: 是")
    
    analyzer = SentimentAnalyzer(use_snownlp=use_snownlp, add_finance_words=True)
    
    print("正在进行情绪分析...")
    result = analyzer.analyze_news_list(news_data["news"])
    
    # 添加元数据
    result["stock_code"] = "000300"
    result["data_source"] = input_file
    result["method"] = "snownlp_hybrid" if use_snownlp else "dictionary_based"
    result["add_finance_words"] = True
    
    # 保存结果 - 基于输入文件名生成输出文件名
    input_basename = os.path.basename(input_file)
    input_filename_without_ext = os.path.splitext(input_basename)[0]  # 获取不含扩展名的文件名
    output_filename = f"{input_filename_without_ext}_sentiment_analyses.json"
    output_file = os.path.join(output_dir, output_filename)
    save_analysis_result(result, output_file)
    
    # 打印摘要结果
    print("\n" + "="*80)
    print("情绪分析结果摘要")
    print("="*80)
    print(f"分析方法: {result['method']}")
    print(f"金融词汇扩展: {result['add_finance_words']}")
    print(f"股票代码: {result['stock_code']}")
    print(f"分析日期: {result['analysis_date']}")
    print(f"新闻总数: {result['total_news_count']}")
    print(f"\n整体情绪: {result['overall_sentiment']['label'].upper()}")
    print(f"整体得分: {result['overall_sentiment']['score']} (0=极负, 0.5=中性, 1=极正)")
    print(f"整体置信度: {result['overall_sentiment']['confidence']}")
    print(f"\n情绪分布:")
    print(f"  - 正面新闻: {result['sentiment_distribution']['positive']} 条")
    print(f"  - 中性新闻: {result['sentiment_distribution']['neutral']} 条")
    print(f"  - 负面新闻: {result['sentiment_distribution']['negative']} 条")
    
    if result['time_series_sentiment']['recent_7d'] is not None:
        print(f"\n时间序列情绪:")
        print(f"  - 近7天平均情绪: {result['time_series_sentiment']['recent_7d']}")
        print(f"  - 近30天平均情绪: {result['time_series_sentiment']['recent_30d']}")
        print(f"  - 全时间平均情绪: {result['time_series_sentiment']['all_time']}")
    
    print(f"\n详细分析已保存到: {output_file}")
    print("="*80)
    
    # 打印前3条新闻的详细分析
    print("\n前3条新闻的详细分析:")
    print("-"*80)
    for i, analysis in enumerate(result['detailed_analyses'][:3], 1):
        print(f"\n[{i}] {analysis['title']}")
        print(f"    发布时间: {analysis['publish_time']}")
        print(f"    新闻来源: {analysis['source']}")
        if analysis['title_analysis']:
            method_info = f" [{analysis['title_analysis'].get('method', 'dict')}]"
            print(f"    标题情绪: {analysis['title_analysis']['sentiment_label']} "
                  f"(得分: {analysis['title_analysis']['sentiment_score']}, "
                  f"置信度: {analysis['title_analysis']['confidence']}){method_info}")
            print(f"    标题关键词: {', '.join(analysis['title_analysis']['matched_words']) if analysis['title_analysis']['matched_words'] else '无'}")
        if analysis['content_analysis']:
            method_info = f" [{analysis['content_analysis'].get('method', 'dict')}]"
            print(f"    内容情绪: {analysis['content_analysis']['sentiment_label']} "
                  f"(得分: {analysis['content_analysis']['sentiment_score']}, "
                  f"置信度: {analysis['content_analysis']['confidence']}){method_info}")
            keywords = ', '.join(analysis['content_analysis']['matched_words'][:5]) if analysis['content_analysis']['matched_words'] else '无'
            print(f"    内容关键词: {keywords}")
        print(f"    最终情绪: {analysis['final_sentiment']['label'].upper()} "
              f"(得分: {analysis['final_sentiment']['score']}, "
              f"置信度: {analysis['final_sentiment']['confidence']})")


if __name__ == "__main__":
    main()
