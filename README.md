# 金融数据与 NLP 工具集
本项目用于抓取并处理股票数据和财经新闻，支持技术指标分析和情绪分析，输出结构化数据用于 NLP 模型或策略研究。

## 项目结构
```
nlp_project_data_prepare/
├── scripts/
│   ├── logging_config.py                        # 日志配置工具
│   └── tools/
│       ├── web_search.py                        # 网页搜索功能（基于Playwright）
│       ├── data_analyzer.py                     # 股票数据技术指标分析           （股票代码改为stock_id，部分指标改为保留8位小数，csv文件里数据结构统一，无文本类型）
│       ├── news_crawler.py                      # 股票相关新闻爬取
│       ├── sentiment_analyzer.py                # 中文情绪分析（SnowNLP + 自定义词典混合方案）
│       ├── financial_data.py                    # 股票价格历史数据获取与处理      （修改了Hurst指数的计算方式）
│       ├── get_em_calendar_image.py             # 查找东方财富财经早餐网页图片链接
│       ├── get_em_listpage_url.py               # 查找东方财富财经早餐网页链接
│       ├── eastmoney_breakfast.py               # 查找东方财富财经早餐  （判读工作日函数有问题，从2022-11-9至2022-12-21无法正确返回网址序号）
│       ├── insight_agent.py                     # 结合情绪数据与市场指标，生成投资洞察与预测结果
│       └── cache/                               # 数据缓存
│           ├── insight_reports                  # 存放 Insight Agent 输出的 JSON 洞察报告
│           ├── news/                
│           │   ├── eastmoney_breakfast/         # 东方财富财经早餐相关链接
│           │   │   └── urls_of_em.json
│           │   └── stock_news/                  #股票新闻，以股票代码划分文件夹
│           ├── stock_price_data/                #股票价格数据，以股票代码划分文件夹
│           └── sentiment_analysis/              #情绪分析结果，以股票代码划分文件夹
└── logs/                                       # 日志文件存储目录（自动生成）
```

## 安装依赖（Dependencies Installation）(运行insight_agent.py前)

1. 确保你已安装 **Python 3.9+**
2. 推荐：创建并激活虚拟环境（可选）
3. 在项目根目录执行以下命令安装所有依赖：

```
pip install -r requirements.txt
```
---

## 更新说明
1.优化了代码结构   
2.financial_data.py中修改了Hurst指数的计算方式，索引没变    
3.提取到的新闻和价格数据，统一保存在cache文件夹中，以股票代码划分文件夹    
4.data_analyzer.py中股票代码改为stock_id，部分指标改为保留8位小数，csv文件里数据结构统一，无文本类型        
- 具体保留八位小数指标如下    
```
decimal_cols = [
    "momentum_1m", "momentum_3m", "momentum_6m", "volume_ma20", "volume_momentum",
    "historical_volatility", "volatility_regime", "volatility_z_score", "atr", "atr_ratio",
    "hurst_exponent", "skewness", "kurtosis", "ma5", "ma10", "ma20", "ma60", "macd",
    "singal_line", "macd_hist", "rsi", "bb_middle", "bb_upper", "bb_lower", "volume_ma5",
    "volume_ratio", "price_momentum", "price_acceleration", "daily_return", "volatility_5d",
    "volatility_20d"
]
```
5.东方财富财经早餐相关功能基本完善，get_em_calendar_image.py和get_em_listpage_url.py在输入url后能正确读取相应链接，eastmoney_breakfast.py 列表网页计算仍存在问题

## 功能说明
1.东方财富早报获取（eastmoney_breakfast.py）  
- 爬取东方财富网早报内容及对应日期的链接  
- 自动处理工作日判断（排除周末和法定节假日）  
- 支持指定日期范围的数据获取  
  
2.股票数据技术指标分析（data_analyzer.py）  
- 计算常见技术指标（MA、MACD、RSI、布林带等）  
- 分析成交量、价格动量、波动率等特征  
- 生成结构化分析结果并保存为 CSV 文件  
  
3.网页搜索功能（web_search.py）  
- 基于 Playwright 实现模拟浏览器搜索  
- 支持自定义搜索选项（结果数量、超时时间等）  
- 反爬虫处理（浏览器指纹模拟、状态保存）  
  
4.股票价格数据获取（financial_data.py）  
- 获取股票历史价格数据（开盘价、收盘价、成交量等）  
- 支持复权类型选择（前复权、后复权、不复权）  
- 计算动量指标、波动率、赫斯特指数等金融特征  
  
5.股票新闻爬取（news_crawler.py）  
- 支持通过 Google 搜索或 AKShare 获取股票相关新闻  
- 自动过滤无效新闻（招聘、广告、开户等）  
- 实现新闻数据缓存机制，避免重复爬取  
  
6.网页渲染与解析（test.py）  
- 使用 Playwright 渲染动态网页内容  
- 解析网页中的新闻列表及日期信息  
- 自动处理浏览器依赖安装  

7.中文情绪分析（sentiment_analyzer.py）  
- 支持 SnowNLP + 自定义词典的混合方案（SnowNLP 权重 60%，自定义词典 40%）  
- 计算置信度（基于匹配度、一致性、强度三个维度）  
- 支持新闻列表的批量分析和时间序列情绪趋势计算  
- 输出包含情绪标签、得分、关键词、置信度等完整信息的 JSON 格式结果

8.投资洞察生成（insight_agent.py）  
- 综合 **情绪分析结果（JSON）** + **技术指标（CSV）**  
- 使用大语言模型(来着Openrouter)进行趋势判断，输出方向：上涨（up） / 下跌（down） / 震荡（flat）  
- 自动生成结构化 JSON，包括：  
  - 趋势方向（direction）  
  - 置信度（confidence）  
  - 关键理由（reason）  
  - 佐证数据（supporting_facts）  
  - 教学提示（teaching_note，解释指标含义）  
- 自动保存输出为 `cache/insight_reports/*.json`
  
## 安装说明
### 前置依赖
Python 3.8+
依赖库：pandas, requests, beautifulsoup4, playwright, akshare, chinese-calendar, numpy, snownlp

## 安装步骤
### 克隆项目代码
```
git clone <项目仓库地址>
cd nlp_project_data_prepare
```
### 安装依赖包
```
pip install pandas requests beautifulsoup4 playwright akshare chinese-calendar numpy snownlp
#安装 Playwright 浏览器（首次使用时需要）

python -m playwright install chromium
```
## 使用示例
1.分析股票数据
```
from datetime import datetime, timedelta
from scripts.tools.data_analyzer import analyze_stock_data
symbol = "600519"  # 贵州茅台
current_date = datetime.now()
end_date = current_date.strftime("%Y-%m-%d")  # 使用今天作为结束日期
start_date = (current_date - timedelta(days=365)).strftime("%Y-%m-%d")

print(f"分析时间范围: {start_date} 至 {end_date}")
analyze_stock_data(symbol, start_date, end_date)
```

2.获取股票新闻
```
from scripts.tools.news_crawler import get_stock_news

# 获取贵州茅台（600519）的最新10条新闻
news_list = get_stock_news("600519", max_news=10)
print(f"获取到 {len(news_list)} 条新闻")
```
3.获取股票历史价格数据
```
from scripts.tools.financial_data import get_price_history

# 获取贵州茅台（600519）的价格历史数据
df = get_price_history("600519")
print(f"获取到 {len(df)} 条价格记录")
```

4.中文情绪分析
```
from scripts.tools.sentiment_analyzer import SentimentAnalyzer
import json

# 载入新闻数据
with open("cache/news/stock_news/000300/000300_news_2025-11-07.json", 'r', encoding='utf-8') as f:
    news_data = json.load(f)

# 创建情绪分析器
analyzer = SentimentAnalyzer(use_snownlp=True, add_finance_words=True)

# 分析新闻列表
result = analyzer.analyze_news_list(news_data["news"])

# 打印整体情绪结果
print(f"整体情绪: {result['overall_sentiment']['label']}")
print(f"整体得分: {result['overall_sentiment']['score']}")
print(f"整体置信度: {result['overall_sentiment']['confidence']}")

```

5.Insight Agent
```
1. 修改配置文件路径
------------------
打开:
scripts/tools/insight_agent.py

将CONFIG内容改成你的文件路径：

"SENTIMENT_SAMPLE_PATH": "cache/sentiment_analysis/xxx.json",
"MARKET_SAMPLE_CSV": "cache/stock_price_data/xxxxxx/xxxxxx.csv"

- SENTIMENT_SAMPLE_PATH → 指向情绪分析 JSON
- MARKET_SAMPLE_CSV → 指向股票技术指标 CSV


2. 运行 Insight Agent 主程序
---------------------------
在项目根目录执行：

python scripts/tools/insight_agent.py
```
