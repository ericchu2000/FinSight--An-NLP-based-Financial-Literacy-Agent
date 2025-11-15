import requests
import json
import time
from typing import Dict, Any, List

class BailianClient:
    """阿里云百炼API客户端"""
    
    def __init__(self, api_key: str):
        """
        初始化百炼客户端
        
        Args:
            api_key: 阿里云百炼API密钥
        """
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable"
        }
    
    def call_qwen_plus(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> Dict[str, Any]:
        """
        调用通义千问-Plus模型
        
        Args:
            prompt: 输入提示词（字符串或messages列表）
            max_tokens: 最大生成token数
            temperature: 温度参数，控制随机性
            
        Returns:
            包含响应结果的字典
        """
        # 支持字符串或messages格式
        if isinstance(prompt, dict) and "messages" in prompt:
            messages = prompt["messages"]
        else:
            messages = [{"role": "user", "content": prompt}]
            
        payload = {
            "model": "qwen-plus",
            "input": {
                "messages": messages
            },
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "result_format": "message"
            }
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 提取响应内容
            if "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                
                return {
                    "success": True,
                    "content": content,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "success": False,
                    "error": f"Unexpected response format: {result}",
                    "content": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "response_time": response.elapsed.total_seconds()
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "response_time": 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "response_time": 0
            }


def create_zero_shot_prompt(problem_description: str, function_signature: str) -> str:
    """
    创建零样本提示词
    
    Args:
        problem_description: 问题描述
        function_signature: 函数签名
        
    Returns:
        格式化后的提示词
    """
    prompt = f"""请根据以下问题描述和函数签名，编写一个Python函数来解决这个问题。

问题描述：
{problem_description}

函数签名：
{function_signature}

要求：
1. 只返回完整的函数代码，不要添加任何解释
2. 确保代码可以直接运行
3. 包含所有必要的import语句
4. 函数名和参数必须与签名完全一致

代码：
"""
    return prompt


def extract_function_code(response_content: str) -> str:
    """
    从AI响应中提取函数代码
    
    Args:
        response_content: AI的响应内容
        
    Returns:
        提取的函数代码
    """
    # 移除可能的markdown代码块标记
    lines = response_content.split('\n')
    
    # 移除开头的```python或```
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    
    # 移除结尾的```
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    
    # 合并代码
    code = '\n'.join(lines).strip()
    
    return code


# 全局客户端实例
bailian_client = BailianClient("sk-00c0fc19147d468cbb23f1c5ed46f0fb")


if __name__ == "__main__":
    # 测试代码
    test_prompt = create_zero_shot_prompt(
        "编写一个函数，接收两个整数a和b，返回它们的和",
        "def add_numbers(a: int, b: int) -> int:"
    )
    
    result = bailian_client.call_qwen_plus(test_prompt)
    
    if result["success"]:
        print("调用成功！")
        print("生成的代码：")
        print(extract_function_code(result["content"]))
        print(f"输入tokens: {result['input_tokens']}")
        print(f"输出tokens: {result['output_tokens']}")
        print(f"总tokens: {result['total_tokens']}")
        print(f"响应时间: {result['response_time']:.2f}秒")
    else:
        print("调用失败：")
        print(result["error"])
