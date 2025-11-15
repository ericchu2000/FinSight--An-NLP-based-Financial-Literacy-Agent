import requests
import json
import time
from typing import Dict, Any, List

# Import shared LLM config from tools package
from scripts.tools.llm_config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL


class OpenAIClient:
    """OpenAI API client for financial education agent."""

    def __init__(self):
        if not OPENAI_API_KEY or "paste-your-openai-key-here" in OPENAI_API_KEY:
            # We don't raise here to allow higher-level code to handle/log it,
            # but any actual call will fail clearly.
            print(
                "[OpenAIClient] WARNING: OPENAI_API_KEY not set or placeholder "
                "(see scripts/tools/llm_config.py)."
            )

        self.api_key = OPENAI_API_KEY
        self.base_url = OPENAI_BASE_URL.rstrip("/")
        self.model = OPENAI_MODEL

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def call_llm(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Call OpenAI chat completion API.

        Args:
            prompt: User prompt (string).
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dict with:
              - success (bool)
              - content (str)
              - input_tokens (int)
              - output_tokens (int)
              - total_tokens (int)
              - response_time (float, seconds)
              - error (str, if any)
        """
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            response_time = time.time() - start_time
            response.raise_for_status()

            result = response.json()

            # Extract content and usage
            try:
                content = result["choices"][0]["message"]["content"]
            except Exception:
                return {
                    "success": False,
                    "error": f"Unexpected response format: {result}",
                    "content": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "response_time": response_time,
                }

            usage = result.get("usage", {})
            return {
                "success": True,
                "content": content,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "response_time": response_time,
            }

        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "response_time": response_time,
            }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "response_time": response_time,
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
    lines = response_content.split("\n")

    # 移除开头的```python或```
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    # 移除结尾的```
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    # 合并代码
    code = "\n".join(lines).strip()
    return code


# 全局客户端实例（可选测试用）
openai_client = OpenAIClient()


if __name__ == "__main__":
    # 简单测试：生成一个函数代码
    test_prompt = create_zero_shot_prompt(
        "编写一个函数，接收两个整数a和b，返回它们的和",
        "def add_numbers(a: int, b: int) -> int:",
    )

    result = openai_client.call_llm(test_prompt)

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
