"""
LLM 客户端模块
封装 OpenAI 兼容 API 调用，供各智能体使用
"""
import os

try:
    import openai
except ImportError:
    openai = None

try:
    import httpx
except ImportError:
    httpx = None

from typing import List, Dict, Optional
import config


class LLMClient:
    """通用 LLM API 客户端（兼容 DeepSeek / GPT / 千问 等）"""

    def __init__(self, model: str = None):
        self.model = model or config.DEFAULT_MODEL_NAME
        # 显式创建 httpx 客户端
        # trust_env: 默认 False（不读取系统代理，解决 Windows 代理干扰问题）
        #            云服务器部署时设置 LLM_TRUST_ENV=true 可按需启用代理
        http_client = None
        if httpx is not None:
            trust_env_str = os.getenv("LLM_TRUST_ENV", "false").lower()
            trust_env = trust_env_str in ("true", "1", "yes")
            timeout_val = float(os.getenv("LLM_TIMEOUT", "60.0"))
            http_client = httpx.Client(
                trust_env=trust_env,
                timeout=timeout_val,
            )
        self.client = openai.OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            http_client=http_client,
        )

    def call(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        通用 API 调用

        Args:
            system_prompt: 系统提示词（设定角色+规则）
            user_prompt: 用户提示词（待决策的具体案例）
            temperature: 随机性参数
            max_tokens: 最大输出 token 数

        Returns:
            模型返回的文本结果
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[API_ERROR] {str(e)}"

    def call_with_json_output(self, system_prompt: str, user_prompt: str,
                              temperature: float = 0.3) -> str:
        """
        期望返回 JSON 格式的调用（用于结构化决策结果）
        提示词中应要求模型返回 JSON
        """
        return self.call(system_prompt, user_prompt,
                         temperature=temperature, max_tokens=4000)
