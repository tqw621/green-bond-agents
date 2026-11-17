"""
LLM 客户端模块
封装 OpenAI 兼容 API 调用，供各智能体使用
"""
import os
import getpass
import warnings

try:
    import openai
except ImportError:
    openai = None

try:
    import httpx
except ImportError:
    httpx = None

import config


def prompt_api_key():
    """Ask interactively without echo or falling back to shared environment credentials."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("请输入你自己的 API Key（输入隐藏，仅本次运行使用）：").strip()
    except (EOFError, getpass.GetPassWarning):
        raise SystemExit("请在交互式终端输入 API Key，或使用网页的密钥输入框。") from None
    if not key or key == "1":
        raise SystemExit("API Key 为空，已停止 LLM 运行。")
    return key


class LLMClient:
    """通用 LLM API 客户端（兼容 DeepSeek / GPT / 千问 等）"""

    def __init__(self, model: str = None, *, api_key: str = None):
        if not isinstance(api_key, str) or not api_key.strip() or api_key.strip() == "1":
            raise ValueError("请提供本次使用者的 API Key")
        self.model = model or config.DEFAULT_MODEL_NAME
        self._api_key = api_key.strip()

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
        client = None
        http_client = None
        try:
            # 显式创建 httpx 客户端
            # trust_env: 默认 False（不读取系统代理，解决 Windows 代理干扰问题）
            #            云服务器部署时设置 LLM_TRUST_ENV=true 可按需启用代理
            client_class = getattr(openai, "DefaultHttpxClient", None)
            if client_class is None and httpx is not None:
                client_class = httpx.Client
            if client_class is not None:
                trust_env_str = os.getenv("LLM_TRUST_ENV", "false").lower()
                trust_env = trust_env_str in ("true", "1", "yes")
                timeout_val = float(os.getenv("LLM_TIMEOUT", "60.0"))
                http_client = client_class(
                    trust_env=trust_env,
                    timeout=timeout_val,
                )
            client = openai.OpenAI(
                api_key=self._api_key,
                base_url=config.LLM_BASE_URL,
                http_client=http_client,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as error:
            # Provider exception bodies may contain credentials. Keep only a safe error category.
            status = getattr(error, "status_code", None)
            code = str(status) if isinstance(status, int) else type(error).__name__
            return f"[API_ERROR] 模型请求失败（{code}），请核对密钥、额度和服务地址"
        finally:
            if client is not None:
                client.close()
            elif http_client is not None:
                http_client.close()

    def call_with_json_output(self, system_prompt: str, user_prompt: str,
                              temperature: float = config.LLM_JSON_TEMPERATURE) -> str:
        """
        期望返回 JSON 格式的调用（用于结构化决策结果）
        提示词中应要求模型返回 JSON
        """
        return self.call(system_prompt, user_prompt,
                         temperature=temperature, max_tokens=4000)
