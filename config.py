"""
全局配置模块
从 .env 加载 LLM API 密钥、模型名称等配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ===== LLM API 配置 =====
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "deepseek-chat")

# ===== 数据库配置 =====
APP_DIR = Path(__file__).resolve().parent
_default_db = str(APP_DIR / "data" / "simulation_results.db")
DB_PATH = os.getenv("DB_PATH", _default_db)

# ===== 模拟实验默认参数 =====
DEFAULT_SIMULATION_ROUNDS = 200       # 每个场景默认运行次数
RANDOM_SEED = 42                      # 随机种子（保证可复现）
