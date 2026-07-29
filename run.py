#!/usr/bin/env python3
"""
绿色债券发行多主体LLM智能体链路模拟系统 — 启动脚本

用法：
    python run.py                  # 启动 Streamlit Web 界面
    python engine.py               # 命令行直接运行一次模拟测试
    python engine.py --llm         # 命令行运行一次 LLM 模式模拟测试
    python run.py --cli 200        # 命令行批量运行 200 次模拟（规则引擎模式）
    python run.py --cli 10 --llm   # 命令行批量运行 10 次模拟（LLM 模式）
"""
import subprocess
import sys
import os

# 修复 Windows 终端 GBK 编码无法输出 emoji 的问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def check_requirements():
    """检查必要的依赖"""
    try:
        import openai
        import streamlit
        import plotly
        import pandas
        print("✅ 所有依赖包已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False


def check_config():
    """检查配置"""
    try:
        import config
        if not config.LLM_API_KEY or config.LLM_API_KEY == "1":
            print("⚠️  警告: LLM_API_KEY 未配置或为默认值")
            print("  规则引擎模式不需要 API Key，如需 LLM 决策请在 .env 中设置有效的 API Key")
        else:
            masked_key = config.LLM_API_KEY[:8] + "***" if len(config.LLM_API_KEY) > 8 else "***"
            print(f"✅ API Key 已配置 ({masked_key})")
            print(f"   Base URL: {config.LLM_BASE_URL}")
            print(f"   Model: {config.DEFAULT_MODEL_NAME}")
        return True
    except ImportError:
        print("❌ 配置文件 config.py 不存在")
        return False


def main():
    print("🌿 启动绿色债券发行链路模拟系统...")
    print("=" * 50)

    if not check_requirements():
        return

    check_config()

    # 解析参数
    use_llm = "--llm" in sys.argv
    rounds = 50
    for i, arg in enumerate(sys.argv):
        if arg == "--cli" and i + 1 < len(sys.argv):
            try:
                rounds = int(sys.argv[i + 1])
            except ValueError:
                pass

    # 支持命令行参数
    if "--cli" in sys.argv:
        print(f"\n📊 命令行批量模拟模式：{rounds} 次 ({'LLM 智能决策' if use_llm else '规则引擎'})")
        print("=" * 50)

        from engine import generate_sample_enterprises, run_batch_simulation, analyze_results

        enterprises = generate_sample_enterprises(20)
        print(f"已生成 {len(enterprises)} 家样本企业")
        print(f"开始模拟...\n")

        results = run_batch_simulation(enterprises, {
            "certifier_type": "moderate",
            "regulator_type": "moderate",
            "underwriter_type": "moderate",
            "investor_type": "balanced",
        }, rounds=rounds, use_llm=use_llm)

        stats = analyze_results(results)
        print(f"\n{'=' * 50}")
        print(f"📈 模拟完成！")
        print(f"   总次数：{stats['total_simulations']}")
        print(f"   成功率：{stats['success_rate']:.1%}")
        print(f"   瓶颈环节：{stats['bottleneck_stage']}")
        for stage, info in stats["stage_failure_rates"].items():
            print(f"     {stage}: 失败 {info['count']} 次 ({info['rate']:.1%})")

        if stats.get("llm_fallback_rate", 0) > 0:
            print(f"   ⚠️  LLM 降级率：{stats['llm_fallback_rate']:.1%}（{stats.get('llm_fallback_count', 0)} 次降级到规则引擎）")

        return

    if "--help" in sys.argv:
        print("用法：")
        print("  python run.py                     启动 Web 界面（仅本机访问）")
        print("  python run.py --production        启动 Web 界面（生产模式，局域网可访问）")
        print("  python run.py --cli N             命令行批量运行 N 次（规则引擎，默认50）")
        print("  python run.py --cli N --llm       命令行批量运行 N 次（LLM 模式）")
        print("  python engine.py                  运行单次规则引擎模拟")
        print("  python engine.py --llm            运行单次 LLM 模式模拟")
        return

    # 默认：启动 Streamlit
    print("🌐 正在启动 Web 界面...")

    # 生产模式：绑定 0.0.0.0 允许外部访问
    is_production = "--production" in sys.argv or os.getenv("PRODUCTION", "").lower() == "true"
    server_address = "0.0.0.0" if is_production else "127.0.0.1"
    server_port = os.getenv("STREAMLIT_PORT", "8504")

    if is_production:
        print("🔓 生产模式：绑定 0.0.0.0（允许外部访问）")
        print(f"📝 访问地址: http://<服务器IP>:{server_port}")
    else:
        print(f"📝 访问地址: http://localhost:{server_port}")
    print("⏹️  按 Ctrl+C 停止服务")
    print("=" * 50)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", server_port,
            "--server.address", server_address,
        ])
    except KeyboardInterrupt:
        print("\n👋 退出系统")


if __name__ == "__main__":
    main()
