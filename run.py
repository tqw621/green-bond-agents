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
        import matplotlib
        import seaborn
        import jieba
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
        print(f"模型服务：{config.LLM_BASE_URL}")
        print(f"模型名称：{config.DEFAULT_MODEL_NAME}")
        print("LLM 模式由使用者在页面或终端输入自己的 API Key。")
        return True
    except ImportError:
        print("❌ 配置文件 config.py 不存在")
        return False


def run_preset_cli():
    """对正式名单逐家运行预设场景，并输出可复核的 JSON。"""
    import argparse
    import json
    from pathlib import Path

    import pandas as pd

    import config
    from engine import analyze_results, load_enterprises_from_df, validate_enterprise_dataframe
    from scenarios import paired_changes, run_preset_experiment

    parser = argparse.ArgumentParser(description="从第一组名单运行 P0–P6")
    parser.add_argument("--preset", required=True, type=Path, help="企业名单 CSV 或 XLSX")
    parser.add_argument("--prefiltered", action="store_true", help="文件已确认仅含愿意发行企业")
    parser.add_argument("--llm", action="store_true", help="使用 LLM 模式")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--output", required=True, type=Path, help="新的 JSON 输出文件")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出文件已存在，请选择新文件名")
    if not args.output.parent.is_dir():
        parser.error("输出文件所在目录不存在")

    try:
        if args.preset.suffix.lower() == ".csv":
            frame = pd.read_csv(args.preset, dtype=str, keep_default_na=False)
        elif args.preset.suffix.lower() == ".xlsx":
            frame = pd.read_excel(args.preset, dtype=str, keep_default_na=False)
        else:
            parser.error("企业名单应为 CSV 或 XLSX")
    except (OSError, ValueError) as error:
        parser.error(f"无法读取名单：{error}")
    valid, message, processed = validate_enterprise_dataframe(
        frame, willing_only=True, prefiltered=args.prefiltered,
    )
    if not valid:
        parser.error(message)

    print(message, flush=True)
    from llm_client import prompt_api_key
    api_key = prompt_api_key() if args.llm else None
    scenarios = run_preset_experiment(load_enterprises_from_df(processed), use_llm=args.llm, seed=args.seed, api_key=api_key)
    summary = {}
    for name, records in scenarios.items():
        stats = analyze_results(records)
        stats["gained"], stats["lost"] = paired_changes(scenarios["P0"], records)
        summary[name] = stats
    output = json.dumps({"summary": summary, "scenarios": scenarios}, ensure_ascii=False, indent=2)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(output)
    print(f"已保存 {sum(map(len, scenarios.values()))} 条模拟记录：{args.output}")


def run_analysis_cli():
    """Analyze an existing JSON export without rerunning simulations."""
    import argparse
    import json
    from pathlib import Path
    from analysis import SOURCES
    from engine import STAGES
    from reporting import render_report

    parser = argparse.ArgumentParser(description="从模拟 JSON 导出静态图表和文本分析 ZIP")
    parser.add_argument("--analyze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="新建的 .zip 文件")
    parser.add_argument("--scope", choices=["failures", "all"], default="failures")
    parser.add_argument("--source", choices=["all", *SOURCES], default="all")
    parser.add_argument("--stage", choices=["all", *[code for code, _, _ in STAGES]], default="all")
    parser.add_argument("--top-n", type=int, choices=range(1, 101), default=20, metavar="1-100")
    parser.add_argument("--stopwords", nargs="*", default=[])
    args = parser.parse_args()
    if args.output.suffix.lower() != ".zip" or args.output.exists() or not args.output.parent.is_dir():
        parser.error("输出应为已有目录中的新 .zip 文件")
    try:
        payload = json.loads(args.analyze.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            scenarios = {}
            for record in payload:
                scenarios.setdefault(record.get("scenario_name") or "export", []).append(record)
        else:
            scenarios = payload["scenarios"]
        if not isinstance(scenarios, dict) or not scenarios:
            raise ValueError("输入缺少场景记录")
        for records in scenarios.values():
            if not isinstance(records, list) or not records or any(
                not isinstance(record, dict) or record.get("final_result") not in ("success", "failed")
                for record in records
            ):
                raise ValueError("各场景必须包含有效的模拟结果列表")
        report = render_report(scenarios, scope=args.scope, source=args.source, stage=args.stage,
                               top_n=args.top_n, extra_stopwords=tuple(args.stopwords))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"无法生成报告：{error}")
    with args.output.open("xb") as handle:
        handle.write(report["archive"])
    print(f"已导出分析报告：{args.output}")
    if report["metadata"]["font"] is None:
        print("当前环境缺少中文绘图字体，请安装 Noto Sans CJK 后重新导出。")


def main():
    print("🌿 启动绿色债券发行链路模拟系统...")
    print("=" * 50)

    if "--analyze" in sys.argv:
        run_analysis_cli()
        return
    if "--preset" in sys.argv:
        run_preset_cli()
        return

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

        from llm_client import prompt_api_key
        api_key = prompt_api_key() if use_llm else None
        results = run_batch_simulation(enterprises, {
            "certifier_type": "moderate",
            "regulator_type": "moderate",
            "underwriter_type": "moderate",
            "investor_type": "balanced",
        }, rounds=rounds, use_llm=use_llm, api_key=api_key)

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
        print("  python run.py --analyze results.json --output report.zip  导出图表和文本分析")
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
