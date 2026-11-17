"""
绿色债券发行多主体LLM智能体链路模拟系统 — Web界面

基于 Streamlit 构建交互式界面：
- 场景配置
- 单次模拟运行
- 批量实验
- 结果统计可视化
"""
import json
import hashlib

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from uuid import uuid4
import config
from analysis import SOURCES, analyze_decision_text
from reporting import render_report
from document_import import (
    FIELD_LABELS, EXTRACTION_DEFAULTS, convert_text, docx_to_text,
    export_csv, export_excel, load_reviewed_enterprises,
)

from engine import (
    generate_sample_enterprises,
    run_single_simulation,
    run_batch_simulation,
    analyze_results,
    validate_enterprise_dataframe,
    load_enterprises_from_df,
    ownership_survival,
    STAGES,
)
from database import SimulationDB
from scenarios import PRESET_SCENARIOS, paired_changes, run_preset_experiment
from agents import (
    CERTIFIER_TYPES,
    REGULATOR_TYPES,
    UNDERWRITER_TYPES,
    INVESTOR_TYPES,
)

# 页面配置
st.set_page_config(
    page_title="绿色债券发行链路模拟系统",
    page_icon="🌿",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background: #f7faf8; }
[data-testid="stSidebar"] { background: #eef5f1; border-right: 1px solid #dce8e0; }
.block-container { max-width: 1320px; padding-top: 2.5rem; padding-bottom: 3rem; }
h1 { font-size: 2.15rem !important; letter-spacing: -.035em; }
h2, h3 { letter-spacing: -.02em; }
[data-testid="stMetric"] { background: white; border: 1px solid #e0eae4; border-radius: 14px; padding: 16px 20px; }
[data-testid="stMetricValue"] { font-size: 1.65rem; }
[data-testid="stExpander"] { background: white; border-radius: 12px; }
.stButton button, .stDownloadButton button { border-radius: 9px; }
[data-baseweb="tab-list"] { gap: 1.5rem; }
[data-baseweb="tab"] { padding-top: .75rem; padding-bottom: .75rem; }
@media (max-width: 700px) {
  .block-container { padding-top: 1.25rem; }
  [data-baseweb="tab-list"] { gap: .4rem; }
}
</style>
""", unsafe_allow_html=True)

# 初始化数据库
db = SimulationDB()


@st.cache_data(show_spinner=False, max_entries=12)
def cached_text_analysis(records, scope, source, stage, top_n, extra_stopwords):
    return analyze_decision_text(
        records, scope=scope, source=source, stage=stage,
        top_n=top_n, extra_stopwords=extra_stopwords,
    )


@st.cache_data(show_spinner=False, max_entries=4)
def cached_report(scenarios, scope, source, stage, top_n, extra_stopwords):
    return render_report(
        scenarios, scope=scope, source=source, stage=stage,
        top_n=top_n, extra_stopwords=extra_stopwords,
    )


# ============================================================
# 侧边栏 — 场景配置
# ============================================================

def clear_api_key():
    st.session_state["user_api_key"] = ""


def clear_extraction_key():
    st.session_state["extraction_api_key"] = ""


st.sidebar.title("绿债研究台")
st.sidebar.caption("链路组 2 · 发行模拟")
st.sidebar.subheader("运行方式")
use_llm = st.sidebar.checkbox(
    "启用 LLM API 决策", value=False, on_change=clear_api_key,
    help="使用自己的模型服务账户，调用费用由该账户承担。",
)
api_key = None
if use_llm:
    api_key = st.sidebar.text_input(
        "你的 API Key", type="password", key="user_api_key",
        placeholder="输入你自己的密钥",
        help="密钥仅保留在当前页面会话内存中，用于本次模型请求。",
    ).strip()
    st.sidebar.button("清除密钥", on_click=clear_api_key, width="stretch")
    from urllib.parse import urlparse
    provider = urlparse(config.LLM_BASE_URL).hostname or "已配置的模型服务"
    st.sidebar.caption(f"发送到 {provider} · {config.DEFAULT_MODEL_NAME}")
    st.sidebar.caption("输入后点击运行才会请求模型。重新连接页面时可能需要再次输入。")
else:
    st.sidebar.caption("使用内置规则与概率判断，可直接运行。")

with st.sidebar.expander("自定义场景参数", expanded=False):
    st.caption("用于单次模拟和单个自定义场景。P0–P6 使用预设参数。")
    certifier_type = st.selectbox(
        "绿色认证机构类型", options=list(CERTIFIER_TYPES),
        format_func=lambda value: CERTIFIER_TYPES[value]["name"], index=1,
    )
    regulator_type = st.selectbox(
        "监管审核机构类型", options=list(REGULATOR_TYPES),
        format_func=lambda value: REGULATOR_TYPES[value]["name"], index=1,
    )
    underwriter_type = st.selectbox(
        "承销商类型", options=list(UNDERWRITER_TYPES),
        format_func=lambda value: UNDERWRITER_TYPES[value]["name"], index=1,
    )
    investor_type = st.selectbox(
        "投资机构类型", options=list(INVESTOR_TYPES),
        format_func=lambda value: INVESTOR_TYPES[value]["name"], index=2,
    )

# ---- 企业数据源 ----
st.sidebar.markdown("---")
st.sidebar.subheader("企业数据")
st.sidebar.caption("选择来源 → 核对字段 → 运行模拟")

data_source = st.sidebar.radio(
    "选择数据来源",
    options=["示例数据（随机生成）", "上传第一组企业名单", "Word / 文字提取"],
    key="data_source_radio",
)

using_group_list = data_source != "示例数据（随机生成）"
uploaded_enterprises = []
uploaded_df = None
if data_source == "上传第一组企业名单":
    uploaded_file = st.sidebar.file_uploader(
        "上传 CSV 或 Excel 文件",
        type=["csv", "xlsx"],
        help="CSV 是 Excel 可打开的逗号分隔表格。请按 README 的字段准备名单，企业编号在 Excel 中应设为文本。",
    )
    prefiltered = st.sidebar.checkbox(
        "已确认名单仅含愿意发行的企业",
        help="名单没有发行决策列时才需要勾选。有决策列时，系统会按该列筛选。",
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl", dtype=str, keep_default_na=False)

            is_valid, msg, processed_df = validate_enterprise_dataframe(
                df, willing_only=True, prefiltered=prefiltered,
            )
            if is_valid:
                uploaded_enterprises = load_enterprises_from_df(processed_df)
                uploaded_df = processed_df
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)
        except Exception as e:
            st.sidebar.error(f"文件读取失败：{e}")
    else:
        st.sidebar.info("请先上传第一组企业名单")
elif data_source == "Word / 文字提取":
    st.sidebar.caption("需要企业名称／编号、发行决策、评级、负债率、绿色程度和三项材料状态。")
    with st.sidebar.expander("查看完整字段清单", expanded=False):
        st.caption("尽量提供下列字段。缺失时仍可试算，系统会标出待核实项。资产负债率可写 65% 或 0.65。")
        for field in ["name", "code", *EXTRACTION_DEFAULTS]:
            label = FIELD_LABELS[field]
            st.markdown(f"- **{label}** `{field}`")
        st.caption("可选补充：企业性质、行业、规模、发行规模、合规评分、项目目录、重大违规、市场基准收益率和定价利差等。")
    source_doc = st.sidebar.file_uploader(
        "上传 Word 文档（.docx）", type=["docx"],
        help="支持正文和表格；旧版 .doc 请先在 Word 中另存为 .docx。",
        key="source_docx",
    )
    source_text = st.sidebar.text_area(
        "或者粘贴企业文字", height=150,
        placeholder="企业名称：示例公司；信用评级：AA；资产负债率：65%；绿色程度：中……",
        help="可同时上传文档和粘贴文字。多家企业请逐家分段，或在 Word 表格中每行写一家。",
    )
    extract_with_llm = st.sidebar.checkbox(
        "使用 LLM 辅助提取",
        help="点击提取时才会将文档文字发送给模型服务。关闭时只在本地按字段标签提取；可独立于模拟决策模式使用。",
    )
    extraction_key = api_key
    if extract_with_llm:
        if not use_llm:
            extraction_key = st.sidebar.text_input(
                "提取用 API Key", type="password", key="extraction_api_key",
                placeholder="输入你自己的密钥",
                help="只用于本次文档提取，保存在当前页面会话内存中。",
            ).strip()
            st.sidebar.button("清除提取密钥", on_click=clear_extraction_key, width="stretch")
        from urllib.parse import urlparse
        st.sidebar.caption(f"点击提取后发送文字到 {urlparse(config.LLM_BASE_URL).hostname or '已配置的模型服务'}")
    content_bytes = source_doc.getvalue() if source_doc is not None else b""
    source_signature = hashlib.sha256(content_bytes + source_text.encode("utf-8")).hexdigest()
    if st.sidebar.button("提取并转换", type="primary", width="stretch"):
        try:
            document_text = docx_to_text(content_bytes) if content_bytes else ""
            combined_text = "\n\n".join(x for x in (document_text, source_text) if x.strip())
            review_df, processed_df, missing, notices = convert_text(
                combined_text, use_llm=extract_with_llm, api_key=extraction_key,
            )
            st.session_state.text_import = (
                source_signature, review_df, processed_df, missing, notices,
            )
        except ValueError as error:
            st.session_state.pop("text_import", None)
            st.sidebar.error(str(error))
    converted = st.session_state.get("text_import")
    if converted and converted[0] == source_signature:
        _, review_df, processed_df, missing, notices = converted
        uploaded_df = processed_df
        uploaded_enterprises = load_reviewed_enterprises(processed_df, missing)
        st.sidebar.success(f"已提取 {len(review_df)} 家企业，{len(uploaded_enterprises)} 家进入模拟")
        for notice in notices:
            st.sidebar.warning(notice)
        st.sidebar.caption("转换文件包含全部提取记录及待核实字段；分析仅使用愿意发行的企业。")
        st.sidebar.download_button(
            "下载 CSV", data=export_csv(review_df), file_name="提取企业数据.csv",
            mime="text/csv", width="stretch",
        )
        st.sidebar.download_button(
            "下载 Excel", data=export_excel(review_df), file_name="提取企业数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
        )
    else:
        st.sidebar.info("上传文档或粘贴文字后，点击“提取并转换”")

if using_group_list:
    simulation_rounds = len(uploaded_enterprises)
else:
    simulation_rounds = st.sidebar.slider("模拟轮数", 50, 500, 200, 50)

simulation_disabled = using_group_list and not uploaded_enterprises
st.sidebar.markdown("---")
st.sidebar.caption("规则模式使用内置假设和概率判断，无需调用模型。")
st.sidebar.caption("LLM 模式调用模型，调用或解析失败的环节会降级到规则判断。")
if use_llm and (not api_key or api_key == "1"):
    st.sidebar.info("输入自己的 API Key 后即可运行 LLM 模拟。")
    simulation_disabled = True


# ============================================================
# 主页面
# ============================================================

st.title("绿色债券发行模拟")
st.markdown("从发行意愿到发行结果，查看每个环节的判断与阻碍。")

# ---- 数据预览 ----
enterprises = uploaded_enterprises
with st.expander("企业名单与数据预览", expanded=False):
    if using_group_list:
        if uploaded_df is None:
            st.info("导入企业名单或提取 Word／文字后可预览")
        else:
            st.dataframe(uploaded_df, hide_index=True, width="stretch")
            if data_source == "Word / 文字提取":
                st.caption("“待核实字段”是原文没有提供的信息。对应模拟值是试算占位值，请结合原文核对。")
    else:
        count_col, button_col = st.columns(2)
        sample_count = int(count_col.number_input(
            "示例企业数量", min_value=1, max_value=500, value=20, step=1,
        ))
        regenerate = button_col.button("重新生成示例数据")
        samples = st.session_state.get("sample_enterprises", [])
        if regenerate or len(samples) != sample_count:
            samples = generate_sample_enterprises(sample_count)
            st.session_state.sample_enterprises = samples
        enterprises = samples
        st.caption(f"当前有 {len(samples)} 家示例企业。单次与批量模拟均使用下表中的企业。")
        sample_df = pd.DataFrame(samples)
        st.dataframe(
            sample_df, hide_index=True, width="stretch",
            column_config={
                "code": "企业编号", "name": "企业名称", "ownership": "企业性质",
                "industry": "行业", "scale": "企业规模", "credit_rating": "信用评级",
                "asset_liability_ratio": st.column_config.NumberColumn("资产负债率", format="percent"),
                "green_level": "绿色程度", "issue_scale": "发行规模（亿元）",
                "has_green_project_cert": "绿色项目证明", "has_financial_report": "财务报告",
                "has_financing_plan": "融资方案", "material_complete": "材料完整",
                "green_project_authentic": "项目证明真实", "consecutive_profit": "连续盈利",
                "compliance_score": "合规评分", "financing_plan_quality": "融资方案质量",
                "willingness_score": "发行意愿得分",
            },
        )

overview = st.columns(3)
overview[0].metric("当前企业", f"{len(enterprises)} 家")
overview[1].metric("数据来源", "Word / 文字" if data_source == "Word / 文字提取" else ("第一组名单" if using_group_list else "合成示例"))
overview[2].metric("决策方式", "LLM + 硬规则" if use_llm else "规则模拟")
st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["单次模拟", "批量实验", "统计与文本分析", "流程与说明"])


# ---- Tab 1: 单次模拟 ----
with tab1:
    st.subheader("单次发行链路模拟")

    if enterprises:
        selected_index = st.selectbox(
            "选择企业", range(len(enterprises)),
            format_func=lambda index: f"{enterprises[index]['code']} · {enterprises[index]['name']}",
        )

    if st.button("运行一次模拟", type="primary", disabled=simulation_disabled):
        st.session_state.single_result = run_single_simulation(
            enterprise=enterprises[selected_index],
            certifier_type=certifier_type,
            regulator_type=regulator_type,
            underwriter_type=underwriter_type,
            investor_type=investor_type,
            use_llm=use_llm,
            api_key=api_key,
        )

    if "single_result" in st.session_state:
        result = st.session_state.single_result
        st.caption(f"最近一次模拟 · {result['enterprise_code']} · {result['enterprise_name']}")
        # 显示结果
        col1, col2 = st.columns(2)

        with col1:
            st.metric("最终结果",
                      "✅ 发行成功" if result["final_result"] == "success" else "❌ 发行失败")
            if result["failure_stage"]:
                stage_names = {
                    "1_enterprise_application": "环节1：企业发起",
                    "2_green_certification": "环节2：绿色认证",
                    "3_regulatory_review": "环节3：监管审核",
                    "4_underwriter_pricing": "环节4：承销商定价",
                    "5_investor_subscription": "环节5：投资认购",
                }
                st.error(f"失败环节：{stage_names.get(result['failure_stage'], result['failure_stage'])}")
                st.warning(f"失败原因：{result['failure_reason']}")

        with col2:
            st.json(result.get("enterprise_data", {}))

        # 各环节流程图
        stages = [
            ("1.企业发起", result.get("stage_1_material", {}).get("passed", False)),
            ("2.绿色认证", result.get("stage_2_certifier", {}).get("passed", False) if result.get("stage_2_certifier") else None),
            ("3.监管审核", result.get("stage_3_regulator", {}).get("passed", False) if result.get("stage_3_regulator") else None),
            ("4.承销定价", result.get("stage_4_underwriter", {}).get("passed", False) if result.get("stage_4_underwriter") else None),
            ("5.投资认购", result.get("stage_5_investor", {}).get("passed", False) if result.get("stage_5_investor") else None),
        ]

        st.markdown("### 发行链路流转")
        cols = st.columns(5)
        for i, (name, status) in enumerate(stages):
            with cols[i]:
                if status is True:
                    st.success(f"✅ {name}")
                elif status is False:
                    st.error(f"❌ {name}")
                else:
                    st.info(f"⏸ {name}")

        with st.expander("查看各环节判断依据"):
            for _, label, key in STAGES:
                stage_result = result.get(key)
                if stage_result:
                    st.markdown(f"**{label}**")
                    st.write(stage_result.get("reason", "历史理由未记录"))
                    detail = stage_result.get("detail") or {}
                    if detail.get("source") in SOURCES:
                        st.caption(SOURCES[detail["source"]])
                    if detail.get("conditions"):
                        st.write("补件条件", detail["conditions"])


# ---- Tab 2: 批量实验 ----
with tab2:
    st.subheader("批量仿真实验")

    batch_mode = st.radio("实验方式", ["P0–P6 场景对照", "单个自定义场景"], horizontal=True)
    use_presets = batch_mode == "P0–P6 场景对照"

    if use_presets:
        cohort_size = len(enterprises)
        if cohort_size:
            st.caption(f"七个场景使用同一批 {cohort_size} 家企业，共 {cohort_size * len(PRESET_SCENARIOS)} 次模拟。")
        if st.button("运行 P0–P6", type="primary", disabled=simulation_disabled):
            experiment_id = f"preset_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
            with st.spinner("正在运行 P0–P6 场景…"):
                scenario_results = run_preset_experiment(enterprises, use_llm=use_llm, api_key=api_key)
                scenario_stats = {
                    scenario_id: analyze_results(results)
                    for scenario_id, results in scenario_results.items()
                }
                for scenario_id, stats in scenario_stats.items():
                    stats["gained"], stats["lost"] = paired_changes(
                        scenario_results["P0"], scenario_results[scenario_id],
                    )
                db.save_experiment({
                    f"{experiment_id}_{scenario_id}": results
                    for scenario_id, results in scenario_results.items()
                })
            st.session_state.preset_summary = (experiment_id, scenario_stats, data_source, use_llm)

        if "preset_summary" in st.session_state:
            experiment_id, scenario_stats, used_source, used_llm = st.session_state.preset_summary
            source = used_source
            decision_mode = "LLM 模式" if used_llm else "规则模式"
            st.caption(f"最近一次场景对照 · {source} · {decision_mode} · {experiment_id}")
            baseline = scenario_stats["P0"]
            rows = [
                {
                    "场景": scenario_id,
                    "条件": PRESET_SCENARIOS[scenario_id][0],
                    "企业数": stats["total_simulations"],
                    "发行成功数": stats["successes"],
                    "成功率 (%)": round(100 * stats["successes"] / stats["total_simulations"], 2),
                    "较 P0 变化 (百分点)": round(
                        100 * (stats["successes"] - baseline["successes"]) / stats["total_simulations"], 2
                    ),
                    "较 P0 新增成功": stats["gained"],
                    "较 P0 转为失败": stats["lost"],
                    "失败最多环节": stats["bottleneck_stage"] if stats["failures"] else "无",
                    "LLM 降级企业数": stats.get("llm_fallback_count", 0),
                }
                for scenario_id, stats in scenario_stats.items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if not use_presets and using_group_list and uploaded_enterprises:
        st.caption(f"本场景将对名单中的 {simulation_rounds} 家企业各运行一次。")

    if not use_presets and st.button("开始批量实验", type="primary", disabled=simulation_disabled):
        with st.spinner(f"正在运行 {simulation_rounds} 次模拟实验..."):
            scenario = {
                "certifier_type": certifier_type,
                "regulator_type": regulator_type,
                "underwriter_type": underwriter_type,
                "investor_type": investor_type,
            }

            results = run_batch_simulation(
                enterprises=enterprises,
                scenario=scenario,
                rounds=simulation_rounds,
                use_llm=use_llm,
                api_key=api_key,
                each_enterprise_once=using_group_list,
            )

            # 保存到数据库
            scenario_name = f"custom_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
            db.save_batch(results, scenario_name)

            # 统计分析
            stats = analyze_results(results)

        st.success(f"✅ 实验完成！共运行 {simulation_rounds} 次模拟")

        # 关键指标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("发行成功率", f"{stats['success_rate']:.1%}")
        with col2:
            st.metric("瓶颈环节", stats['bottleneck_stage'])
        with col3:
            st.metric("总失败次数", stats['failures'])

        # 环节失败率柱状图
        st.markdown("### 各环节失败率")
        stage_data = []
        for stage, info in stats["stage_failure_rates"].items():
            stage_data.append({"环节": stage, "失败率": info["rate"], "失败次数": info["count"]})
        df_stage = pd.DataFrame(stage_data)

        fig = px.bar(df_stage, x="环节", y="失败率", text="失败次数",
                     title="各环节失败率分布",
                     color="失败率", color_continuous_scale="Reds")
        st.plotly_chart(fig, width="stretch")

        # 失败原因饼图
        st.markdown("### 失败原因归因")
        reason_data = []
        for reason, count in stats["failure_reasons"].items():
            reason_data.append({"失败原因": reason, "次数": count})
        df_reason = pd.DataFrame(reason_data)
        if not df_reason.empty:
            fig2 = px.pie(df_reason, values="次数", names="失败原因",
                         title="失败原因分类占比")
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("本次实验全部成功。")


# ---- Tab 3: 统计分析 ----
with tab3:
    st.subheader("历史实验统计")
    saved_scenarios = db.list_scenarios()
    if not saved_scenarios:
        st.info("运行一次批量实验后可查看历史统计。")
    else:
        selected = st.selectbox("选择历史场景", saved_scenarios)
        records = db.load_scenario(selected)
        with st.expander("查看该场景实际使用的企业数据"):
            st.dataframe(pd.DataFrame([record["enterprise_data"] for record in records]),
                         hide_index=True, width="stretch")
        stats = analyze_results(records)
        st.metric("发行成功率", f"{stats['success_rate']:.1%}")
        stage_df = pd.DataFrame([
            {"环节": name, "到达": data["reached"], "通过": data["passed"],
             "失败": data["count"], "占全样本失败率": data["rate"],
             "到达后失败率": data["conditional_failure_rate"]}
            for name, data in stats["stage_failure_rates"].items()
        ])
        st.plotly_chart(px.funnel(stage_df, y="环节", x="通过", title="各环节通过数"), width="stretch")
        st.dataframe(stage_df, hide_index=True, width="stretch",
                     column_config={
                         "占全样本失败率": st.column_config.NumberColumn(format="percent"),
                         "到达后失败率": st.column_config.NumberColumn(format="percent"),
                     })
        ownership = pd.DataFrame(ownership_survival(records))
        figure = px.line(ownership, x="环节", y="留存比例", color="企业性质", markers=True,
                         hover_data=["企业数"], title="不同企业性质的阶段留存")
        figure.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(figure, width="stretch")
        if stats["failure_reasons"]:
            st.dataframe(pd.DataFrame(stats["failure_reasons"].items(), columns=["环节及原因", "失败数"]),
                         hide_index=True, width="stretch")
        experiment_id, _, suffix = selected.rpartition("_")
        related = [name for name in saved_scenarios if name.rpartition("_")[0] == experiment_id]
        if experiment_id.startswith("preset_") and suffix in PRESET_SCENARIOS:
            histories = {
                name.rsplit("_", 1)[-1]: db.load_scenario(name)
                for name in sorted(related)
            }
            heatmap = pd.DataFrame({
                key: {stage: values["conditional_failure_rate"]
                      for stage, values in analyze_results(history)["stage_failure_rates"].items()}
                for key, history in histories.items()
            }).T
            st.plotly_chart(px.imshow(heatmap, zmin=0, zmax=1, color_continuous_scale="Reds",
                                       title="各场景到达后失败率", aspect="auto",
                                       labels={"x": "环节", "y": "场景", "color": "失败率"}),
                            width="stretch")
            scenario_records = [record for history in histories.values() for record in history]
        else:
            scenario_records = records
        st.download_button("下载本次实验明细 JSON",
                           data=json.dumps(scenario_records, ensure_ascii=False, indent=2),
                           file_name=f"{experiment_id if len(related) > 1 else selected}.json",
                           mime="application/json")

        st.divider()
        st.subheader("决策理由文本分析")
        st.caption("jieba 对已保存的判断理由分词。词频按理由条数统计，重复实验按记录计数。")
        include_batch = st.checkbox(
            "分析同批次全部 P0–P6 场景", value=False,
            disabled=not (experiment_id.startswith("preset_") and suffix in PRESET_SCENARIOS),
        )
        analysis_scenarios = histories if include_batch else {selected: records}
        text_records = [
            {**record, "scenario_name": name}
            for name, items in analysis_scenarios.items() for record in items
        ]
        scope_col, source_col, stage_col = st.columns(3)
        scope = scope_col.selectbox(
            "理由范围", ["failures", "all"],
            format_func=lambda value: {"failures": "首次失败理由", "all": "全部已到达环节理由"}[value],
        )
        source = source_col.selectbox(
            "判断来源", ["all", *SOURCES],
            format_func=lambda value: {"all": "全部来源", **SOURCES}[value],
        )
        stage_labels = {"all": "全部环节", **{code: label for code, label, _ in STAGES}}
        stage_filter = stage_col.selectbox(
            "分析环节", list(stage_labels), format_func=stage_labels.get,
        )
        top_n = st.slider("显示高频词数量", min_value=5, max_value=40, value=20)
        extra_stopwords = tuple(st.text_input(
            "额外停用词", help="用空格分隔，填写本次分析希望排除的词。"
        ).split())
        text_result = cached_text_analysis(
            text_records, scope, source, stage_filter, top_n, extra_stopwords,
        )
        st.caption(f"当前筛选共 {text_result['record_count']} 条模拟记录、{text_result['document_count']} 条理由。")
        if text_result["source_counts"]:
            st.write("理由来源", {SOURCES[key]: count for key, count in text_result["source_counts"].items()})
        terms = pd.DataFrame(text_result["terms"])
        if terms.empty:
            st.info("当前筛选没有可统计的词语。可调整范围、来源或停用词。")
        else:
            display_terms = terms.rename(columns={
                "term": "词语", "count": "出现次数",
                "document_count": "包含该词的理由数", "document_rate": "理由覆盖率",
            })
            st.plotly_chart(px.bar(
                display_terms, x="包含该词的理由数", y="词语", orientation="h",
                category_orders={"词语": display_terms["词语"].tolist()},
            ), width="stretch")
            st.dataframe(display_terms, hide_index=True, width="stretch",
                         column_config={"理由覆盖率": st.column_config.NumberColumn(format="percent")})
            st.download_button("下载词频 CSV", terms.to_csv(index=False).encode("utf-8-sig"),
                               file_name="reason_terms.csv", mime="text/csv")
        with st.expander("核对原始理由"):
            st.dataframe(pd.DataFrame(text_result["documents"]), hide_index=True, width="stretch")
        st.download_button("下载文本分析 JSON",
                           json.dumps(text_result, ensure_ascii=False, indent=2),
                           file_name="text_analysis.json", mime="application/json")

        st.subheader("汇报用静态图表")
        st.caption("使用 matplotlib／seaborn 生成 PNG，并打包统计表、文本原文和分析口径。")
        if st.checkbox("生成并预览静态报告"):
            with st.spinner("正在生成静态图表…"):
                report = cached_report(
                    analysis_scenarios, scope, source, stage_filter, top_n, extra_stopwords,
                )
            if report["metadata"]["font"] is None:
                st.warning("当前环境缺少中文绘图字体，请安装 Noto Sans CJK 后重启应用。")
            for image_name, payload in report["images"].items():
                st.image(payload)
                st.download_button("下载 " + image_name, payload,
                                   file_name=image_name, mime="image/png", key=image_name)
            st.download_button("下载完整分析报告 ZIP", report["archive"],
                               file_name="analysis_report.zip", mime="application/zip")


# ---- Tab 4: 发行链路说明 ----
with tab4:
    st.subheader("绿色债券发行全流程")

    st.markdown("**数据准备**：使用 CSV／Excel 名单，或上传 `.docx`／粘贴文字，提取后核对待核实字段。转换后的 CSV／Excel 可下载；缺失字段使用明确标注的模拟占位值继续试算。明确不发行的企业不进入链路。")

    flow = [
        ("企业发起", "核对绿色项目证明、财务报告和融资方案。"),
        ("绿色认证", "检查目录匹配、材料真实性和项目绿色程度。"),
        ("监管审核", "评估财务、合规与信息披露，读取认证意见。"),
        ("承销定价", "决定是否承销，形成票面利率和路演信息。"),
        ("投资认购", "综合评估风险和收益，判断认购是否足额。"),
    ]
    for column, (title, description) in zip(st.columns(5), flow):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)
    st.caption("各环节依次执行。拒绝时记录失败并终止，足额认购后记为发行成功。")
    st.markdown("""
    ### 核心判定逻辑
    - **硬规则**：材料、目录及财务等程序检查。80% 负债率和连续两年盈利是当前模拟假设，正式实验需核验适用口径
    - **概率性规则**：基于主体类型+企业特征的概率判定
    - **交互性规则**：后一环节的结果依赖前一环节的输出（如认购概率受定价利率影响）
    - **失败终止**：任一环节不通过，整个链路终止
    - **有条件通过**：携带补件意见继续模拟，当前版本尚无实际补件复审回路
    - **文本分析**：jieba 统计判断理由用词，可按环节和来源筛选，并核对原文。词频本身不能验证因果
    - **图表导出**：Plotly 用于交互查看，matplotlib／seaborn 生成汇报用 PNG 和 ZIP 报告
    """)
