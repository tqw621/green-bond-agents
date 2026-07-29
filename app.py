"""
绿色债券发行多主体LLM智能体链路模拟系统 — Web界面

基于 Streamlit 构建交互式界面：
- 场景配置
- 单次模拟运行
- 批量实验
- 结果统计可视化
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from engine import (
    generate_sample_enterprises,
    run_single_simulation,
    run_batch_simulation,
    analyze_results,
    validate_enterprise_dataframe,
    load_enterprises_from_df,
)
from database import SimulationDB
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

# 初始化数据库
db = SimulationDB()


def get_enterprises(n: int = 20) -> list:
    """获取企业数据：优先使用上传数据，否则随机生成"""
    if st.session_state.data_source_mode == "uploaded" and st.session_state.uploaded_enterprises:
        enterprises = st.session_state.uploaded_enterprises
        # 如果上传数据不够 n 条，循环使用
        if len(enterprises) < n:
            return enterprises * (n // len(enterprises) + 1)
        return enterprises
    return generate_sample_enterprises(n)

# 初始化会话状态（上传数据）
if "uploaded_enterprises" not in st.session_state:
    st.session_state.uploaded_enterprises = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "data_source_mode" not in st.session_state:
    st.session_state.data_source_mode = "sample"  # "sample" 或 "uploaded"


# ============================================================
# 侧边栏 — 场景配置
# ============================================================

st.sidebar.title("🌿 绿色债券发行链路模拟")

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 场景参数配置")

certifier_type = st.sidebar.selectbox(
    "绿色认证机构类型",
    options=list(CERTIFIER_TYPES.keys()),
    format_func=lambda x: CERTIFIER_TYPES[x]["name"],
    index=1,
)

regulator_type = st.sidebar.selectbox(
    "监管审核机构类型",
    options=list(REGULATOR_TYPES.keys()),
    format_func=lambda x: REGULATOR_TYPES[x]["name"],
    index=1,
)

underwriter_type = st.sidebar.selectbox(
    "承销商类型",
    options=list(UNDERWRITER_TYPES.keys()),
    format_func=lambda x: UNDERWRITER_TYPES[x]["name"],
    index=1,
)

investor_type = st.sidebar.selectbox(
    "投资机构类型",
    options=list(INVESTOR_TYPES.keys()),
    format_func=lambda x: INVESTOR_TYPES[x]["name"],
    index=2,
)

simulation_rounds = st.sidebar.slider("模拟轮数", 50, 500, 200, 50)
use_llm = st.sidebar.checkbox("启用 LLM API 决策", value=False,
                              help="勾选后将调用 LLM API 进行决策（消耗 tokens），不勾选则使用规则引擎")

# ---- 企业数据源 ----
st.sidebar.markdown("---")
st.sidebar.subheader("📂 企业数据源")

data_source = st.sidebar.radio(
    "选择数据来源",
    options=["示例数据（随机生成）", "上传真实数据"],
    horizontal=True,
    key="data_source_radio",
)

if data_source == "上传真实数据":
    uploaded_file = st.sidebar.file_uploader(
        "上传 CSV 或 Excel 文件",
        type=["csv", "xlsx", "xls"],
        help="支持中/英文列名。必填列：企业编号、信用评级、资产负债率、绿色程度、绿色项目证明、财务报告、融资方案",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")

            is_valid, msg, processed_df = validate_enterprise_dataframe(df)

            if is_valid:
                st.session_state.uploaded_df = processed_df
                st.session_state.uploaded_enterprises = load_enterprises_from_df(processed_df)
                st.session_state.data_source_mode = "uploaded"
                st.sidebar.success(f"✅ {msg}")
            else:
                st.session_state.data_source_mode = "sample"
                st.session_state.uploaded_enterprises = None
                st.sidebar.error(f"❌ {msg}")
        except Exception as e:
            st.session_state.data_source_mode = "sample"
            st.session_state.uploaded_enterprises = None
            st.sidebar.error(f"❌ 文件读取失败: {e}")

# 当前数据模式提示
if st.session_state.data_source_mode == "uploaded" and st.session_state.uploaded_enterprises:
    n_ents = len(st.session_state.uploaded_enterprises)
    st.sidebar.success(f"📌 当前使用上传数据（{n_ents} 家企业）")
    if st.sidebar.button("🔄 切换回示例数据", use_container_width=True):
        st.session_state.uploaded_enterprises = None
        st.session_state.uploaded_df = None
        st.session_state.data_source_mode = "sample"
        st.rerun()
else:
    st.sidebar.info("📌 当前使用随机示例数据")

st.sidebar.markdown("---")
st.sidebar.caption("💡 非 LLM 模式下使用内置概率规则引擎，速度快且零成本")
st.sidebar.caption("🤖 LLM 模式下调用大模型进行智能决策，结果更贴近真实行为")


# ============================================================
# 主页面
# ============================================================

st.title("🌿 绿色债券发行多主体智能体链路模拟系统")
st.markdown("模拟企业从「有发行意愿」到「发行成功/失败」的完整链路，识别核心梗阻点")

# ---- 数据预览 ----
with st.expander("📋 当前数据预览", expanded=False):
    if st.session_state.data_source_mode == "uploaded" and st.session_state.uploaded_enterprises:
        st.success(f"✅ 已加载 {len(st.session_state.uploaded_enterprises)} 条真实企业数据")
        if st.session_state.uploaded_df is not None:
            st.dataframe(st.session_state.uploaded_df, use_container_width=True)
    else:
        st.info("📊 使用随机生成示例数据（每次模拟随机生成 20 家样本企业）")
        with st.columns(3)[0]:
            st.caption("示例企业特征：")
            st.caption("• 央企/国企/民企混合")
            st.caption("• 6 大绿色行业随机分布")
            st.caption("• 信用评级 AAA~BB")
            st.caption("• 资产负债率 30%~90%")
            st.caption("• 85% 材料完整率")

tab1, tab2, tab3, tab4 = st.tabs(["🚀 单次模拟", "📊 批量实验", "📈 统计分析", "📋 发行链路说明"])


# ---- Tab 1: 单次模拟 ----
with tab1:
    st.subheader("单次发行链路模拟")

    if st.button("▶️ 运行一次模拟", type="primary"):
        enterprises = get_enterprises(1)
        result = run_single_simulation(
            enterprise=enterprises[0],
            certifier_type=certifier_type,
            regulator_type=regulator_type,
            underwriter_type=underwriter_type,
            investor_type=investor_type,
            use_llm=use_llm,
        )

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


# ---- Tab 2: 批量实验 ----
with tab2:
    st.subheader("批量仿真实验")

    if st.button("🚀 开始批量实验", type="primary"):
        with st.spinner(f"正在运行 {simulation_rounds} 次模拟实验..."):
            enterprises = get_enterprises(20)
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
            )

            # 保存到数据库
            scenario_name = f"{certifier_type}_{regulator_type}_{underwriter_type}_{investor_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
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
        st.plotly_chart(fig, use_container_width=True)

        # 失败原因饼图
        st.markdown("### 失败原因归因")
        reason_data = []
        for reason, count in stats["failure_reasons"].items():
            reason_data.append({"失败原因": reason, "次数": count})
        df_reason = pd.DataFrame(reason_data)
        fig2 = px.pie(df_reason, values="次数", names="失败原因",
                      title="失败原因分类占比")
        st.plotly_chart(fig2, use_container_width=True)


# ---- Tab 3: 统计分析 ----
with tab3:
    st.subheader("历史实验统计")
    st.info("📌 在此查看数据库中保存的历史模拟实验记录和统计结果。")
    st.markdown("（后续版本将支持多场景对比分析、企业异质性分析等功能）")


# ---- Tab 4: 发行链路说明 ----
with tab4:
    st.subheader("绿色债券发行全流程")

    st.markdown("""
    ### 发行链路（严格按顺序）

    ```
    企业发起申请
        │
        ▼
    ┌─────────────────────────────┐
    │ 环节1：企业发起              │
    │ 提交绿色项目证明、财务报告、   │
    │ 融资方案等发行材料            │
    │ 判定：材料完整性             │
    └─────────────┬───────────────┘
                  │ ✅ 材料完整
                  ▼
    ┌─────────────────────────────┐
    │ 环节2：绿色认证机构审核      │
    │ 审核项目绿色程度、材料真实性  │
    │ 类型：严格型/中等型/宽松型    │
    └─────────────┬───────────────┘
                  │ ✅ 审核通过
                  ▼
    ┌─────────────────────────────┐
    │ 环节3：监管部门审核          │
    │ 审核发行资质、融资方案合规性  │
    │ 类型：严监管/中监管/松监管    │
    └─────────────┬───────────────┘
                  │ ✅ 审核通过
                  ▼
    ┌─────────────────────────────┐
    │ 环节4：承销商定价&路演       │
    │ 确定发行利率、开展路演        │
    │ 类型：激进型/稳健型/保守型    │
    └─────────────┬───────────────┘
                  │ ✅ 定价完成
                  ▼
    ┌─────────────────────────────┐
    │ 环节5：投资机构认购          │
    │ 决定是否认购及认购规模        │
    │ 类型：绿色偏好/收益偏好/均衡  │
    └─────────────┬───────────────┘
                  │
          ┌───────┴───────┐
          ▼               ▼
     认购达标          认购不足
     ✅ 发行成功       ❌ 发行失败
    ```

    ### 核心判定逻辑
    - **刚性规则**：合规红线（如资产负债率>80%直接不通过）
    - **概率性规则**：基于主体类型+企业特征的概率判定
    - **交互性规则**：后一环节的结果依赖前一环节的输出（如认购概率受定价利率影响）
    - **失败终止**：任一环节不通过，整个链路终止
    """)
