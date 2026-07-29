"""
绿色债券发行全链路模拟引擎

按照以下顺序依次执行：
  企业发起 → 绿色认证 → 监管审核 → 承销商定价&路演 → 投资机构认购 → 发行成功/失败

核心逻辑：
- 顺序交互：前一环节通过后，才进入下一环节
- 结果传递：前一环节决策结果自动传递给后续环节
- 失败终止：任一环节不通过，整个链路终止

两种模式：
- 规则引擎模式（use_llm=False）：纯概率表驱动，无需 API Key
- LLM 模式（use_llm=True）：调用大模型进行智能决策，需配置 .env + API Key
"""

import json
import re
import time
import random
from datetime import datetime
from typing import Dict, Tuple, List, Optional
try:
    import pandas as pd
except ImportError:
    pd = None  # 统计分析功能需要 pandas，核心引擎不依赖

from rules import (
    enterprise_material_check,
    certifier_check,
    regulator_check,
    underwriter_price,
    investor_subscribe,
)
import config

random.seed(config.RANDOM_SEED)


# ============================================================
# LLM 响应解析工具
# ============================================================

def _extract_json(text: str) -> Optional[Dict]:
    """
    从 LLM 返回的文本中提取 JSON 对象

    兼容以下格式：
    1. 纯 JSON 字符串
    2. markdown 代码块包裹的 JSON（```json ... ```）
    3. 文本中夹杂的 JSON 对象
    """
    if not text or "[API_ERROR]" in text:
        return None

    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试提取 markdown 代码块中的 JSON
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试用正则匹配最外层花括号
    brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _parse_llm_result(llm_text: str, key_map: Dict[str, str]) -> Tuple[bool, str, Dict]:
    """
    通用 LLM 结果解析器

    Args:
        llm_text: LLM 返回的原始文本
        key_map: JSON key → 中文标签映射，如 {"passed": "通过", "reason": "原因"}

    Returns:
        (是否解析成功, 原始文本/错误信息, 解析后的字典)
    """
    parsed = _extract_json(llm_text)
    if parsed is None:
        return False, f"[LLM_PARSE_ERROR] 无法解析 JSON: {llm_text[:200]}", {}

    # 标准化 passed/will_subscribe/success 等布尔字段
    for bool_key in ["passed", "will_subscribe", "success"]:
        if bool_key in parsed:
            val = parsed[bool_key]
            if isinstance(val, str):
                parsed[bool_key] = val.lower() in ("true", "yes", "是")
            elif isinstance(val, bool):
                parsed[bool_key] = val

    return True, "", parsed


# ============================================================
# 企业数据模板（示例）
# ============================================================

def generate_sample_enterprises(n: int = 10) -> List[Dict]:
    """
    生成有发行意愿的样本企业数据

    在实际实验中，这里应该用「决策模拟组」筛选出的有发行意愿的企业数据
    """
    ownerships = ["央企", "国企", "民企"]
    industries = ["清洁能源", "节能环保", "绿色建筑", "新能源汽车", "传统制造转型", "生态修复"]
    scales = ["大型", "中型", "小型"]
    credit_ratings = ["AAA", "AA", "A", "BBB", "BB"]
    green_levels = ["high", "medium", "low"]

    enterprises = []
    for i in range(n):
        credit = random.choice(credit_ratings)
        green = random.choice(green_levels)
        # 信用评级与资产负债率相关
        if credit in ("AAA", "AA"):
            asset_liability = round(random.uniform(0.30, 0.60), 2)
        elif credit in ("A",):
            asset_liability = round(random.uniform(0.50, 0.75), 2)
        else:
            asset_liability = round(random.uniform(0.65, 0.90), 2)

        ent = {
            "code": f"ENT{i+1:04d}",
            "name": f"样本企业{i+1}号",
            "ownership": random.choice(ownerships),
            "industry": random.choice(industries),
            "scale": random.choice(scales),
            "credit_rating": credit,
            "asset_liability_ratio": asset_liability,
            "consecutive_profit": random.random() < 0.85,      # 85%概率连续盈利
            "compliance_score": round(random.uniform(0.4, 1.0), 2),
            "green_level": green,
            "issue_scale": random.choice([3, 5, 10, 15, 20]),  # 发行规模（亿元）
            "has_green_project_cert": random.random() < 0.90,  # 90%有绿色证明
            "green_project_authentic": random.random() < 0.95, # 95%为真
            "has_financial_report": random.random() < 0.95,
            "has_financing_plan": random.random() < 0.90,
            "material_complete": random.random() < 0.85,
            "financing_plan_quality": random.choice(["完备", "一般", "待完善"]),
            "willingness_score": round(random.uniform(0.7, 1.0), 2),
        }
        enterprises.append(ent)
    return enterprises


# ============================================================
# 真实数据加载与校验
# ============================================================

# 中英文列名映射（用户用中文列名也能识别）
COLUMN_ALIASES = {
    "企业编号": "code", "企业名称": "name", "企业性质": "ownership",
    "所有制": "ownership", "行业": "industry", "所属行业": "industry",
    "规模": "scale", "企业规模": "scale", "信用评级": "credit_rating",
    "资产负债率": "asset_liability_ratio", "连续盈利": "consecutive_profit",
    "合规评分": "compliance_score", "绿色程度": "green_level",
    "发行规模": "issue_scale", "发行规模(亿)": "issue_scale",
    "绿色项目证明": "has_green_project_cert", "财务报告": "has_financial_report",
    "融资方案": "has_financing_plan", "材料完整": "material_complete",
    "绿色项目真实性": "green_project_authentic", "融资方案质量": "financing_plan_quality",
    "发行意愿": "willingness_score",
}

# 必填字段（缺失任何一个都会校验失败）
REQUIRED_FIELDS = [
    "code", "credit_rating", "asset_liability_ratio", "green_level",
    "has_green_project_cert", "has_financial_report", "has_financing_plan",
]

# 可选字段及其默认值
OPTIONAL_DEFAULTS = {
    "name": "未命名企业", "ownership": "民企", "industry": "传统制造转型",
    "scale": "中型", "consecutive_profit": True, "compliance_score": 0.70,
    "issue_scale": 10, "material_complete": True, "green_project_authentic": True,
    "financing_plan_quality": "一般", "willingness_score": 0.85,
}


def _parse_bool(value) -> bool:
    """将各种布尔表达统一转为 bool：支持 是/否、有/无、true/false、1/0"""
    if isinstance(value, bool):
        return value
    if pd is not None and hasattr(pd, 'isna') and pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s in ("true", "yes", "是", "有", "1", "1.0")


def validate_enterprise_dataframe(df: "pd.DataFrame"):
    """
    校验上传的 DataFrame 并标准化列名。

    1. 将中文列名映射为英文
    2. 检查必填列是否齐全
    3. 验证数据类型和取值范围
    4. 为缺失的可选字段填充默认值

    Returns:
        (is_valid: bool, message: str, processed_df: pd.DataFrame | None)
    """
    if pd is None:
        return False, "pandas 未安装，无法处理上传文件", None

    df = df.copy()

    # --- 1. 列名标准化 ---
    df.columns = [str(c).strip() for c in df.columns]
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    # --- 2. 检查必填列 ---
    existing_cols = set(df.columns)
    missing = [c for c in REQUIRED_FIELDS if c not in existing_cols]
    if missing:
        # 给出友好的中文提示
        reverse_map = {v: k for k, v in COLUMN_ALIASES.items()}
        missing_cn = [reverse_map.get(m, m) for m in missing]
        return False, f"缺少必填列: {', '.join(missing_cn)}（英文列名: {', '.join(missing)}）", None

    # --- 3. 类型转换 ---
    # 布尔字段
    bool_fields = ["consecutive_profit", "has_green_project_cert",
                   "green_project_authentic", "has_financial_report",
                   "has_financing_plan", "material_complete"]
    for col in bool_fields:
        if col in df.columns:
            df[col] = df[col].apply(_parse_bool)

    # 数值字段
    for col in ["asset_liability_ratio", "compliance_score", "issue_scale", "willingness_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 文本字段
    for col in ["credit_rating", "green_level", "ownership", "industry", "scale",
                "financing_plan_quality", "code", "name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # --- 4. 取值范围校验 ---
    errors = []
    if df["asset_liability_ratio"].isna().any() or \
       (df["asset_liability_ratio"] < 0).any() or (df["asset_liability_ratio"] > 1.5).any():
        errors.append("资产负债率应在 0-1.5 之间")

    if "compliance_score" in df.columns:
        if df["compliance_score"].isna().any() or \
           (df["compliance_score"] < 0).any() or (df["compliance_score"] > 1).any():
            errors.append("合规评分应在 0-1 之间")

    # 绿色程度：支持中英文
    if not df["green_level"].isin(["high", "medium", "low"]).all():
        green_cn_map = {"高": "high", "中": "medium", "低": "low"}
        if df["green_level"].isin(green_cn_map.keys()).all():
            df["green_level"] = df["green_level"].map(green_cn_map)
        else:
            errors.append("绿色程度必须为 high/medium/low 或 高/中/低")

    # 信用评级
    valid_ratings = {"AAA", "AA", "A", "BBB", "BB"}
    if not df["credit_rating"].isin(valid_ratings).all():
        errors.append("信用评级必须为 AAA/AA/A/BBB/BB 之一")

    if errors:
        return False, "数据校验失败: " + "; ".join(errors), None

    # --- 5. 填充可选字段默认值 ---
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # --- 6. 自动生成 code ---
    if "code" not in df.columns or df["code"].isna().any():
        df["code"] = [f"ENT{i+1:04d}" for i in range(len(df))]

    return True, f"校验通过：{len(df)} 条记录，{len(df.columns)} 个字段", df


def load_enterprises_from_df(df: "pd.DataFrame") -> List[Dict]:
    """
    将校验后的 DataFrame 转换为企业字典列表。

    确保所有字段都有正确的 Python 原生类型（非 numpy 类型），
    以便与现有规则引擎和 LLM 链路无缝对接。
    """
    enterprises = []
    for _, row in df.iterrows():
        ent = {
            "code": str(row.get("code", "")),
            "name": str(row.get("name", "未命名")),
            "ownership": str(row.get("ownership", "民企")),
            "industry": str(row.get("industry", "")),
            "scale": str(row.get("scale", "中型")),
            "credit_rating": str(row.get("credit_rating", "A")),
            "asset_liability_ratio": float(row.get("asset_liability_ratio", 0.6)),
            "consecutive_profit": bool(row.get("consecutive_profit", True)),
            "compliance_score": float(row.get("compliance_score", 0.7)),
            "green_level": str(row.get("green_level", "medium")),
            "issue_scale": float(row.get("issue_scale", 10)),
            "has_green_project_cert": bool(row.get("has_green_project_cert", True)),
            "green_project_authentic": bool(row.get("green_project_authentic", True)),
            "has_financial_report": bool(row.get("has_financial_report", True)),
            "has_financing_plan": bool(row.get("has_financing_plan", True)),
            "material_complete": bool(row.get("material_complete", True)),
            "financing_plan_quality": str(row.get("financing_plan_quality", "一般")),
            "willingness_score": float(row.get("willingness_score", 0.8)),
        }
        enterprises.append(ent)
    return enterprises


# ============================================================
# 单次发行链路模拟
# ============================================================

def run_single_simulation(
    enterprise: Dict,
    certifier_type: str = "moderate",
    regulator_type: str = "moderate",
    underwriter_type: str = "moderate",
    investor_type: str = "balanced",
    use_llm: bool = False,
) -> Dict:
    """
    执行一次完整的绿色债券发行链路模拟

    Args:
        enterprise: 企业数据字典
        certifier_type: 认证机构类型 (strict/moderate/lenient)
        regulator_type: 监管类型 (strict/moderate/lenient)
        underwriter_type: 承销商类型 (aggressive/moderate/conservative)
        investor_type: 投资机构类型 (green_focused/return_focused/balanced)
        use_llm: 是否调用 LLM API 进行决策（False 时仅用规则引擎）

    Returns:
        完整的模拟结果记录
    """
    result = {
        "simulation_time": datetime.now().isoformat(),
        "enterprise_code": enterprise["code"],
        "enterprise_name": enterprise["name"],
        "enterprise_data": enterprise,
        # 场景参数
        "certifier_type": certifier_type,
        "regulator_type": regulator_type,
        "underwriter_type": underwriter_type,
        "investor_type": investor_type,
        "use_llm": use_llm,
        # 各环节结果
        "stage_1_material": None,
        "stage_2_certifier": None,
        "stage_3_regulator": None,
        "stage_4_underwriter": None,
        "stage_5_investor": None,
        # 最终结果
        "final_result": None,       # "success" / "failed"
        "failure_stage": None,       # 失败所在环节
        "failure_reason": None,      # 失败原因
    }

    # ---- 环节1：企业发起（材料完整性检查）----
    # 无论是规则模式还是 LLM 模式，材料检查都用刚性规则（有/无，AI 没必要判断）
    passed, reason = enterprise_material_check(enterprise)
    result["stage_1_material"] = {"passed": passed, "reason": reason}
    if not passed:
        result["final_result"] = "failed"
        result["failure_stage"] = "1_enterprise_application"
        result["failure_reason"] = reason
        return result

    if use_llm:
        return _run_llm_simulation(enterprise, result, certifier_type,
                                   regulator_type, underwriter_type, investor_type)
    else:
        return _run_rules_simulation(enterprise, result, certifier_type,
                                     regulator_type, underwriter_type, investor_type)


def _run_rules_simulation(
    enterprise: Dict, result: Dict,
    certifier_type: str, regulator_type: str,
    underwriter_type: str, investor_type: str,
) -> Dict:
    """规则引擎模式：使用 rules.py 中的概率表进行决策（原有逻辑）"""

    # ---- 环节2：绿色认证机构审核 ----
    passed, reason, detail = certifier_check(enterprise, certifier_type)
    result["stage_2_certifier"] = {"passed": passed, "reason": reason, "detail": detail}
    if not passed:
        result["final_result"] = "failed"
        result["failure_stage"] = "2_green_certification"
        result["failure_reason"] = reason
        return result

    # ---- 环节3：监管部门审核 ----
    passed, reason, detail = regulator_check(enterprise, regulator_type)
    result["stage_3_regulator"] = {"passed": passed, "reason": reason, "detail": detail}
    if not passed:
        result["final_result"] = "failed"
        result["failure_stage"] = "3_regulatory_review"
        result["failure_reason"] = reason
        return result

    # ---- 环节4：承销商定价&路演 ----
    passed, reason, detail = underwriter_price(enterprise, underwriter_type)
    final_rate = detail["final_rate"]
    roadshow_quality = detail["roadshow_quality"]
    result["stage_4_underwriter"] = {"passed": passed, "reason": reason, "detail": detail}
    if not passed:
        result["final_result"] = "failed"
        result["failure_stage"] = "4_underwriter_pricing"
        result["failure_reason"] = reason
        return result

    # ---- 环节5：投资机构认购 ----
    issue_scale = enterprise.get("issue_scale", 10)
    passed, reason, detail = investor_subscribe(
        enterprise, investor_type, final_rate, roadshow_quality, issue_scale
    )
    result["stage_5_investor"] = {"passed": passed, "reason": reason, "detail": detail}
    if not passed:
        result["final_result"] = "failed"
        result["failure_stage"] = "5_investor_subscription"
        result["failure_reason"] = reason
        return result

    # ---- 全部通过 → 发行成功 ----
    result["final_result"] = "success"
    return result


def _run_llm_simulation(
    enterprise: Dict, result: Dict,
    certifier_type: str, regulator_type: str,
    underwriter_type: str, investor_type: str,
) -> Dict:
    """
    LLM 模式：调用 agents.py 中的 LLM 智能体进行决策

    智能体顺序：
    认证机构 → 监管机构 → 承销商 → 投资机构
    每个智能体收到前一环节的结果作为上下文
    刚性红线（资产负债率>80%等）在调用 LLM 前先拦截
    """
    # 延迟导入 agents 模块（仅在 LLM 模式下需要，含 openai 依赖）
    from agents import (
        EnterpriseAgent, CertifierAgent, RegulatorAgent,
        UnderwriterAgent, InvestorAgent,
    )

    # 构建企业上下文（供所有 LLM 智能体使用）
    enterprise_ctx = EnterpriseAgent(enterprise).to_prompt_context()

    # ================================================================
    # 环节2：绿色认证机构审核（LLM）
    # ================================================================
    try:
        certifier = CertifierAgent(certifier_type)
        llm_raw = certifier.review(enterprise_ctx)
        ok, err, parsed = _parse_llm_result(llm_raw, {})
        if ok:
            cert_passed = parsed.get("passed", False)
            cert_reason = parsed.get("reason", llm_raw[:300])
            result["stage_2_certifier"] = {
                "passed": cert_passed,
                "reason": f"[LLM] {cert_reason}",
                "detail": {
                    "certifier_type": certifier_type,
                    "llm_raw_response": llm_raw[:500],
                    "llm_parsed": parsed,
                    "source": "llm",
                },
            }
        else:
            # LLM 返回解析失败，降级到规则引擎
            print(f"  ⚠️  认证机构 LLM 返回解析失败，降级到规则引擎: {err}")
            passed, reason, detail = certifier_check(enterprise, certifier_type)
            result["stage_2_certifier"] = {
                "passed": passed, "reason": f"[降级-规则] {reason}",
                "detail": {**detail, "source": "fallback_rules", "llm_raw": llm_raw[:300]},
            }
            cert_passed = passed
    except Exception as e:
        print(f"  ❌ 认证机构 LLM 调用异常，降级到规则引擎: {e}")
        passed, reason, detail = certifier_check(enterprise, certifier_type)
        result["stage_2_certifier"] = {
            "passed": passed, "reason": f"[降级-规则] {reason}",
            "detail": {**detail, "source": "fallback_rules", "error": str(e)},
        }
        cert_passed = passed

    if not result["stage_2_certifier"]["passed"]:
        result["final_result"] = "failed"
        result["failure_stage"] = "2_green_certification"
        result["failure_reason"] = result["stage_2_certifier"]["reason"]
        return result

    # 构建认证结果摘要，传给后续环节
    cert_summary = json.dumps({
        "passed": result["stage_2_certifier"]["passed"],
        "reason": result["stage_2_certifier"]["reason"],
        "detail": {k: v for k, v in result["stage_2_certifier"]["detail"].items()
                   if k != "llm_raw_response"},
    }, ensure_ascii=False, indent=2)

    # ================================================================
    # 环节3：监管部门审核（刚性红线 + LLM）
    # ================================================================
    # 刚性红线前置检查（不浪费 LLM 调用）
    asset_liability = enterprise.get("asset_liability_ratio", 0.5)
    if asset_liability > 0.80:
        result["stage_3_regulator"] = {
            "passed": False,
            "reason": f"刚性驳回：资产负债率 {asset_liability:.0%} 超过监管红线 80%",
            "detail": {"rigid_reject": True, "source": "rigid_rule"},
        }
        result["final_result"] = "failed"
        result["failure_stage"] = "3_regulatory_review"
        result["failure_reason"] = result["stage_3_regulator"]["reason"]
        return result

    if not enterprise.get("consecutive_profit", True):
        result["stage_3_regulator"] = {
            "passed": False,
            "reason": "刚性驳回：未满足连续两年盈利要求",
            "detail": {"rigid_reject": True, "source": "rigid_rule"},
        }
        result["final_result"] = "failed"
        result["failure_stage"] = "3_regulatory_review"
        result["failure_reason"] = result["stage_3_regulator"]["reason"]
        return result

    # 通过刚性检查，进入 LLM 审核
    try:
        regulator = RegulatorAgent(regulator_type)
        llm_raw = regulator.review(enterprise_ctx, cert_summary)
        ok, err, parsed = _parse_llm_result(llm_raw, {})
        if ok:
            reg_passed = parsed.get("passed", False)
            reg_reason = parsed.get("reason", llm_raw[:300])
            reg_concerns = parsed.get("concerns", [])
            result["stage_3_regulator"] = {
                "passed": reg_passed,
                "reason": f"[LLM] {reg_reason}",
                "detail": {
                    "regulator_type": regulator_type,
                    "concerns": reg_concerns,
                    "rigid_reject": False,
                    "llm_raw_response": llm_raw[:500],
                    "source": "llm",
                },
            }
        else:
            print(f"  ⚠️  监管机构 LLM 返回解析失败，降级到规则引擎: {err}")
            passed, reason, detail = regulator_check(enterprise, regulator_type)
            result["stage_3_regulator"] = {
                "passed": passed, "reason": f"[降级-规则] {reason}",
                "detail": {**detail, "source": "fallback_rules", "llm_raw": llm_raw[:300]},
            }
    except Exception as e:
        print(f"  ❌ 监管机构 LLM 调用异常，降级到规则引擎: {e}")
        passed, reason, detail = regulator_check(enterprise, regulator_type)
        result["stage_3_regulator"] = {
            "passed": passed, "reason": f"[降级-规则] {reason}",
            "detail": {**detail, "source": "fallback_rules", "error": str(e)},
        }

    if not result["stage_3_regulator"]["passed"]:
        result["final_result"] = "failed"
        result["failure_stage"] = "3_regulatory_review"
        result["failure_reason"] = result["stage_3_regulator"]["reason"]
        return result

    # 构建监管审核摘要
    reg_summary = json.dumps({
        "passed": result["stage_3_regulator"]["passed"],
        "reason": result["stage_3_regulator"]["reason"],
    }, ensure_ascii=False, indent=2)

    # ================================================================
    # 环节4：承销商定价&路演（LLM）
    # ================================================================
    try:
        underwriter = UnderwriterAgent(underwriter_type)
        llm_raw = underwriter.price_and_roadshow(enterprise_ctx, cert_summary, reg_summary)
        ok, err, parsed = _parse_llm_result(llm_raw, {})
        if ok:
            issue_rate = float(parsed.get("issue_rate", 0.045))
            rate_rationale = parsed.get("rate_rationale", llm_raw[:300])
            roadshow_strategy = parsed.get("roadshow_strategy", "")
            # 承销商定价环节一般都会通过（定价本身不会"失败"），但检查利率是否合理
            uw_passed = 0.01 <= issue_rate <= 0.20  # 利率在 1%-20% 之间视为合理
            result["stage_4_underwriter"] = {
                "passed": uw_passed,
                "reason": f"[LLM] 发行利率 {issue_rate:.2%} — {rate_rationale}",
                "detail": {
                    "underwriter_type": underwriter_type,
                    "final_rate": round(issue_rate, 4),
                    "rate_rationale": rate_rationale,
                    "roadshow_strategy": roadshow_strategy,
                    "roadshow_quality": 0.75,  # LLM 模式下给一个基准路演质量
                    "llm_raw_response": llm_raw[:500],
                    "source": "llm",
                },
            }
            final_rate = issue_rate
        else:
            print(f"  ⚠️  承销商 LLM 返回解析失败，降级到规则引擎: {err}")
            passed, reason, detail = underwriter_price(enterprise, underwriter_type)
            final_rate = detail["final_rate"]
            result["stage_4_underwriter"] = {
                "passed": passed, "reason": f"[降级-规则] {reason}",
                "detail": {**detail, "source": "fallback_rules", "llm_raw": llm_raw[:300]},
            }
    except Exception as e:
        print(f"  ❌ 承销商 LLM 调用异常，降级到规则引擎: {e}")
        passed, reason, detail = underwriter_price(enterprise, underwriter_type)
        final_rate = detail["final_rate"]
        result["stage_4_underwriter"] = {
            "passed": passed, "reason": f"[降级-规则] {reason}",
            "detail": {**detail, "source": "fallback_rules", "error": str(e)},
        }

    if not result["stage_4_underwriter"]["passed"]:
        result["final_result"] = "failed"
        result["failure_stage"] = "4_underwriter_pricing"
        result["failure_reason"] = result["stage_4_underwriter"]["reason"]
        return result

    roadshow_quality = result["stage_4_underwriter"]["detail"].get("roadshow_quality", 0.75)

    # 构建承销商结果摘要
    uw_summary = json.dumps({
        "passed": result["stage_4_underwriter"]["passed"],
        "issue_rate": final_rate,
        "reason": result["stage_4_underwriter"]["reason"],
    }, ensure_ascii=False, indent=2)

    # ================================================================
    # 环节5：投资机构认购（LLM）
    # ================================================================
    try:
        investor = InvestorAgent(investor_type)
        llm_raw = investor.make_decision(enterprise_ctx, uw_summary, final_rate)
        ok, err, parsed = _parse_llm_result(llm_raw, {})
        if ok:
            will_subscribe = parsed.get("will_subscribe", False)
            subscribe_amount = float(parsed.get("subscribe_amount", 0))
            decision_rationale = parsed.get("decision_rationale", llm_raw[:300])
            issue_scale = enterprise.get("issue_scale", 10)
            inv_passed = will_subscribe and subscribe_amount >= issue_scale
            result["stage_5_investor"] = {
                "passed": inv_passed,
                "reason": f"[LLM] {decision_rationale}",
                "detail": {
                    "investor_type": investor_type,
                    "will_subscribe": will_subscribe,
                    "subscribe_amount": subscribe_amount,
                    "issue_scale": issue_scale,
                    "success": inv_passed,
                    "rate_adjust": 0.0,
                    "roadshow_bonus": 0.0,
                    "llm_raw_response": llm_raw[:500],
                    "source": "llm",
                },
            }
        else:
            print(f"  ⚠️  投资机构 LLM 返回解析失败，降级到规则引擎: {err}")
            issue_scale = enterprise.get("issue_scale", 10)
            passed, reason, detail = investor_subscribe(
                enterprise, investor_type, final_rate, roadshow_quality, issue_scale
            )
            result["stage_5_investor"] = {
                "passed": passed, "reason": f"[降级-规则] {reason}",
                "detail": {**detail, "source": "fallback_rules", "llm_raw": llm_raw[:300]},
            }
    except Exception as e:
        print(f"  ❌ 投资机构 LLM 调用异常，降级到规则引擎: {e}")
        issue_scale = enterprise.get("issue_scale", 10)
        passed, reason, detail = investor_subscribe(
            enterprise, investor_type, final_rate, roadshow_quality, issue_scale
        )
        result["stage_5_investor"] = {
            "passed": passed, "reason": f"[降级-规则] {reason}",
            "detail": {**detail, "source": "fallback_rules", "error": str(e)},
        }

    if not result["stage_5_investor"]["passed"]:
        result["final_result"] = "failed"
        result["failure_stage"] = "5_investor_subscription"
        result["failure_reason"] = result["stage_5_investor"]["reason"]
        return result

    # ---- 全部通过 → 发行成功 ----
    result["final_result"] = "success"
    return result


# ============================================================
# 批量模拟实验
# ============================================================

def run_batch_simulation(
    enterprises: List[Dict],
    scenario: Dict,
    rounds: int = 200,
    use_llm: bool = False,
) -> List[Dict]:
    """
    批量运行模拟实验

    Args:
        enterprises: 企业列表
        scenario: 场景参数 {"certifier_type": "...", "regulator_type": "...", ...}
        rounds: 每个场景运行次数
        use_llm: 是否调用 LLM API

    Returns:
        所有模拟结果的列表
    """
    results = []
    for i in range(rounds):
        # 随机选择一个企业（带替换）
        enterprise = random.choice(enterprises).copy()
        # 为每次模拟生成唯一标识
        original_code = enterprise["code"]
        enterprise["code"] = f"{original_code}_R{i+1}"

        sim_result = run_single_simulation(
            enterprise=enterprise,
            certifier_type=scenario.get("certifier_type", "moderate"),
            regulator_type=scenario.get("regulator_type", "moderate"),
            underwriter_type=scenario.get("underwriter_type", "moderate"),
            investor_type=scenario.get("investor_type", "balanced"),
            use_llm=use_llm,
        )
        results.append(sim_result)

        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/{rounds} 次模拟...")

    return results


# ============================================================
# 统计分析方法
# ============================================================

def analyze_results(results: List[Dict]) -> Dict:
    """
    对模拟结果进行统计分析

    Returns:
        统计结果字典，包含：
        - 整体成功率
        - 各环节失败率
        - 失败原因归因
        - 企业异质性分析
    """
    total = len(results)
    successes = sum(1 for r in results if r["final_result"] == "success")
    failures = total - successes

    # 各环节失败率
    stage_names = {
        "1_enterprise_application": "企业发起",
        "2_green_certification": "绿色认证",
        "3_regulatory_review": "监管审核",
        "4_underwriter_pricing": "承销商定价",
        "5_investor_subscription": "投资认购",
    }
    stage_failures = {}
    for key, name in stage_names.items():
        count = sum(1 for r in results if r["failure_stage"] == key)
        stage_failures[name] = {
            "count": count,
            "rate": round(count / total, 4),
        }

    # 失败原因归因
    failure_reasons = {}
    for r in results:
        if r["failure_reason"]:
            reason = r["failure_reason"]
            if "材料" in reason:
                category = "材料不完整"
            elif "绿色" in reason or "认证" in reason:
                category = "绿色认证不通过"
            elif "资产" in reason or "盈利" in reason or "合规" in reason or "监管" in reason:
                category = "监管审核不通过"
            elif "定价" in reason or "利率" in reason:
                category = "定价失败"
            elif "认购" in reason:
                category = "认购不足"
            else:
                category = "其他"
            failure_reasons[category] = failure_reasons.get(category, 0) + 1

    # LLM 降级统计
    llm_fallbacks = sum(
        1 for r in results
        if r.get("use_llm") and any(
            "降级" in str(r.get(f"stage_{i}_{name}", {}).get("reason", ""))
            for i, name in [(2, "certifier"), (3, "regulator"), (4, "underwriter"), (5, "investor")]
        )
    )

    stats = {
        "total_simulations": total,
        "successes": successes,
        "failures": failures,
        "success_rate": round(successes / total, 4),
        "stage_failure_rates": stage_failures,
        "failure_reasons": failure_reasons,
        "bottleneck_stage": max(stage_failures, key=lambda k: stage_failures[k]["rate"]),
    }
    if llm_fallbacks > 0:
        stats["llm_fallback_count"] = llm_fallbacks
        stats["llm_fallback_rate"] = round(llm_fallbacks / total, 4)
    return stats


# ============================================================
# 便捷运行入口
# ============================================================

if __name__ == "__main__":
    import sys

    use_llm = "--llm" in sys.argv
    print("=" * 60)
    print(f"绿色债券发行链路模拟引擎 - 单次测试 ({'LLM模式' if use_llm else '规则引擎模式'})")
    print("=" * 60)

    # 生成样本企业
    enterprises = generate_sample_enterprises(5)

    # 单次模拟
    print("\n📋 单次模拟：")
    result = run_single_simulation(
        enterprise=enterprises[0],
        certifier_type="moderate",
        regulator_type="moderate",
        underwriter_type="moderate",
        investor_type="balanced",
        use_llm=use_llm,
    )

    print(f"\n模拟结果：{'✅ 成功' if result['final_result'] == 'success' else '❌ 失败'}")
    if result['failure_stage']:
        print(f"失败环节：{result['failure_stage']}")
        print(f"失败原因：{result['failure_reason']}")
    print(f"\n各环节详情：")
    for stage_key in ["stage_1_material", "stage_2_certifier", "stage_3_regulator",
                       "stage_4_underwriter", "stage_5_investor"]:
        s = result.get(stage_key, {})
        if s:
            source = s.get("detail", {}).get("source", "rules")
            print(f"  {stage_key}: {'✅ 通过' if s.get('passed') else '❌ 不通过'} | 来源={source} | {s.get('reason', '')[:100]}")

    # 如果是 LLM 模式，单次测试就够了（避免浪费 token）
    if use_llm:
        print("\n⚠️  LLM 模式下不自动批量运行，如需批量测试请使用 run.py --cli N --llm")
        sys.exit(0)

    # 规则引擎模式：小批量测试
    print("\n" + "=" * 60)
    print("批量模拟测试（50次）")
    print("=" * 60)
    batch_results = run_batch_simulation(
        enterprises=enterprises,
        scenario={
            "certifier_type": "moderate",
            "regulator_type": "moderate",
            "underwriter_type": "moderate",
            "investor_type": "balanced",
        },
        rounds=50,
        use_llm=False,
    )
    stats = analyze_results(batch_results)
    print(f"\n统计结果：成功率 {stats['success_rate']:.1%}")
    print(f"瓶颈环节：{stats['bottleneck_stage']}")
    for stage, info in stats["stage_failure_rates"].items():
        print(f"  {stage}: 失败 {info['count']} 次 ({info['rate']:.1%})")
