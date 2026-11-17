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
- LLM 模式（use_llm=True）：调用大模型进行智能决策，需在本次调用中传入使用者的 API Key
"""

import json
import math
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
    certification_rejection,
    regulatory_rejection,
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
    if not isinstance(parsed, dict):
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
    "firm_id": "code", "firm_name": "name",
    "发行决策": "decision", "是否发行": "decision",
    "符合项目目录": "project_catalog_eligible", "重大违规": "major_violation",
    "市场基准收益率": "market_benchmark_yield", "定价利差(bp)": "pricing_spread_bp",
    "发行人票息上限": "issuer_rate_cap",
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


def validate_enterprise_dataframe(df: "pd.DataFrame", *, willing_only: bool = False, prefiltered: bool = False):
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
    df = df.rename(columns=COLUMN_ALIASES)

    if df.columns.has_duplicates:
        duplicates = df.columns[df.columns.duplicated()].unique()
        return False, f"重复字段: {', '.join(duplicates)}", None

    if df.empty:
        return False, "文件里没有企业记录", None

    original_count = len(df)
    if willing_only and "decision" in df.columns:
        decisions = df["decision"].astype("string").str.strip().str.lower()
        willing = decisions.isin({"发行", "愿意发行", "是", "true", "1", "1.0", "yes"})
        unwilling = decisions.isin({"不发行", "不愿意发行", "否", "false", "0", "0.0", "no"})
        invalid = ~(willing | unwilling)
        if invalid.any():
            rows = ", ".join(str(i + 2) for i, bad in enumerate(invalid) if bad)
            return False, f"发行决策为空或无法识别（表格行号: {rows}）", None
        df = df.loc[willing].copy()
    elif willing_only and not prefiltered:
        return False, "缺少发行决策列。若第一组已筛选名单，请勾选已确认名单仅含愿意发行企业", None

    if df.empty:
        return False, "名单中没有愿意发行的企业", None

    # --- 2. 检查必填列 ---
    existing_cols = set(df.columns)
    missing = [c for c in REQUIRED_FIELDS if c not in existing_cols]
    if missing:
        # 给出友好的中文提示
        reverse_map = {v: k for k, v in COLUMN_ALIASES.items()}
        missing_cn = [reverse_map.get(m, m) for m in missing]
        return False, f"缺少必填列: {', '.join(missing_cn)}（英文列名: {', '.join(missing)}）", None

    if willing_only:
        codes = df["code"].astype("string").str.strip()
        if codes.isna().any() or codes.eq("").any():
            return False, "愿意发行的企业必须填写原始企业编号", None
        if codes.duplicated().any():
            duplicates = ", ".join(codes[codes.duplicated()].unique())
            return False, f"企业编号重复: {duplicates}。请提供每家企业一条最终决策", None
        df["code"] = codes

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

    if "issue_scale" in df and not df["issue_scale"].map(lambda x: math.isfinite(x) and x > 0).all():
        errors.append("发行规模须为正数（亿元）")

    green_cn_map = {"高": "high", "中": "medium", "低": "low"}
    df["green_level"] = df["green_level"].str.lower().replace(green_cn_map)
    if not df["green_level"].isin(["high", "medium", "low"]).all():
        errors.append("绿色程度必须为 high/medium/low 或 高/中/低")

    # 信用评级
    valid_ratings = {"AAA", "AA+", "AA", "A", "BBB", "BB"}
    if not df["credit_rating"].isin(valid_ratings).all():
        errors.append("信用评级必须为 AAA/AA+/AA/A/BBB/BB 之一")

    yes_no = {"是": True, "否": False, "true": True, "false": False, "1": True, "0": False}
    for name in ("project_catalog_eligible", "major_violation"):
        if name in df:
            values = df[name].fillna("").astype(str).str.strip().str.lower()
            if not values.isin({"", *yes_no}).all():
                errors.append(f"{name} 必须为是/否或留空")
            df[name] = values.map(yes_no)

    numeric_bounds = {"market_benchmark_yield": (0, 0.20), "pricing_spread_bp": (-1000, 2000), "issuer_rate_cap": (0.0001, 0.20)}
    for name, (lower, upper) in numeric_bounds.items():
        if name in df:
            raw = df[name].fillna("").astype(str).str.strip()
            values = pd.to_numeric(raw, errors="coerce")
            if (raw.ne("") & ~values.between(lower, upper)).any():
                errors.append(f"{name} 的取值范围为 {lower}–{upper}")
            df[name] = values

    benchmark = df.get("market_benchmark_yield", pd.Series(index=df.index, dtype=float))
    spread = df.get("pricing_spread_bp", pd.Series(index=df.index, dtype=float))
    if (benchmark.notna() != spread.notna()).any():
        errors.append("市场基准收益率与定价利差必须成对填写")
    if ((benchmark + spread / 10000).dropna() <= 0).any():
        errors.append("市场基准收益率加定价利差须大于零")

    if errors:
        return False, "数据校验失败: " + "; ".join(errors), None

    # --- 5. 填充可选字段默认值 ---
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # --- 6. 自动生成 code ---
    if "code" not in df.columns or df["code"].isna().any():
        df["code"] = [f"ENT{i+1:04d}" for i in range(len(df))]

    if willing_only:
        return True, f"名单共 {original_count} 条，纳入 {len(df)} 家愿意发行的企业", df
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
        for name in ("project_catalog_eligible", "major_violation"):
            if pd.notna(row.get(name)):
                ent[name] = bool(row[name])
        for name in ("market_benchmark_yield", "pricing_spread_bp", "issuer_rate_cap"):
            if pd.notna(row.get(name)):
                ent[name] = float(row[name])
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
    *,
    rule_seed: Optional[int] = None,
    api_key: Optional[str] = None,
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
    if use_llm and (not isinstance(api_key, str) or not api_key.strip() or api_key.strip() == "1"):
        raise ValueError("LLM 模式需要输入你自己的 API Key")
    api_key = api_key.strip() if api_key else None
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
        "model": config.DEFAULT_MODEL_NAME if use_llm else None,
        "model_temperature": config.LLM_JSON_TEMPERATURE if use_llm else None,
        "rule_seed": rule_seed,
        "rules_version": "2",
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

    # 每个环节独立取随机数，避免某场景提前退出影响后续企业或环节的对照。
    rule_rngs = (
        [random] * 4 if rule_seed is None
        else [random.Random(f"{rule_seed}:{stage}") for stage in range(4)]
    )
    simulate = _run_llm_simulation if use_llm else _run_rules_simulation
    credentials = {"api_key": api_key} if use_llm else {}
    return simulate(
        enterprise, result, certifier_type, regulator_type,
        underwriter_type, investor_type, rule_rngs, **credentials,
    )


def _run_rules_simulation(
    enterprise: Dict, result: Dict,
    certifier_type: str, regulator_type: str,
    underwriter_type: str, investor_type: str, rule_rngs: list,
) -> Dict:
    """规则引擎模式：使用 rules.py 中的概率表进行决策（原有逻辑）"""

    # ---- 环节2：绿色认证机构审核 ----
    passed, reason, detail = certifier_check(enterprise, certifier_type, rng=rule_rngs[0])
    result["stage_2_certifier"] = {"passed": passed, "reason": reason, "detail": detail}
    if not passed:
        result["final_result"] = "failed"
        result["failure_stage"] = "2_green_certification"
        result["failure_reason"] = reason
        return result

    # ---- 环节3：监管部门审核 ----
    passed, reason, detail = regulator_check(enterprise, regulator_type, rng=rule_rngs[1])
    result["stage_3_regulator"] = {"passed": passed, "reason": reason, "detail": detail}
    if not passed:
        result["final_result"] = "failed"
        result["failure_stage"] = "3_regulatory_review"
        result["failure_reason"] = reason
        return result

    # ---- 环节4：承销商定价&路演 ----
    passed, reason, detail = underwriter_price(enterprise, underwriter_type, rng=rule_rngs[2])
    result["stage_4_underwriter"] = {"passed": passed, "reason": reason, "detail": detail}
    if not passed:
        result["final_result"] = "failed"
        result["failure_stage"] = "4_underwriter_pricing"
        result["failure_reason"] = reason
        return result

    # ---- 环节5：投资机构认购 ----
    issue_scale = enterprise.get("issue_scale", 10)
    passed, reason, detail = investor_subscribe(
        enterprise, investor_type, detail["final_rate"], detail["roadshow_quality"], issue_scale, rng=rule_rngs[3]
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


def _review_decision(payload):
    decision = payload.get("decision")
    if decision is None and isinstance(payload.get("passed"), bool):
        decision = "approved" if payload["passed"] else "rejected"
    if decision not in ("approved", "conditional", "rejected"):
        raise ValueError("审核结果缺少有效的 decision")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("审核结果缺少理由")
    conditions = payload.get("conditions", payload.get("concerns", []))
    if not isinstance(conditions, list) or not all(isinstance(item, str) for item in conditions):
        raise ValueError("补件条件须为文本列表")
    if decision == "conditional" and not any(item.strip() for item in conditions):
        raise ValueError("有条件通过需要补件条件")
    return decision != "rejected", reason, {"decision": decision, "conditions": conditions}


def _underwriting_decision(payload, enterprise):
    decision = payload.get("decision", "underwrite" if "issue_rate" in payload else None)
    reason = payload.get("rate_rationale") or payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("承销结果缺少理由")
    if decision == "declined":
        return False, reason, {"decision": decision}
    if decision != "underwrite":
        raise ValueError("承销结果缺少有效的 decision")
    rate = float(payload["issue_rate"])
    if not math.isfinite(rate) or not 0 < rate <= 0.20:
        raise ValueError("发行票息须为 0–20% 之间的小数")
    rate_cap = enterprise.get("issuer_rate_cap")
    if rate_cap is not None and rate > rate_cap:
        return False, f"不承销：票息 {rate:.2%} 超过发行人上限 {rate_cap:.2%}", {
            "decision": "declined", "quoted_rate": rate, "issuer_rate_cap": rate_cap,
        }
    return True, reason, {
        "decision": decision, "final_rate": rate, "roadshow_quality": 0.75,
        "roadshow_strategy": payload.get("roadshow_strategy", ""),
    }


def _investment_decision(payload, issue_scale):
    willing = payload.get("will_subscribe")
    amount = float(payload["subscribe_amount"])
    reason = payload.get("decision_rationale")
    if not isinstance(willing, bool) or not math.isfinite(amount) or amount < 0:
        raise ValueError("认购决策或金额无效")
    if (willing and amount == 0) or (not willing and amount > 0):
        raise ValueError("认购决定与金额不一致")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("认购结果缺少理由")
    passed = willing and amount >= issue_scale
    return passed, reason, {
        "decision": "full" if passed else "partial" if willing else "declined",
        "will_subscribe": willing, "subscribe_amount": amount, "issue_scale": issue_scale,
        "success": passed, "rate_adjust": 0.0, "roadshow_bonus": 0.0,
    }


def _llm_stage(call, parse, fallback, rejection=None, *, api_key=None):
    if rejection:
        return {"passed": False, "reason": rejection, "detail": {
            "source": "rigid_rule", "decision": "rejected", "rigid_reject": True,
        }}
    try:
        raw = call()
        if api_key:
            raw = raw.replace(api_key, "[密钥已隐藏]")
        ok, error, payload = _parse_llm_result(raw, {})
        if not ok:
            raise ValueError(error)
        passed, reason, detail = parse(payload)
        return {"passed": passed, "reason": reason, "detail": {
            **detail, "source": "llm", "llm_raw_response": raw[:2000],
        }}
    except Exception as error:
        message = str(error).replace(api_key, "[密钥已隐藏]") if api_key else str(error)
        passed, reason, detail = fallback()
        return {"passed": passed, "reason": reason, "detail": {
            **detail, "source": "fallback_rules", "error": message[:500],
        }}


def _run_llm_simulation(
    enterprise: Dict, result: Dict, certifier_type: str, regulator_type: str,
    underwriter_type: str, investor_type: str, rule_rngs: list, *, api_key: str,
) -> Dict:
    from agents import EnterpriseAgent, CertifierAgent, RegulatorAgent, UnderwriterAgent, InvestorAgent

    context = EnterpriseAgent(enterprise).to_prompt_context()
    benchmark = enterprise.get("market_benchmark_yield")
    if benchmark is not None:
        context += f"\n市场基准收益率：{benchmark:.3%}；定价利差：{enterprise['pricing_spread_bp']:.1f} bp"
    rate_cap = enterprise.get("issuer_rate_cap")
    if rate_cap is not None:
        context += f"\n发行人票息上限：{rate_cap:.3%}"
    issue_scale = enterprise.get("issue_scale", 10)
    summaries = {}
    stages = [
        ("stage_2_certifier", "2_green_certification",
         lambda: CertifierAgent(certifier_type, api_key=api_key).review(context),
         _review_decision,
         lambda: certifier_check(enterprise, certifier_type, rng=rule_rngs[0]),
         certification_rejection),
        ("stage_3_regulator", "3_regulatory_review",
         lambda: RegulatorAgent(regulator_type, api_key=api_key).review(context, summaries["stage_2_certifier"]),
         _review_decision,
         lambda: regulator_check(enterprise, regulator_type, rng=rule_rngs[1]),
         regulatory_rejection),
        ("stage_4_underwriter", "4_underwriter_pricing",
         lambda: UnderwriterAgent(underwriter_type, api_key=api_key).price_and_roadshow(
             context, summaries["stage_2_certifier"], summaries["stage_3_regulator"],
         ),
         lambda payload: _underwriting_decision(payload, enterprise),
         lambda: underwriter_price(enterprise, underwriter_type, rng=rule_rngs[2]),
         None),
        ("stage_5_investor", "5_investor_subscription",
         lambda: InvestorAgent(investor_type, api_key=api_key).make_decision(
             context, summaries["stage_4_underwriter"], pricing["final_rate"],
             certifier_result=summaries["stage_2_certifier"],
             regulator_result=summaries["stage_3_regulator"],
         ),
         lambda payload: _investment_decision(payload, issue_scale),
         lambda: investor_subscribe(
             enterprise, investor_type, pricing["final_rate"], pricing["roadshow_quality"],
             issue_scale, rng=rule_rngs[3],
         ),
         None),
    ]
    pricing = {}
    for key, failure_stage, call, parse, fallback, reject in stages:
        stage = _llm_stage(call, parse, fallback, reject(enterprise) if reject else None, api_key=api_key)
        result[key] = stage
        if not stage["passed"]:
            result.update(final_result="failed", failure_stage=failure_stage, failure_reason=stage["reason"])
            return result
        detail = {k: v for k, v in stage["detail"].items() if k not in ("llm_raw_response", "error")}
        summaries[key] = json.dumps({"reason": stage["reason"], **detail}, ensure_ascii=False)
        if key == "stage_4_underwriter":
            pricing = detail

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
    *,
    each_enterprise_once: bool = False,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    批量运行模拟实验

    Args:
        enterprises: 企业列表
        scenario: 场景参数 {"certifier_type": "...", "regulator_type": "...", ...}
        rounds: 每个场景运行次数
        use_llm: 是否调用 LLM API
        each_enterprise_once: 按名单逐家运行一次，并保留原始企业编号

    Returns:
        所有模拟结果的列表
    """
    results = []
    total_runs = len(enterprises) if each_enterprise_once else rounds
    for i in range(total_runs):
        if each_enterprise_once:
            enterprise = enterprises[i]
        else:
            enterprise = random.choice(enterprises).copy()
            enterprise["code"] = f"{enterprise['code']}_R{i+1}"

        sim_result = run_single_simulation(
            enterprise=enterprise,
            certifier_type=scenario.get("certifier_type", "moderate"),
            regulator_type=scenario.get("regulator_type", "moderate"),
            underwriter_type=scenario.get("underwriter_type", "moderate"),
            investor_type=scenario.get("investor_type", "balanced"),
            use_llm=use_llm,
            api_key=api_key,
        )
        results.append(sim_result)

        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/{total_runs} 次模拟...")

    return results


# ============================================================
# 统计分析方法
# ============================================================

STAGES = [
    ("1_enterprise_application", "企业发起", "stage_1_material"),
    ("2_green_certification", "绿色认证", "stage_2_certifier"),
    ("3_regulatory_review", "监管审核", "stage_3_regulator"),
    ("4_underwriter_pricing", "承销定价", "stage_4_underwriter"),
    ("5_investor_subscription", "投资认购", "stage_5_investor"),
]


def ownership_survival(results):
    groups = {}
    for record in results:
        ownership = record.get("enterprise_data", {}).get("ownership") or "未填写"
        groups.setdefault(ownership, []).append(record)
    return [
        {"企业性质": ownership, "环节": label, "企业数": len(records),
         "留存比例": sum(bool((r.get(key) or {}).get("passed")) for r in records) / len(records)}
        for ownership, records in groups.items()
        for _, label, key in STAGES
    ]


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
    if not total:
        raise ValueError("分析至少需要一条模拟记录")
    successes = sum(1 for r in results if r["final_result"] == "success")
    failures = total - successes

    # 各环节失败率
    stage_failures = {}
    failure_reasons = {}
    for key, name, stage_key in STAGES:
        reached = sum(r.get(stage_key) is not None for r in results)
        passed = sum(bool((r.get(stage_key) or {}).get("passed")) for r in results)
        count = sum(r["failure_stage"] == key for r in results)
        stage_failures[name] = {
            "count": count, "rate": round(count / total, 4),
            "reached": reached, "passed": passed,
            "conditional_failure_rate": round(count / reached, 4) if reached else None,
        }
        for result in results:
            if result["failure_stage"] == key:
                reason = result.get("failure_reason") or "原因未记录"
                stage = result.get(stage_key) or {}
                detail = stage.get("detail") or {}
                if key == "1_enterprise_application":
                    category = "申请材料不齐"
                elif "目录" in reason:
                    category = "目录不符"
                elif "不真实" in reason:
                    category = "证明真实性"
                elif "违规" in reason:
                    category = "重大违规"
                elif "资产负债" in reason or "盈利" in reason:
                    category = "财务约束"
                elif detail.get("decision") == "declined" and key == "4_underwriter_pricing":
                    category = "承销拒绝"
                elif key == "5_investor_subscription":
                    category = "认购不足"
                else:
                    category = "其他审核或定价原因"
                label = f"{name} · {category}"
                failure_reasons[label] = failure_reasons.get(label, 0) + 1

    # LLM 降级统计
    llm_fallbacks = sum(
        any((record.get(key) or {}).get("detail", {}).get("source") == "fallback_rules"
            for _, _, key in STAGES[1:])
        for record in results
    )

    stats = {
        "total_simulations": total,
        "successes": successes,
        "failures": failures,
        "success_rate": round(successes / total, 4),
        "stage_failure_rates": stage_failures,
        "failure_reasons": failure_reasons,
        "bottleneck_stage": max(stage_failures, key=lambda k: stage_failures[k]["rate"]) if failures else "无",
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
    from llm_client import prompt_api_key
    api_key = prompt_api_key() if use_llm else None
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
        api_key=api_key,
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
