"""
智能体行为规则模块
定义各环节的刚性规则 + 概率性规则
规则以自然语言提示词 + 量化判定公式的形式嵌入各智能体

规则体系说明：
- 刚性规则：触发后直接判定"不通过"，不进入概率判定
- 概率性规则：基于企业特征、主体类型，按概率判定
- 交互性规则：后一环节的规则基于前一环节结果调整
"""

import random
from typing import Dict, Optional, Tuple
import config

# 设置随机种子
random.seed(config.RANDOM_SEED)


# ============================================================
# 环节1：企业发起 → 材料完整性刚性检查
# ============================================================

def enterprise_material_check(enterprise: Dict) -> Tuple[bool, str]:
    """
    企业发行申请 - 刚性材料检查

    刚性规则：
    - 无绿色项目证明材料 → 直接不通过
    - 无财务报告 → 直接不通过
    - 无融资方案 → 直接不通过
    """
    missing = []
    if not enterprise.get("has_green_project_cert", False):
        missing.append("绿色项目证明材料")
    if not enterprise.get("has_financial_report", False):
        missing.append("财务报告")
    if not enterprise.get("has_financing_plan", False):
        missing.append("融资方案")

    if missing:
        return False, f"材料不完整，缺少: {', '.join(missing)}"
    return True, "材料完整，进入绿色认证审核"


# ============================================================
# 环节2：绿色认证机构审核
# ============================================================

CERTIFIER_PROB_TABLE = {
    # (认证机构严格度, 项目绿色程度) → 通过概率
    ("strict", "high"):   0.90,
    ("strict", "medium"): 0.50,
    ("strict", "low"):    0.20,
    ("moderate", "high"):   0.95,
    ("moderate", "medium"): 0.75,
    ("moderate", "low"):    0.40,
    ("lenient", "high"):   0.98,
    ("lenient", "medium"): 0.90,
    ("lenient", "low"):    0.60,
}

# 材料不完整额外扣减
MATERIAL_INCOMPLETE_PENALTY = 0.30


def certification_rejection(enterprise: Dict) -> Optional[str]:
    if enterprise.get("project_catalog_eligible") is False:
        return "刚性驳回：项目不符合所选绿色项目目录"
    if not enterprise.get("green_project_authentic", True):
        return "刚性驳回：绿色项目证明材料不真实"
    return None


def certifier_check(enterprise: Dict, certifier_type: str, *, rng=random) -> Tuple[bool, str, Dict]:
    """
    绿色认证机构审核

    Args:
        enterprise: 企业智能体数据
        certifier_type: 认证机构类型 strict/moderate/lenient

    Returns:
        (是否通过, 原因, 决策详情)
    """
    green_level = enterprise.get("green_level", "medium")       # high/medium/low
    material_complete = enterprise.get("material_complete", True)

    rejection = certification_rejection(enterprise)
    if rejection:
        return False, rejection, {"rigid_reject": True, "decision": "rejected"}

    # 概率判定
    base_prob = CERTIFIER_PROB_TABLE.get((certifier_type, green_level), 0.5)

    # 材料不完整扣减
    if not material_complete:
        base_prob = max(0, base_prob - MATERIAL_INCOMPLETE_PENALTY)

    passed = rng.random() < base_prob

    detail = {
        "certifier_type": certifier_type,
        "green_level": green_level,
        "material_complete": material_complete,
        "pass_probability": round(base_prob, 2),
        "rigid_reject": False,
    }
    conditional = passed and not material_complete
    detail["decision"] = "conditional" if conditional else "approved" if passed else "rejected"
    detail["conditions"] = ["补充项目证明或环境效益测算"] if conditional else []
    label = "有条件通过" if conditional else "通过" if passed else "不通过"
    reason = f"{label}（{certifier_type}型认证机构，绿色程度={green_level}，通过概率={base_prob:.0%}）"
    return passed, reason, detail


# ============================================================
# 环节3：监管部门审核
# ============================================================

# 刚性红线
REGULATOR_RIGID_RULES = {
    "asset_liability_ratio_max": 0.80,   # 资产负债率超过80%直接不通过
    "profit_requirement": True,           # 必须连续两年盈利
}


def regulatory_rejection(enterprise: Dict) -> Optional[str]:
    if enterprise.get("major_violation") is True:
        return "模拟刚性驳回：企业存在重大违规"
    limit = REGULATOR_RIGID_RULES["asset_liability_ratio_max"]
    ratio = enterprise.get("asset_liability_ratio", 0.5)
    if ratio > limit:
        return f"模拟刚性驳回：资产负债率 {ratio:.0%} 超过 {limit:.0%} 阈值"
    if REGULATOR_RIGID_RULES["profit_requirement"] and not enterprise.get("consecutive_profit", True):
        return "模拟刚性驳回：未满足连续两年盈利假设"
    return None


def regulator_check(enterprise: Dict, regulator_type: str, *, rng=random) -> Tuple[bool, str, Dict]:
    """
    监管部门审核

    Args:
        enterprise: 企业智能体数据
        regulator_type: 监管类型 strict/moderate/lenient

    Returns:
        (是否通过, 原因, 决策详情)
    """
    rejection = regulatory_rejection(enterprise)
    if rejection:
        return False, rejection, {"rigid_reject": True, "decision": "rejected"}
    asset_liability = enterprise.get("asset_liability_ratio", 0.5)

    # 概率判定（基于企业合规性评分 + 监管类型）
    compliance_score = enterprise.get("compliance_score", 0.7)  # 0-1
    strictness_map = {"strict": 0.85, "moderate": 0.70, "lenient": 0.55}
    threshold = strictness_map.get(regulator_type, 0.70)

    passed = compliance_score >= threshold or (
        compliance_score >= threshold - 0.15 and rng.random() < 0.5
    )

    detail = {
        "regulator_type": regulator_type,
        "asset_liability_ratio": asset_liability,
        "compliance_score": compliance_score,
        "threshold": threshold,
        "rigid_reject": False,
    }
    conditional = passed and compliance_score < threshold
    detail["decision"] = "conditional" if conditional else "approved" if passed else "rejected"
    detail["conditions"] = ["补充合规和信息披露材料"] if conditional else []
    label = "有条件通过" if conditional else "通过" if passed else "不通过"
    reason = f"{label}（{regulator_type}型监管，合规评分={compliance_score:.2f}，阈值={threshold:.2f}）"
    return passed, reason, detail


# ============================================================
# 环节4：承销商定价 & 路演
# ============================================================

# 基准利率 + 企业风险溢价 + 绿色溢价扣减
BASE_INTEREST_RATE = 0.035      # 基准利率 3.5%
RISK_PREMIUM_MAP = {
    "AAA": 0.005, "AA+": 0.0075, "AA": 0.010, "A": 0.020, "BBB": 0.030, "BB": 0.050,
}
GREEN_PREMIUM_DISCOUNT = 0.005  # 绿色债券利率可低 0.5%


def underwriter_price(enterprise: Dict, underwriter_type: str, *, rng=random) -> Tuple[bool, str, Dict]:
    """
    承销商定价与路演

    Args:
        enterprise: 企业智能体数据
        underwriter_type: 承销商类型 aggressive/moderate/conservative

    Returns:
        (是否成功定价并路演, 原因, 定价详情)
    """
    credit_rating = enterprise.get("credit_rating", "A")
    green_level = enterprise.get("green_level", "medium")

    market_yield = enterprise.get("market_benchmark_yield")
    market_spread = enterprise.get("pricing_spread_bp")
    if market_yield is not None and market_spread is not None:
        base_rate = market_yield + market_spread / 10000
        pricing_source = "market_benchmark"
    else:
        base_rate = BASE_INTEREST_RATE + RISK_PREMIUM_MAP.get(credit_rating, 0.025)
        if green_level == "high":
            base_rate -= GREEN_PREMIUM_DISCOUNT
        elif green_level == "medium":
            base_rate -= GREEN_PREMIUM_DISCOUNT * 0.5
        pricing_source = "fixed_assumptions"

    # 承销商风格调整
    style_adjust = {"aggressive": -0.003, "moderate": 0.0, "conservative": 0.005}
    minimum_rate = 0.0001 if pricing_source == "market_benchmark" else 0.01
    final_rate = max(minimum_rate, base_rate + style_adjust.get(underwriter_type, 0.0))
    rate_cap = enterprise.get("issuer_rate_cap")
    if rate_cap is not None and final_rate > rate_cap:
        return False, f"不承销：估计票息 {final_rate:.2%} 超过发行人上限 {rate_cap:.2%}", {
            "quoted_rate": round(final_rate, 6), "issuer_rate_cap": rate_cap,
            "pricing_source": pricing_source, "decision": "declined",
        }

    # 路演效果（随机 + 企业资质）
    roadshow_quality = rng.uniform(0.5, 1.0)
    if credit_rating in ("AAA", "AA+", "AA"):
        roadshow_quality += 0.15
    roadshow_quality = min(1.0, roadshow_quality)

    detail = {
        "underwriter_type": underwriter_type,
        "credit_rating": credit_rating,
        "base_rate": round(base_rate, 6),
        "final_rate": round(final_rate, 6),
        "roadshow_quality": round(roadshow_quality, 2),
        "pricing_source": pricing_source,
        "green_discount_applied": pricing_source == "fixed_assumptions" and green_level in ("high", "medium"),
        "decision": "underwrite",
    }
    reason = f"定价完成：发行利率 {final_rate:.2%}，路演质量 {roadshow_quality:.0%}"
    return True, reason, detail


# ============================================================
# 环节5：投资机构认购
# ============================================================

INVESTOR_PROB_BASE = {
    # (投资机构类型, 项目绿色程度) → 基础认购概率
    ("green_focused", "high"):   0.90,
    ("green_focused", "medium"): 0.70,
    ("green_focused", "low"):    0.40,
    ("return_focused", "high"):   0.75,
    ("return_focused", "medium"): 0.65,
    ("return_focused", "low"):    0.45,
    ("balanced", "high"):   0.85,
    ("balanced", "medium"): 0.70,
    ("balanced", "low"):    0.50,
}

# 利率对认购概率的影响系数
INTEREST_RATE_SENSITIVITY = {
    "green_focused": 0.3,    # 绿色偏好型对利率不那么敏感
    "return_focused": 0.8,   # 收益偏好型对利率高度敏感
    "balanced": 0.5,
}


def investor_subscribe(enterprise: Dict, investor_type: str,
                       final_rate: float, roadshow_quality: float,
                       issue_scale: float, *, rng=random) -> Tuple[bool, str, Dict]:
    """
    投资机构认购决策

    Args:
        enterprise: 企业智能体数据
        investor_type: 投资机构类型 green_focused/return_focused/balanced
        final_rate: 发行利率
        roadshow_quality: 路演质量 (0-1)
        issue_scale: 发行规模（亿元）

    Returns:
        (是否认购成功, 原因, 决策详情)
    """
    green_level = enterprise.get("green_level", "medium")

    # 基础认购概率
    base_prob = INVESTOR_PROB_BASE.get((investor_type, green_level), 0.6)

    # 利率调整：利率越高，认购概率越高
    benchmark = enterprise.get("market_benchmark_yield")
    spread = enterprise.get("pricing_spread_bp")
    comparison_rate = max(0.0001, benchmark + spread / 10000) if benchmark is not None and spread is not None else BASE_INTEREST_RATE
    rate_factor = final_rate / comparison_rate
    sensitivity = INTEREST_RATE_SENSITIVITY.get(investor_type, 0.5)
    rate_adjust = (rate_factor - 1.0) * sensitivity

    # 路演效果加成
    roadshow_bonus = (roadshow_quality - 0.5) * 0.2

    # 企业信用评级调整
    credit_adjust = {"AAA": 0.10, "AA+": 0.075, "AA": 0.05, "A": 0.0, "BBB": -0.05, "BB": -0.10}
    credit_bonus = credit_adjust.get(enterprise.get("credit_rating", "A"), 0.0)

    final_prob = min(0.98, max(0.05, base_prob + rate_adjust + roadshow_bonus + credit_bonus))

    # 认购规模
    if rng.random() < final_prob:
        subscribe_ratio = rng.uniform(0.8, 1.2)
        subscribe_amount = round(issue_scale * subscribe_ratio, 2)
        success = subscribe_amount >= issue_scale
    else:
        subscribe_amount = round(issue_scale * rng.uniform(0, 0.5), 2)
        success = False

    detail = {
        "investor_type": investor_type,
        "base_prob": round(base_prob, 2),
        "final_prob": round(final_prob, 2),
        "rate_adjust": round(rate_adjust, 3),
        "roadshow_bonus": round(roadshow_bonus, 3),
        "subscribe_amount": subscribe_amount,
        "issue_scale": issue_scale,
        "success": success,
    }

    if success:
        reason = f"认购成功：认购 {subscribe_amount}亿 / 发行 {issue_scale}亿"
    else:
        reason = f"认购失败：仅认购 {subscribe_amount}亿，未达发行规模 {issue_scale}亿（认购概率={final_prob:.0%}）"

    return success, reason, detail
