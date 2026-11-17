"""
多主体智能体定义模块

五类智能体：
1. 企业智能体         — 绿色债券发行申请人
2. 绿色认证机构智能体  — 审核绿色项目资质
3. 监管审核机构智能体  — 审核发行合规性
4. 承销商智能体       — 定价与路演
5. 投资机构智能体     — 认购决策

每类智能体包含：
- 标签化人设（type + traits）
- 对应的 System Prompt 模板
"""

from llm_client import LLMClient
from typing import Dict, Optional
import config


# ============================================================
# 1. 企业智能体
# ============================================================

class EnterpriseAgent:
    """
    企业智能体 — 有绿色债券发行意愿的企业

    标签化维度：
    - ownership: 企业性质（央企/国企/民企）
    - industry: 行业（清洁能源/节能环保/绿色建筑/传统制造转型/...）
    - scale: 规模（大型/中型/小型）
    - credit_rating: 信用评级（AAA/AA/A/BBB/BB）
    - green_level: 项目绿色程度（high/medium/low）
    - financial_health: 财务状况（asset_liability_ratio, consecutive_profit, ...）
    """

    def __init__(self, enterprise_data: Dict):
        self.data = enterprise_data
        self.name = enterprise_data.get("name", "未命名企业")
        self.code = enterprise_data.get("code", "ENT000")

    def to_prompt_context(self) -> str:
        """将企业数据转为 prompt 可用的文本描述"""
        d = self.data
        return f"""
【企业基本信息】
- 企业名称：{d.get('name', 'N/A')}
- 企业性质：{d.get('ownership', 'N/A')}
- 所属行业：{d.get('industry', 'N/A')}
- 企业规模：{d.get('scale', 'N/A')}
- 信用评级：{d.get('credit_rating', 'N/A')}
- 资产负债率：{d.get('asset_liability_ratio', 0):.1%}
- 近两年连续盈利：{'是' if d.get('consecutive_profit', True) else '否'}
- 合规性评分：{d.get('compliance_score', 0):.2f}（满分1.0）

【绿色债券发行相关信息】
- 项目绿色程度：{d.get('green_level', 'N/A')}（high/medium/low）
- 拟发行规模：{d.get('issue_scale', 0)} 亿元
- 绿色项目证明：{'有' if d.get('has_green_project_cert', False) else '无'}
- 材料完整度：{'完整' if d.get('material_complete', True) else '不完整'}
- 融资方案完备度：{d.get('financing_plan_quality', 'N/A')}
"""

    def get_willingness_score(self) -> float:
        """返回企业发行意愿评分（0-1）"""
        return self.data.get("willingness_score", 0.8)


# ============================================================
# 2. 绿色认证机构智能体
# ============================================================

CERTIFIER_TYPES = {
    "strict": {
        "name": "严格型认证机构",
        "traits": "审核标准严苛，对材料完整性和绿色程度要求高",
        "base_pass_rate": 0.60,
    },
    "moderate": {
        "name": "中等型认证机构",
        "traits": "兼顾标准与效率，对中等绿色项目有一定容忍度",
        "base_pass_rate": 0.75,
    },
    "lenient": {
        "name": "宽松型认证机构",
        "traits": "积极推动绿色债券市场扩容，审核标准相对灵活",
        "base_pass_rate": 0.90,
    },
}


class CertifierAgent:
    """绿色认证机构智能体"""

    def __init__(self, certifier_type: str = "moderate", model: str = None, *, api_key: str = None):
        self.certifier_type = certifier_type
        self.type_info = CERTIFIER_TYPES.get(certifier_type, CERTIFIER_TYPES["moderate"])
        self.llm = LLMClient(model=model, api_key=api_key)

    def get_system_prompt(self) -> str:
        return f"""你是一名{self.type_info['name']}的审核员。
你的核心职责是审核企业绿色债券发行的项目绿色程度和材料完整性。
审核风格：{self.type_info['traits']}

【决策流程】
1. 刚性检查：若企业无绿色项目真实证明材料 → 直接不通过
2. 材料完整性检查：材料不完整 → 降低通过概率
3. 概率判定：根据项目绿色程度和材料完整度，按概率判定通过/不通过

请严格按照规则做出判定，并以 JSON 格式返回结果：
{{"decision": "approved/conditional/rejected", "reason": "判定理由", "conditions": ["需补充的材料"]}}
有条件通过须列出补件条件，其余情况 conditions 为空数组。"""

    def review(self, enterprise_context: str) -> str:
        """调用 LLM 进行审核判定"""
        return self.llm.call_with_json_output(
            self.get_system_prompt(),
            f"请审核以下企业的绿色债券发行申请：\n{enterprise_context}"
        )


# ============================================================
# 3. 监管审核机构智能体
# ============================================================

REGULATOR_TYPES = {
    "strict": {
        "name": "严监管型监管部门",
        "traits": "对公司治理、财务状况、合规性均有高标准，审批周期长",
        "compliance_threshold": 0.85,
    },
    "moderate": {
        "name": "中监管型监管部门",
        "traits": "依规审核，适度灵活，关注企业信用和市场稳定",
        "compliance_threshold": 0.70,
    },
    "lenient": {
        "name": "松监管型监管部门",
        "traits": "鼓励绿色金融创新，在合规基础上简化审批流程",
        "compliance_threshold": 0.55,
    },
}


class RegulatorAgent:
    """监管审核机构智能体"""

    def __init__(self, regulator_type: str = "moderate", model: str = None, *, api_key: str = None):
        self.regulator_type = regulator_type
        self.type_info = REGULATOR_TYPES.get(regulator_type, REGULATOR_TYPES["moderate"])
        self.llm = LLMClient(model=model, api_key=api_key)

    def get_system_prompt(self) -> str:
        return f"""你是一名{self.type_info['name']}的审核官员。
你的核心职责是审核企业绿色债券发行的合规性。

程序已处理模拟硬约束。根据财务、合规材料及前序审核意见完成剩余判断。

【审核要点】
1. 企业资质：信用评级、行业属性、资产规模
2. 财务健康：资产负债率、盈利能力、现金流
3. 合规性：信息披露完整性、历史违规记录
4. 融资方案：发行规模合理性、资金用途清晰度

请严格按照规则做出判定，并以 JSON 格式返回结果：
{{"decision": "approved/conditional/rejected", "reason": "判定理由", "conditions": ["需补充的材料"]}}
有条件通过须列出补件条件，其余情况 conditions 为空数组。"""

    def review(self, enterprise_context: str, certifier_result: str) -> str:
        """调用 LLM 进行监管审核"""
        return self.llm.call_with_json_output(
            self.get_system_prompt(),
            f"【前一环节结果：绿色认证审核】\n{certifier_result}\n\n【待审核企业】\n{enterprise_context}"
        )


# ============================================================
# 4. 承销商智能体
# ============================================================

UNDERWRITER_TYPES = {
    "aggressive": {
        "name": "激进型承销商",
        "traits": "愿意承担较高承销风险，综合权衡发行人融资成本和投资者需求",
        "style_multiplier": -0.003,
    },
    "moderate": {
        "name": "稳健型承销商",
        "traits": "综合考量企业风险与市场需求，定价中性",
        "style_multiplier": 0.0,
    },
    "conservative": {
        "name": "保守型承销商",
        "traits": "注重风险覆盖，定价偏高，给投资者更大的安全边际",
        "style_multiplier": 0.005,
    },
}


class UnderwriterAgent:
    """承销商智能体"""

    def __init__(self, underwriter_type: str = "moderate", model: str = None, *, api_key: str = None):
        self.underwriter_type = underwriter_type
        self.type_info = UNDERWRITER_TYPES.get(underwriter_type, UNDERWRITER_TYPES["moderate"])
        self.llm = LLMClient(model=model, api_key=api_key)

    def get_system_prompt(self) -> str:
        return f"""你是一名{self.type_info['name']}的债券承销负责人。
定价风格：{self.type_info['traits']}

【定价参考因素】
1. 输入中提供的市场基准收益率和定价利差
2. 企业信用评级和风险溢价
3. 绿色债券溢价扣减（绿色程度越高，利率可越低）
4. 市场流动性环境

请根据企业实际情况，确定发行利率和路演策略，以 JSON 返回：
{{"decision": "underwrite/declined", "issue_rate": 0.045, "rate_rationale": "承销或拒绝的理由", "roadshow_strategy": "路演策略描述"}}
拒绝承销时省略 issue_rate。"""

    def price_and_roadshow(self, enterprise_context: str, certifier_result: str,
                           regulator_result: str) -> str:
        """调用 LLM 进行定价和路演决策"""
        return self.llm.call_with_json_output(
            self.get_system_prompt(),
            f"""请根据以下信息确定发行利率和路演策略：
【企业信息】
{enterprise_context}
【绿色认证结果】
{certifier_result}
【监管审核结果】
{regulator_result}"""
        )


# ============================================================
# 5. 投资机构智能体
# ============================================================

INVESTOR_TYPES = {
    "green_focused": {
        "name": "绿色偏好型投资机构",
        "traits": "ESG导向，优先关注项目绿色程度和环境效益，利率敏感度较低",
    },
    "return_focused": {
        "name": "收益偏好型投资机构",
        "traits": "以投资回报率为核心决策依据，对利率高度敏感",
    },
    "balanced": {
        "name": "均衡型投资机构",
        "traits": "兼顾绿色价值与投资收益，综合评估后决策",
    },
}


class InvestorAgent:
    """投资机构智能体"""

    def __init__(self, investor_type: str = "balanced", model: str = None, *, api_key: str = None):
        self.investor_type = investor_type
        self.type_info = INVESTOR_TYPES.get(investor_type, INVESTOR_TYPES["balanced"])
        self.llm = LLMClient(model=model, api_key=api_key)

    def get_system_prompt(self) -> str:
        return f"""你是一名{self.type_info['name']}的投资经理。
投资风格：{self.type_info['traits']}

【认购决策参考因素】
1. 项目绿色程度和认证结果
2. 发行利率（与市场基准对比）
3. 企业信用评级和财务状况
4. 路演质量和信息披露透明度
5. 当前绿色债券市场供需环境

请做出认购决策，以 JSON 返回：
{{"will_subscribe": true/false, "subscribe_amount": 5.0, "decision_rationale": "决策理由"}}"""

    def make_decision(self, enterprise_context: str, underwriter_result: str,
                      issue_rate: float, *, certifier_result: str = "",
                      regulator_result: str = "") -> str:
        """调用 LLM 进行认购决策"""
        return self.llm.call_with_json_output(
            self.get_system_prompt(),
            f"""请做出认购决策：
【企业信息】
{enterprise_context}
【绿色认证结果】
{certifier_result}
【监管审核结果】
{regulator_result}
【承销商定价与路演结果】
{underwriter_result}
当前发行利率：{issue_rate:.2%}"""
        )
