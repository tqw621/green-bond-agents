"""第二组研究方案 v2 的预设场景。"""
import random

import config
from engine import run_single_simulation


BASELINE = {
    "certifier_type": "moderate",
    "regulator_type": "moderate",
    "underwriter_type": "moderate",
    "investor_type": "balanced",
}

PRESET_SCENARIOS = {
    "P0": ("基准", {}),
    "P1": ("放宽认证", {"certifier_type": "lenient"}),
    "P2": ("收紧认证", {"certifier_type": "strict"}),
    "P3": ("放宽监管", {"regulator_type": "lenient"}),
    "P4": ("增强投资者 ESG 偏好", {"investor_type": "green_focused"}),
    "P5": ("提高承销商风险偏好", {"underwriter_type": "aggressive"}),
    "P6": ("放宽认证与监管，增强投资者 ESG 偏好", {
        "certifier_type": "lenient",
        "regulator_type": "lenient",
        "investor_type": "green_focused",
    }),
}


def run_preset_experiment(enterprises, *, use_llm=False, seed=config.RANDOM_SEED):
    """在同一企业池上运行 P0–P6。种子用于规则判断及 LLM 降级后的规则判断。"""
    if not enterprises:
        raise ValueError("预设场景实验至少需要一家企业")

    codes = [enterprise["code"] for enterprise in enterprises]
    if len(set(codes)) != len(codes):
        raise ValueError("预设场景实验要求企业编号唯一")

    rng = random.Random(seed)
    rule_seeds = {code: rng.getrandbits(64) for code in sorted(codes)}
    return {
        scenario_id: [
            {
                **run_single_simulation(
                    enterprise, **{**BASELINE, **overrides},
                    use_llm=use_llm, rule_seed=rule_seeds[enterprise["code"]],
                ),
                "experiment_seed": seed,
            }
            for enterprise in enterprises
        ]
        for scenario_id, (_, overrides) in PRESET_SCENARIOS.items()
    }


def paired_changes(baseline, scenario):
    """逐企业统计相对基准转为成功或转为失败的数量。"""
    outcomes = {record["enterprise_code"]: record["final_result"] for record in baseline}
    gained = lost = 0
    for record in scenario:
        before = outcomes[record["enterprise_code"]]
        after = record["final_result"]
        gained += before != "success" and after == "success"
        lost += before == "success" and after != "success"
    return gained, lost
