"""Decision-reason analysis shared by the UI and command-line reports."""
from collections import Counter
from functools import lru_cache
import re

import jieba

from engine import STAGES


SOURCES = {
    "llm": "模型判断",
    "rules": "规则判断",
    "rigid_rule": "硬规则",
    "fallback_rules": "调用失败后规则降级",
    "unknown": "历史来源未记录",
}
STOP_WORDS = frozenset(
    "的 了 和 与 在 是 为 对 及 等 将 该 其 以 于 但 并 或 而 可 "
    "一个 进行 根据 因此 由于 当前 本次 建议 需要 可以 已经 "
    "企业 项目 通过 不通过 成功 失败 审核 结果 判断 类型 "
    "strict moderate lenient aggressive conservative balanced "
    "green_focused return_focused".split()
)
DOMAIN_WORDS = (
    "绿色债券", "绿色项目", "资产负债率", "信息披露", "融资方案", "信用评级",
    "财务报告", "环境效益", "重大违规", "连续盈利", "绿色认证", "认购不足",
    "发行规模", "票面利率", "流动性", "信用风险", "合规评分", "项目目录",
)


@lru_cache(maxsize=1)
def _tokenizer():
    tokenizer = jieba.Tokenizer()
    for word in DOMAIN_WORDS:
        tokenizer.add_word(word, freq=100000)
    return tokenizer


def decision_documents(records):
    """Extract one reason per reached stage, with a legacy final-reason fallback."""
    documents = []
    for index, record in enumerate(records):
        failure_seen = False
        for code, label, key in STAGES:
            stage = record.get(key) or {}
            reason = stage.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                continue
            detail = stage.get("detail") or {}
            source = detail.get("source")
            if source not in SOURCES:
                if key == "stage_1_material" or detail.get("rigid_reject"):
                    source = "rigid_rule"
                elif record.get("use_llm") is False:
                    source = "rules"
                else:
                    source = "unknown"
            is_failure = record.get("failure_stage") == code
            failure_seen = failure_seen or is_failure
            documents.append({
                "record_index": index,
                "enterprise_code": record.get("enterprise_code", ""),
                "scenario": record.get("scenario_name", ""),
                "stage": code,
                "stage_label": label,
                "source": source,
                "failed": is_failure,
                "reason": reason.strip(),
            })
        reason = record.get("failure_reason")
        if not failure_seen and isinstance(reason, str) and reason.strip():
            code = record.get("failure_stage")
            label = next((label for stage, label, _ in STAGES if stage == code), "历史失败环节")
            documents.append({
                "record_index": index,
                "enterprise_code": record.get("enterprise_code", ""),
                "scenario": record.get("scenario_name", ""),
                "stage": code,
                "stage_label": label,
                "source": "unknown",
                "failed": True,
                "reason": reason.strip(),
            })
    return documents


def analyze_decision_text(records, *, scope="failures", source="all", stage="all",
                          top_n=20, extra_stopwords=()):
    """Return filtered evidence, token counts, and document-level frequencies."""
    if scope not in ("failures", "all"):
        raise ValueError("scope 必须为 failures 或 all")
    if source != "all" and source not in SOURCES:
        raise ValueError("source 无效")
    if stage != "all" and stage not in {code for code, _, _ in STAGES}:
        raise ValueError("stage 无效")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or not 1 <= top_n <= 100:
        raise ValueError("top_n 必须为 1–100 的整数")
    extra_stopwords = tuple(extra_stopwords)
    documents = [
        doc for doc in decision_documents(records)
        if (scope == "all" or doc["failed"])
        and (source == "all" or doc["source"] == source)
        and (stage == "all" or doc["stage"] == stage)
    ]
    stopwords = STOP_WORDS | {word.strip().lower() for word in extra_stopwords if word.strip()}
    counts, document_counts = Counter(), Counter()
    tokenizer = _tokenizer()
    for doc in documents:
        tokens = [
            word.lower().strip() for word in tokenizer.cut(doc["reason"], HMM=False)
        ]
        tokens = [
            word for word in tokens
            if len(word) > 1 and word not in stopwords
            and re.fullmatch(r"[\u4e00-\u9fffA-Za-z]+", word)
        ]
        counts.update(tokens)
        document_counts.update(set(tokens))
    ranked = sorted(counts, key=lambda word: (-document_counts[word], -counts[word], word))
    return {
        "scope": scope, "source": source, "stage": stage, "top_n": top_n,
        "extra_stopwords": sorted(set(extra_stopwords)),
        "document_count": len(documents),
        "record_count": len({doc["record_index"] for doc in documents}),
        "source_counts": dict(Counter(doc["source"] for doc in documents)),
        "terms": [
            {"term": word, "count": counts[word], "document_count": document_counts[word],
             "document_rate": document_counts[word] / len(documents)}
            for word in ranked[:top_n]
        ],
        "documents": documents,
    }
