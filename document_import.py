"""Turn a Word document or pasted enterprise description into reviewable simulation input."""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter
from xml.etree import ElementTree as ET

import pandas as pd

from engine import COLUMN_ALIASES, load_enterprises_from_df, validate_enterprise_dataframe


FIELD_LABELS = {
    "code": "企业编号", "name": "企业名称", "decision": "发行决策",
    "credit_rating": "信用评级", "asset_liability_ratio": "资产负债率",
    "green_level": "绿色程度", "has_green_project_cert": "绿色项目证明",
    "has_financial_report": "财务报告", "has_financing_plan": "融资方案",
    "ownership": "企业性质", "industry": "所属行业", "scale": "企业规模",
    "issue_scale": "发行规模(亿)", "consecutive_profit": "连续盈利",
    "compliance_score": "合规评分", "material_complete": "材料完整",
    "green_project_authentic": "绿色项目真实性",
    "financing_plan_quality": "融资方案质量",
    "project_catalog_eligible": "符合项目目录", "major_violation": "重大违规",
    "market_benchmark_yield": "市场基准收益率", "pricing_spread_bp": "定价利差(bp)",
    "issuer_rate_cap": "发行人票息上限",
}

# These are simulation placeholders, never claims about the source document.
EXTRACTION_DEFAULTS = {
    "credit_rating": "A", "asset_liability_ratio": 0.60,
    "green_level": "medium", "has_green_project_cert": "是",
    "has_financial_report": "是", "has_financing_plan": "是",
    "decision": "发行",
}

_ALIASES = {**{field: field for field in FIELD_LABELS}, **COLUMN_ALIASES,
            "公司名称": "name", "公司": "name", "企业代码": "code",
            "债券发行规模": "issue_scale", "项目绿色程度": "green_level"}
_LABEL_PATTERN = re.compile(
    r"(?P<label>" + "|".join(re.escape(x) for x in sorted(_ALIASES, key=len, reverse=True)) +
    r")\s*(?:为|是|[:：=])\s*(?P<value>[^，,、；;。\n\t|]+)"
)
_ROOT = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_to_text(content: bytes) -> str:
    """Read paragraphs and tables from a .docx without creating a server-side file."""
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Word 文件不能超过 20MB")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as document:
            xml = document.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("无法读取 Word 文档，请另存为 .docx 后重试") from exc
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError("Word 文档内容损坏，请重新保存 .docx") from exc
    body = root.find(f"{_ROOT}body")
    if body is None:
        return ""

    def cell_text(node):
        return "".join(t.text or "" for t in node.iter(f"{_ROOT}t")).strip()

    lines = []
    for node in body:
        if node.tag == f"{_ROOT}p":
            value = cell_text(node)
            if value:
                lines.append(value)
        elif node.tag == f"{_ROOT}tbl":
            rows = [[cell_text(cell) for cell in row.findall(f"{_ROOT}tc")]
                    for row in node.findall(f"{_ROOT}tr")]
            if not rows:
                continue
            headers = [_ALIASES.get(x.strip()) for x in rows[0]]
            if len(headers) >= 2 and sum(x is not None for x in headers) >= 2:
                for row in rows[1:]:
                    lines.append("；".join(f"{header}：{value}" for header, value in zip(headers, row)
                                          if header and value.strip()))
                    lines.append("")
            else:
                for row in rows:
                    lines.append("；".join(f"{row[i]}：{row[i+1]}" for i in range(0, len(row)-1, 2)))
                lines.append("")
    return "\n".join(lines).strip()


def _blocks(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    blocks = []
    for section in re.split(r"\n\s*\n", text):
        names = list(re.finditer(r"(?:企业名称|公司名称|firm_name|name)\s*[:：=]", section))
        ids = list(re.finditer(r"(?m)^\s*(?:企业编号|firm_id|code)\s*[:：=]", section))
        numbered = list(re.finditer(r"(?m)^\s*\d+[.、)]\s*", section))
        # Fields written before the first name still belong to that firm.
        starts = ids if len(ids) > 1 else (numbered if len(numbered) > 1 else names)
        cuts = [0, *(match.start() for match in starts[1:]), len(section)]
        blocks.extend(section[cuts[i]:cuts[i + 1]].strip()
                      for i in range(len(cuts) - 1) if section[cuts[i]:cuts[i + 1]].strip())
    return blocks


def _parse_local(text: str) -> list[dict]:
    records = []
    for block in _blocks(text):
        fields = {}
        for match in _LABEL_PATTERN.finditer(block):
            value = match.group("value").strip()
            if value:
                fields[_ALIASES[match.group("label")]] = value
        narrative = {
            "credit_rating": r"(?:信用)?评级\s*(?:为|是|[:：])?\s*(AAA|AA\+|AA|A|BBB|BB)(?![A-Z+])",
            "asset_liability_ratio": r"(?:资产)?负债率\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?%?)",
            "green_level": r"(?:绿色程度|绿色等级)\s*(?:为|是|[:：])?\s*(高|中|低|high|medium|low)",
        }
        for key, pattern in narrative.items():
            if key not in fields:
                found = re.search(pattern, block, re.I)
                if found:
                    fields[key] = found.group(1)
        if "decision" not in fields:
            if re.search(r"(?:不愿意|暂不|不计划|不拟)发行", block):
                fields["decision"] = "不发行"
            elif re.search(r"(?:愿意|计划|拟)发行", block):
                fields["decision"] = "发行"
        for key, phrase in (("has_green_project_cert", "绿色项目证明"),
                            ("has_financial_report", "财务报告"),
                            ("has_financing_plan", "融资方案")):
            if key not in fields:
                if re.search(rf"(?:没有|无|缺少){phrase}", block):
                    fields[key] = "否"
                elif re.search(rf"(?:已有|具备|持有|有){phrase}", block):
                    fields[key] = "是"
        if "name" not in fields:
            # A short narrative about one enterprise may omit its name label.
            name = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{1,40}(?:公司|集团|企业))", block)
            if name:
                fields["name"] = name.group(1)
        if fields:
            records.append(fields)
    return records


def _parse_model(text: str, api_key: str) -> list[dict]:
    from llm_client import LLMClient

    prompt = (
        "从企业材料提取绿色债券模拟输入。只输出 JSON 对象，格式为 "
        '{"enterprises":[{"name":"...","credit_rating":"..."}]}。'
        "每家企业一项，只用以下英文键：" + ", ".join(FIELD_LABELS) + "。"
        "仅填写材料明确陈述的值，未知不要输出该键。不得猜测、推断或补齐。"
        "资产负债率等比例可用原文百分数；发行规模单位为亿元。"
    )
    reply = LLMClient(api_key=api_key).call_with_json_output(prompt, text[:30000], temperature=0)
    if reply.startswith("[API_ERROR]"):
        raise ValueError(reply)
    if api_key in reply:
        raise ValueError("模型响应包含密钥，已弃用该响应")
    reply = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    data = json.loads(reply)
    if not isinstance(data, dict) or not isinstance(data.get("enterprises"), list):
        raise ValueError("模型没有返回企业数组")
    return [{_ALIASES[k]: str(v).strip() for k, v in item.items()
             if k in _ALIASES and v is not None and str(v).strip()}
            for item in data["enterprises"] if isinstance(item, dict)]


def _normalize(record: dict) -> dict:
    normalized = {}
    for key, value in record.items():
        if key not in FIELD_LABELS:
            continue
        value = str(value).strip()
        if not value:
            continue
        if key in {"asset_liability_ratio", "compliance_score", "market_benchmark_yield", "issuer_rate_cap"}:
            try:
                value = str(float(value.rstrip("%")) / (100 if value.endswith("%") else 1))
            except ValueError:
                pass
        if key == "issue_scale":
            value = value.removesuffix("亿元").removesuffix("亿").strip()
        if key == "credit_rating":
            value = value.upper()
        if key == "green_level":
            value = {"高": "high", "中": "medium", "低": "low"}.get(value, value.lower())
        if key == "decision":
            value = "不发行" if value in {"否", "不发行", "不愿意发行", "0", "false"} else (
                "发行" if value in {"是", "发行", "愿意发行", "1", "true"} else value)
        normalized[key] = value
    return normalized


def convert_text(text: str, *, use_llm: bool = False, api_key: str | None = None):
    """Return (review dataframe, simulation dataframe, missing-by-firm, notices).

    Unknown mandatory values get explicit scenario placeholders. The review table
    carries provenance, and the standard validator still checks all supplied values.
    """
    if not text.strip():
        raise ValueError("请先输入文字或上传 Word 文档")
    notices = []
    records = []
    if use_llm:
        if not api_key:
            raise ValueError("使用 LLM 提取前，请输入自己的 API Key")
        if len(text) > 30000:
            notices.append("文档较长，LLM 仅处理前 30000 字；请分批提取其余内容。")
        try:
            records = _parse_model(text, api_key)
            if not records:
                raise ValueError("模型未识别出企业")
        except (ValueError, json.JSONDecodeError) as exc:
            notices.append(f"LLM 提取未完成，已改用本地字段提取：{exc}")
    if not records:
        records = _parse_local(text)
    if not records:
        raise ValueError("未识别到企业。请至少写出“企业名称：…”或“信用评级：…”等字段")

    seen = set()
    missing_by_firm = {}
    rows = []
    for index, record in enumerate(records, 1):
        row = _normalize(record)
        if not row.get("code"):
            row["code"] = f"TEXT{index:04d}"
            missing = ["code"]
        else:
            missing = []
        if row["code"] in seen:
            raise ValueError(f"企业编号重复：{row['code']}")
        seen.add(row["code"])
        for key, default in EXTRACTION_DEFAULTS.items():
            if key not in row:
                row[key] = default
                missing.append(key)
        if "name" not in row:
            row["name"] = f"未命名企业 {index}"
            missing.append("name")
        row["待核实字段"] = "、".join(FIELD_LABELS[key] for key in missing)
        missing_by_firm[row["code"]] = missing
        rows.append(row)

    review = pd.DataFrame(rows)
    simulation_input = review.copy()
    benchmark = "market_benchmark_yield"
    spread = "pricing_spread_bp"
    if benchmark in simulation_input.columns or spread in simulation_input.columns:
        for name in (benchmark, spread):
            if name not in simulation_input.columns:
                simulation_input[name] = ""
        for index, row in simulation_input.iterrows():
            has_benchmark = bool(str(row.get(benchmark, "")).strip()) and pd.notna(row.get(benchmark))
            has_spread = bool(str(row.get(spread, "")).strip()) and pd.notna(row.get(spread))
            if has_benchmark != has_spread:
                absent = spread if has_benchmark else benchmark
                code = row["code"]
                missing_by_firm[code].append(absent)
                review.at[index, "待核实字段"] += "、" + FIELD_LABELS[absent]
                simulation_input.at[index, benchmark] = ""
                simulation_input.at[index, spread] = ""
                notices.append(f"{code} 只提供了定价参数的一半，本次改用模拟定价；转换文件保留原输入。")
    valid, message, processed = validate_enterprise_dataframe(simulation_input, willing_only=True)
    if not valid:
        raise ValueError(message)
    excluded = len(review) - len(processed)
    if excluded:
        notices.append(f"原文中 {excluded} 家明确不发行，已从模拟名单排除；转换文件仍保留原记录。")
    counts = Counter(field for fields in missing_by_firm.values() for field in fields)
    if counts:
        notices.append("缺失字段已填入模拟占位值，请核实后再用于正式结论：" +
                       "、".join(f"{FIELD_LABELS[key]}（{count} 家）" for key, count in counts.items()))
    return review, processed, missing_by_firm, notices


def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def export_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="企业数据")
    return output.getvalue()


def load_reviewed_enterprises(processed: pd.DataFrame, missing_by_firm: dict) -> list[dict]:
    enterprises = load_enterprises_from_df(processed)
    for enterprise in enterprises:
        missing = missing_by_firm.get(enterprise["code"], [])
        enterprise["assumed_input_fields"] = missing
    return enterprises
