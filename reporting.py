"""Matplotlib/seaborn reports with traceable text statistics and offline exports."""
from io import BytesIO
import json
from threading import RLock
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

from analysis import analyze_decision_text
from engine import analyze_results, ownership_survival


_PLOT_LOCK = RLock()


def render_report(scenarios, *, scope="failures", source="all", stage="all",
                  top_n=20, extra_stopwords=()):
    """Build PNGs, CSV/JSON data and a ZIP in memory. Empty scenarios are rejected."""
    if not scenarios or any(not records for records in scenarios.values()):
        raise ValueError("报告需要至少一个含模拟记录的场景")
    records = [
        {**record, "scenario_name": name}
        for name, rows in scenarios.items() for record in rows
    ]
    text_analysis = analyze_decision_text(
        records, scope=scope, source=source, stage=stage,
        top_n=top_n, extra_stopwords=extra_stopwords,
    )
    stats = {name: analyze_results(rows) for name, rows in scenarios.items()}
    stage_rows = [
        {"场景": name, "环节": label, "到达": values["reached"],
         "通过": values["passed"], "失败": values["count"],
         "占全样本失败率": values["rate"],
         "到达后失败率": values["conditional_failure_rate"]}
        for name, summary in stats.items()
        for label, values in summary["stage_failure_rates"].items()
    ]
    stage_frame = pd.DataFrame(stage_rows)
    heatmap = stage_frame.pivot(index="场景", columns="环节", values="到达后失败率")
    heatmap = heatmap.reindex(index=list(scenarios), columns=list(next(iter(stats.values()))["stage_failure_rates"]))
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in (
        "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei",
        "PingFang SC", "WenQuanYi Micro Hei",
    ) if name in fonts), None)
    images = {}
    # Matplotlib uses shared style state, so concurrent Streamlit sessions serialize rendering.
    with _PLOT_LOCK, matplotlib.rc_context({
        "font.family": [font or "DejaVu Sans"], "axes.unicode_minus": False,
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    }):
        figure = Figure(figsize=(10, 5.6), layout="constrained")
        axis = figure.subplots()
        for name, summary in stats.items():
            values = summary["stage_failure_rates"]
            axis.plot(["进入模拟", *values], [summary["total_simulations"],
                      *[item["passed"] for item in values.values()]], marker="o", label=name)
        axis.set(title="各场景阶段留存", ylabel="模拟记录数")
        axis.legend(fontsize=8)
        images["stage_survival.png"] = _png(figure)

        figure = Figure(figsize=(10, max(3.5, len(scenarios) * 0.55 + 1.8)), layout="constrained")
        axis = figure.subplots()
        sns.heatmap(heatmap, mask=heatmap.isna(), annot=True, fmt=".1%",
                    vmin=0, vmax=1, cmap="Reds", ax=axis,
                    cbar_kws={"format": PercentFormatter(1)})
        axis.set(title="到达后失败率 · 空白表示无人到达", xlabel="", ylabel="")
        images["failure_heatmap.png"] = _png(figure)

        if len(scenarios) == 1:
            figure = Figure(figsize=(10, 5.6), layout="constrained")
            axis = figure.subplots()
            survival = pd.DataFrame(ownership_survival(records))
            for ownership, group in survival.groupby("企业性质", sort=False):
                axis.plot(group["环节"], group["留存比例"], marker="o",
                          label=f"{ownership} (n={int(group['企业数'].iloc[0])})")
            axis.set(title="按企业性质的阶段留存", ylabel="占本类进入模拟记录的比例", ylim=(0, 1.05))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.legend(fontsize=9)
            images["ownership_survival.png"] = _png(figure)

        if text_analysis["terms"]:
            terms = pd.DataFrame(text_analysis["terms"])
            figure = Figure(figsize=(10, max(4, len(terms) * 0.30 + 1.5)), layout="constrained")
            axis = figure.subplots()
            sns.barplot(data=terms, x="document_count", y="term",
                        color="#27836b", errorbar=None, ax=axis)
            axis.set(title=f"决策理由高频词 · 共 {text_analysis['document_count']} 条理由",
                     xlabel="包含该词的理由条数", ylabel="")
            images["reason_terms.png"] = _png(figure)

    metadata = {
        "scenarios": list(scenarios), "record_count": len(records),
        "font": font, "text_filters": {key: text_analysis[key] for key in (
            "scope", "source", "stage", "top_n", "extra_stopwords")},
        "note": "词频描述记录中的用词。重复实验按记录计数，审核理由按环节计数。词频本身不能验证因果。",
    }
    files = dict(images)
    files["stage_statistics.csv"] = stage_frame.to_csv(index=False).encode("utf-8-sig")
    terms_frame = pd.DataFrame(text_analysis["terms"], columns=["term", "count", "document_count", "document_rate"])
    files["reason_terms.csv"] = terms_frame.to_csv(index=False).encode("utf-8-sig")
    for name, payload in (("text_analysis.json", text_analysis), ("metadata.json", metadata),
                          ("scenario_summary.json", stats)):
        files[name] = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name, payload in files.items():
            bundle.writestr(name, payload)
    return {"images": images, "archive": archive.getvalue(),
            "text_analysis": text_analysis, "metadata": metadata}


def _png(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, facecolor="white")
    figure.clear()
    return buffer.getvalue()
