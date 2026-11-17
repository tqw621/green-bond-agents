"""Document intake must stay usable and disclose every assumed input."""

import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import pandas as pd

from document_import import (
    convert_text, docx_to_text, export_csv, export_excel, load_reviewed_enterprises,
)
from engine import run_single_simulation
import config


class DocumentImportTests(unittest.TestCase):
    def test_missing_fields_are_warned_exported_and_simulated(self):
        review, processed, missing, notices = convert_text("企业名称：示例企业甲；信用评级：AA")
        self.assertEqual(len(processed), 1)
        self.assertIn("资产负债率", review.loc[0, "待核实字段"])
        self.assertIn("decision", missing["TEXT0001"])
        self.assertTrue(any("模拟占位值" in notice for notice in notices))
        enterprise = load_reviewed_enterprises(processed, missing)[0]
        self.assertIn("asset_liability_ratio", enterprise["assumed_input_fields"])
        result = run_single_simulation(enterprise)
        self.assertIn(result["final_result"], {"success", "failed"})
        self.assertIn("asset_liability_ratio", result["enterprise_data"]["assumed_input_fields"])
        csv_roundtrip = pd.read_csv(io.BytesIO(export_csv(review)), dtype=str)
        xlsx_roundtrip = pd.read_excel(io.BytesIO(export_excel(review)), dtype=str)
        self.assertEqual(csv_roundtrip.loc[0, "待核实字段"], xlsx_roundtrip.loc[0, "待核实字段"])

    def test_multiple_firms_prose_and_explicit_no(self):
        text = ("企业编号：0001；企业名称：甲公司；信用评级：AAA；资产负债率为65%；"
                "绿色程度：高；发行决策：发行\n"
                "企业编号：0002；企业名称：乙公司；发行决策：不发行")
        review, processed, missing, notices = convert_text(text)
        self.assertEqual(len(review), 2)
        self.assertEqual(processed["code"].tolist(), ["0001"])
        self.assertAlmostEqual(float(processed.iloc[0]["asset_liability_ratio"]), .65)
        self.assertTrue(any("明确不发行" in notice for notice in notices))

    def test_docx_paragraph_and_table(self):
        xml = ("<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
               "<w:body><w:p><w:r><w:t>企业资料</w:t></w:r></w:p>"
               "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>企业名称</w:t></w:r></w:p></w:tc>"
               "<w:tc><w:p><w:r><w:t>信用评级</w:t></w:r></w:p></w:tc></w:tr>"
               "<w:tr><w:tc><w:p><w:r><w:t>甲公司</w:t></w:r></w:p></w:tc>"
               "<w:tc><w:p><w:r><w:t>AA</w:t></w:r></w:p></w:tc></w:tr>"
               "</w:tbl></w:body></w:document>")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as document:
            document.writestr("word/document.xml", xml)
        extracted = docx_to_text(buffer.getvalue())
        self.assertIn("name：甲公司", extracted)
        self.assertEqual(convert_text(extracted)[0].loc[0, "credit_rating"], "AA")
        with self.assertRaisesRegex(ValueError, "另存为"):
            docx_to_text(b"not a document")

    def test_unpaired_market_input_warns_and_continues(self):
        review, processed, missing, notices = convert_text(
            "企业名称：甲公司；市场基准收益率：1.5%"
        )
        self.assertEqual(len(processed), 1)
        self.assertIn("定价利差(bp)", review.loc[0, "待核实字段"])
        self.assertIn("pricing_spread_bp", missing["TEXT0001"])
        self.assertTrue(any("模拟定价" in item for item in notices))
        self.assertTrue(pd.isna(processed.iloc[0]["market_benchmark_yield"]))

    def test_narrative_and_optional_model_fallback(self):
        narrative = "甲公司评级AA，负债率65%，具备财务报告，计划发行绿色债券。"
        review, processed, missing, notices = convert_text(narrative)
        self.assertEqual(review.loc[0, "name"], "甲公司")
        self.assertEqual(review.loc[0, "credit_rating"], "AA")
        self.assertAlmostEqual(float(processed.iloc[0]["asset_liability_ratio"]), .65)
        with patch("document_import._parse_model", side_effect=ValueError("模型未返回企业数组")):
            _, processed, _, notices = convert_text(narrative, use_llm=True, api_key="fixture-key")
        self.assertEqual(len(processed), 1)
        self.assertTrue(any("本地字段提取" in item for item in notices))
        self.assertNotIn("fixture-key", repr(notices))

    def test_ui_pasted_text_runs_with_missing_fields(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as folder, patch.object(
            config, "DB_PATH", str(Path(folder) / "ui.db")
        ):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                                    default_timeout=60).run()
            next(r for r in app.radio if r.label == "选择数据来源").set_value("Word / 文字提取").run()
            next(area for area in app.text_area if area.label == "或者粘贴企业文字").input(
                "企业名称：测试企业；信用评级：AA"
            ).run()
            next(button for button in app.button if button.label == "提取并转换").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(next(button for button in app.button if button.label == "运行一次模拟").disabled)
            self.assertTrue(any("模拟占位值" in item.value for item in app.warning))
            next(item for item in app.checkbox if item.label == "使用 LLM 辅助提取").check().run()
            self.assertTrue(any(item.label == "提取用 API Key" for item in app.text_input))
            self.assertFalse(next(button for button in app.button if button.label == "运行一次模拟").disabled)


if __name__ == "__main__":
    unittest.main()
