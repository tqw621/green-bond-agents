"""Regression checks for counts, source attribution, exports and UI integration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from io import BytesIO
from zipfile import ZipFile

from analysis import analyze_decision_text, decision_documents
from engine import analyze_results, STAGES
from reporting import render_report
from database import SimulationDB


def failure_record():
    return {
        "enterprise_code": "000001", "enterprise_name": "测试企业",
        "simulation_time": "2026-11-17T10:00:00",
        "final_result": "failed", "failure_stage": "2_green_certification",
        "failure_reason": "绿色项目 信息披露不足，信息披露需要补充",
        "use_llm": True,
        "stage_1_material": {"passed": True, "reason": "材料完整"},
        "stage_2_certifier": {
            "passed": False, "reason": "绿色项目 信息披露不足，信息披露需要补充",
            "detail": {"source": "llm"},
        },
    }


class AnalysisTests(unittest.TestCase):
    def test_failure_reason_not_double_counted_and_terms_track_documents(self):
        result = analyze_decision_text([failure_record()], source="llm")
        self.assertEqual(result["document_count"], 1)
        term = next(row for row in result["terms"] if row["term"] == "信息披露")
        self.assertEqual(term["count"], 2)
        self.assertEqual(term["document_count"], 1)
        self.assertEqual(term["document_rate"], 1)

    def test_source_stage_filters_and_empty_output(self):
        record = failure_record()
        record["stage_2_certifier"]["detail"]["source"] = "fallback_rules"
        self.assertEqual(analyze_decision_text([record], source="llm")["terms"], [])
        self.assertEqual(analyze_decision_text([record], stage="3_regulatory_review")["document_count"], 0)
        result = analyze_decision_text([record], extra_stopwords=("信息披露",))
        self.assertNotIn("信息披露", [item["term"] for item in result["terms"]])
        self.assertEqual(result["source_counts"], {"fallback_rules": 1})
        self.assertEqual(analyze_decision_text([])["document_count"], 0)

    def test_legacy_reason_and_metadata_are_conservative(self):
        record = failure_record()
        del record["stage_2_certifier"]
        documents = decision_documents([record])
        self.assertEqual(documents[-1]["source"], "unknown")
        self.assertEqual(documents[-1]["reason"], record["failure_reason"])

    def test_repeated_runs_count_as_separate_records(self):
        result = analyze_decision_text([failure_record(), failure_record()])
        self.assertEqual(result["document_count"], 2)
        self.assertEqual(result["record_count"], 2)

    def test_all_success_has_no_bottleneck(self):
        record = failure_record()
        record.update(final_result="success", failure_stage=None, failure_reason=None)
        for _, _, key in STAGES:
            record[key] = {"passed": True, "reason": "通过"}
        self.assertEqual(analyze_results([record])["bottleneck_stage"], "无")

    def test_report_and_cli_use_existing_json(self):
        with tempfile.TemporaryDirectory() as folder:
            report = render_report({"P0": [failure_record()]})
            self.assertTrue(report["images"]["failure_heatmap.png"].startswith(b"\x89PNG"))
            with ZipFile(BytesIO(report["archive"])) as archive:
                result = json.loads(archive.read("text_analysis.json"))
                self.assertEqual(result["document_count"], 1)
                self.assertIn("stage_statistics.csv", archive.namelist())
            source = Path(folder) / "input.json"
            target = Path(folder) / "report.zip"
            source.write_text(json.dumps({"scenarios": {"P0": [failure_record()]}}), encoding="utf-8")
            process = subprocess.run(
                [sys.executable, "run.py", "--analyze", str(source), "--output", str(target)],
                capture_output=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr.decode("utf-8", errors="replace"))
            self.assertTrue(target.is_file())
            overwrite = subprocess.run(
                [sys.executable, "run.py", "--analyze", str(source), "--output", str(target)],
                capture_output=True,
            )
            self.assertNotEqual(overwrite.returncode, 0)

    def test_legacy_database_detail_survives_load(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "legacy.db")
            db = SimulationDB(path)
            db.save_result(failure_record(), "old")
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(
                    "UPDATE simulation_records SET stage_2_detail = ?",
                    (json.dumps({"decision": "rejected"}),),
                )
            result = db.load_scenario("old")[0]
            self.assertFalse(result["stage_2_certifier"]["passed"])
            self.assertEqual(analyze_decision_text([result])["document_count"], 1)

    def test_streamlit_analysis_filters_and_report(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "ui.db")
            db = SimulationDB(path)
            db.save_experiment({"preset_test_P0": [failure_record()], "preset_test_P1": [failure_record()]})
            previous = os.environ.get("DB_PATH")
            os.environ["DB_PATH"] = path
            # config is already imported by engine in this process.
            import config
            original_path = config.DB_PATH
            config.DB_PATH = path
            try:
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
                self.assertEqual(len(app.exception), 0, list(app.exception))
                preview = next(item for item in app.checkbox if item.label == "生成并预览静态报告")
                preview.check().run()
                self.assertEqual(len(app.exception), 0, list(app.exception))
                source = next(item for item in app.selectbox if item.label == "判断来源")
                source.select("fallback_rules").run()
                self.assertEqual(len(app.exception), 0, list(app.exception))
                batch = next(item for item in app.checkbox if item.label == "分析同批次全部 P0–P6 场景")
                batch.check().run()
                self.assertEqual(len(app.exception), 0, list(app.exception))
            finally:
                config.DB_PATH = original_path
                if previous is None:
                    os.environ.pop("DB_PATH", None)
                else:
                    os.environ["DB_PATH"] = previous


if __name__ == "__main__":
    unittest.main()
