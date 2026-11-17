"""Per-user credential isolation through simulation and UI entry points."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import config
from engine import run_single_simulation, run_batch_simulation, _llm_stage
from llm_client import LLMClient, prompt_api_key
from scenarios import run_preset_experiment


FIRM = {
    "code": "000001", "name": "虚构测试企业", "credit_rating": "AAA",
    "asset_liability_ratio": 0.4, "green_level": "high",
    "has_green_project_cert": True, "has_financial_report": True,
    "has_financing_plan": True, "consecutive_profit": True,
    "compliance_score": 1, "issue_scale": 10,
}


class FakeClient:
    def __init__(self, **kwargs):
        self.key = kwargs["api_key"]
        self.http_client = kwargs.get("http_client")
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if "认购决策" in system or "投资经理" in system:
            payload = {"will_subscribe": True, "subscribe_amount": 10,
                       "decision_rationale": self.key + " 收益合理"}
        elif "承销" in system:
            payload = {"decision": "underwrite", "issue_rate": 0.03,
                       "rate_rationale": self.key + " 风险可控"}
        else:
            payload = {"decision": "approved", "reason": self.key + " 审核通过", "conditions": []}
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))
        ])

    def close(self):
        if self.http_client is not None:
            self.http_client.close()


class CredentialTests(unittest.TestCase):
    def test_environment_key_does_not_enable_llm(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "environment-only-fixture"}):
            with self.assertRaises(ValueError):
                LLMClient()
            with self.assertRaises(ValueError):
                run_single_simulation(FIRM, use_llm=True)
        self.assertIn(run_single_simulation(FIRM)["final_result"], ("success", "failed"))

    @patch("llm_client.openai.OpenAI", side_effect=FakeClient)
    def test_concurrent_keys_are_isolated_and_absent_from_results(self, factory):
        keys = ["test-person-one", "test-person-two"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(
                lambda key: run_single_simulation(FIRM, use_llm=True, api_key=key), keys,
            ))
        self.assertEqual([record["final_result"] for record in results], ["success", "success"])
        serialized = json.dumps(results)
        for key in keys:
            self.assertNotIn(key, serialized)
        passed_keys = [call.kwargs["api_key"] for call in factory.call_args_list]
        for key in keys:
            self.assertEqual(passed_keys.count(key), 4)

    @patch("llm_client.openai.OpenAI", side_effect=FakeClient)
    def test_batch_and_presets_forward_user_key(self, factory):
        batch = run_batch_simulation([FIRM], {}, use_llm=True,
                                     each_enterprise_once=True, api_key="batch-fixture")
        presets = run_preset_experiment([FIRM], use_llm=True, api_key="preset-fixture")
        self.assertEqual(batch[0]["final_result"], "success")
        self.assertTrue(all(rows[0]["final_result"] == "success" for rows in presets.values()))
        self.assertEqual(factory.call_count, 32)

    @patch("llm_client.openai.OpenAI", side_effect=RuntimeError("private-fixture in provider error"))
    def test_client_repeated_calls_hide_provider_error(self, factory):
        client = LLMClient(api_key="private-fixture")
        for _ in range(2):
            response = client.call("test", "test")
            self.assertIn("[API_ERROR]", response)
            self.assertNotIn("private-fixture", response)
        self.assertEqual(factory.call_count, 2)

    @patch("llm_client.openai.OpenAI", side_effect=FakeClient)
    def test_client_supports_repeated_successful_calls(self, factory):
        client = LLMClient(api_key="reusable-fixture")
        for _ in range(2):
            self.assertEqual(json.loads(client.call("审核", "测试"))["decision"], "approved")
        self.assertEqual(factory.call_count, 2)

    def test_stage_exception_is_redacted(self):
        def fail():
            raise RuntimeError("request contained fixture-secret")
        result = _llm_stage(fail, None, lambda: (True, "规则通过", {}),
                            api_key="fixture-secret")
        self.assertNotIn("fixture-secret", json.dumps(result))
        self.assertEqual(result["detail"]["source"], "fallback_rules")

    @patch("llm_client.getpass.getpass", return_value=" terminal-fixture ")
    def test_cli_asks_for_private_input(self, prompt):
        self.assertEqual(prompt_api_key(), "terminal-fixture")
        prompt.assert_called_once()

    def test_ui_requires_key_and_clear_removes_it(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as folder, patch.object(
            config, "DB_PATH", str(Path(folder) / "ui.db")
        ):
            script = str(Path(__file__).resolve().parents[1] / "app.py")
            first = AppTest.from_file(script, default_timeout=60).run()
            next(item for item in first.checkbox if item.label == "启用 LLM API 决策").check().run()
            run = next(item for item in first.button if item.label == "运行一次模拟")
            self.assertTrue(run.disabled)
            first.text_input(key="user_api_key").input("session-fixture").run()
            self.assertFalse(next(item for item in first.button if item.label == "运行一次模拟").disabled)
            second = AppTest.from_file(script, default_timeout=60).run()
            next(item for item in second.checkbox if item.label == "启用 LLM API 决策").check().run()
            self.assertEqual(second.text_input(key="user_api_key").value, "")
            next(item for item in first.button if item.label == "清除密钥").click().run()
            self.assertEqual(first.text_input(key="user_api_key").value, "")
            self.assertTrue(next(item for item in first.button if item.label == "运行一次模拟").disabled)
            self.assertEqual(len(first.exception), 0)
