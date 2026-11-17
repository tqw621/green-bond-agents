"""
结果数据库模块
存储和管理模拟实验结果
"""
from contextlib import closing
import sqlite3
import json
import os
import time
from datetime import datetime
from typing import Dict, List
import config


def _connect(db_path: str) -> sqlite3.Connection:
    """建立数据库连接并启用 WAL 模式（提升并发性能）"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _execute_with_retry(cursor, sql: str, params: tuple, max_retries: int = 3):
    """执行 SQL 写入，遇到锁时自动重试"""
    for attempt in range(max_retries):
        try:
            cursor.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
                continue
            raise


class SimulationDB:
    """模拟实验结果数据库"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        # 确保数据目录存在
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with closing(_connect(self.db_path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            self._create_tables(conn.cursor())

    def _create_tables(self, cursor):
        # 模拟实验记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_time TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                enterprise_code TEXT NOT NULL,
                enterprise_name TEXT,
                enterprise_data TEXT,
                certifier_type TEXT,
                regulator_type TEXT,
                underwriter_type TEXT,
                investor_type TEXT,
                run_metadata TEXT,
                stage_1_passed INTEGER DEFAULT 0,
                stage_1_detail TEXT,
                stage_2_passed INTEGER DEFAULT 0,
                stage_2_detail TEXT,
                stage_3_passed INTEGER DEFAULT 0,
                stage_3_detail TEXT,
                stage_4_passed INTEGER DEFAULT 0,
                stage_4_detail TEXT,
                stage_5_passed INTEGER DEFAULT 0,
                stage_5_detail TEXT,
                final_result TEXT NOT NULL,
                failure_stage TEXT,
                failure_reason TEXT,
                created_at TEXT NOT NULL
            )
        ''')

        # 场景配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenario_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                certifier_type TEXT DEFAULT 'moderate',
                regulator_type TEXT DEFAULT 'moderate',
                underwriter_type TEXT DEFAULT 'moderate',
                investor_type TEXT DEFAULT 'balanced',
                total_rounds INTEGER DEFAULT 200,
                created_at TEXT NOT NULL
            )
        ''')

        columns = {row[1] for row in cursor.execute("PRAGMA table_info(simulation_records)")}
        if "run_metadata" not in columns:
            cursor.execute("ALTER TABLE simulation_records ADD COLUMN run_metadata TEXT")

    def save_result(self, result: Dict, scenario_name: str = "default"):
        self.save_batch([result], scenario_name)

    def _insert_result(self, cursor, result, scenario_name):
        stage_values = []
        for stage_key in (
            "stage_1_material", "stage_2_certifier", "stage_3_regulator",
            "stage_4_underwriter", "stage_5_investor",
        ):
            stage = result.get(stage_key)
            passed = None if stage is None else int(bool(stage.get("passed")))
            stage_values.extend((passed, json.dumps(stage, ensure_ascii=False)))

        _execute_with_retry(cursor, '''
            INSERT INTO simulation_records
            (simulation_time, scenario_name, enterprise_code, enterprise_name,
             enterprise_data, certifier_type, regulator_type, underwriter_type,
             investor_type, run_metadata, stage_1_passed, stage_1_detail, stage_2_passed,
             stage_2_detail, stage_3_passed, stage_3_detail, stage_4_passed,
             stage_4_detail, stage_5_passed, stage_5_detail, final_result,
             failure_stage, failure_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get("simulation_time"),
            scenario_name,
            result.get("enterprise_code"),
            result.get("enterprise_name"),
            json.dumps(result.get("enterprise_data", {}), ensure_ascii=False),
            result.get("certifier_type"),
            result.get("regulator_type"),
            result.get("underwriter_type"),
            result.get("investor_type"),
            json.dumps({key: result[key] for key in ("use_llm", "model", "rule_seed", "experiment_seed", "model_temperature", "rules_version") if key in result},
                       ensure_ascii=False),
            *stage_values,
            result.get("final_result"),
            result.get("failure_stage"),
            result.get("failure_reason"),
            datetime.now().isoformat(),
        ))

    def save_batch(self, results: List[Dict], scenario_name: str = "default"):
        self.save_experiment({scenario_name: results})

    def save_experiment(self, scenarios):
        """同一批次的所有场景一同提交，写入失败时回滚。"""
        with closing(_connect(self.db_path)) as conn, conn:
            cursor = conn.cursor()
            for scenario_name, records in scenarios.items():
                for record in records:
                    self._insert_result(cursor, record, scenario_name)

    def list_scenarios(self):
        with closing(_connect(self.db_path)) as conn:
            return [row[0] for row in conn.execute(
                "SELECT scenario_name FROM simulation_records GROUP BY scenario_name ORDER BY MAX(id) DESC"
            )]

    def load_scenario(self, scenario_name):
        with closing(_connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM simulation_records WHERE scenario_name = ? ORDER BY id", (scenario_name,),
            ).fetchall()
        keys = ("stage_1_material", "stage_2_certifier", "stage_3_regulator",
                "stage_4_underwriter", "stage_5_investor")
        records = []
        for row in rows:
            record = dict(row)
            record["enterprise_data"] = json.loads(record["enterprise_data"] or "{}")
            record.update(json.loads(record.pop("run_metadata") or "{}"))
            for i, key in enumerate(keys, 1):
                detail = json.loads(record.pop(f"stage_{i}_detail") or "null")
                passed = record.pop(f"stage_{i}_passed")
                if detail and "passed" not in detail:
                    # Earlier versions stored stage details without the outer result envelope.
                    detail = {"passed": bool(passed), "detail": detail}
                    if str(record.get("failure_stage") or "").startswith(f"{i}_"):
                        detail["reason"] = record.get("failure_reason") or ""
                record[key] = detail
            records.append(record)
        return records

    def get_scenario_stats(self, scenario_name: str) -> Dict:
        with closing(_connect(self.db_path)) as conn:
            total, successes = conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(final_result = 'success'), 0)
                   FROM simulation_records WHERE scenario_name = ?""", (scenario_name,),
            ).fetchone()
        return {"total": total, "successes": successes,
                "success_rate": round(successes / total, 4) if total else 0}
