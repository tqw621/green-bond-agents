"""
结果数据库模块
存储和管理模拟实验结果
"""
import sqlite3
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
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
        conn = _connect(self.db_path)
        cursor = conn.cursor()

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

        conn.commit()
        conn.close()

    def save_result(self, result: Dict, scenario_name: str = "default"):
        """保存单条模拟结果"""
        conn = _connect(self.db_path)
        cursor = conn.cursor()

        _execute_with_retry(cursor, '''
            INSERT INTO simulation_records
            (simulation_time, scenario_name, enterprise_code, enterprise_name,
             enterprise_data, certifier_type, regulator_type, underwriter_type,
             investor_type, stage_1_passed, stage_1_detail, stage_2_passed,
             stage_2_detail, stage_3_passed, stage_3_detail, stage_4_passed,
             stage_4_detail, stage_5_passed, stage_5_detail, final_result,
             failure_stage, failure_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            1 if result.get("stage_1_material", {}).get("passed") else 0,
            json.dumps(result.get("stage_1_material", {}), ensure_ascii=False),
            1 if result.get("stage_2_certifier", {}).get("passed") else 0,
            json.dumps(result.get("stage_2_certifier", {}), ensure_ascii=False),
            1 if result.get("stage_3_regulator", {}).get("passed") else 0,
            json.dumps(result.get("stage_3_regulator", {}), ensure_ascii=False),
            1 if result.get("stage_4_underwriter", {}).get("passed") else 0,
            json.dumps(result.get("stage_4_underwriter", {}), ensure_ascii=False),
            1 if result.get("stage_5_investor", {}).get("passed") else 0,
            json.dumps(result.get("stage_5_investor", {}), ensure_ascii=False),
            result.get("final_result"),
            result.get("failure_stage"),
            result.get("failure_reason"),
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

    def save_batch(self, results: List[Dict], scenario_name: str = "default"):
        """批量保存模拟结果"""
        for r in results:
            self.save_result(r, scenario_name)

    def get_scenario_stats(self, scenario_name: str) -> Dict:
        """获取指定场景的统计信息"""
        conn = _connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*),
                   SUM(CASE WHEN final_result='success' THEN 1 ELSE 0 END),
                   failure_stage, COUNT(*) as cnt
            FROM simulation_records
            WHERE scenario_name = ?
            GROUP BY failure_stage
        ''', (scenario_name,))
        rows = cursor.fetchall()
        conn.close()

        total = rows[0][0] if rows else 0
        successes = rows[0][1] if rows else 0

        return {
            "total": total,
            "successes": successes,
            "success_rate": round(successes / total, 4) if total > 0 else 0,
        }
