"""
Feedback Storage Backends: SQLite (local dev) and PostgreSQL (production).

Usage:
    # Local development
    storage = SQLiteStorage("feedback.db")
    storage.initialize()

    # Production with EvenUp
    storage = PostgresStorage(
        host="your-postgres-host",
        database="evenup_db",
        user="app_user",
        password=os.environ["POSTGRES_PASSWORD"]
    )
    storage.initialize()
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from .feedback import (
    FeedbackStorage,
    StrategyOutcome,
    StrategyStats,
)


# =============================================================================
# SQLite Storage (Local Development)
# =============================================================================

class SQLiteStorage(FeedbackStorage):
    """
    SQLite-based feedback storage for local development.

    Zero setup - just specify a file path.
    """

    def __init__(self, db_path: str = "feedback.db"):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_intent TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    strategy_config TEXT,
                    docs_retrieved INTEGER DEFAULT 0,
                    docs_after_filter INTEGER DEFAULT 0,
                    novelty_score REAL DEFAULT 0.0,
                    rerank_top_score REAL DEFAULT 0.0,
                    user_feedback REAL,
                    implicit_success REAL,
                    latency_ms REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_intent
                ON strategy_outcomes(query_intent)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_strategy
                ON strategy_outcomes(strategy_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp
                ON strategy_outcomes(timestamp)
            """)

    def log_outcome(self, outcome: StrategyOutcome) -> None:
        """Store a strategy outcome."""
        data = outcome.to_dict()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO strategy_outcomes (
                    session_id, query_id, query_text, query_intent,
                    strategy_type, strategy_config, docs_retrieved,
                    docs_after_filter, novelty_score, rerank_top_score,
                    user_feedback, implicit_success, latency_ms,
                    timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["session_id"], data["query_id"], data["query_text"],
                data["query_intent"], data["strategy_type"], data["strategy_config"],
                data["docs_retrieved"], data["docs_after_filter"],
                data["novelty_score"], data["rerank_top_score"],
                data["user_feedback"], data["implicit_success"],
                data["latency_ms"], data["timestamp"], data["metadata"]
            ))

    def get_strategy_stats(
        self,
        intent: Optional[str] = None,
        strategy: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[StrategyStats]:
        """Get aggregated statistics for strategies."""
        query = """
            SELECT
                query_intent,
                strategy_type,
                COUNT(*) as total_uses,
                AVG(COALESCE(user_feedback, implicit_success, rerank_top_score)) as avg_success,
                AVG(latency_ms) as avg_latency,
                AVG(docs_retrieved) as avg_docs,
                MAX(timestamp) as last_used
            FROM strategy_outcomes
            WHERE 1=1
        """
        params = []

        if intent:
            query += " AND query_intent = ?"
            params.append(intent)

        if strategy:
            query += " AND strategy_type = ?"
            params.append(strategy)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " GROUP BY query_intent, strategy_type"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            StrategyStats(
                intent=row["query_intent"],
                strategy=row["strategy_type"],
                total_uses=row["total_uses"],
                avg_success=row["avg_success"] or 0.0,
                avg_latency_ms=row["avg_latency"] or 0.0,
                avg_docs_retrieved=row["avg_docs"] or 0.0,
                last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else datetime.utcnow()
            )
            for row in rows
        ]

    def get_recent_outcomes(
        self,
        limit: int = 100,
        intent: Optional[str] = None
    ) -> List[StrategyOutcome]:
        """Get recent outcomes for analysis."""
        query = """
            SELECT * FROM strategy_outcomes
            WHERE 1=1
        """
        params = []

        if intent:
            query += " AND query_intent = ?"
            params.append(intent)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [StrategyOutcome.from_dict(dict(row)) for row in rows]

    def get_best_strategy(self, intent: str) -> Optional[str]:
        """Get the historically best-performing strategy for an intent."""
        stats = self.get_strategy_stats(intent=intent)

        if not stats:
            return None

        # Find strategy with highest average success
        best = max(stats, key=lambda s: s.avg_success)
        return best.strategy if best.total_uses >= 5 else None


# =============================================================================
# PostgreSQL Storage (Production)
# =============================================================================

class PostgresStorage(FeedbackStorage):
    """
    PostgreSQL-based feedback storage for production.

    Designed to integrate with existing infrastructure (e.g., EvenUp).
    """

    def __init__(
        self,
        host: str = None,
        port: int = 5432,
        database: str = None,
        user: str = None,
        password: str = None,
        schema: str = "agentic_search"
    ):
        """
        Initialize PostgreSQL storage.

        Args:
            host: Database host (defaults to PG_HOST env var)
            port: Database port (defaults to PG_PORT env var or 5432)
            database: Database name (defaults to PG_DATABASE env var)
            user: Database user (defaults to PG_USER env var)
            password: Database password (defaults to POSTGRES_PASSWORD env var)
            schema: Schema name for tables (default: agentic_search)
        """
        self.host = host or os.environ.get("PG_HOST", "localhost")
        self.port = port or int(os.environ.get("PG_PORT", "5432"))
        self.database = database or os.environ.get("PG_DATABASE", "postgres")
        self.user = user or os.environ.get("PG_USER", "postgres")
        self.password = password or os.environ.get("POSTGRES_PASSWORD", "")
        self.schema = schema

        self._pool = None

    def _get_connection_string(self) -> str:
        """Build connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL storage. "
                "Install it with: pip install psycopg2-binary"
            )

        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create schema and tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create schema
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

            # Create table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.strategy_outcomes (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    query_id VARCHAR(255) NOT NULL,
                    query_text TEXT NOT NULL,
                    query_intent VARCHAR(100) NOT NULL,
                    strategy_type VARCHAR(50) NOT NULL,
                    strategy_config JSONB DEFAULT '{{}}',
                    docs_retrieved INTEGER DEFAULT 0,
                    docs_after_filter INTEGER DEFAULT 0,
                    novelty_score FLOAT DEFAULT 0.0,
                    rerank_top_score FLOAT DEFAULT 0.0,
                    user_feedback FLOAT,
                    implicit_success FLOAT,
                    latency_ms FLOAT DEFAULT 0.0,
                    timestamp TIMESTAMPTZ NOT NULL,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Indexes
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_outcomes_intent
                ON {self.schema}.strategy_outcomes(query_intent)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_outcomes_strategy
                ON {self.schema}.strategy_outcomes(strategy_type)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp
                ON {self.schema}.strategy_outcomes(timestamp)
            """)
            # Composite index for common queries
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_outcomes_intent_strategy
                ON {self.schema}.strategy_outcomes(query_intent, strategy_type)
            """)

    def log_outcome(self, outcome: StrategyOutcome) -> None:
        """Store a strategy outcome."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {self.schema}.strategy_outcomes (
                    session_id, query_id, query_text, query_intent,
                    strategy_type, strategy_config, docs_retrieved,
                    docs_after_filter, novelty_score, rerank_top_score,
                    user_feedback, implicit_success, latency_ms,
                    timestamp, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                outcome.session_id, outcome.query_id, outcome.query_text,
                outcome.query_intent, outcome.strategy_type,
                json.dumps(outcome.strategy_config),
                outcome.docs_retrieved, outcome.docs_after_filter,
                outcome.novelty_score, outcome.rerank_top_score,
                outcome.user_feedback, outcome.implicit_success,
                outcome.latency_ms, outcome.timestamp,
                json.dumps(outcome.metadata)
            ))

    def get_strategy_stats(
        self,
        intent: Optional[str] = None,
        strategy: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[StrategyStats]:
        """Get aggregated statistics for strategies."""
        query = f"""
            SELECT
                query_intent,
                strategy_type,
                COUNT(*) as total_uses,
                AVG(COALESCE(user_feedback, implicit_success, rerank_top_score)) as avg_success,
                AVG(latency_ms) as avg_latency,
                AVG(docs_retrieved) as avg_docs,
                MAX(timestamp) as last_used
            FROM {self.schema}.strategy_outcomes
            WHERE 1=1
        """
        params = []

        if intent:
            query += " AND query_intent = %s"
            params.append(intent)

        if strategy:
            query += " AND strategy_type = %s"
            params.append(strategy)

        if since:
            query += " AND timestamp >= %s"
            params.append(since)

        query += " GROUP BY query_intent, strategy_type"

        with self._get_connection() as conn:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [
            StrategyStats(
                intent=row["query_intent"],
                strategy=row["strategy_type"],
                total_uses=row["total_uses"],
                avg_success=float(row["avg_success"] or 0.0),
                avg_latency_ms=float(row["avg_latency"] or 0.0),
                avg_docs_retrieved=float(row["avg_docs"] or 0.0),
                last_used=row["last_used"] or datetime.utcnow()
            )
            for row in rows
        ]

    def get_recent_outcomes(
        self,
        limit: int = 100,
        intent: Optional[str] = None
    ) -> List[StrategyOutcome]:
        """Get recent outcomes for analysis."""
        query = f"""
            SELECT * FROM {self.schema}.strategy_outcomes
            WHERE 1=1
        """
        params = []

        if intent:
            query += " AND query_intent = %s"
            params.append(intent)

        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)

        with self._get_connection() as conn:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()

        results = []
        for row in rows:
            row_dict = dict(row)
            # Handle JSONB fields
            if isinstance(row_dict.get("strategy_config"), dict):
                row_dict["strategy_config"] = json.dumps(row_dict["strategy_config"])
            if isinstance(row_dict.get("metadata"), dict):
                row_dict["metadata"] = json.dumps(row_dict["metadata"])
            # Handle timestamp
            if isinstance(row_dict.get("timestamp"), datetime):
                row_dict["timestamp"] = row_dict["timestamp"].isoformat()
            results.append(StrategyOutcome.from_dict(row_dict))

        return results

    def get_best_strategy(self, intent: str) -> Optional[str]:
        """Get the historically best-performing strategy for an intent."""
        stats = self.get_strategy_stats(intent=intent)

        if not stats:
            return None

        best = max(stats, key=lambda s: s.avg_success)
        return best.strategy if best.total_uses >= 5 else None


# =============================================================================
# Factory Function
# =============================================================================

def get_feedback_storage(
    backend: str = "sqlite",
    **kwargs
) -> FeedbackStorage:
    """
    Factory function to get the appropriate storage backend.

    Args:
        backend: "sqlite" or "postgres"
        **kwargs: Backend-specific configuration

    Returns:
        Configured FeedbackStorage instance

    Usage:
        # Local dev
        storage = get_feedback_storage("sqlite", db_path="./feedback.db")

        # Production
        storage = get_feedback_storage("postgres", host="db.evenup.com", database="prod")
    """
    if backend == "sqlite":
        return SQLiteStorage(db_path=kwargs.get("db_path", "feedback.db"))

    elif backend == "postgres":
        return PostgresStorage(
            host=kwargs.get("host"),
            port=kwargs.get("port", 5432),
            database=kwargs.get("database"),
            user=kwargs.get("user"),
            password=kwargs.get("password"),
            schema=kwargs.get("schema", "agentic_search")
        )

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'sqlite' or 'postgres'.")
