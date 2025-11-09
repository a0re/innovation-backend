"""
Simplified database operations for spam detection API.
Combines repository pattern with simple interface.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from threading import Lock

logger = logging.getLogger(__name__)


class SimpleDatabase:
    """
    Simplified database interface for predictions.

    Uses a thread-safe connection pattern suitable for SQLite with FastAPI.
    For production with high concurrency, consider PostgreSQL with asyncpg.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, db_path: str = "spam_detection.db"):
        """Singleton pattern to ensure single database instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = "spam_detection.db"):
        """Initialize database connection."""
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self.db_path = db_path
            self._connection_lock = Lock()
            self._init_tables()
            self._initialized = True

    @contextmanager
    def _get_connection(self):
        """
        Get database connection with row factory.
        Uses lock to ensure thread-safe access to SQLite.
        """
        with self._connection_lock:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent read performance
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}")
                raise
            finally:
                conn.close()

    def _init_tables(self):
        """Initialize database tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    processed_message TEXT,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_spam BOOLEAN NOT NULL,
                    model_results TEXT,
                    cluster_id INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_feedback TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_is_spam ON predictions(is_spam)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster ON predictions(cluster_id)")

            logger.info("Database tables initialized")

    def save_prediction(
        self,
        message: str,
        prediction: str,
        confidence: float,
        is_spam: bool,
        processed_message: str = None,
        model_results: Dict = None,
        cluster_id: int = None,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Save a prediction to the database.

        Args:
            message: Original message text
            prediction: Prediction result (spam/ham)
            confidence: Confidence score
            is_spam: Boolean spam flag
            processed_message: Preprocessed message
            model_results: Optional dict of all model results
            cluster_id: Optional cluster ID
            timestamp: Optional timestamp for the prediction (ISO format)

        Returns:
            ID of created record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            model_results_json = json.dumps(model_results) if model_results else None
            ts = timestamp or datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO predictions
                (message, processed_message, prediction, confidence, is_spam, model_results, cluster_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (message, processed_message, prediction, confidence, is_spam, model_results_json, cluster_id, ts))

            return cursor.lastrowid

    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall prediction statistics.

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Basic counts
            total = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            spam = cursor.execute("SELECT COUNT(*) FROM predictions WHERE is_spam = 1").fetchone()[0]
            ham = total - spam

            # Average confidence
            avg_conf = cursor.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0

            # Top clusters (if available)
            top_clusters = cursor.execute("""
                SELECT cluster_id, COUNT(*) as count
                FROM predictions
                WHERE cluster_id IS NOT NULL AND is_spam = 1
                GROUP BY cluster_id
                ORDER BY count DESC
                LIMIT 5
            """).fetchall()

            return {
                "total_predictions": total,
                "spam_count": spam,
                "ham_count": ham,
                "spam_rate": round(spam / total, 3) if total > 0 else 0,
                "avg_confidence": round(avg_conf, 3),
                "top_clusters": [{"cluster_id": row[0], "count": row[1]} for row in top_clusters]
            }

    def get_trends(self, period: str = "day", limit: int = 30) -> Dict[str, Any]:
        """
        Get prediction trends over time.

        Args:
            period: Time period grouping (hour, day, week, month)
            limit: Maximum number of periods to return

        Returns:
            Dictionary with trend data
        """
        # Whitelist allowed period formats to prevent SQL injection
        period_formats = {
            "hour": "strftime('%Y-%m-%d %H:00', timestamp)",
            "day": "DATE(timestamp)",
            "week": "strftime('%Y-W%W', timestamp)",
            "month": "strftime('%Y-%m', timestamp)"
        }

        # Validate period input
        if period not in period_formats:
            period = "day"

        interval = period_formats[period]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get counts per period - safe because interval is from whitelist
            query = f"""
                SELECT
                    {interval} as period,
                    COUNT(*) as total,
                    SUM(CASE WHEN is_spam = 1 THEN 1 ELSE 0 END) as spam_count,
                    AVG(confidence) as avg_confidence
                FROM predictions
                GROUP BY period
                ORDER BY period DESC
                LIMIT ?
            """
            trends = cursor.execute(query, (limit,)).fetchall()

            # Get cluster distribution per period if available
            cluster_query = f"""
                SELECT
                    {interval} as period,
                    cluster_id,
                    COUNT(*) as count
                FROM predictions
                WHERE cluster_id IS NOT NULL AND is_spam = 1
                GROUP BY period, cluster_id
                ORDER BY period DESC, count DESC
            """
            cluster_trends = cursor.execute(cluster_query).fetchall()

            # Format results
            result = {
                "period": period,
                "data": []
            }

            for row in trends:
                period_key, total, spam, avg_conf = row
                result["data"].append({
                    "period": period_key,
                    "total": total,
                    "spam_count": spam,
                    "ham_count": total - spam,
                    "spam_rate": round(spam / total, 3) if total > 0 else 0,
                    "avg_confidence": round(avg_conf, 3) if avg_conf else 0
                })

            # Add cluster data
            cluster_data = {}
            for row in cluster_trends:
                period_key, cluster_id, count = row
                if period_key not in cluster_data:
                    cluster_data[period_key] = []
                cluster_data[period_key].append({
                    "cluster_id": cluster_id,
                    "count": count
                })

            for item in result["data"]:
                item["clusters"] = cluster_data.get(item["period"], [])

            return result

    def get_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent predictions.

        Args:
            limit: Maximum number of predictions to return

        Returns:
            List of prediction dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            predictions = cursor.execute("""
                SELECT
                    id, message, prediction, confidence, is_spam,
                    cluster_id, timestamp, user_feedback
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (min(limit, 1000),)).fetchall()

            return [dict(row) for row in predictions]

    def get_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific prediction by ID.

        Args:
            prediction_id: Prediction ID

        Returns:
            Prediction dictionary or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            row = cursor.execute("""
                SELECT * FROM predictions WHERE id = ?
            """, (prediction_id,)).fetchone()

            return dict(row) if row else None

    def delete_all(self):
        """Delete all predictions (for reset)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM predictions")
            logger.info("All predictions deleted")

    def is_empty(self) -> bool:
        """Check if database is empty."""
        with self._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            return count == 0
    
    def get_cluster_distribution(self) -> List[Dict[str, Any]]:
        """
        Get cluster distribution from database.
        
        Returns:
            List of dictionaries with cluster_id and count
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cluster_counts = cursor.execute("""
                SELECT cluster_id, COUNT(*) as count
                FROM predictions
                WHERE cluster_id IS NOT NULL AND is_spam = 1
                GROUP BY cluster_id
                ORDER BY cluster_id
            """).fetchall()
            
            return [{"cluster_id": row[0], "count": row[1]} for row in cluster_counts]