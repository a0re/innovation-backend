"""
SQLite database module for spam detection API.
Stores prediction history and statistics.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import os
import logging

logger = logging.getLogger(__name__)

# Database file location
DB_FILE = os.path.join(os.path.dirname(__file__), "spam_detection.db")


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """Initialize the database schema"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Create predictions table
        cursor.execute("""
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

        # Create index on timestamp for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON predictions(timestamp)
        """)

        # Create index on is_spam for statistics
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_is_spam
            ON predictions(is_spam)
        """)

        logger.info(f"Database initialized at {DB_FILE}")


def save_prediction(
    message: str,
    processed_message: str,
    prediction: str,
    confidence: float,
    is_spam: bool,
    model_results: Optional[Dict[str, Any]] = None,
    cluster_id: Optional[int] = None
) -> int:
    """
    Save a prediction to the database.

    Args:
        message: Original message text
        processed_message: Preprocessed message
        prediction: Prediction result (spam/not spam)
        confidence: Confidence score
        is_spam: Boolean spam indicator
        model_results: JSON of individual model predictions
        cluster_id: Cluster ID if applicable

    Returns:
        ID of the inserted record
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        model_results_json = json.dumps(model_results) if model_results else None

        cursor.execute("""
            INSERT INTO predictions
            (message, processed_message, prediction, confidence, is_spam, model_results, cluster_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message,
            processed_message,
            prediction,
            confidence,
            is_spam,
            model_results_json,
            cluster_id,
            datetime.now().isoformat()
        ))

        return cursor.lastrowid


def get_statistics() -> Dict[str, Any]:
    """
    Get prediction statistics from database.

    Returns:
        Dictionary with total predictions, spam/not spam counts, and average confidence
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get total predictions
        cursor.execute("SELECT COUNT(*) as total FROM predictions")
        total = cursor.fetchone()['total']

        # Get spam count
        cursor.execute("SELECT COUNT(*) as spam FROM predictions WHERE is_spam = 1")
        spam_count = cursor.fetchone()['spam']

        # Get not spam count
        cursor.execute("SELECT COUNT(*) as not_spam FROM predictions WHERE is_spam = 0")
        not_spam_count = cursor.fetchone()['not_spam']

        # Get average confidence
        cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
        result = cursor.fetchone()
        avg_confidence = result['avg_conf'] if result['avg_conf'] else 0.0

        return {
            "total_predictions": total,
            "spam_detected": spam_count,
            "not_spam_detected": not_spam_count,
            "average_confidence": float(avg_confidence)
        }


def get_recent_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent predictions from database.

    Args:
        limit: Maximum number of predictions to return

    Returns:
        List of prediction dictionaries
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, message, prediction, confidence, is_spam,
                   cluster_id, timestamp, user_feedback
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_predictions_by_date_range(
    start_date: str,
    end_date: str
) -> List[Dict[str, Any]]:
    """
    Get predictions within a date range.

    Args:
        start_date: Start date (ISO format)
        end_date: End date (ISO format)

    Returns:
        List of prediction dictionaries
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, message, prediction, confidence, is_spam,
                   cluster_id, timestamp, user_feedback
            FROM predictions
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """, (start_date, end_date))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def update_user_feedback(prediction_id: int, feedback: str) -> bool:
    """
    Update user feedback for a prediction.

    Args:
        prediction_id: ID of the prediction
        feedback: User feedback (e.g., "correct", "incorrect")

    Returns:
        True if successful, False otherwise
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE predictions
            SET user_feedback = ?
            WHERE id = ?
        """, (feedback, prediction_id))

        return cursor.rowcount > 0


def get_spam_examples(limit: int = 10) -> List[str]:
    """
    Get example spam messages from database.

    Args:
        limit: Maximum number of examples

    Returns:
        List of spam message texts
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT message
            FROM predictions
            WHERE is_spam = 1
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        return [row['message'] for row in rows]


def get_not_spam_examples(limit: int = 10) -> List[str]:
    """
    Get example not spam messages from database.

    Args:
        limit: Maximum number of examples

    Returns:
        List of not spam message texts
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT message
            FROM predictions
            WHERE is_spam = 0
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        return [row['message'] for row in rows]


def get_prediction_by_id(prediction_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a single prediction by ID.

    Args:
        prediction_id: ID of the prediction

    Returns:
        Prediction dictionary or None if not found
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM predictions
            WHERE id = ?
        """, (prediction_id,))

        row = cursor.fetchone()
        return dict(row) if row else None


def delete_old_predictions(days: int = 90) -> int:
    """
    Delete predictions older than specified days.

    Args:
        days: Number of days to keep

    Returns:
        Number of deleted records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM predictions
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))

        return cursor.rowcount


def clear_all_predictions() -> int:
    """
    Clear all predictions from database.

    Returns:
        Number of deleted records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("DELETE FROM predictions")

        return cursor.rowcount
