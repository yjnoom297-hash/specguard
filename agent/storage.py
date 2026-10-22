"""SQLite persistence for reviews, approvals, and local issues."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class ReviewStore:
    def __init__(self, path: Path | None = None):
        self.path = path or Path(__file__).parent.parent / "specguard.db"
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS reviews (review_id TEXT PRIMARY KEY, thread_id TEXT,
          document_id TEXT, domain TEXT, requirement_text TEXT, result_json TEXT, created TEXT);
        CREATE TABLE IF NOT EXISTS approvals (review_id TEXT, finding_id TEXT, approved INTEGER,
          created TEXT, PRIMARY KEY(review_id, finding_id));
        CREATE TABLE IF NOT EXISTS issues (issue_id TEXT PRIMARY KEY, review_id TEXT,
          finding_id TEXT, payload_json TEXT, created TEXT);
        CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT PRIMARY KEY,
          review_id TEXT, state_json TEXT, updated TEXT);
        """)
        self.conn.commit()

    def save_review(self, review_id: str, thread_id: str, document_id: str,
                    domain: str, text: str, result: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?)",
                          (review_id, thread_id, document_id, domain, text,
                           json.dumps(result, ensure_ascii=False), datetime.now().isoformat()))
        self.conn.commit()

    def get_review(self, review_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM reviews WHERE review_id=?", (review_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["result"] = json.loads(item.pop("result_json"))
        item["approved_finding_ids"] = sorted(self.approved_ids(review_id))
        return item

    def approve(self, review_id: str, finding_ids: list[str]) -> None:
        self.conn.execute("DELETE FROM approvals WHERE review_id=?", (review_id,))
        now = datetime.now().isoformat()
        self.conn.executemany("INSERT INTO approvals VALUES (?,?,1,?)",
                              [(review_id, finding_id, now) for finding_id in finding_ids])
        self.conn.commit()

    def approved_ids(self, review_id: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT finding_id FROM approvals WHERE review_id=? AND approved=1", (review_id,)
        ).fetchall()
        return {row["finding_id"] for row in rows}

    def save_issue(self, review_id: str, finding_id: str, payload: dict,
                   issue_id: str | None = None) -> str:
        existing = self.conn.execute(
            "SELECT issue_id FROM issues WHERE review_id=? AND finding_id=?", (review_id, finding_id)
        ).fetchone()
        if existing:
            return existing["issue_id"]
        issue_id = issue_id or f"SG-{uuid4().hex[:6].upper()}"
        self.conn.execute("INSERT INTO issues VALUES (?,?,?,?,?)",
                          (issue_id, review_id, finding_id, json.dumps(payload, ensure_ascii=False),
                           datetime.now().isoformat()))
        self.conn.commit()
        return issue_id

    def list_issues(self, limit: int = 50, document_id: str | None = None) -> list[dict]:
        sql = ("SELECT i.issue_id, i.review_id, i.finding_id, i.payload_json, i.created, "
               "r.document_id FROM issues i LEFT JOIN reviews r ON r.review_id=i.review_id ")
        params: tuple = ()
        if document_id:
            sql += "WHERE r.document_id=? "
            params = (document_id,)
        sql += "ORDER BY i.created DESC LIMIT ?"
        rows = self.conn.execute(sql, (*params, limit)).fetchall()
        return [{
            "issue_id": row["issue_id"],
            "review_id": row["review_id"],
            "finding_id": row["finding_id"],
            "document_id": row["document_id"],
            "finding": json.loads(row["payload_json"]),
            "created": row["created"],
        } for row in rows]

    def list_issue_document_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT r.document_id FROM issues i "
            "JOIN reviews r ON r.review_id=i.review_id "
            "WHERE r.document_id IS NOT NULL AND r.document_id<>'' "
            "ORDER BY r.document_id"
        ).fetchall()
        return [row["document_id"] for row in rows]

    def save_checkpoint(self, thread_id: str, review_id: str, state: dict) -> None:
        """Persist the latest completed graph state for a conversation thread."""
        self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?)",
            (thread_id, review_id, json.dumps(state, ensure_ascii=False),
             datetime.now().isoformat()),
        )
        self.conn.commit()

    def get_checkpoint(self, thread_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM checkpoints WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if row is None:
            return None
        return {"thread_id": row["thread_id"], "review_id": row["review_id"],
                "state": json.loads(row["state_json"]), "updated": row["updated"]}

    def close(self) -> None:
        self.conn.close()
