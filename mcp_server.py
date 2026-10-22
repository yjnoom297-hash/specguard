"""SpecGuard MCP server exposing Tool, Resource, and Prompt capabilities."""
import json
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP(name="specguard-requirements")
DATA_DIR = Path(__file__).parent / "data"


@mcp.tool()
def get_requirement_context(document_id: str) -> dict:
    """Return requirements and metadata for a document."""
    documents = json.loads((DATA_DIR / "requirements.json").read_text(encoding="utf-8"))
    return documents.get(document_id, {"document_id": document_id, "requirements": []})


@mcp.resource("quality-rule://requirements/{rule_type}")
def get_quality_rule(rule_type: str) -> str:
    """Return a static quality rule document."""
    path = DATA_DIR / "quality_rules" / f"{rule_type}.md"
    return path.read_text(encoding="utf-8") if path.exists() else "# Rule not found"


@mcp.prompt()
def requirement_review_prompt(document_id: str, domain: str) -> str:
    """Return the standardized requirement review procedure."""
    return f"""문서 {document_id}({domain})를 검토하세요.
1. 원문에 실제로 존재하는 문제 구간을 인용합니다.
2. 모호성, 누락, 충돌을 분리합니다.
3. 각 판정에 quality-rule Resource 근거를 연결합니다.
4. 측정 가능한 수정안과 Given/When/Then 테스트를 작성합니다.
5. 근거 없는 추측은 finding으로 확정하지 않습니다."""


@mcp.tool()
def create_review_issue(review_id: str, finding: dict) -> dict:
    """Persist an approved finding as a local issue. API approval is required first."""
    path = DATA_DIR / "review_issues.json"
    issues = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    issue_id = f"SG-{len(issues) + 1:03d}"
    issues.append({"issue_id": issue_id, "review_id": review_id,
                   "finding": finding, "created": datetime.now().isoformat()})
    path.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"issue_id": issue_id, "status": "created"}


if __name__ == "__main__":
    try:
        mcp.run(show_banner=False)
    except TypeError:
        mcp.run()
