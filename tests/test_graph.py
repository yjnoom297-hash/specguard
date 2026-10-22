import asyncio

from agent.graph import _conflict_findings, _omission_findings, build_graph
from agent.runtime import ModelDecision


class FakeMCP:
    async def get_prompt(self, document_id, domain):
        return f"review {document_id} {domain}"

    async def call_tool(self, name, arguments):
        return {"document_id": arguments["document_id"], "requirements": []}

    async def read_rule(self, rule_type):
        return f"# {rule_type} 품질 기준"


def test_three_defect_types_are_found():
    graph = build_graph(FakeMCP())
    text = ("R-01 빠르게 주문을 처리한다. R-02 결제에 실패하면 필요한 처리를 한다. "
            "R-03 주문 취소는 24시간 이내다. R-04 주문 취소는 48시간 이내다.")
    result = asyncio.run(graph.ainvoke({"query": text, "thread_id": "test",
        "document_id": "ORDER-001", "domain": "주문·결제", "scope": "all",
        "events": [], "replan_rounds": 0, "validation_attempts": 0}))
    assert {item["type"] for item in result["final"]["findings"]} == {
        "ambiguity", "omission", "conflict"
    }
    assert all(item["revision"] and item["test"]["then"] for item in result["final"]["findings"])


def test_scope_limits_workers():
    graph = build_graph(FakeMCP())
    result = asyncio.run(graph.ainvoke({"query": "시스템은 빠르게 요청을 처리한다.",
        "thread_id": "test", "document_id": "X", "domain": "회원",
        "scope": "ambiguity", "events": [], "replan_rounds": 0,
        "validation_attempts": 0}))
    assert all(item["type"] == "ambiguity" for item in result["final"]["findings"])


class NoAnalysisModel:
    async def json(self, *, tier, system, user):
        if "route" in system:
            return ModelDecision({"route": "review", "reasoning": "test"}, "fast-test")
        return ModelDecision({"needs_analysis": False,
                              "tasks": [{"kind": "ambiguity"}]}, "fast-test")


def test_authoritative_gate_removes_contradictory_llm_tasks():
    graph = build_graph(FakeMCP(), NoAnalysisModel())
    result = asyncio.run(graph.ainvoke({"query": "빠르게 처리한다.", "thread_id": "test",
        "document_id": "X", "domain": "회원", "scope": "all", "events": [],
        "replan_rounds": 0, "validation_attempts": 0, "writing_attempts": 0,
        "degraded_reasons": []}))
    assert result["tasks"] == []
    assert result["final"]["findings"] == []
    assert any(event["type"] == "GateApplied" and event["removed_tasks"] == 1
               for event in result["events"])


def test_omission_revision_keeps_original_domain_context():
    findings = _omission_findings("로그인에 실패하면 처리를 종료한다.", "# omission")
    assert len(findings) == 1
    assert "로그인" in findings[0]["revision"]
    assert "결제" not in findings[0]["revision"]


def test_conflict_uses_real_ids_and_same_business_target():
    text = ("R-11. 주문 취소는 24시간 이내다. "
            "R-12. 주문 취소는 48시간 이내다. "
            "R-13. 세션 보관은 72시간 이내다.")
    findings = _conflict_findings(text, "# conflict")
    assert len(findings) == 1
    assert findings[0]["related_requirement_ids"] == ["R-11", "R-12"]


def test_unrelated_time_constraints_are_not_a_conflict():
    text = "R-01. 주문 취소는 24시간 이내다. R-02. 세션 보관은 48시간 이내다."
    assert _conflict_findings(text, "# conflict") == []
