"""LangGraph workflow for evidence-backed requirement review."""
import asyncio
import operator
import re
from functools import partial
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from agent.config import prompt, review_rules
from agent.runtime import TieredModelClient, ToolError, with_retry

MAX_REPLAN_ROUNDS = 1
MAX_WRITING_ATTEMPTS = 2
MAX_VALIDATION_ROUNDS = 3


class ReviewState(TypedDict, total=False):
    query: str
    thread_id: str
    document_id: str
    domain: str
    scope: str
    route: str
    route_reasoning: str
    needs_analysis: bool
    tasks: list[dict]
    context: dict
    rules: dict[str, str]
    worker_findings: Annotated[list[dict], operator.add]
    worker_events: Annotated[list[dict], operator.add]
    findings: list[dict]
    replan_rounds: int
    replan_decision: dict
    degraded_reasons: list[str]
    writing_attempts: int
    validation_attempts: int
    validation_feedback: str
    final: dict
    events: list[dict]


class WorkerState(TypedDict):
    query: str
    task: dict
    rule: str
    worker_findings: Annotated[list[dict], operator.add]
    worker_events: Annotated[list[dict], operator.add]


def _events(state: ReviewState, event: dict) -> list[dict]:
    return [*state.get("events", []), event]


async def route_node(state: ReviewState, models: TieredModelClient) -> dict:
    text = state.get("query", "").strip()
    decision = await models.json(
        tier="fast",
        system=prompt("routing"),
        user=text,
    )
    if decision and decision.value.get("route") in {"review", "chat"}:
        route = decision.value["route"]
        reasoning = str(decision.value.get("reasoning", "LLM 라우팅"))
        model = decision.model
    else:
        review_cues = ("시스템", "API", "요구", "해야", "이내", "실패", "처리", "허용", "표시")
        chat_cues = ("안녕", "고마워", "반가워")
        is_chat = any(cue in text for cue in chat_cues) and not any(cue in text for cue in review_cues)
        route = "chat" if is_chat else "review"
        reasoning = "요구사항 의미 단서 기반 로컬 라우팅"
        model = "deterministic-fallback"
    return {"route": route, "route_reasoning": reasoning,
            "events": _events(state, {"type": "RouteDecided", "route": route,
                                        "reasoning": reasoning, "model": model})}


async def chat_node(state: ReviewState) -> dict:
    final = {"summary": "검토할 요구사항 문장을 입력해 주세요.", "findings": []}
    return {"final": final, "events": _events(state, {"type": "Finalized", "count": 0})}


async def plan_node(state: ReviewState, mcp, models: TieredModelClient) -> dict:
    degraded = list(state.get("degraded_reasons", []))
    try:
        procedure = await with_retry(
            lambda: mcp.get_prompt(state["document_id"], state["domain"]),
            tool_name="MCP review prompt",
        )
    except ToolError as exc:
        procedure = "원문을 인용하고 측정 가능한 수정안과 테스트를 작성합니다."
        degraded.append(exc.user_message)
    selected = ["ambiguity", "omission", "conflict"]
    if state.get("scope") not in (None, "all"):
        selected = [state["scope"]]
    decision = await models.json(
        tier="fast",
        system=prompt("planning"),
        user=state["query"],
    )
    needs_analysis = True
    raw_kinds = selected
    model = "deterministic-fallback"
    if decision:
        needs_analysis = bool(decision.value.get("needs_analysis", True))
        raw_kinds = [str(t.get("kind")) for t in decision.value.get("tasks", []) if isinstance(t, dict)]
        raw_kinds = raw_kinds or selected
        model = decision.model
    tasks = [{"id": f"task-{index}-{kind}", "kind": kind, "status": "pending"}
             for index, kind in enumerate(raw_kinds, 1)]
    return {"tasks": tasks, "needs_analysis": needs_analysis,
            "degraded_reasons": degraded,
            "context": {"procedure": procedure},
            "events": _events(state, {"type": "PlanCreated", "tasks": raw_kinds,
                                        "needs_analysis": needs_analysis, "model": model})}


async def gate_node(state: ReviewState) -> dict:
    """Code-level authority gate: only known, requested worker kinds may execute."""
    allowed = set(review_rules()["allowed_types"])
    if state.get("scope") not in (None, "all"):
        allowed &= {state["scope"]}
    tasks = [task for task in state.get("tasks", []) if task.get("kind") in allowed]
    if not state.get("needs_analysis", False):
        tasks = []
    removed = len(state.get("tasks", [])) - len(tasks)
    return {"tasks": tasks,
            "events": _events(state, {"type": "GateApplied", "removed_tasks": removed})}


def _after_gate(state: ReviewState) -> str:
    return "research" if state.get("tasks") else "finalize"


async def research_node(state: ReviewState, mcp) -> dict:
    degraded = list(state.get("degraded_reasons", []))
    try:
        context, rules = await asyncio.gather(
            with_retry(lambda: mcp.call_tool("get_requirement_context", {
                "document_id": state["document_id"]}), tool_name="MCP requirement context"),
            with_retry(lambda: _load_rules(mcp, [task["kind"] for task in state["tasks"]]),
                       tool_name="MCP quality rules"),
        )
    except ToolError as exc:
        context = {"document_id": state["document_id"], "requirements": []}
        rules = {task["kind"]: f"# {task['kind']} fallback rule" for task in state["tasks"]}
        degraded.append(exc.user_message)
    return {"context": {**state.get("context", {}), "document": context}, "rules": rules,
            "degraded_reasons": degraded,
            "events": _events(state, {"type": "MCPContextLoaded",
                                      "rule_types": list(rules),
                                      "related_count": len(context.get("requirements", [])),
                                      "degraded": bool(degraded)})}


async def _load_rules(mcp, kinds: list[str]) -> dict[str, str]:
    values = await asyncio.gather(*(mcp.read_rule(kind) for kind in kinds))
    return dict(zip(kinds, values))


async def replan_check_node(state: ReviewState) -> dict:
    expected = {task["kind"] for task in state.get("tasks", [])}
    missing = sorted(expected - set(state.get("rules", {})))
    enough = not missing
    decision = {"is_sufficient": enough, "missing_rule_types": missing,
                "reasoning": "모든 작업의 판정 규칙이 준비됨" if enough else "판정 규칙이 누락됨"}
    return {"replan_decision": decision,
            "events": _events(state, {"type": "ContextChecked", **decision,
                                        "round": state.get("replan_rounds", 0)})}


async def replan_node(state: ReviewState) -> dict:
    rounds = state.get("replan_rounds", 0) + 1
    missing = state.get("replan_decision", {}).get("missing_rule_types", [])
    tasks = list(state.get("tasks", []))
    existing = {task["kind"] for task in tasks}
    for kind in missing:
        if kind not in existing:
            tasks.append({"id": f"task-gap-{rounds}-{kind}", "kind": kind, "status": "pending"})
    return {"tasks": tasks, "replan_rounds": rounds,
            "events": _events(state, {"type": "PlanRevised", "round": rounds,
                                        "gap_tasks": missing})}


def _after_replan(state: ReviewState) -> str:
    if state.get("replan_decision", {}).get("is_sufficient", False):
        return "dispatch_workers"
    if state.get("replan_rounds", 0) >= MAX_REPLAN_ROUNDS:
        return "dispatch_workers"
    return "replan"


async def dispatch_workers_node(state: ReviewState) -> dict:
    kinds = [task["kind"] for task in state.get("tasks", [])]
    return {"worker_findings": [], "worker_events": [],
            "events": _events(state, {"type": "WorkersDispatched", "workers": kinds})}


def dispatch_workers(state: ReviewState) -> list[Send]:
    """LangGraph map step: create one real graph execution per specialist worker."""
    return [
        Send("review_worker", {
            "query": state["query"],
            "task": task,
            "rule": state.get("rules", {}).get(task["kind"], ""),
            "worker_findings": [],
            "worker_events": [],
        })
        for task in state.get("tasks", [])
    ]


async def review_worker_node(state: WorkerState) -> dict:
    """LangGraph map worker. Results are merged by Annotated reducers."""
    await asyncio.sleep(0)
    kind = state["task"]["kind"]
    if kind == "ambiguity":
        findings = _ambiguity_findings(state["query"], state["rule"])
    elif kind == "omission":
        findings = _omission_findings(state["query"], state["rule"])
    else:
        findings = _conflict_findings(state["query"], state["rule"])
    return {
        "worker_findings": findings,
        "worker_events": [{"type": "WorkerCompleted", "worker": kind,
                           "finding_count": len(findings)}],
    }


def _finding(kind: str, evidence: str, problem: str, revision: str,
             given: str, when: str, then: str, rule: str,
             related_ids: list[str] | None = None) -> dict:
    return {
        "finding_id": "",
        "type": kind,
        "severity": review_rules()["severity"][kind],
        "evidence": evidence,
        "problem": problem,
        "rule": rule.splitlines()[0].lstrip("# ") if rule else f"{kind} rule",
        "related_requirement_ids": related_ids or [],
        "revision": revision,
        "test": {"given": given, "when": when, "then": then},
    }


def _ambiguity_findings(text: str, rule: str) -> list[dict]:
    findings = []
    policy = review_rules()["ambiguity"]
    for word in policy["expressions"]:
        if word in text:
            replacement = policy["replacement"].get(word, "명시된 조건과 성공 기준에 따라")
            findings.append(_finding(
                "ambiguity", word, f"'{word}'의 측정 기준이 정의되지 않았습니다.",
                text.replace(word, replacement),
                "정상 입력과 처리 가능한 시스템 상태가 주어지고", "사용자가 기능을 실행하면",
                "정의된 시간과 결과 기준을 만족해야 합니다.", rule))
    return findings


def _omission_findings(text: str, rule: str) -> list[dict]:
    policy = review_rules()["omission"]
    findings = []
    for req in _parse_requirements(text):
        body = req["body"]
        if not any(term in body for term in policy["failure_terms"]):
            continue
        missing = [name for name, words in policy["required_dimensions"].items()
                   if not any(word in body for word in words)]
        if not missing:
            continue
        target = _failure_target(body)
        labels = {"state": "실패 상태", "guidance": "사용자 안내", "retry": "재시도 정책"}
        findings.append(_finding(
            "omission", body, f"{target} 실패 후 " + ", ".join(labels[item] for item in missing) + "이(가) 누락되었습니다.",
            f"{target} 실패 시 실패 상태를 저장하고 원인을 안내하며 정책에서 허용한 횟수만큼 재시도할 수 있게 한다.",
            f"{target} 처리 중인 상태가 주어지고", f"{target} 처리가 실패하면",
            "실패 상태·원인·재시도 정책이 정의된 결과로 표시되어야 합니다.", rule, [req["id"]]))
    return findings


def _conflict_findings(text: str, rule: str) -> list[dict]:
    constraints = [_constraint(req) for req in _parse_requirements(text)]
    constraints = [item for item in constraints if item]
    findings = []
    for index, left in enumerate(constraints):
        for right in constraints[index + 1:]:
            same_target = (left["subject"], left["action"], left["unit"]) == (right["subject"], right["action"], right["unit"])
            if not same_target or left["value"] == right["value"]:
                continue
            target = " ".join(part for part in (left["subject"], left["action"]) if part)
            findings.append(_finding(
                "conflict", left["body"],
                f"{target} 기준이 {left['value']}{left['unit']}과 {right['value']}{right['unit']}로 서로 다르게 정의되었습니다.",
                f"{target} 기준을 정책 담당자가 확정한 하나의 값으로 통일한다.",
                f"{target} 대상이 주어지고", "정책 경계값에서 기능을 요청하면",
                "확정된 단일 정책에 따라 결과가 결정되어야 합니다.", rule,
                [left["id"], right["id"]]))
    return findings


def _parse_requirements(text: str) -> list[dict[str, str]]:
    matches = list(re.finditer(r"(?m)(R-\d+)\s*[.:]?\s*", text))
    if matches:
        return [{"id": match.group(1),
                 "body": text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)].strip(" .\n")}
                for index, match in enumerate(matches)]
    bodies = [part.strip() for part in re.split(r"[.\n]+", text) if part.strip()]
    return [{"id": f"AUTO-{index:03d}", "body": body} for index, body in enumerate(bodies, 1)]


def _failure_target(body: str) -> str:
    match = re.search(r"([A-Za-z0-9가-힣· _-]+?)(?:에|가|이)?\s*(?:실패|오류|예외)", body)
    return match.group(1).strip() if match and match.group(1).strip() else "해당 기능"


def _constraint(req: dict[str, str]) -> dict | None:
    policy = review_rules()["conflict"]
    units = "|".join(re.escape(unit) for unit in policy["units"])
    match = re.search(rf"(\d+(?:\.\d+)?)\s*({units})\s*(?:이내|이상|이하|미만|초과)?", req["body"])
    if not match:
        return None
    subject = next((word for word in policy["subjects"] if word in req["body"]), "")
    action = next((word for word in policy["actions"] if word in req["body"]), "")
    if not subject and action:
        subject = action
    return {**req, "subject": subject, "action": action,
            "value": float(match.group(1)), "unit": match.group(2)}


async def write_node(state: ReviewState) -> dict:
    findings = []
    seen = set()
    for finding in state.get("worker_findings", []):
        key = (finding["type"], finding["evidence"])
        if key in seen:
            continue
        seen.add(key)
        item = dict(finding)
        item["finding_id"] = f"F-{len(findings) + 1:03d}"
        findings.append(item)
    events = list(state.get("events", []))
    events.append({"type": "DraftCreated", "finding_count": len(findings)})
    return {"findings": findings, "writing_attempts": state.get("writing_attempts", 0) + 1,
            "events": events}


async def validate_node(state: ReviewState, models: TieredModelClient) -> dict:
    required = {"finding_id", "type", "severity", "evidence", "problem", "rule", "revision", "test"}
    errors = []
    for finding in state.get("findings", []):
        missing = required - finding.keys()
        if missing or set(finding.get("test", {})) != {"given", "when", "then"}:
            errors.append(f"{finding.get('finding_id', '?')}: {sorted(missing)}")
            continue
        if finding["evidence"] not in state.get("query", ""):
            errors.append(f"{finding['finding_id']}: 증거가 원문에 없음")
        if finding["revision"].strip() == state.get("query", "").strip():
            errors.append(f"{finding['finding_id']}: 수정안이 원문과 같음")
        if not all(str(value).strip() for value in finding["test"].values()):
            errors.append(f"{finding['finding_id']}: Given/When/Then이 비어 있음")
    model_name = "semantic-rules"
    if not errors and state.get("findings"):
        decision = await models.json(
            tier="quality",
            system=prompt("validation"),
            user=str({"requirement": state.get("query"), "findings": state.get("findings")}),
        )
        if decision:
            model_name = decision.model
            if not bool(decision.value.get("is_acceptable", True)):
                errors.append(str(decision.value.get("feedback", "의미 검증 실패")))
    attempts = state.get("validation_attempts", 0) + 1
    passed = not errors
    return {"validation_attempts": attempts, "validation_feedback": "; ".join(errors),
            "events": _events(state, {"type": "ValidationPassed" if passed else "ValidationFailed",
                                      "attempt": attempts, "errors": errors,
                                      "model": model_name})}


async def rewrite_node(state: ReviewState) -> dict:
    """Apply validator feedback instead of blindly producing the same draft."""
    source = state.get("query", "")
    repaired = []
    for finding in state.get("findings", []):
        if finding.get("evidence") not in source:
            continue
        item = dict(finding)
        item.setdefault("related_requirement_ids", [])
        repaired.append(item)
    return {"findings": repaired,
            "events": _events(state, {"type": "DraftRewritten",
                                        "feedback": state.get("validation_feedback", "")})}


def _after_validate(state: ReviewState) -> str:
    if (state.get("validation_feedback")
            and state.get("validation_attempts", 0) < MAX_VALIDATION_ROUNDS
            and state.get("writing_attempts", 0) < MAX_WRITING_ATTEMPTS):
        return "rewrite"
    return "finalize"


async def finalize_node(state: ReviewState) -> dict:
    findings = state.get("findings", [])
    final = {
        "summary": f"총 {len(findings)}건의 요구사항 결함을 찾았습니다.",
        "document_id": state["document_id"],
        "domain": state["domain"],
        "findings": findings,
        "metrics": {"replan_rounds": state.get("replan_rounds", 0),
                    "validation_attempts": state.get("validation_attempts", 0),
                    "writing_attempts": state.get("writing_attempts", 0)},
        "quality_warning": state.get("validation_feedback", "") or None,
        "degraded_reasons": state.get("degraded_reasons", []),
    }
    return {"final": final,
            "events": _events(state, {"type": "Finalized", "finding_count": len(findings)})}


def build_graph(mcp, models: TieredModelClient | None = None):
    models = models or TieredModelClient()
    graph = StateGraph(ReviewState)
    graph.add_node("route", partial(route_node, models=models))
    graph.add_node("chat", chat_node)
    graph.add_node("plan", partial(plan_node, mcp=mcp, models=models))
    graph.add_node("gate", gate_node)
    graph.add_node("research", partial(research_node, mcp=mcp))
    graph.add_node("replan_check", replan_check_node)
    graph.add_node("replan", replan_node)
    graph.add_node("dispatch_workers", dispatch_workers_node)
    graph.add_node("review_worker", review_worker_node)
    graph.add_node("write", write_node)
    graph.add_node("validate", partial(validate_node, models=models))
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("route")
    graph.add_conditional_edges("route", lambda s: s["route"], {"chat": "chat", "review": "plan"})
    graph.add_edge("chat", END)
    graph.add_edge("plan", "gate")
    graph.add_conditional_edges("gate", _after_gate,
                                {"research": "research", "finalize": "finalize"})
    graph.add_edge("research", "replan_check")
    graph.add_conditional_edges("replan_check", _after_replan,
                                {"replan": "replan", "dispatch_workers": "dispatch_workers"})
    graph.add_edge("replan", "research")
    graph.add_conditional_edges("dispatch_workers", dispatch_workers, ["review_worker"])
    graph.add_edge("review_worker", "write")
    graph.add_edge("write", "validate")
    graph.add_conditional_edges("validate", _after_validate,
                                {"rewrite": "rewrite", "finalize": "finalize"})
    graph.add_edge("rewrite", "validate")
    graph.add_edge("finalize", END)
    return graph.compile()
