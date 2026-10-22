"""FastAPI entry point for the SpecGuard web MVP."""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.runtime import TieredModelClient
from agent.storage import ReviewStore

ROOT = Path(__file__).parent


class ReviewRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    domain: str = Field(default="주문·결제", min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=20000)
    scope: str = Field(default="all", pattern="^(all|ambiguity|omission|conflict)$")
    thread_id: str = Field(default="default", min_length=1, max_length=100)


class ApprovalRequest(BaseModel):
    finding_ids: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    mcp = MCPClient()
    await mcp.start()
    app.state.mcp = mcp
    app.state.store = ReviewStore()
    app.state.models = TieredModelClient()
    app.state.graph = build_graph(mcp, app.state.models)
    yield
    app.state.store.close()
    await mcp.stop()


app = FastAPI(title="SpecGuard", version="0.1.0", lifespan=lifespan)


def initial_state(req: ReviewRequest) -> dict:
    return {"query": req.text, "thread_id": req.thread_id,
            "document_id": req.document_id, "domain": req.domain,
            "scope": req.scope, "events": [], "replan_rounds": 0,
            "validation_attempts": 0, "writing_attempts": 0,
            "degraded_reasons": []}


@app.get("/")
async def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "specguard", "mcp": "connected", "version": "v2"}


@app.get("/version")
async def version():
    return {"supervisor_version": "v2", "service_type": "pc-web"}


@app.post("/reviews")
async def create_review(req: ReviewRequest):
    review_id = f"REV-{uuid4().hex[:8].upper()}"
    result = await app.state.graph.ainvoke(initial_state(req))
    final = result["final"]
    app.state.store.save_review(review_id, req.thread_id, req.document_id,
                                req.domain, req.text, final)
    app.state.store.save_checkpoint(req.thread_id, review_id, result)
    return {"review_id": review_id, "review": final,
            "events": [*result.get("events", []), *result.get("worker_events", [])],
            "metadata": {"route": result.get("route"),
                         "validation_attempts": result.get("validation_attempts", 0),
                         "replan_rounds": result.get("replan_rounds", 0)}}


@app.get("/reviews/stream")
async def review_stream(document_id: str, domain: str, text: str,
                        scope: str = "all", thread_id: str = "default"):
    try:
        req = ReviewRequest(document_id=document_id, domain=domain, text=text,
                            scope=scope, thread_id=thread_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    review_id = f"REV-{uuid4().hex[:8].upper()}"

    async def generate():
        seen = 0
        final = None
        try:
            async for update in app.state.graph.astream(initial_state(req), stream_mode="updates"):
                for node, delta in update.items():
                    events = delta.get("events") or []
                    for event in events[seen:]:
                        yield f"data: {json.dumps({**event, 'node': node}, ensure_ascii=False)}\n\n"
                    seen = max(seen, len(events))
                    for event in delta.get("worker_events") or []:
                        yield f"data: {json.dumps({**event, 'node': node}, ensure_ascii=False)}\n\n"
                    if delta.get("final"):
                        final = delta["final"]
            if final is None:
                raise RuntimeError("final result was not produced")
            app.state.store.save_review(review_id, req.thread_id, req.document_id,
                                        req.domain, req.text, final)
            app.state.store.save_checkpoint(req.thread_id, review_id, {"final": final})
            payload = {"type": "FinalAnswer", "review_id": review_id, "review": final}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'Error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/reviews/{review_id}")
async def get_review(review_id: str):
    review = app.state.store.get_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
    return review


@app.get("/threads/{thread_id}/checkpoint")
async def get_thread_checkpoint(thread_id: str):
    checkpoint = app.state.store.get_checkpoint(thread_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="thread checkpoint not found")
    return checkpoint


@app.get("/issues")
async def list_issues(limit: int = 50, document_id: str | None = None):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    normalized = document_id.strip() if document_id else None
    return {"document_id": normalized,
            "document_ids": app.state.store.list_issue_document_ids(),
            "issues": app.state.store.list_issues(limit, normalized)}


@app.post("/reviews/{review_id}/approve")
async def approve_review(review_id: str, req: ApprovalRequest):
    review = app.state.store.get_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
    valid_ids = {item["finding_id"] for item in review["result"]["findings"]}
    invalid = set(req.finding_ids) - valid_ids
    if invalid:
        raise HTTPException(status_code=400, detail=f"unknown finding ids: {sorted(invalid)}")
    app.state.store.approve(review_id, req.finding_ids)
    return {"review_id": review_id, "approved_finding_ids": req.finding_ids}


@app.post("/reviews/{review_id}/issues")
async def create_issues(review_id: str, req: ApprovalRequest):
    review = app.state.store.get_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
    approved = app.state.store.approved_ids(review_id)
    requested = set(req.finding_ids)
    if not requested or not requested.issubset(approved):
        raise HTTPException(status_code=409, detail="only approved findings can become issues")
    by_id = {item["finding_id"]: item for item in review["result"]["findings"]}
    issues = []
    for finding_id in req.finding_ids:
        finding = by_id.get(finding_id)
        if finding is None:
            raise HTTPException(status_code=400, detail=f"unknown finding id: {finding_id}")
        created = await app.state.mcp.call_tool(
            "create_review_issue", {"review_id": review_id, "finding": finding})
        issue_id = app.state.store.save_issue(review_id, finding_id, finding, created["issue_id"])
        issues.append({"finding_id": finding_id, "issue_id": issue_id})
    return {"review_id": review_id, "issues": issues}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8600)
