"""Evaluate the actual SpecGuard graph on 30 reproducible examples."""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from agent.graph import build_graph

NORMAL = [
    "API는 요청 후 2초 이내에 200 응답을 반환한다.",
    "결제 실패 시 주문 상태를 PAYMENT_FAILED로 저장하고 오류 원인을 안내하며 1회 재시도한다.",
    "관리자 역할 사용자만 환불 승인 버튼을 사용할 수 있다.",
    "비밀번호는 8자 이상이며 영문과 숫자를 포함한다.",
    "재고가 0이면 주문 버튼을 비활성화하고 품절 문구를 표시한다.",
    "로그인 5회 실패 시 상태를 LOCKED로 저장하고 원인을 안내하며 1회 재시도한다.",
    "배송 완료 후 7일 이내에만 반품 요청을 허용한다.",
    "파일 크기가 10MB를 넘으면 업로드를 거부한다.",
]
AMBIGUITY = [f"시스템은 {word} 요청을 처리한다." for word in
             ["빠르게", "적절하게", "가능한 경우", "필요한 처리"]] * 2
OMISSION = [f"{target}에 실패하면 처리를 종료한다." for target in
            ["결제", "로그인", "업로드", "환불", "배송 요청", "인증", "주문 저장", "알림 전송"]]
CONFLICT = [f"R-01 주문 취소는 {a}시간 이내다. R-02 주문 취소는 {b}시간 이내다."
            for a, b in [(24, 48), (3, 7), (10, 24), (30, 72), (1, 2), (12, 36)]]
DATASET = ([(text, "normal") for text in NORMAL] +
           [(text, "ambiguity") for text in AMBIGUITY] +
           [(text, "omission") for text in OMISSION] +
           [(text, "conflict") for text in CONFLICT])


class EvaluationMCP:
    async def get_prompt(self, document_id, domain):
        return "원문 근거와 측정 가능한 수정안을 제시하세요."

    async def call_tool(self, name, arguments):
        return {"document_id": arguments["document_id"], "requirements": []}

    async def read_rule(self, rule_type):
        return f"# {rule_type} quality rule"


def scores(expected, predicted):
    labels = sorted(set(expected) | set(predicted))
    per_label = {}
    for label in labels:
        tp = sum(e == label and p == label for e, p in zip(expected, predicted))
        fp = sum(e != label and p == label for e, p in zip(expected, predicted))
        fn = sum(e == label and p != label for e, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": round(precision, 3), "recall": round(recall, 3),
                            "f1": round(f1, 3), "support": expected.count(label)}
    return {"count": len(expected),
            "accuracy": round(sum(e == p for e, p in zip(expected, predicted)) / len(expected), 3),
            "macro_f1": round(sum(item["f1"] for item in per_label.values()) / len(per_label), 3),
            "per_label": per_label}


async def evaluate_graph():
    graph = build_graph(EvaluationMCP())
    expected, predicted = [], []
    counters = defaultdict(int)
    for index, (text, label) in enumerate(DATASET, 1):
        result = await graph.ainvoke({
            "query": text, "thread_id": "evaluation", "document_id": f"E-{index:03d}",
            "domain": "주문·결제", "scope": "all", "events": [], "replan_rounds": 0,
            "validation_attempts": 0, "writing_attempts": 0, "degraded_reasons": [],
        })
        types = [item["type"] for item in result["final"]["findings"]]
        predicted.append(types[0] if len(types) == 1 else "normal" if not types else "multiple")
        expected.append(label)
        counters["validation"] += result.get("validation_attempts", 0)
        counters["replan"] += result.get("replan_rounds", 0)
    result = scores(expected, predicted)
    result["average_validation_attempts"] = round(counters["validation"] / len(DATASET), 3)
    result["average_replan_rounds"] = round(counters["replan"] / len(DATASET), 3)
    return {"system": "specguard-langgraph", "metrics": result}


if __name__ == "__main__":
    result = asyncio.run(evaluate_graph())
    output = Path(__file__).parent / "evaluation_result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
