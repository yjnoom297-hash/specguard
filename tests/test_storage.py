from agent.storage import ReviewStore


def test_human_approval_is_persisted(tmp_path):
    store = ReviewStore(tmp_path / "test.db")
    result = {"findings": [{"finding_id": "F-001"}]}
    store.save_review("REV-1", "t", "D", "주문", "요구사항 본문입니다.", result)
    assert store.approved_ids("REV-1") == set()
    store.approve("REV-1", ["F-001"])
    assert store.approved_ids("REV-1") == {"F-001"}
    issue_id = store.save_issue("REV-1", "F-001", result["findings"][0], "SG-001")
    assert issue_id == "SG-001"
    issues = store.list_issues()
    assert issues[0]["issue_id"] == "SG-001"
    assert issues[0]["document_id"] == "D"
    assert issues[0]["finding"] == result["findings"][0]
    assert store.list_issues(document_id="D")[0]["issue_id"] == "SG-001"
    assert store.list_issues(document_id="OTHER") == []
    assert store.list_issue_document_ids() == ["D"]
    store.save_checkpoint("t", "REV-1", {"final": result})
    assert store.get_checkpoint("t")["state"] == {"final": result}
    store.close()
