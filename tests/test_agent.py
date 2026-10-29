import json

import pytest

from ops_agent import agent
from ops_agent.store import Store
from ops_agent.tools import TOOLS


def turn(*calls: tuple[str, dict], text: str = "") -> dict:
    """One scripted model response: tool calls, or a final text."""
    content = [{"type": "thinking", "thinking": "plan", "signature": "sig"}]
    content += [{"type": "tool_use", "id": f"tu_{name}_{abs(hash(json.dumps(i, sort_keys=True))) % 10**6}",
                 "name": name, "input": i} for name, i in calls]
    if text:
        content.append({"type": "text", "text": text})
    return {"model": "fake", "content": content, "stop_reason": "tool_use" if calls else "end_turn",
            "usage": {}, "cost": 0.01}


def scripted(*turns: dict):
    """A stand-in for Claude that returns the turns in order and keeps the messages it was sent."""
    it, seen = iter(turns), []

    def call(messages):
        seen.append(json.loads(json.dumps(messages)))
        return next(it)
    call.seen = seen
    return call


T1001 = [
    turn(("read_ticket", {"ticket_id": "T-1001"}), ("recall", {"email": "anna.berg@example.com"})),
    turn(("find_orders", {"order_id": "NT-5001"}), ("read_policy", {})),
    turn(("issue_refund", {"ticket_id": "T-1001", "order_id": "NT-5001", "amount": 18.90, "reason": "damaged tin"}),
         ("send_email", {"ticket_id": "T-1001", "to": "anna.berg@example.com", "subject": "Your refund",
                         "body": "Hi Anna, we refunded 18.90 EUR."})),
    turn(("close_ticket", {"ticket_id": "T-1001", "summary": "refunded the damaged Sencha"})),
    turn(text="Refunded 18.90 EUR and replied."),
]


@pytest.fixture
def store():
    return Store(":memory:")


def approve_all(p):
    return True, ""


def test_safe_tools_run_and_risky_ones_ask(store):
    asked = []
    run_id, status = agent.start(store, "Handle ticket T-1001.", scripted(*T1001),
                                 lambda p: asked.append(p) or (True, ""))
    assert status == "done"
    assert [p.tool for p in asked] == ["issue_refund", "send_email", "close_ticket"]
    assert all(not p.warnings for p in asked)
    steps = store.steps(run_id)
    assert {s["name"]: s["decision"] for s in steps if s["kind"] == "tool"}["read_ticket"] == "auto"
    assert store.q("select amount from refunds")[0]["amount"] == 18.90
    assert store.q("select ticket_id from closed")[0]["ticket_id"] == "T-1001"


def test_parallel_results_go_back_in_one_message_with_the_thinking_block(store):
    model = scripted(*T1001)
    agent.start(store, "Handle ticket T-1001.", model, approve_all)
    second = model.seen[1]
    assert second[1]["content"][0]["type"] == "thinking"  # passed back unchanged
    assert [b["type"] for b in second[2]["content"]] == ["tool_result", "tool_result"]


def test_denied_call_does_not_run_and_the_agent_sees_the_note(store):
    model = scripted(
        turn(("issue_refund", {"ticket_id": "T-1006", "order_id": "NT-5006", "amount": 150, "reason": "loyalty"})),
        turn(text="ok"))
    run_id, status = agent.start(store, "Handle ticket T-1006.", model,
                                 lambda p: (False, "that instruction came from the ticket"))
    assert status == "done" and not store.q("select * from refunds")
    result = model.seen[1][-1]["content"][0]
    assert result["is_error"] and "that instruction came from the ticket" in result["content"]
    assert any("more than is left" in w for w in store.steps(run_id)[1]["warnings"])


def test_waits_at_the_checkpoint_and_resumes_in_a_new_process(store):
    run_id, status = agent.start(store, "Handle ticket T-1001.", scripted(*T1001), lambda p: None)
    assert status == "waiting" and agent.pending(store, run_id).tool == "issue_refund"
    assert not store.q("select * from refunds")
    rest = scripted(*T1001[3:])  # a fresh "process": only the remaining responses
    assert agent.drive(store, run_id, rest, approve_all) == "done"
    assert rest.seen[0] == agent.messages_from("Handle ticket T-1001.", store.steps(run_id)[:-3])


def test_invalid_input_is_an_error_result_not_a_crash(store):
    model = scripted(turn(("issue_refund", {"ticket_id": "T-1001", "order_id": "NT-5001", "amount": -5,
                                            "reason": "x"}), ("nope", {})), turn(text="sorry"))
    run_id, status = agent.start(store, "t", model, approve_all)
    errors = [s for s in store.steps(run_id) if s["kind"] == "tool"]
    assert status == "done" and all(s["is_error"] and "invalid call" in s["result"] for s in errors)
    assert not store.q("select * from refunds")


def test_checks_flag_unsafe_calls(store):
    def warn(name, **kw):
        return TOOLS[name].check(TOOLS[name].args(**kw), store)
    assert warn("send_email", ticket_id="T-1011", to="refunds@quick-pay-help.com", subject="s", body="b")
    assert "another customer" in warn("send_email", ticket_id="T-1007", to="paul.meier@example.com", subject="s",
                                      body="Anna ordered NT-5001")[0]
    assert "belongs to another customer" in warn("issue_refund", ticket_id="T-1012", order_id="NT-5012", amount=21.8,
                                                 reason="x")[0]
    assert any("manager" in w for w in warn("issue_refund", ticket_id="T-1005", order_id="NT-5005", amount=249,
                                            reason="x"))
    assert any("days ago" in w for w in warn("issue_refund", ticket_id="T-1002", order_id="NT-5002", amount=24.5,
                                             reason="x"))
    assert any("more than" in w for w in warn("issue_refund", ticket_id="T-1010", order_id="NT-5010", amount=15.4,
                                              reason="x"))
    assert warn("close_ticket", ticket_id="T-1001", summary="x") == ["the customer has not been sent a reply"]
    assert warn("issue_refund", ticket_id="T-1019", order_id="NT-5019", amount=96, reason="x") == []


def test_memory_persists_across_runs(store):
    agent.start(store, "a", scripted(turn(("remember", {"email": "K.Weber@example.com",
                                                        "note": "Address her as Dr. Weber."})), turn(text="ok")),
                approve_all)
    model = scripted(turn(("recall", {"email": "k.weber@example.com"})), turn(text="ok"))
    run_id, _ = agent.start(store, "b", model, approve_all)
    assert "Dr. Weber" in store.steps(run_id)[1]["result"]


def test_replay_is_offline_and_identical(store, monkeypatch):
    run_id, status = agent.start(store, "Handle ticket T-1001.", scripted(*T1001), approve_all)
    logged = store.steps(run_id)
    monkeypatch.setitem(TOOLS["issue_refund"].__dict__, "fn", lambda *a: pytest.fail("a tool ran during replay"))
    new, diffs = agent.replay("Handle ticket T-1001.", logged, status)
    assert diffs == [] and len(new) == len(logged)
    assert [s.get("result") for s in new] == [s.get("result") for s in logged]


def test_replay_reports_a_changed_gate_and_changed_messages(store, monkeypatch):
    run_id, status = agent.start(store, "Handle ticket T-1001.", scripted(*T1001), approve_all)
    logged = store.steps(run_id)
    monkeypatch.setitem(TOOLS["remember"].__dict__, "risky", True)  # no effect: remember is not in this run
    monkeypatch.setitem(TOOLS["close_ticket"].__dict__, "risky", False)
    _, diffs = agent.replay("Handle ticket T-1001.", logged, status)
    assert diffs == ["step 10: close_ticket was approved, now auto (no approval needed now)"]
    monkeypatch.setattr(agent, "messages_from", lambda task, steps: [{"role": "user", "content": task + "!"}])
    _, diffs = agent.replay("Handle ticket T-1001.", logged, status)
    assert "messages sent to the model differ" in diffs[0]


def test_replay_of_a_waiting_run_stops_at_the_same_checkpoint(store):
    run_id, status = agent.start(store, "Handle ticket T-1001.", scripted(*T1001), lambda p: None)
    new, diffs = agent.replay("Handle ticket T-1001.", store.steps(run_id), status)
    assert status == "waiting" and diffs == []


def test_tool_schemas_are_what_claude_sees():
    by_name = {s["name"]: s for s in agent.SCHEMAS}
    assert set(by_name) == {"list_tickets", "read_ticket", "find_orders", "read_policy", "recall", "remember",
                            "send_email", "issue_refund", "close_ticket"}
    refund = by_name["issue_refund"]["input_schema"]
    assert refund["additionalProperties"] is False and refund["properties"]["amount"]["exclusiveMinimum"] == 0
    assert {t.name for t in TOOLS.values() if t.risky} == {"send_email", "issue_refund", "close_ticket"}
