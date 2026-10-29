import builtins

import pytest
from streamlit.testing.v1 import AppTest

from ops_agent import agent, cli
from ops_agent.store import Store
from tests.test_agent import T1001, scripted, turn


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("OPS_AGENT_DB", str(tmp_path / "demo.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return AppTest.from_file(str(cli.DEMO_APP), default_timeout=30)  # the file `ops-agent demo` starts


def test_runs_until_the_checkpoint_then_approve_and_deny(app, monkeypatch):
    responses = scripted(*T1001[:3], turn(("close_ticket", {"ticket_id": "T-1001", "summary": "done"})),
                         turn(text="Refunded 18.90 EUR."))
    monkeypatch.setattr(agent, "claude", lambda key=None: responses)
    at = app.run()
    assert not at.exception
    at.sidebar.text_input[0].input("x").run()
    next(b for b in at.button if b.label.startswith("Run the agent")).click().run()
    assert not at.exception and any("issue_refund" in m.value for m in at.markdown)
    next(b for b in at.button if b.label == "Approve").click().run()  # the refund
    assert any("anna.berg@example.com" in m.value for m in at.markdown)  # now the email waits
    at.text_input(key=next(t.key for t in at.text_input if t.key and t.key.startswith("note-"))).input("too short")
    next(b for b in at.button if b.label == "Deny").click().run()
    next(b for b in at.button if b.label == "Approve").click().run()  # close_ticket
    assert not at.exception and any("done" in m.value for m in at.markdown)
    store = Store()
    tools = [s for s in store.steps(1) if s["kind"] == "tool"]
    assert [s["decision"] for s in tools if s["decision"] != "auto"] == ["approved", "denied", "approved"]
    assert tools[-2]["note"] == "too short"
    assert any("customer has not been sent a reply" in w for w in tools[-1]["warnings"])


def test_replay_tab_without_key(app, monkeypatch):
    store = Store()
    agent.start(store, "Handle ticket T-1001.", scripted(*T1001), lambda p: (True, ""))
    at = app.run()
    next(b for b in at.button if b.label == "Replay offline").click().run()
    assert not at.exception and any("identical" in s.value for s in at.success)


def test_demo_without_streamlit_gives_install_hint(monkeypatch):
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                        lambda n, *a, **k: (_ for _ in ()).throw(ImportError()) if n == "streamlit" else real(n, *a, **k))
    with pytest.raises(SystemExit, match="pip install"):
        cli.demo()
