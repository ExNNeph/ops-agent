"""Local web demo, started with `ops-agent demo`."""
import json
import os
from pathlib import Path

import streamlit as st

from ops_agent import agent
from ops_agent.store import Store
from ops_agent.tools import TICKETS, TOOLS

EXAMPLES = Path(__file__).parents[2] / "examples"  # recorded runs, replayable without a key
STATUS = {"done": "✅ done", "waiting": "⏸ waiting for you", "running": "▶ running", "refusal": "🚫 declined",
          "max_tokens": "✂ cut off", "too many steps": "⛔ too many steps", "log ended": "■ log ended"}

st.set_page_config(page_title="Ops agent demo", page_icon="🫖", layout="wide")
st.title("🫖 Support agent with a human checkpoint")
st.caption("Nordlicht Tee, a small (fictional) tea shop. Claude handles a support ticket with typed tools: it reads "
           "the ticket, looks up orders and the policy, remembers customers in SQLite, and proposes emails, refunds "
           "and ticket closes. **Those three wait for your approval**, with warnings from deterministic checks. "
           "Every step is logged, and any run can be replayed offline from its log.")


@st.cache_resource
def load() -> Store:
    return Store()


store = load()

with st.sidebar:
    api_key = st.text_input("Anthropic API key", type="password",
                            help="Used only for your requests in this local session, never stored.")
    if api_key or os.getenv("ANTHROPIC_API_KEY"):
        st.success(f"🔑 Key set ({'sidebar' if api_key else 'environment'}): Claude ({agent.MODEL}) runs the agent.")
    else:
        st.info("No key: browse the tickets, and replay the recorded runs under **Runs**. Paste a key and press "
                "Enter to run the agent.")
    spent = sum(s["cost"] for r in store.q("select id from runs") for s in store.steps(r["id"]) if s["kind"] == "model")
    st.metric("spent in this database", f"${spent:.3f}")
    st.caption(f"Database: `{Path(store.path).name}` (the CLI uses the same file).")
    if st.button("Reset demo data", help="Deletes all runs, emails, refunds, closed tickets and memory."):
        for table in ("runs", "steps", "memory", "outbox", "refunds", "closed"):
            store.x(f"delete from {table}")
        st.session_state.pop("run_id", None)
        st.rerun()


def show_steps(steps: list[dict]) -> None:
    for s in steps:
        if s["kind"] == "model":
            for b in s["content"]:
                if b["type"] == "thinking" and b.get("thinking"):
                    st.caption(f"💭 {b['thinking']}")
                elif b["type"] == "text" and b["text"].strip():
                    st.markdown(f"🫖 {b['text']}")
        else:
            badge = {"auto": "", "approved": " · ✅ approved", "denied": " · ⛔ denied"}[s["decision"]]
            args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in s["input"].items())
            label = f"🔧 {s['name']}({args[:90] + '…' if len(args) > 90 else args}){badge}"
            with st.expander(("❌ " if s["is_error"] and s["decision"] == "auto" else "") + label):
                for w in s["warnings"]:
                    st.warning(w)
                if s.get("note"):
                    st.markdown(f"**Your note:** {s['note']}")
                st.code(s["result"] if len(s["result"]) < 4000 else s["result"][:4000] + "\n…", language=None,
                        wrap_lines=True)
    model = [s for s in steps if s["kind"] == "model"]
    if model:
        st.caption(f"{len(model)} model calls · ${sum(s['cost'] for s in model):.4f}")


def show_proposal(p: agent.Proposal) -> None:
    i = p.input
    with st.container(border=True):
        st.markdown(f"#### ⏸ Your approval: `{p.tool}`")
        if p.tool == "send_email":
            st.markdown(f"**To:** {i.get('to')}  \n**Subject:** {i.get('subject')}")
            st.text(i.get("body", ""))
        elif p.tool == "issue_refund":
            st.markdown(f"**Refund {float(i.get('amount', 0)):.2f} EUR** on order **{i.get('order_id')}**  \n"
                        f"Reason: {i.get('reason')}")
        else:
            st.markdown(f"**Close {i.get('ticket_id')}**: {i.get('summary')}")
        for w in p.warnings:
            st.warning(w)
        if not p.warnings:
            st.success("The checks found nothing unusual.")
        note = st.text_input("Note for the agent (sent with a denial)", key=f"note-{p.tool_use_id}")
        a, b, _ = st.columns([1, 1, 2])
        decision = (True, "") if a.button("Approve", type="primary", key=f"ok-{p.tool_use_id}") else \
            (False, note) if b.button("Deny", key=f"no-{p.tool_use_id}") else None
    if decision:
        drive(p.run_id, lambda q: decision if q.tool_use_id == p.tool_use_id else None)


def drive(run_id: int, approve) -> None:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        st.error("Paste an API key in the sidebar to continue this run.")
        return
    model = agent.claude(key)
    with st.status("Claude is working…") as box:
        def live(messages):
            box.update(label=f"Claude is working… (model call {sum(m['role'] == 'assistant' for m in messages) + 1})")
            return model(messages)
        try:
            agent.drive(store, run_id, live, approve)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}"[:400])
            return
    st.rerun()


tab_ticket, tab_runs, tab_state = st.tabs(["Ticket", "Runs and replay", "Outbox, refunds, memory"])

if "run_id" not in st.session_state:  # pick up a run left waiting (by the CLI's "later", or a closed tab)
    waiting = store.q("select id, task from runs where status = 'waiting' order by id desc limit 1")
    st.session_state.run_id = waiting[0]["id"] if waiting else None
    if waiting and waiting[0]["task"].split()[-1].rstrip(".") in TICKETS:
        st.session_state.ticket = waiting[0]["task"].split()[-1].rstrip(".")

with tab_ticket:
    closed = {r["ticket_id"] for r in store.q("select ticket_id from closed")}
    left, right = st.columns([2, 3], gap="large")
    with left:
        tid = st.selectbox("Ticket", list(TICKETS), key="ticket", format_func=lambda t: f"{t} · {TICKETS[t]['subject']}"
                           + (" (closed)" if t in closed else ""))
        t = TICKETS[tid]
        with st.container(border=True):
            st.markdown(f"**From:** {t['email']}  \n**Subject:** {t['subject']}  \n**Received:** {t['received']}")
            st.text(t["body"])
        if st.button("Run the agent on this ticket", type="primary"):
            st.session_state.run_id = store.new_run(f"Handle ticket {tid}.")
            drive(st.session_state.run_id, lambda p: None)  # runs until the first checkpoint
    with right:
        run_id = st.session_state.get("run_id")
        if run_id and store.run(run_id):
            run = store.run(run_id)
            st.markdown(f"**Run {run_id}** · {run['task']} · {STATUS.get(run['status'], run['status'])}")
            show_steps(store.steps(run_id))
            p = agent.pending(store, run_id) if run["status"] == "waiting" else None
            if p:
                show_proposal(p)
        else:
            st.info("Pick a ticket and run the agent. Good ones to start with: **T-1001** (a plain refund), "
                    "**T-1011** (the ticket asks to email someone else), **T-1005** (over the manager limit), "
                    "**T-1006** (a prompt injection).")

with tab_runs:
    runs = store.q("select * from runs order by id desc")
    files = sorted(EXAMPLES.glob("*.jsonl")) if EXAMPLES.exists() else []
    options = [f"run {r['id']} · {r['task']} · {r['status']}" for r in runs] + [f"example · {f.stem}" for f in files]
    if not options:
        st.info("No runs yet.")
    else:
        choice = st.selectbox("Run", options)
        if choice.startswith("example"):
            lines = [json.loads(x) for x in files[options.index(choice) - len(runs)].read_text().splitlines() if x]
            task, status, logged = lines[0]["task"], lines[0]["status"], lines[1:]
        else:
            r = runs[options.index(choice)]
            task, status, logged = r["task"], r["status"], store.steps(r["id"])
        st.markdown(f"**{task}** · {STATUS.get(status, status)} · {len(logged)} steps")
        if st.button("Replay offline", help="Drives the loop again from the log: no API call, no tool runs."):
            new, diffs = agent.replay(task, logged, status)
            n = sum(s["kind"] == "model" for s in new)
            if diffs:
                st.error("The current code differs from the log:\n\n" + "\n".join(f"- {d}" for d in diffs))
            else:
                st.success(f"Replayed {n} model calls and {len(new) - n} tool calls from the log, without the API "
                           "and without side effects: identical. The messages sent to the model hash the same, "
                           "and the same calls stopped at the checkpoint.")
        show_steps(logged)

with tab_state:
    for name, sql in (("Outbox", "select ticket_id, to_addr, subject, body, created from outbox order by id desc"),
                      ("Refunds", "select ticket_id, order_id, amount, reason, created from refunds order by id desc"),
                      ("Closed tickets", "select ticket_id, summary, created from closed order by created desc"),
                      ("Memory", "select email, note, created from memory order by id desc")):
        st.subheader(name)
        rows = store.q(sql)
        st.dataframe(rows, hide_index=True, width="stretch") if rows else st.caption("nothing yet")
    with st.expander("Tool schemas Claude sees"):
        st.json([dict(t.schema, risky=t.risky) for t in TOOLS.values()], expanded=False)
