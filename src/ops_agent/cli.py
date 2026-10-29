import argparse
import json
import subprocess
import sys
from pathlib import Path

DEMO_APP = Path(__file__).with_name("demo_app.py")


def demo() -> None:
    """Start the local web demo on http://localhost:8501 and open the browser."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit('The demo needs Streamlit: pip install -e ".[demo]"')
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(DEMO_APP), "--server.address", "localhost",
                    "--server.port", "8501", "--browser.gatherUsageStats", "false",
                    "--client.toolbarMode", "minimal",  # hides Streamlit's "Deploy" button
                    "--server.fileWatcherType", "none"])  # no code reloading needed, and no "install Watchdog" hint


def task_for(arg: str) -> str:
    return f"Handle ticket {arg}." if arg.upper().startswith("T-") and " " not in arg else arg


def short(v, n: int = 160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def show_step(i: int, s: dict) -> None:
    if s["kind"] == "model":
        print(f"{i:>3} model  {s['stop_reason']}, ${s['cost']:.4f}")
        for b in s["content"]:
            if b["type"] == "thinking" and b.get("thinking"):
                print(f"      thinks: {short(b['thinking'])}")
            elif b["type"] == "text" and b["text"].strip():
                print(f"      says:   {short(b['text'], 400)}")
            elif b["type"] == "tool_use":
                print(f"      calls:  {b['name']}({short(b['input'])})")
    else:
        gate = "" if s["decision"] == "auto" else f" [{s['decision'].upper()} by human" + (
            f": {s['note']}" if s.get("note") else "") + "]"
        print(f"{i:>3} tool   {s['name']}{gate}" + ("  ERROR" if s["is_error"] else ""))
        for w in s["warnings"]:
            print(f"      ⚠ {w}")
        print(f"      -> {short(s['result'])}")


def ask(p) -> tuple[bool, str] | None:
    """The checkpoint in the terminal."""
    print(f"\n⏸  {p.tool} needs your approval:")
    for k, v in p.input.items():
        print(f"   {k}: " + str(v).replace("\n", "\n      "))
    for w in p.warnings:
        print(f"   ⚠ {w}")
    while True:
        a = input("[a]pprove, [d]eny, or [l]ater (the run waits; continue with `ops-agent resume`): ").strip().lower()
        if a[:1] == "a":
            return True, ""
        if a[:1] == "d":
            return False, input("note for the agent (optional): ").strip()
        if a[:1] == "l":
            return None


def report(store, run_id: int, status: str) -> int:
    steps = store.steps(run_id)
    for i, s in enumerate(steps):
        show_step(i, s)
    cost = sum(s["cost"] for s in steps if s["kind"] == "model")
    print(f"\nrun {run_id}: {status}, {sum(s['kind'] == 'model' for s in steps)} model calls, ${cost:.4f}")
    return 0 if status in ("done", "waiting") else 1


def load_log(ref: str, store) -> tuple[str, str, list[dict]]:
    if ref.isdigit():
        run = store.run(int(ref))
        if not run:
            raise SystemExit(f"no run {ref}")
        return run["task"], run["status"], store.steps(int(ref))
    lines = [json.loads(line) for line in Path(ref).read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[0]["task"], lines[0]["status"], lines[1:]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="ops-agent", description="A support agent for a small tea shop: typed tools, "
                                "SQLite memory, a human checkpoint before risky actions, and a replayable step log.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tickets", help="list the tickets and their status")
    sub.add_parser("run", help="handle a ticket (T-1001) or any task in quotes; needs ANTHROPIC_API_KEY").add_argument(
        "task")
    sub.add_parser("resume", help="continue a run that waits for your approval").add_argument("run_id", type=int)
    sub.add_parser("runs", help="list the runs in the database")
    sub.add_parser("show", help="print a run's step log").add_argument("run_id", type=int)
    sub.add_parser("replay", help="re-drive a run from its log, offline, and report differences").add_argument(
        "run", help="a run id, or a .jsonl file from `export`")
    sub.add_parser("export", help="write a run's log as JSON lines to stdout").add_argument("run_id", type=int)
    sub.add_parser("outbox", help="emails sent, refunds paid, tickets closed, and the agent's memory")
    sub.add_parser("demo", help="start the local web demo on http://localhost:8501")
    args = p.parse_args(argv)

    if args.cmd == "demo":
        return demo()
    from . import agent
    from .store import Store
    from .tools import TICKETS
    store = Store()
    if args.cmd == "tickets":
        closed = {r["ticket_id"] for r in store.q("select ticket_id from closed")}
        for t in TICKETS.values():
            print(f"{t['id']}  {'closed' if t['id'] in closed else 'open  '}  {t['email']:<28} {t['subject']}")
    elif args.cmd == "runs":
        for r in store.q("select r.id, r.task, r.status, count(s.n) n from runs r left join steps s "
                         "on s.run_id = r.id group by r.id order by r.id"):
            cost = sum(s["cost"] for s in store.steps(r["id"]) if s["kind"] == "model")
            print(f"{r['id']:>4}  {r['status']:<9} {r['n']:>3} steps  ${cost:.4f}  {r['task']}")
    elif args.cmd == "show":
        run = store.run(args.run_id) or sys.exit(f"no run {args.run_id}")
        print(f"task: {run['task']}\n")
        return report(store, args.run_id, run["status"])
    elif args.cmd == "export":
        run = store.run(args.run_id) or sys.exit(f"no run {args.run_id}")
        print(json.dumps({"task": run["task"], "status": run["status"]}, ensure_ascii=False))
        for s in store.steps(args.run_id):
            print(json.dumps(s, ensure_ascii=False))
    elif args.cmd == "replay":
        task, status, logged = load_log(args.run, store)
        new, diffs = agent.replay(task, logged, status)
        print(f"task: {task}\n")
        for i, s in enumerate(new):
            show_step(i, s)
        n = sum(s["kind"] == "model" for s in new)
        print(f"\nreplayed offline: {n} model calls and {len(new) - n} tool calls from the log, no API, no side effects")
        print("identical to the log" if not diffs else "differences:\n  " + "\n  ".join(diffs))
        return 1 if diffs else 0
    elif args.cmd == "outbox":
        for name, sql in (("emails", "select ticket_id, to_addr, subject from outbox"),
                          ("refunds", "select ticket_id, order_id, amount from refunds"),
                          ("closed", "select ticket_id, summary from closed"),
                          ("memory", "select email, note from memory")):
            print(f"\n{name}:")
            for r in store.q(sql):
                print("  " + "  ".join(str(v) for v in r.values()))
    else:  # run, resume
        try:
            model = agent.claude()
            if args.cmd == "run":
                run_id, status = agent.start(store, task_for(args.task), model, ask)
            else:
                run_id = args.run_id
                status = agent.drive(store, run_id, model, ask)
        except Exception as e:
            if type(e).__name__ in ("AuthenticationError", "APIConnectionError") or "api_key" in str(e):
                raise SystemExit(f"{type(e).__name__}: is ANTHROPIC_API_KEY set and valid?")
            raise
        print()
        return report(store, run_id, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
