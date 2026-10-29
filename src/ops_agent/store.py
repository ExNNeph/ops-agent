"""One SQLite file holds everything: the step log of every run, the agent's memory, and what the risky tools
changed (outbox, refunds, closed tickets). Tickets, orders and the policy are read-only files in data/."""
import json
import os
import sqlite3

DB = os.getenv("OPS_AGENT_DB", "ops-agent.db")

SCHEMA = """
create table if not exists runs (id integer primary key, task text, status text default 'running',
                                 created text default current_timestamp);
create table if not exists steps (run_id integer, n integer, kind text, data text,
                                  created text default current_timestamp, primary key (run_id, n));
create table if not exists memory (id integer primary key, email text, note text, run_id integer,
                                   created text default current_timestamp);
create table if not exists outbox (id integer primary key, ticket_id text, to_addr text, subject text, body text,
                                   run_id integer, created text default current_timestamp);
create table if not exists refunds (id integer primary key, order_id text, amount real, reason text,
                                    ticket_id text, run_id integer, created text default current_timestamp);
create table if not exists closed (ticket_id text primary key, summary text, run_id integer,
                                   created text default current_timestamp);
"""


class Store:
    def __init__(self, path: str = DB):
        self.db = sqlite3.connect(path, check_same_thread=False)  # the demo reuses one store across reruns
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def q(self, sql: str, *args) -> list[dict]:
        return [dict(r) for r in self.db.execute(sql, args)]

    def x(self, sql: str, *args) -> int:
        with self.db:
            return self.db.execute(sql, args).lastrowid

    def new_run(self, task: str) -> int:
        return self.x("insert into runs (task) values (?)", task)

    def run(self, run_id: int) -> dict | None:
        rows = self.q("select * from runs where id = ?", run_id)
        return rows[0] if rows else None

    def set_status(self, run_id: int, status: str) -> str:
        self.x("update runs set status = ? where id = ?", status, run_id)
        return status

    def add_step(self, run_id: int, kind: str, data: dict) -> None:
        n = self.q("select count(*) c from steps where run_id = ?", run_id)[0]["c"]
        self.x("insert into steps (run_id, n, kind, data) values (?, ?, ?, ?)", run_id, n, kind,
               json.dumps(data, ensure_ascii=False))

    def steps(self, run_id: int) -> list[dict]:
        return [{"kind": r["kind"], **json.loads(r["data"])}
                for r in self.q("select kind, data from steps where run_id = ? order by n", run_id)]
