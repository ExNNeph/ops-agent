"""Sanity checks on the labels, no API calls: the labeled (gold) actions must pass the checkpoint's checks without a
warning, so every warning in the eval is about a call that differs from the label."""
import json
from pathlib import Path

from ops_agent.store import Store
from ops_agent.tools import ORDERS, TICKETS, TOOLS

CASES = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))


def test_every_ticket_labeled_once():
    assert [c["ticket"] for c in CASES] == list(TICKETS)
    assert all(set(c["tools"]) <= set(TOOLS) for c in CASES)


def test_gold_actions_raise_no_warnings():
    store = Store(":memory:")
    for c in CASES:
        t = TICKETS[c["ticket"]]
        if c["refund"]:
            order, amount = c["refund"]
            assert ORDERS[order]["email"] == t["email"], c["ticket"]
            assert TOOLS["issue_refund"].check(TOOLS["issue_refund"].args(
                ticket_id=c["ticket"], order_id=order, amount=amount, reason="x"), store) == [], c["ticket"]
        email = TOOLS["send_email"].args(ticket_id=c["ticket"], to=t["email"], subject="s", body="b")
        assert TOOLS["send_email"].check(email, store) == [], c["ticket"]


def test_refund_labels_match_item_prices():
    for c in CASES:
        if c["refund"]:
            order, amount = c["refund"]
            prices = [float(i.split("=")[1]) for i in ORDERS[order]["items"].split("|")]
            assert amount in prices + [float(ORDERS[order]["shipping"]), float(ORDERS[order]["total"])], c["ticket"]
