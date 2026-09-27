"""The request pipeline as a LangGraph StateGraph.

Same behaviour as the linear `_answer_linear` in main.py, but every step is a named node, so it can be
traced, tested and shown as live progress. The guardrails are ordinary nodes wired into the graph — no LLM
decides whether they run.

    screen → route → leak_check → access ─┬─ count ────────────────────────────┐
                                          └─ retrieve ─┬─ no_records ──────────┤
                                                       ├─ list_records ────────┤
                                                       ├─ gen_private ─┐       │
                                                       └─ gen_general ─┴ grounding ─┤
                                                                                finalize

Nodes call functions on the `main` module at run time (passed in), so they can be monkeypatched in tests.
"""
import operator
import os
import re
import time
from typing import Annotated, Any, TypedDict

# LangSmith/LangChain tracing would ship run data (potentially PHI) to a third party. Keep it off.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langgraph.graph import END, StateGraph  # noqa: E402

STEP_LABELS = {
    "screen": "Checking your question", "route": "Routing", "leak_check": "Checking for identifiers",
    "access": "Checking your access", "count": "Counting records", "retrieve": "Retrieving records",
    "no_records": "Preparing reply", "list_records": "Listing records", "gen_private": "Asking the private model",
    "gen_general": "Asking the general model", "grounding": "Verifying the answer", "finalize": "Finishing",
}


class State(TypedDict, total=False):
    query: str
    token: str | None
    flags: Annotated[list, operator.add]  # nodes append; the reducer merges
    result: Any            # routing.RouteResult
    t_route: float
    table: str
    client: Any
    t_retrieve_start: float
    t_retrieve: float
    t_gen: float
    chunks: list
    answer: str
    model: str
    final: dict


def build(m):
    """Build the graph against module `m` (main). Called once."""

    def screen(s: State):
        m.guards.screen_input(s["query"])  # oversized input / injection patterns: stop first
        return {}

    def route(s: State):
        t0 = time.time()
        return {"result": m.route(s["query"], m.registry), "t_route": time.time() - t0}

    def leak_check(s: State):
        # Belt and suspenders behind the router: a question that itself looks like it carries an identifier
        # is treated as PHI and kept off the hosted model.
        r = s["result"]
        if not r.phi:
            leaks = m.preflight_leak_check(s["query"])
            if leaks:
                return {"result": m.RouteResult(True, r.intent, r.tables, f"pre-flight leak check flagged {leaks}"),
                        "flags": ["leak_check_redirected_to_private"]}
        return {}

    def access(s: State):
        r = s["result"]
        table = f"{m.CATALOG}.clinical.phi_chunks" if r.phi else f"{m.CATALOG}.operational.general_chunks"
        client = None
        if r.phi and not m.LOCAL_DEV:
            # PHI is only ever read as the signed-in user. No identity -> refuse, never fall back to the app's access.
            client = m.user_client(s.get("token"))
            if client is None:
                raise PermissionError("Your identity was not forwarded to the app, so clinical records can't be read.")
        return {"table": table, "client": client, "t_retrieve_start": time.time()}

    def after_access(s: State):
        return "count" if s["result"].phi and m.COUNT_RE.search(s["query"]) else "retrieve"

    def count(s: State):
        # Counting question: "top matches" can't answer it. Count over everything this user may see.
        rows = m.read_chunks(s["table"], s["client"], True)
        units = {u.lower(): u for c in rows for u in re.findall(r" in ([A-Za-z]+), diagnosis", c[1])}
        unit = next((units[u] for u in units if re.search(rf"\b{re.escape(u)}\b", s["query"].lower())), None)
        rows = [r for r in rows if not unit or f" in {unit}," in r[1]]
        patients = {mt.group(1) for r in rows if (mt := re.search(r"Patient (.+?) \(MRN", r[1]))}
        answer = (f"{len(patients)} patient{'s' if len(patients) != 1 else ''} with encounters on record"
                  f"{' in ' + unit if unit else ''}, among the records you have access to."
                  " (The data has no discharge status, so this counts everyone with an encounter on record, not current occupancy.)")
        return {"answer": answer, "model": "none (count from records)", "chunks": [{"chunk_id": r[0]} for r in rows],
                "t_retrieve": time.time() - s["t_retrieve_start"], "t_gen": 0.0}

    def retrieve(s: State):
        chunks = m.retrieve(s["table"], s["query"], s["client"], phi=s["result"].phi)
        return {"chunks": chunks, "t_retrieve": time.time() - s["t_retrieve_start"]}

    def after_retrieve(s: State):
        if not s["chunks"]:
            return "no_records"
        if s["result"].phi:
            return "list_records" if len(s["chunks"]) > 1 else "gen_private"
        return "gen_general"

    def no_records(s: State):
        # Nothing to ground an answer in — don't let a model improvise from general knowledge.
        return {"answer": "I couldn't find any matching records for that question.",
                "model": "none (no records matched)", "t_gen": 0.0}

    def list_records(s: State):
        # Several patients matched: list the records verbatim. A small model tends to trim or garble lists.
        cs = s["chunks"]
        return {"answer": f"**{len(cs)} matching records**\n\n" + m.format_records([c["content"] for c in cs]),
                "model": "none (records listed as-is)", "t_gen": 0.0}

    def gen_private(s: State):
        t = time.time()
        answer = m.llm.generate(s["query"], [c["content"] for c in s["chunks"]])
        return {"answer": answer, "model": m.llm.PRIVATE_ENDPOINT, "t_gen": time.time() - t}

    def gen_general(s: State):
        t = time.time()
        ctx = "\n".join(f"- {c['content']}" for c in s["chunks"]) or "(no matching records)"
        leaks = m.preflight_leak_check(ctx)
        if leaks:  # general chunks should never hold identifiers — refuse rather than send them out
            raise RuntimeError(f"Blocked before sending to the hosted model: context flagged {leaks}")
        answer = m.ask_general_model(s["query"], ctx)
        out_leaks = m.preflight_leak_check(answer)  # the hosted model's reply must not carry identifiers either
        if out_leaks:
            raise m.GuardrailBlocked(f"The response was withheld because it contained identifiers ({', '.join(out_leaks)}).")
        return {"answer": answer, "model": m.GENERAL_ENDPOINT, "t_gen": time.time() - t}

    def grounding(s: State):
        # A number, ID, code, time or date the model states must appear in the records, or its answer is replaced
        # by the records themselves.
        answer, ungrounded = m._grounded_check(s["answer"], s["chunks"])
        return {"answer": answer, "flags": ["grounding_failed"] if ungrounded else []}

    def finalize(s: State):
        r = s["result"]
        # Never log the question text: on the PHI path it may contain PHI.
        print(f"route phi={r.phi} intent={r.intent} model={s['model']} route={s['t_route']:.1f}s "
              f"retrieve={s['t_retrieve']:.1f}s generate={s['t_gen']:.1f}s", flush=True)
        return {"final": {
            "path": "phi" if r.phi else "general", "intent": r.intent, "reason": r.reason, "model": s["model"],
            "answer": s["answer"], "sources": [c["chunk_id"] for c in s["chunks"]], "flags": s.get("flags", []),
            "timings_s": {"route": round(s["t_route"], 1), "retrieve": round(s["t_retrieve"], 1),
                          "generate": round(s["t_gen"], 1)}}}

    g = StateGraph(State)
    for fn in (screen, route, leak_check, access, count, retrieve, no_records, list_records, gen_private,
               gen_general, grounding, finalize):
        g.add_node(fn.__name__, fn)
    g.set_entry_point("screen")
    g.add_edge("screen", "route")
    g.add_edge("route", "leak_check")
    g.add_edge("leak_check", "access")
    g.add_conditional_edges("access", after_access, {"count": "count", "retrieve": "retrieve"})
    g.add_conditional_edges("retrieve", after_retrieve, {"no_records": "no_records", "list_records": "list_records",
                                                         "gen_private": "gen_private", "gen_general": "gen_general"})
    for n in ("gen_private", "gen_general"):
        g.add_edge(n, "grounding")
    for n in ("count", "no_records", "list_records", "grounding"):
        g.add_edge(n, "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_graph = None


def run(m, query: str, token: str | None, progress=None) -> dict:
    """Run the pipeline. `progress(step_label)` is called as each node starts to finish."""
    global _graph
    if _graph is None:
        _graph = build(m)
    final = None
    for update in _graph.stream({"query": query, "token": token, "flags": []}, stream_mode="updates"):
        for node, out in update.items():
            if progress:
                progress(STEP_LABELS.get(node, node))
            if node == "finalize":
                final = out["final"]
    return final
