"""Gradio web UI for the hospital chat app. All logic lives in main.py; this file only presents it.

Run with `python ui.py` (see app.yaml). Extra routes /health, /status and /whoami are kept for diagnostics."""
import base64
import json
import os
import time

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request

import llm
import main

THEME = gr.themes.Soft(
    primary_hue="teal", secondary_hue="cyan", neutral_hue="slate",
    font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
)

CSS = """
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }
.hero { padding: 22px 26px; border-radius: 16px; color: #fff;
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%); margin-bottom: 8px; }
.hero h1 { margin: 0 0 4px 0; font-size: 1.7rem; color: #fff; }
.hero p { margin: 0; opacity: .92; }
.badges { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
.badge { background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
         padding: 3px 11px; border-radius: 999px; font-size: .8rem; }
.statusline { font-size: .85rem; color: #64748b; margin: 2px 4px 6px 4px; }
.footnote { text-align: center; color: #94a3b8; font-size: .78rem; margin-top: 10px; }
"""

HEADER = """
<div class="hero">
  <h1>🏥 Hospital Assistant</h1>
  <p>Ask about patients, beds, facilities or policy. Patient questions are answered by a private model
     inside the workspace; general questions use a separate model.</p>
  <div class="badges">
    <span class="badge">🔒 PHI stays in the workspace</span>
    <span class="badge">🛡️ Guardrails on every request</span>
    <span class="badge">🧾 Audited</span>
  </div>
</div>
"""

EXAMPLES = [
    "What was Marco Bianchi's creatinine result?",
    "Which patients are in the ICU?",
    "What is Nadia Karim's diagnosis?",
    "How many patients are admitted?",
    "How many ICU beds are available?",
    "When does the cafeteria open?",
    "What is the visitor policy?",
    "Is the pharmacy open on Sundays?",
]

DESCRIPTION = """
**How it works.** Your question is routed first: anything that touches patient data goes to the private model
and is read *as you*, so you only see records you're allowed to. Everything else goes to a general model.
Answers come only from retrieved records, and any number or ID a model states is checked against them.
"""


def _footer(res: dict) -> str:
    path = "🔒 PHI path" if res["path"] == "phi" else "🌐 General path"
    total = sum(res["timings_s"].values())
    bits = [path, res["model"], f"{total:.0f}s"]
    if "leak_check_redirected_to_private" in res.get("flags", []):
        bits.append("⚠️ identifier detected → kept on the private model")
    if "grounding_failed" in res.get("flags", []):
        bits.append("⚠️ model answer failed the grounding check → showing records")
    src = ", ".join(res.get("sources", [])[:4])
    return f"\n\n<sub>{' · '.join(bits)}{' · sources: ' + src if src else ''}</sub>"


def respond(message, history, request: gr.Request):
    user = request.headers.get("x-forwarded-email") or "unknown"
    token = request.headers.get("x-forwarded-access-token")
    try:
        job = main.submit(message, user, token)
    except main.RateLimited as e:
        yield f"⏱️ {e}"
        return
    started = time.time()
    while main.JOBS[job]["state"] == "running":
        elapsed = int(time.time() - started)
        step = main.JOBS[job].get("step")  # set by the pipeline as each graph node finishes
        hint = " The private model is waking up — the first patient question after idle can take a few minutes." if elapsed > 15 else ""
        yield f"⏳ {step + ' ✓ · ' if step else ''}Working… {elapsed}s{hint}"
        time.sleep(1)
    j = main.JOBS[job]
    if j["state"] == "done":
        yield j["result"]["answer"] + _footer(j["result"])
    elif j.get("kind") == "blocked":
        yield f"🛡️ **Blocked by a guardrail.** {j['error']}"
    elif j.get("kind") == "denied":
        yield f"🔐 **Access denied.** {j['error']}"
    else:
        yield f"⚠️ **Something went wrong.** {j['error']}"


def status_line(request: gr.Request) -> str:
    llm.refresh_status()
    state = llm.status["state"]
    icon = {"ready": "🟢", "failed": "🔴"}.get(state, "🟡")
    user = request.headers.get("x-forwarded-email") or "local user"
    return f"<div class='statusline'>{icon} Private model endpoint: <b>{state}</b> &nbsp;·&nbsp; signed in as <b>{user}</b></div>"


with gr.Blocks(title="Hospital Assistant", fill_width=False) as demo:
    gr.HTML(HEADER)
    status = gr.HTML()
    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=430, placeholder="**Ask a question** — or pick an example below.", show_label=False),
        textbox=gr.Textbox(placeholder="e.g. What was Marco Bianchi's creatinine result?", show_label=False, scale=8),
        examples=EXAMPLES,
        cache_examples=False,
        description=DESCRIPTION,
        autofocus=True,
    )
    gr.HTML("<div class='footnote'>Demo with fictional data. Not for clinical decision-making.</div>")
    demo.load(status_line, None, status)
    gr.Timer(15).tick(status_line, None, status)

demo.queue(default_concurrency_limit=8)

api = FastAPI()


@api.get("/health")
def health():
    return {"ok": True}


@api.get("/status")
def status_route():
    llm.refresh_status()
    return {"model": llm.status, "registry_loaded": bool(main.registry.known), "general_endpoint": main.GENERAL_ENDPOINT}


@api.get("/whoami")
def whoami(request: Request):
    """Diagnostic: which scopes does the signed-in user's token carry? Returns claims only, never the token."""
    token = request.headers.get("x-forwarded-access-token")
    out = {"email": request.headers.get("x-forwarded-email"), "token_forwarded": bool(token)}
    if token:
        try:
            body = token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
            out.update(scopes=str(claims.get("scope", "")).split(), issued_at=claims.get("iat"), expires_at=claims.get("exp"))
        except Exception as e:
            out["decode_error"] = type(e).__name__
    return out


app = gr.mount_gradio_app(api, demo, path="/", theme=THEME, css=CSS, ssr_mode=False)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", os.environ.get("PORT", 8000))),
                proxy_headers=True, forwarded_allow_ips="*")
