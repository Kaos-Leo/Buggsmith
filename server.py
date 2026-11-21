"""Zel Beta - Real-Time Forensic Web Visualizer & DAG Inspector.

Zero-dependency HTTP server providing an interactive browser dashboard:
- Interactive Git Commit DAG Timeline (Horizontal commit stream with trailer inspection)
- 2D Attribution Grid (4 Horizontal Swimlanes: SPEC, ARCH, AST, EXEC with vertical causality routing)
- One-Click Demo Scenario Buttons (Scenario A: Convergence, B: Bisect + ddmin, C: Anti-Goodhart)
- Live Paired DPO Preference Inspector (Chosen vs. Rejected diffs)
- Server-Sent Events (SSE) live telemetry stream
- Human-in-the-Loop Natural-Language Constraint Synthesis

Run with:
    python server.py --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.demo_scenarios import DemoRunner
from Simulation.Constraints.generator import ConstraintManager
from Simulation.Grader import GitTrailersParser

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zel-Beta // Git-Forensic Swarm Visualizer</title>
    <style>
        :root {
            --bg: #0b0f19;
            --surface: #131b2e;
            --surface-border: #1e293b;
            --text: #f1f5f9;
            --text-dim: #94a3b8;
            --accent-arch: #06b6d4;
            --accent-ast: #10b981;
            --accent-exec: #f59e0b;
            --accent-spec: #ec4899;
            --praise: #10b981;
            --penalty: #ef4444;
            --reconcile: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; }
        
        .header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid var(--surface-border); margin-bottom: 20px; }
        .logo { font-size: 22px; font-weight: 800; letter-spacing: 1px; color: #38bdf8; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700; background: #1e293b; color: #94a3b8; }
        
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px; }
        .panel { background: var(--surface); border: 1px solid var(--surface-border); border-radius: 10px; padding: 18px; }
        .panel-title { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }
        
        /* 2D Grid Swimlanes */
        .swimlane-container { display: flex; flex-direction: column; gap: 10px; }
        .swimlane { display: flex; align-items: center; background: #0f172a; border-radius: 6px; padding: 12px; border-left: 4px solid var(--text-dim); }
        .swimlane.SPEC { border-left-color: var(--accent-spec); }
        .swimlane.ARCH { border-left-color: var(--accent-arch); }
        .swimlane.AST { border-left-color: var(--accent-ast); }
        .swimlane.EXEC { border-left-color: var(--accent-exec); }
        .swimlane-label { width: 90px; font-weight: 800; font-size: 13px; }
        .swimlane-content { flex: 1; display: flex; gap: 8px; overflow-x: auto; align-items: center; }
        .node-pill { padding: 6px 12px; border-radius: 6px; font-size: 12px; font-family: monospace; background: #1e293b; border: 1px solid #334155; white-space: nowrap; }
        .node-pill.active { border-color: #38bdf8; box-shadow: 0 0 10px rgba(56, 189, 248, 0.4); }
        
        /* Buttons */
        .btn-group { display: flex; flex-direction: column; gap: 10px; }
        .btn { padding: 12px 16px; border: none; border-radius: 6px; font-weight: 600; font-size: 13px; cursor: pointer; transition: all 0.2s; display: flex; justify-content: space-between; align-items: center; text-align: left; }
        .btn-primary { background: #2563eb; color: white; }
        .btn-primary:hover { background: #1d4ed8; }
        .btn-warning { background: #b45309; color: white; }
        .btn-warning:hover { background: #92400e; }
        .btn-danger { background: #b91c1c; color: white; }
        .btn-danger:hover { background: #991b1b; }
        
        /* Terminal & Diff output */
        .console-box { background: #020617; border: 1px solid #1e293b; border-radius: 6px; padding: 12px; font-family: 'Courier New', monospace; font-size: 12px; color: #38bdf8; max-height: 220px; overflow-y: auto; white-space: pre-wrap; }
        .diff-box { background: #020617; border-left: 3px solid #ef4444; padding: 8px 12px; font-family: monospace; font-size: 12px; color: #f87171; margin-top: 8px; border-radius: 4px; }
        
        /* DPO Cards */
        .dpo-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px; }
        .dpo-card { background: #0f172a; border-radius: 6px; padding: 12px; font-size: 12px; border: 1px solid #1e293b; }
        .dpo-card.chosen { border-left: 4px solid var(--praise); }
        .dpo-card.rejected { border-left: 4px solid var(--penalty); }
        
        .input-group { display: flex; gap: 8px; margin-top: 10px; }
        .input-text { flex: 1; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; color: white; font-size: 13px; }
        .btn-sub { background: #0284c7; color: white; border: none; padding: 10px 16px; border-radius: 6px; font-weight: 600; cursor: pointer; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="logo">⚡ ZEL-BETA FORENSIC MISSION CONTROL</div>
            <div style="font-size: 12px; color: var(--text-dim); margin-top: 4px;">Stratified Git-Forensic Agent Swarm // 2D Attribution & Self-Supervised Preference Engine</div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span class="badge" id="sse-status" style="background: #14532d; color: #86efac;">● SSE ACTIVE</span>
            <span class="badge" id="contract-badge">Contract-Hash: SYNCED</span>
        </div>
    </div>

    <div class="grid">
        <!-- Left: 2D Attribution Swimlanes -->
        <div class="panel">
            <div class="panel-title">
                <span>1. 2D Forensic Attribution Grid (Horizontal DAG × Vertical Tiers)</span>
                <span style="font-size: 11px; text-transform: none; color: var(--text-dim);">Deterministic Zero-LLM Bisection</span>
            </div>
            
            <div class="swimlane-container">
                <div class="swimlane SPEC">
                    <div class="swimlane-label" style="color: var(--accent-spec);">[SPEC]</div>
                    <div class="swimlane-content" id="lane-SPEC">
                        <div class="node-pill">Contract Guard</div>
                        <div class="node-pill">Parent-Contract-Hash: Active</div>
                    </div>
                </div>
                <div class="swimlane ARCH">
                    <div class="swimlane-label" style="color: var(--accent-arch);">[ARCH]</div>
                    <div class="swimlane-content" id="lane-ARCH">
                        <div class="node-pill">arch-mutator-01</div>
                        <div class="node-pill">Set Lookups: O(1)</div>
                    </div>
                </div>
                <div class="swimlane AST">
                    <div class="swimlane-label" style="color: var(--accent-ast);">[AST]</div>
                    <div class="swimlane-content" id="lane-AST">
                        <div class="node-pill">ast-mutator-01</div>
                        <div class="node-pill">SlotsInjection</div>
                        <div class="node-pill">ListCompVectorizer</div>
                    </div>
                </div>
                <div class="swimlane EXEC">
                    <div class="swimlane-label" style="color: var(--accent-exec);">[EXEC]</div>
                    <div class="swimlane-content" id="lane-EXEC">
                        <div class="node-pill">exec-mutator-01</div>
                        <div class="node-pill">@lru_cache Injector</div>
                        <div class="node-pill">Global Binder</div>
                    </div>
                </div>
            </div>

            <div style="margin-top: 18px;">
                <div class="panel-title">Live Activity Log & Forensic Console</div>
                <div class="console-box" id="live-console">[Zel-Beta Online] Telemetry engine ready. Select a scenario or run simulation...</div>
            </div>
        </div>

        <!-- Right: Demo Triggers & Human Constraint -->
        <div class="panel">
            <div class="panel-title">Demo Controls (Judges Panel)</div>
            <div class="btn-group">
                <button class="btn btn-primary" onclick="runScenario('A')">
                    <div>
                        <div style="font-weight: 700;">Scenario A: Valid Convergence</div>
                        <div style="font-size: 11px; opacity: 0.8;">AST + EXEC loop vectorization & slots injection</div>
                    </div>
                    <span>▶</span>
                </button>
                <button class="btn btn-warning" onclick="runScenario('B')">
                    <div>
                        <div style="font-weight: 700;">Scenario B: Multi-Hunk Regression</div>
                        <div style="font-size: 11px; opacity: 0.8;">git bisect + ddmin isolates 1 bad hunk out of 5</div>
                    </div>
                    <span>▶</span>
                </button>
                <button class="btn btn-danger" onclick="runScenario('C')">
                    <div>
                        <div style="font-weight: 700;">Scenario C: Anti-Goodhart Tamper Defense</div>
                        <div style="font-size: 11px; opacity: 0.8;">EXEC tier mutator touches tests/ -> Blocked</div>
                    </div>
                    <span>🛡️</span>
                </button>
            </div>

            <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--surface-border);">
                <div class="panel-title">Human-in-the-Loop Constraint Synthesis</div>
                <div style="font-size: 12px; color: var(--text-dim); margin-bottom: 8px;">Translate human natural language into executable Python contract:</div>
                <div class="input-group">
                    <input type="text" id="constraint-input" class="input-text" placeholder="e.g. Ensure solve() handles negative numbers">
                    <button class="btn-sub" onclick="submitConstraint()">Compile</button>
                </div>
                <div id="constraint-result" style="font-size: 11px; color: #38bdf8; margin-top: 6px;"></div>
            </div>
        </div>
    </div>

    <!-- Bottom: DPO Dataset Stream & Causal Rules -->
    <div class="panel">
        <div class="panel-title">
            <span>2. Self-Supervised Paired DPO Preference Minting (training_pairs.jsonl)</span>
            <span style="font-size: 11px; color: #10b981;">100% Zero-LLM Hard Grader Verified</span>
        </div>
        <div class="dpo-grid" id="dpo-container">
            <div class="dpo-card chosen">
                <div style="font-weight: 700; color: #10b981; margin-bottom: 4px;">CHOSEN (Verified Milestone)</div>
                <div id="dpo-chosen" style="color: var(--text-dim); font-family: monospace;">Commit: c1837ca | Agent: ast-mutator-01 (AST) | Slots + ListComp verified</div>
            </div>
            <div class="dpo-card rejected">
                <div style="font-weight: 700; color: #ef4444; margin-bottom: 4px;">REJECTED (ddmin Isolated Hunk)</div>
                <div id="dpo-rejected" style="color: var(--text-dim); font-family: monospace;">Commit: b7d2ffd7 | + return [v / 0.0 for v in values] # fastmath division bug</div>
            </div>
        </div>
    </div>

    <script>
        function log(msg) {
            const box = document.getElementById('live-console');
            const time = new Date().toLocaleTimeString();
            box.innerText += `\\n[${time}] ${msg}`;
            box.scrollTop = box.scrollHeight;
        }

        async function runScenario(id) {
            log(`Triggering Scenario ${id}...`);
            try {
                const res = await fetch('/api/scenario', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({scenario: id})
                });
                const data = await res.json();
                log(`Scenario ${id} finished: ${JSON.stringify(data, null, 2)}`);
                if (data.minimal_diff) {
                    document.getElementById('dpo-rejected').innerText = `Commit: ${data.root_commit || ''} | Minimal Diff: ${data.minimal_diff}`;
                }
                if (data.final_sha) {
                    document.getElementById('dpo-chosen').innerText = `Commit: ${data.final_sha} | Praises: ${data.praises || 1} | Status: Converged`;
                }
            } catch (err) {
                log(`Error: ${err}`);
            }
        }

        async function submitConstraint() {
            const input = document.getElementById('constraint-input');
            const prompt = input.value.trim();
            if (!prompt) return;
            log(`Synthesizing constraint: "${prompt}"`);
            const res = await fetch('/api/constraint', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt})
            });
            const data = await res.json();
            document.getElementById('constraint-result').innerText = `Synthesized ${data.filename} (Hash: ${data.contract_hash})`;
            document.getElementById('contract-badge').innerText = `Contract-Hash: ${data.contract_hash}`;
            log(`Constraint saved: hash=${data.contract_hash}`);
            input.value = '';
        }

        // Connect to SSE stream
        const evtSource = new EventSource('/stream');
        evtSource.onmessage = function(e) {
            try {
                const eventData = JSON.parse(e.data);
                log(`[TELEMETRY] ${eventData.node || 'event'}: ${JSON.stringify(eventData)}`);
            } catch (err) {
                log(`[TELEMETRY] ${e.data}`);
            }
        };
        evtSource.onerror = function() {
            document.getElementById('sse-status').style.background = '#7f1d1d';
            document.getElementById('sse-status').innerText = '● SSE OFFLINE';
        };
    </script>
</body>
</html>
"""


class ForensicServerHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP handler for Zel-Beta web visualizer & SSE stream."""

    runner = DemoRunner()
    constraint_mgr = ConstraintManager()
    active_listeners: list[Any] = []
    listener_lock = threading.Lock()

    @classmethod
    def broadcast_sse(cls, payload: dict[str, Any]) -> None:
        """Broadcast live JSON event to all open EventSource client streams."""
        message = f"data: {json.dumps(payload)}\\n\\n".encode("utf-8")
        with cls.listener_lock:
            dead: list[Any] = []
            for wfile in cls.active_listeners:
                try:
                    wfile.write(message)
                    wfile.flush()
                except Exception:
                    dead.append(wfile)
            for d in dead:
                cls.active_listeners.remove(d)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/stream":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with self.listener_lock:
                self.active_listeners.append(self.wfile)

            # Send initial hello
            init_msg = f"data: {json.dumps({'status': 'connected', 'timestamp': time.time()})}\\n\\n".encode("utf-8")
            try:
                self.wfile.write(init_msg)
                self.wfile.flush()
                # Keep thread alive to stream
                while True:
                    time.sleep(1)
            except Exception:
                with self.listener_lock:
                    if self.wfile in self.active_listeners:
                        self.active_listeners.remove(self.wfile)

        elif path == "/api/dpo":
            dpo_file = PROJECT_ROOT / "training_pairs.jsonl"
            records = []
            if dpo_file.exists():
                try:
                    with open(dpo_file, "r", encoding="utf-8") as f:
                        records = [json.loads(line) for line in f if line.strip()][-20:]
                except Exception:
                    pass
            self._send_json(records)

        elif path == "/api/causal_rules":
            rules_file = PROJECT_ROOT / "causal_rules.json"
            rules = []
            if rules_file.exists():
                try:
                    rules = json.loads(rules_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            self._send_json(rules)

        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            data = {}

        if path == "/api/scenario":
            sc_id = data.get("scenario", "A").upper()

            def run_in_bg() -> None:
                if sc_id == "A":
                    res = self.runner.run_scenario_a(max_generations=2)
                elif sc_id == "B":
                    res = self.runner.run_scenario_b()
                elif sc_id == "C":
                    res = self.runner.run_scenario_c()
                else:
                    res = {"error": "Unknown scenario"}
                self.broadcast_sse({"node": "scenario_complete", "result": res})

            t = threading.Thread(target=run_in_bg, daemon=True)
            t.start()
            self._send_json({"status": "started", "scenario": sc_id})

        elif path == "/api/constraint":
            prompt = data.get("prompt", "Default contract")
            out_path = self.constraint_mgr.generate_and_save(prompt)
            contract_hash = self.constraint_mgr.compute_contract_hash()
            resp = {
                "status": "synthesized",
                "filename": out_path.name,
                "contract_hash": contract_hash,
            }
            self.broadcast_sse({"node": "new_constraint", **resp})
            self._send_json(resp)

        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _send_json(self, data: Any) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet standard HTTP access log spam
        pass


def run_server(port: int = 8000) -> None:
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, ForensicServerHandler)
    print(f"[*] Zel-Beta Forensic Web Visualizer online at: http://localhost:{port}")
    print("[*] Press Ctrl+C to terminate.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zel-Beta Web Visualizer Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    run_server(port=args.port)
