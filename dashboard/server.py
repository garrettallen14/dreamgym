#!/usr/bin/env python3
"""Lightweight dashboard server for monitoring training runs.

Usage:
    uv run python dashboard/server.py --port 3000
    
Then open http://localhost:3000 (or your RunPod forwarded port)
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="DreamGym Dashboard")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Directories to monitor
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "models" / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
for d in [LOGS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/status")
async def get_status():
    """Get overall system status."""
    # Check for running processes
    running = []
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train-experience-model|train-agent|generate-synthetic|evaluate"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            running = result.stdout.strip().split("\n")
    except:
        pass
    
    # Count data files
    trajectories = list(DATA_DIR.glob("*.jsonl"))
    models = list(MODELS_DIR.glob("*/"))
    experiments = list(EXPERIMENTS_DIR.glob("*/")) if EXPERIMENTS_DIR.exists() else []
    
    return {
        "timestamp": datetime.now().isoformat(),
        "running_processes": len(running),
        "data_files": len(trajectories),
        "models": len(models),
        "experiments": len(experiments),
        "gpu_available": check_gpu(),
    }


def check_gpu():
    """Check if GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


@app.get("/api/experiments")
async def list_experiments():
    """List all experiments and their status."""
    experiments = []
    
    # Check models/experiments directory
    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
            if not exp_dir.is_dir():
                continue
            
            exp = {"name": exp_dir.name, "path": str(exp_dir)}
            
            # Load config if exists
            config_file = exp_dir / "experiment_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    exp["config"] = json.load(f)
            
            # Load results if exists
            results_file = exp_dir / "experiment_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    exp["results"] = json.load(f)
            
            # Check trainer state for metrics
            trainer_state = exp_dir / "trainer_state.json"
            if trainer_state.exists():
                with open(trainer_state) as f:
                    state = json.load(f)
                    exp["metrics"] = {
                        "global_step": state.get("global_step"),
                        "epoch": state.get("epoch"),
                        "best_metric": state.get("best_metric"),
                    }
                    # Get last few log entries
                    log_history = state.get("log_history", [])
                    exp["recent_logs"] = log_history[-10:]
            
            experiments.append(exp)
    
    return {"experiments": experiments}


@app.get("/api/logs")
async def list_logs():
    """List available log files."""
    logs = []
    
    # Check logs directory
    for log_file in sorted(LOGS_DIR.glob("*.log"), reverse=True):
        logs.append({
            "name": log_file.name,
            "path": str(log_file),
            "size": log_file.stat().st_size,
            "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
        })
    
    # Also check for trainer logs in experiment dirs
    if EXPERIMENTS_DIR.exists():
        for exp_dir in EXPERIMENTS_DIR.iterdir():
            for log_file in exp_dir.glob("*.log"):
                logs.append({
                    "name": f"{exp_dir.name}/{log_file.name}",
                    "path": str(log_file),
                    "size": log_file.stat().st_size,
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                })
    
    return {"logs": logs[:50]}  # Limit to 50 most recent


@app.get("/api/logs/{log_name:path}")
async def get_log_content(log_name: str, tail: int = 200):
    """Get content of a specific log file."""
    # Try logs dir first
    log_path = LOGS_DIR / log_name
    if not log_path.exists():
        # Try experiments dir
        log_path = EXPERIMENTS_DIR / log_name
    if not log_path.exists():
        log_path = PROJECT_ROOT / log_name
    
    if not log_path.exists():
        return JSONResponse({"error": "Log not found"}, status_code=404)
    
    # Read last N lines
    try:
        with open(log_path) as f:
            lines = f.readlines()
            return {"lines": lines[-tail:], "total_lines": len(lines)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/logs/{log_name:path}/stream")
async def stream_log(log_name: str):
    """Stream log file updates in real-time via SSE."""
    log_path = LOGS_DIR / log_name
    if not log_path.exists():
        log_path = EXPERIMENTS_DIR / log_name
    if not log_path.exists():
        log_path = PROJECT_ROOT / log_name
    
    async def generate() -> AsyncGenerator[str, None]:
        last_size = 0
        while True:
            try:
                if log_path.exists():
                    current_size = log_path.stat().st_size
                    if current_size > last_size:
                        with open(log_path) as f:
                            f.seek(last_size)
                            new_content = f.read()
                            if new_content:
                                yield f"data: {json.dumps({'content': new_content})}\n\n"
                        last_size = current_size
            except:
                pass
            await asyncio.sleep(1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/samples/{experiment_name}")
async def get_samples(experiment_name: str):
    """Get training samples for an experiment showing model progress."""
    samples_file = EXPERIMENTS_DIR / experiment_name / "samples.jsonl"
    
    if not samples_file.exists():
        return {"samples": [], "message": "No samples yet (training may still be starting)"}
    
    samples = []
    try:
        with open(samples_file) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
    # Group by step for easier viewing
    by_step = {}
    for s in samples:
        step = s.get("step", 0)
        if step not in by_step:
            by_step[step] = []
        by_step[step].append(s)
    
    return {
        "samples": samples,
        "by_step": by_step,
        "total": len(samples),
        "steps_with_samples": sorted(by_step.keys()),
    }


@app.get("/api/samples")
async def list_all_samples():
    """List all experiments with samples available."""
    experiments_with_samples = []
    
    if EXPERIMENTS_DIR.exists():
        for exp_dir in EXPERIMENTS_DIR.iterdir():
            samples_file = exp_dir / "samples.jsonl"
            if samples_file.exists():
                # Count samples
                with open(samples_file) as f:
                    count = sum(1 for line in f if line.strip())
                experiments_with_samples.append({
                    "name": exp_dir.name,
                    "samples_count": count,
                    "modified": datetime.fromtimestamp(samples_file.stat().st_mtime).isoformat(),
                })
    
    return {"experiments": experiments_with_samples}


@app.get("/api/outputs")
async def list_outputs():
    """List generated outputs (samples, predictions, etc.)."""
    outputs = []
    
    # Check outputs directory
    for output_file in sorted(OUTPUTS_DIR.glob("*.json"), reverse=True)[:20]:
        try:
            with open(output_file) as f:
                data = json.load(f)
            outputs.append({
                "name": output_file.name,
                "path": str(output_file),
                "preview": str(data)[:500] + "..." if len(str(data)) > 500 else str(data),
            })
        except:
            pass
    
    # Check for sample generations in experiment dirs
    if EXPERIMENTS_DIR.exists():
        for exp_dir in EXPERIMENTS_DIR.iterdir():
            samples_file = exp_dir / "sample_generations.json"
            if samples_file.exists():
                try:
                    with open(samples_file) as f:
                        data = json.load(f)
                    outputs.append({
                        "name": f"{exp_dir.name}/sample_generations.json",
                        "path": str(samples_file),
                        "data": data[:5] if isinstance(data, list) else data,
                    })
                except:
                    pass
    
    return {"outputs": outputs}


@app.get("/api/data")
async def list_data_files():
    """List data files."""
    files = []
    for f in DATA_DIR.glob("*.jsonl"):
        # Count lines
        try:
            with open(f) as file:
                line_count = sum(1 for _ in file)
        except:
            line_count = -1
        
        files.append({
            "name": f.name,
            "path": str(f),
            "size_mb": f.stat().st_size / 1024 / 1024,
            "lines": line_count,
        })
    return {"files": files}


@app.get("/api/data/{filename}/sample")
async def sample_data(filename: str, n: int = 5):
    """Get sample rows from a data file."""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    samples = []
    try:
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                samples.append(json.loads(line))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
    return {"samples": samples}


@app.get("/api/gpu")
async def get_gpu_status():
    """Get detailed GPU status."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "temperature": int(parts[2]),
                        "utilization": int(parts[3]),
                        "memory_used": int(parts[4]),
                        "memory_total": int(parts[5]),
                    })
            return {"gpus": gpus}
    except Exception as e:
        return {"error": str(e), "gpus": []}
    
    return {"gpus": []}


# ============================================================================
# HTML Dashboard
# ============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DreamGym Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117; color: #c9d1d9; line-height: 1.5;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        h2 { color: #8b949e; font-size: 14px; text-transform: uppercase; margin: 20px 0 10px; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { 
            background: #161b22; border: 1px solid #30363d; border-radius: 8px; 
            padding: 16px; overflow: hidden;
        }
        .card-title { color: #58a6ff; font-weight: 600; margin-bottom: 12px; display: flex; justify-content: space-between; }
        .card-title .badge { 
            background: #238636; color: white; padding: 2px 8px; border-radius: 12px; 
            font-size: 12px; font-weight: normal;
        }
        .card-title .badge.warning { background: #9e6a03; }
        .card-title .badge.error { background: #da3633; }
        
        .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #21262d; }
        .stat:last-child { border-bottom: none; }
        .stat-label { color: #8b949e; }
        .stat-value { color: #c9d1d9; font-weight: 500; }
        
        .log-viewer {
            background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace; font-size: 12px;
            height: 400px; overflow-y: auto; padding: 12px; white-space: pre-wrap;
            word-wrap: break-word;
        }
        .log-line { margin: 2px 0; }
        .log-line.error { color: #f85149; }
        .log-line.warning { color: #d29922; }
        .log-line.info { color: #58a6ff; }
        .log-line.success { color: #3fb950; }
        
        .tabs { display: flex; gap: 4px; margin-bottom: 12px; flex-wrap: wrap; }
        .tab { 
            background: #21262d; border: none; color: #8b949e; padding: 8px 16px; 
            border-radius: 6px; cursor: pointer; font-size: 13px;
        }
        .tab:hover { background: #30363d; }
        .tab.active { background: #388bfd; color: white; }
        
        .experiment-item {
            background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
            padding: 12px; margin-bottom: 8px;
        }
        .experiment-name { color: #58a6ff; font-weight: 500; }
        .experiment-meta { color: #8b949e; font-size: 12px; margin-top: 4px; }
        .experiment-metrics { display: flex; gap: 16px; margin-top: 8px; font-size: 13px; }
        .metric { color: #c9d1d9; }
        .metric-label { color: #8b949e; }
        
        .gpu-bar { 
            background: #21262d; height: 20px; border-radius: 4px; overflow: hidden; margin: 8px 0;
        }
        .gpu-bar-fill { 
            height: 100%; background: linear-gradient(90deg, #238636, #3fb950); 
            transition: width 0.5s;
        }
        .gpu-bar-fill.high { background: linear-gradient(90deg, #9e6a03, #d29922); }
        .gpu-bar-fill.critical { background: linear-gradient(90deg, #da3633, #f85149); }
        
        .sample-box {
            background: #0d1117; border: 1px solid #30363d; border-radius: 4px;
            padding: 12px; margin: 8px 0; font-family: monospace; font-size: 12px;
            max-height: 200px; overflow-y: auto;
        }
        
        .refresh-btn {
            background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
            padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 13px;
        }
        .refresh-btn:hover { background: #30363d; }
        
        .nav { 
            display: flex; gap: 8px; margin-bottom: 20px; padding-bottom: 16px; 
            border-bottom: 1px solid #21262d; flex-wrap: wrap;
        }
        .nav-btn {
            background: transparent; border: 1px solid #30363d; color: #c9d1d9;
            padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px;
        }
        .nav-btn:hover { background: #21262d; }
        .nav-btn.active { background: #238636; border-color: #238636; }
        
        .section { display: none; }
        .section.active { display: block; }
        
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .log-viewer { height: 300px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏋️ DreamGym Dashboard</h1>
        
        <nav class="nav">
            <button class="nav-btn active" onclick="showSection('overview')">Overview</button>
            <button class="nav-btn" onclick="showSection('experiments')">Experiments</button>
            <button class="nav-btn" onclick="showSection('samples')">📊 Samples</button>
            <button class="nav-btn" onclick="showSection('logs')">Logs</button>
            <button class="nav-btn" onclick="showSection('data')">Data</button>
            <button class="nav-btn" onclick="showSection('outputs')">Outputs</button>
            <span style="flex:1"></span>
            <button class="refresh-btn" onclick="refreshAll()">↻ Refresh</button>
        </nav>
        
        <!-- Overview Section -->
        <div id="overview" class="section active">
            <div class="grid">
                <div class="card">
                    <div class="card-title">System Status <span class="badge" id="status-badge">Loading</span></div>
                    <div id="status-content">Loading...</div>
                </div>
                
                <div class="card">
                    <div class="card-title">GPU Status</div>
                    <div id="gpu-content">Loading...</div>
                </div>
                
                <div class="card" style="grid-column: span 2;">
                    <div class="card-title">Recent Activity</div>
                    <div class="log-viewer" id="recent-logs">Loading...</div>
                </div>
            </div>
        </div>
        
        <!-- Experiments Section -->
        <div id="experiments" class="section">
            <div class="card">
                <div class="card-title">Experiments</div>
                <div id="experiments-list">Loading...</div>
            </div>
        </div>
        
        <!-- Logs Section -->
        <div id="logs" class="section">
            <div class="card">
                <div class="card-title">Log Files</div>
                <div class="tabs" id="log-tabs"></div>
                <div class="log-viewer" id="log-content">Select a log file...</div>
            </div>
        </div>
        
        <!-- Data Section -->
        <div id="data" class="section">
            <div class="card">
                <div class="card-title">Data Files</div>
                <div id="data-list">Loading...</div>
            </div>
        </div>
        
        <!-- Outputs Section -->
        <div id="outputs" class="section">
            <div class="card">
                <div class="card-title">Generated Outputs</div>
                <div id="outputs-list">Loading...</div>
            </div>
        </div>
        
        <!-- Samples Section -->
        <div id="samples" class="section">
            <div class="card">
                <div class="card-title">Training Samples - Model Progress vs Ground Truth</div>
                <div class="tabs" id="samples-tabs"></div>
                <div id="samples-content">
                    <p style="color: #8b949e;">Select an experiment to view samples...</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentLogStream = null;
        
        function showSection(name) {
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(name).classList.add('active');
            event.target.classList.add('active');
            
            // Load section data
            if (name === 'experiments') loadExperiments();
            if (name === 'samples') loadSamplesList();
            if (name === 'logs') loadLogs();
            if (name === 'data') loadData();
            if (name === 'outputs') loadOutputs();
        }
        
        async function loadStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                const badge = document.getElementById('status-badge');
                badge.textContent = data.running_processes > 0 ? 'Running' : 'Idle';
                badge.className = 'badge ' + (data.running_processes > 0 ? '' : 'warning');
                
                document.getElementById('status-content').innerHTML = `
                    <div class="stat"><span class="stat-label">Running Processes</span><span class="stat-value">${data.running_processes}</span></div>
                    <div class="stat"><span class="stat-label">Data Files</span><span class="stat-value">${data.data_files}</span></div>
                    <div class="stat"><span class="stat-label">Models</span><span class="stat-value">${data.models}</span></div>
                    <div class="stat"><span class="stat-label">Experiments</span><span class="stat-value">${data.experiments}</span></div>
                `;
            } catch (e) {
                document.getElementById('status-content').innerHTML = 'Error loading status';
            }
        }
        
        async function loadGPU() {
            try {
                const res = await fetch('/api/gpu');
                const data = await res.json();
                
                if (data.gpus && data.gpus.length > 0) {
                    document.getElementById('gpu-content').innerHTML = data.gpus.map(gpu => {
                        const pct = Math.round(gpu.memory_used / gpu.memory_total * 100);
                        const cls = pct > 90 ? 'critical' : pct > 70 ? 'high' : '';
                        return `
                            <div style="margin-bottom: 12px;">
                                <div style="display: flex; justify-content: space-between; font-size: 13px;">
                                    <span>${gpu.name}</span>
                                    <span>${gpu.temperature}°C | ${gpu.utilization}% util</span>
                                </div>
                                <div class="gpu-bar"><div class="gpu-bar-fill ${cls}" style="width: ${pct}%"></div></div>
                                <div style="font-size: 12px; color: #8b949e;">${gpu.memory_used} / ${gpu.memory_total} MB (${pct}%)</div>
                            </div>
                        `;
                    }).join('');
                } else {
                    document.getElementById('gpu-content').innerHTML = '<div style="color: #8b949e;">No GPU detected</div>';
                }
            } catch (e) {
                document.getElementById('gpu-content').innerHTML = '<div style="color: #8b949e;">GPU status unavailable</div>';
            }
        }
        
        async function loadExperiments() {
            try {
                const res = await fetch('/api/experiments');
                const data = await res.json();
                
                if (data.experiments.length === 0) {
                    document.getElementById('experiments-list').innerHTML = '<div style="color: #8b949e;">No experiments found</div>';
                    return;
                }
                
                document.getElementById('experiments-list').innerHTML = data.experiments.map(exp => {
                    const status = exp.results?.status || 'unknown';
                    const statusClass = status === 'success' ? 'success' : status === 'failed' ? 'error' : 'warning';
                    const metrics = exp.metrics || {};
                    const recentLoss = exp.recent_logs?.filter(l => l.loss)?.slice(-1)[0];
                    
                    return `
                        <div class="experiment-item">
                            <div class="experiment-name">${exp.name}</div>
                            <div class="experiment-meta">Status: <span class="log-line ${statusClass}">${status}</span></div>
                            <div class="experiment-metrics">
                                ${metrics.global_step ? `<div class="metric"><span class="metric-label">Step:</span> ${metrics.global_step}</div>` : ''}
                                ${metrics.epoch ? `<div class="metric"><span class="metric-label">Epoch:</span> ${metrics.epoch.toFixed(2)}</div>` : ''}
                                ${recentLoss ? `<div class="metric"><span class="metric-label">Loss:</span> ${recentLoss.loss.toFixed(4)}</div>` : ''}
                            </div>
                        </div>
                    `;
                }).join('');
            } catch (e) {
                document.getElementById('experiments-list').innerHTML = 'Error loading experiments';
            }
        }
        
        async function loadLogs() {
            try {
                const res = await fetch('/api/logs');
                const data = await res.json();
                
                const tabs = document.getElementById('log-tabs');
                tabs.innerHTML = data.logs.map((log, i) => 
                    `<button class="tab ${i === 0 ? 'active' : ''}" onclick="selectLog('${log.name}')">${log.name}</button>`
                ).join('');
                
                if (data.logs.length > 0) {
                    selectLog(data.logs[0].name);
                }
            } catch (e) {
                document.getElementById('log-tabs').innerHTML = 'Error loading logs';
            }
        }
        
        async function selectLog(name) {
            // Update tab styling
            document.querySelectorAll('#log-tabs .tab').forEach(t => {
                t.classList.toggle('active', t.textContent === name);
            });
            
            // Load log content
            try {
                const res = await fetch(`/api/logs/${encodeURIComponent(name)}?tail=500`);
                const data = await res.json();
                
                const viewer = document.getElementById('log-content');
                viewer.innerHTML = data.lines.map(line => {
                    let cls = '';
                    if (line.includes('ERROR') || line.includes('error')) cls = 'error';
                    else if (line.includes('WARNING') || line.includes('warning')) cls = 'warning';
                    else if (line.includes('INFO')) cls = 'info';
                    else if (line.includes('success') || line.includes('complete')) cls = 'success';
                    return `<div class="log-line ${cls}">${escapeHtml(line)}</div>`;
                }).join('');
                viewer.scrollTop = viewer.scrollHeight;
                
                // Start streaming
                startLogStream(name);
            } catch (e) {
                document.getElementById('log-content').innerHTML = 'Error loading log';
            }
        }
        
        function startLogStream(name) {
            if (currentLogStream) currentLogStream.close();
            
            currentLogStream = new EventSource(`/api/logs/${encodeURIComponent(name)}/stream`);
            currentLogStream.onmessage = (event) => {
                const data = JSON.parse(event.data);
                const viewer = document.getElementById('log-content');
                viewer.innerHTML += escapeHtml(data.content);
                viewer.scrollTop = viewer.scrollHeight;
            };
        }
        
        async function loadData() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();
                
                document.getElementById('data-list').innerHTML = data.files.map(f => `
                    <div class="experiment-item">
                        <div class="experiment-name">${f.name}</div>
                        <div class="experiment-meta">${f.size_mb.toFixed(2)} MB | ${f.lines.toLocaleString()} lines</div>
                        <button class="refresh-btn" style="margin-top: 8px;" onclick="sampleData('${f.name}')">View Sample</button>
                        <div class="sample-box" id="sample-${f.name.replace('.', '-')}" style="display: none;"></div>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('data-list').innerHTML = 'Error loading data';
            }
        }
        
        async function sampleData(filename) {
            const box = document.getElementById('sample-' + filename.replace('.', '-'));
            box.style.display = box.style.display === 'none' ? 'block' : 'none';
            
            if (box.innerHTML) return;
            
            try {
                const res = await fetch(`/api/data/${filename}/sample?n=3`);
                const data = await res.json();
                box.innerHTML = JSON.stringify(data.samples, null, 2);
            } catch (e) {
                box.innerHTML = 'Error loading sample';
            }
        }
        
        async function loadOutputs() {
            try {
                const res = await fetch('/api/outputs');
                const data = await res.json();
                
                if (data.outputs.length === 0) {
                    document.getElementById('outputs-list').innerHTML = '<div style="color: #8b949e;">No outputs found</div>';
                    return;
                }
                
                document.getElementById('outputs-list').innerHTML = data.outputs.map(o => `
                    <div class="experiment-item">
                        <div class="experiment-name">${o.name}</div>
                        <div class="sample-box">${escapeHtml(typeof o.data === 'object' ? JSON.stringify(o.data, null, 2) : o.preview)}</div>
                    </div>
                `).join('');
            } catch (e) {
                document.getElementById('outputs-list').innerHTML = 'Error loading outputs';
            }
        }
        
        async function loadRecentLogs() {
            // Try to find and display most recent training log
            try {
                const res = await fetch('/api/logs');
                const data = await res.json();
                if (data.logs.length > 0) {
                    const logRes = await fetch(`/api/logs/${encodeURIComponent(data.logs[0].name)}?tail=50`);
                    const logData = await logRes.json();
                    document.getElementById('recent-logs').innerHTML = logData.lines.map(line => {
                        let cls = '';
                        if (line.includes('ERROR')) cls = 'error';
                        else if (line.includes('WARNING')) cls = 'warning';
                        else if (line.includes('INFO')) cls = 'info';
                        return `<div class="log-line ${cls}">${escapeHtml(line)}</div>`;
                    }).join('');
                } else {
                    document.getElementById('recent-logs').innerHTML = '<div style="color: #8b949e;">No logs available</div>';
                }
            } catch (e) {
                document.getElementById('recent-logs').innerHTML = '<div style="color: #8b949e;">No recent activity</div>';
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        async function loadSamplesList() {
            try {
                const res = await fetch('/api/samples');
                const data = await res.json();
                
                if (!data.experiments || data.experiments.length === 0) {
                    document.getElementById('samples-tabs').innerHTML = '';
                    document.getElementById('samples-content').innerHTML = '<p style="color: #8b949e;">No samples found. Start training with --sample-every flag to generate samples.</p>';
                    return;
                }
                
                document.getElementById('samples-tabs').innerHTML = data.experiments.map((exp, i) => 
                    `<button class="tab ${i === 0 ? 'active' : ''}" onclick="loadSamples('${exp.name}')">${exp.name.split('_').slice(0,3).join('_')} (${exp.samples_count})</button>`
                ).join('');
                
                // Load first experiment's samples
                if (data.experiments.length > 0) {
                    loadSamples(data.experiments[0].name);
                }
            } catch (e) {
                document.getElementById('samples-content').innerHTML = 'Error loading samples list';
            }
        }
        
        async function loadSamples(experimentName) {
            // Update active tab
            document.querySelectorAll('#samples-tabs .tab').forEach(t => t.classList.remove('active'));
            event?.target?.classList?.add('active');
            
            try {
                const res = await fetch(`/api/samples/${encodeURIComponent(experimentName)}`);
                const data = await res.json();
                
                if (!data.samples || data.samples.length === 0) {
                    document.getElementById('samples-content').innerHTML = '<p style="color: #8b949e;">No samples yet (training may still be starting)</p>';
                    return;
                }
                
                // Group by step
                const steps = data.steps_with_samples || [];
                
                let html = '<div style="margin-bottom: 16px;">';
                html += `<strong>Steps with samples:</strong> ${steps.join(', ')}`;
                html += '</div>';
                
                // Show samples grouped by step
                for (const step of steps) {
                    const stepSamples = data.by_step[step] || [];
                    html += `<div style="margin: 16px 0; padding: 12px; background: #0d1117; border-radius: 6px; border: 1px solid #30363d;">`;
                    html += `<h3 style="color: #58a6ff; margin: 0 0 12px 0;">Step ${step}</h3>`;
                    
                    for (const sample of stepSamples) {
                        html += `<div style="margin: 8px 0; padding: 8px; background: #161b22; border-radius: 4px;">`;
                        html += `<div style="color: #8b949e; font-size: 11px; margin-bottom: 4px;">Sample #${sample.sample_idx}</div>`;
                        html += `<div style="margin: 8px 0;"><strong style="color: #d29922;">Prompt:</strong><pre style="margin: 4px 0; padding: 8px; background: #0d1117; border-radius: 4px; white-space: pre-wrap; font-size: 12px; max-height: 100px; overflow-y: auto;">${escapeHtml(sample.prompt)}</pre></div>`;
                        html += `<div style="margin: 8px 0;"><strong style="color: #3fb950;">Ground Truth:</strong><pre style="margin: 4px 0; padding: 8px; background: #0d1117; border-radius: 4px; white-space: pre-wrap; font-size: 12px; max-height: 150px; overflow-y: auto;">${escapeHtml(sample.ground_truth)}</pre></div>`;
                        html += `<div style="margin: 8px 0;"><strong style="color: #58a6ff;">Generated:</strong><pre style="margin: 4px 0; padding: 8px; background: #0d1117; border-radius: 4px; white-space: pre-wrap; font-size: 12px; max-height: 150px; overflow-y: auto;">${escapeHtml(sample.generated)}</pre></div>`;
                        html += `</div>`;
                    }
                    html += `</div>`;
                }
                
                document.getElementById('samples-content').innerHTML = html;
            } catch (e) {
                document.getElementById('samples-content').innerHTML = 'Error loading samples: ' + e.message;
            }
        }
        
        function refreshAll() {
            loadStatus();
            loadGPU();
            loadRecentLogs();
        }
        
        // Initial load
        refreshAll();
        
        // Auto-refresh every 10 seconds
        setInterval(() => {
            loadStatus();
            loadGPU();
        }, 10000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    return DASHBOARD_HTML


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DreamGym Dashboard Server")
    parser.add_argument("--port", type=int, default=3000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  DreamGym Dashboard")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
