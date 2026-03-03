import datetime
import json
import altair as alt
import pandas as pd
import streamlit as st
import torch
import hydra
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from odd_gen import load_model

# IMPORT THE NEW MODULAR FILES
from feature_extractor import FeatureExtractor
from strategies import get_strategy
from app_generator import AppGenerator

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def load_config():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")
    return cfg


@st.cache_resource
def get_model_resources():
    print("Loading model...")
    cfg = load_config()
    print("Model loaded")
    return load_model(cfg)


def generate_viz_html(history):
    json_data = json.dumps(history)
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        :root {{
            --bg: #0e1117; --panel: #1e2127; --border: #30363d;
            --accent: #2c93ff; --text-main: #c9d1d9; --text-muted: #8b949e;
            --danger: #ff4b4b; --success: #2ea043; --gold: #ffd700;
        }}
        body {{
            background: var(--bg); color: var(--text-main); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0; padding: 0; display: flex; height: 100vh; overflow: hidden;
        }}
        .sidebar {{ width: 340px; background: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 20px; overflow-y: auto; flex-shrink: 0; font-size: 14px; }}
        .main-content {{ flex-grow: 1; display: flex; flex-direction: column; overflow: hidden; }}
        .header {{ padding: 10px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 20px; background: var(--bg); }}
        .grid-container {{ flex-grow: 1; overflow-y: auto; padding: 20px; }}
        h2 {{ margin-top: 0; font-size: 1.1em; color: var(--accent); }}
        h3 {{ margin-top: 15px; margin-bottom: 5px; font-size: 0.95em; color: var(--text-main); border-bottom: 1px solid #333; padding-bottom: 3px; }}
        .metric-box {{ background: #252a33; padding: 12px; border-radius: 6px; margin-bottom: 15px; }}
        .stat-label {{ color: #8b949e; font-size: 0.9em; display: inline-block; min-width: 90px; }}
        .badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; font-weight: bold; margin-bottom: 5px; }}

        .batch-row {{ margin-bottom: 35px; background: #1a1d24; padding: 10px; border-radius: 8px; border: 1px solid #2a2e37; }}
        .batch-label {{ color: var(--accent); font-size: 0.85em; margin-bottom: 12px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }}

        .token-stream {{ display: flex; flex-wrap: wrap; gap: 18px 4px; margin-bottom: 8px; }}

        .t-cell {{
            font-family: 'Fira Code', monospace; font-size: 13px;
            padding: 4px 7px; border-radius: 3px; cursor: crosshair;
            border: 1px solid transparent; transition: all 0.1s;
            min-width: 10px; text-align: center; position: relative;
        }}
        .t-cell:hover {{ border-color: var(--accent); transform: scale(1.1); z-index: 10; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }}
        .t-cell.mask {{ background: #222 !important; color: #555; }}

        .cf-badge {{
            position: absolute; top: -16px; left: 50%; transform: translateX(-50%);
            font-size: 10px; color: #ff6b6b; background: #222; border: 1px solid #ff6b6b;
            border-radius: 3px; padding: 0 4px; text-decoration: line-through;
            white-space: nowrap; z-index: 20; box-shadow: 0 2px 4px rgba(0,0,0,0.5); pointer-events: none;
        }}
        .cf-badge.orig-only {{ color: #aaa; border-color: #ff6b6b; background: #333; text-decoration: none; }}

        @keyframes pulse-gold {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7); }}
            70% {{ box-shadow: 0 0 0 4px rgba(255, 215, 0, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); }}
        }}
        @keyframes pulse-blue {{
            0% {{ box-shadow: 0 0 0 0 rgba(44, 147, 255, 0.7); }}
            70% {{ box-shadow: 0 0 0 4px rgba(44, 147, 255, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(44, 147, 255, 0); }}
        }}

        /* Structural positioning classes */
        .pos-both {{ border: 1px solid var(--gold); animation: pulse-gold 1s infinite; }}
        .pos-odd {{ border: 1px solid var(--accent); background: rgba(44, 147, 255, 0.15) !important; animation: pulse-blue 1s infinite; }}
        .pos-std {{ border: 1px dashed #ff6b6b; background: #222 !important; }}

        /* Token modification class */
        .token-flipped {{
            text-decoration: underline;
            text-decoration-color: var(--danger);
            text-decoration-thickness: 2px;
            text-underline-offset: 3px;
        }}

        /* Probability Tables */
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; margin-bottom: 10px; background: #1a1d24; border-radius: 4px; overflow: hidden; }}
        th, td {{ padding: 4px 6px; border-bottom: 1px solid #333; }}
        th {{ background: #2a2e37; color: #8b949e; font-weight: normal; }}
        td.val {{ text-align: right; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="sidebar" id="inspector">
        <h2>Token Inspector</h2>
        <div style="color:#666; font-size:0.9em; margin-bottom:10px; line-height:1.6;">
            <span style="color:var(--gold); border: 1px solid var(--gold); padding: 0 2px;">Gold Border</span> = Decoded by both<br>
            <span style="color:var(--accent); border: 1px solid var(--accent); padding: 0 2px;">Blue Border</span> = Position ONLY unmasked by ODD<br>
            <span style="color:#ff6b6b; border:1px dashed #ff6b6b; padding:0 2px;">Mask</span> = Standard WOULD have unmasked<br>
            <span style="text-decoration:underline; text-decoration-color:var(--danger); text-decoration-thickness:2px;">Underline</span> = Different Token Sampled
        </div>
        <div id="inspector-content">Hover over tokens to inspect details.</div>
    </div>
    <div class="main-content">
        <div class="header">
            <div style="display:flex; gap:10px; align-items:center;">
                <button id="playBtn" style="background:var(--accent); border:none; color:white; padding:5px 12px; border-radius:4px; cursor:pointer;">Play</button>
                <select id="speedSelect" style="background:var(--panel); color:white; border:1px solid var(--border); border-radius:4px; padding:4px;">
                    <option value="1000">Very Slow</option>
                    <option value="500">Slow</option>
                    <option value="200" selected>Normal</option>
                    <option value="50">Fast</option>
                </select>
                <input type="range" id="slider" min="0" max="{len(history) - 1}" value="0">
                <span id="stepLabel" style="font-family:monospace; min-width: 80px;">Step 0</span>
            </div>
        </div>
        <div class="grid-container" id="grid"></div>
    </div>
<script>
    const history = {json_data};
    const grid = document.getElementById('grid');
    const inspectorContent = document.getElementById('inspector-content');
    const slider = document.getElementById('slider');
    const stepLabel = document.getElementById('stepLabel');
    const playBtn = document.getElementById('playBtn');
    const speedSelect = document.getElementById('speedSelect');

    let isPlaying = false;
    let playInterval;

    function renderStep(step) {{
        const frame = history[step];
        stepLabel.innerText = `Step ${{frame.step}}`;
        grid.innerHTML = '';

        frame.batches.forEach((batch, bIdx) => {{
            const row = document.createElement('div');
            row.className = 'batch-row';
            row.innerHTML = `<div class="batch-label">Batch ${{bIdx}}</div>`;

            const stream = document.createElement('div');
            stream.className = 'token-stream';

            batch.tokens.forEach((text, tIdx) => {{
                const el = document.createElement('div');
                el.className = 't-cell';
                el.innerText = text;

                const isMaskBefore = batch.is_mask && batch.is_mask[tIdx];
                const isTransferred = batch.is_unmasked_next && batch.is_unmasked_next[tIdx];
                const isTransferredOrig = batch.is_unmasked_next_orig && batch.is_unmasked_next_orig[tIdx];
                const isFlipped = batch.is_flip && batch.is_flip[tIdx];

                if (isMaskBefore && !isTransferred) {{
                    el.classList.add('mask');
                }}

                const ent = batch.entropy ? batch.entropy[tIdx] : 0;
                let bg = el.classList.contains('mask') ? '#222' : '#2b2b2b';
                if (ent > 0.1 && !el.classList.contains('mask')) {{
                    const op = Math.min(ent / 3, 0.6);
                    bg = `rgba(200, 50, 50, ${{op}})`;
                }}
                el.style.backgroundColor = bg;

                // Orthogonal Positional Logic and Token Flip Logic
                if (isTransferred && isTransferredOrig) {{
                    el.classList.add('pos-both');
                    if (isFlipped && batch.orig_sampled_tokens && batch.orig_sampled_tokens[tIdx] !== text) {{
                        el.classList.add('token-flipped');
                        const cf = document.createElement('div');
                        cf.className = 'cf-badge';
                        cf.innerText = batch.orig_sampled_tokens[tIdx];
                        el.appendChild(cf);
                    }}
                }} else if (isTransferred && !isTransferredOrig) {{
                    el.classList.add('pos-odd');
                    if (isFlipped && batch.orig_sampled_tokens && batch.orig_sampled_tokens[tIdx] !== text) {{
                        el.classList.add('token-flipped');
                        const cf = document.createElement('div');
                        cf.className = 'cf-badge';
                        cf.innerText = batch.orig_sampled_tokens[tIdx];
                        el.appendChild(cf);
                    }}
                }} else if (!isTransferred && isTransferredOrig) {{
                    el.classList.add('pos-std');
                    const cf = document.createElement('div');
                    cf.className = 'cf-badge orig-only';
                    cf.innerText = batch.orig_sampled_tokens[tIdx];
                    el.appendChild(cf);
                }}

                el.onmouseenter = () => updateInspector(step, bIdx, tIdx);
                stream.appendChild(el);
            }});

            row.appendChild(stream);
            grid.appendChild(row);
        }});
    }}

    function updateInspector(step, bIdx, tIdx) {{
        const batch = history[step].batches[bIdx];
        let html = `<div style="padding-bottom:10px;"><b>Batch ${{bIdx}} : Pos ${{tIdx}}</b></div>`;

        const isTransferred = batch.is_unmasked_next && batch.is_unmasked_next[tIdx];
        const isTransferredOrig = batch.is_unmasked_next_orig && batch.is_unmasked_next_orig[tIdx];
        const isFlipped = batch.is_flip && batch.is_flip[tIdx];

        html += `<div class="metric-box">
            <div style="margin-bottom: 5px;"><span class="stat-label">Final Token:</span> <span style="color:#fff; font-weight:bold">${{batch.tokens[tIdx]}}</span></div>`;

        if (batch.orig_sampled_tokens) {{
            const cText = batch.orig_sampled_tokens[tIdx];
            if ((isTransferredOrig && !isTransferred) || ((isTransferredOrig || isTransferred) && cText !== batch.tokens[tIdx])) {{
                html += `<div style="margin-bottom: 5px;"><span class="stat-label">Original Prediction:</span> <span style="color:var(--danger);">${{cText}}</span></div>`;
            }}
        }}

        html += `<div><span class="stat-label">Entropy:</span> ${{(batch.entropy[tIdx]||0).toFixed(2)}}</div>
        </div>`;

        if (isTransferred && isTransferredOrig && isFlipped) {{
             html += `<div class="badge" style="background: rgba(255, 75, 75, 0.2); color: var(--danger); border: 1px solid var(--danger);">FLIPPED & DECODED</div>`;
        }} else if (isTransferred && !isTransferredOrig) {{
             html += `<div class="badge" style="background: rgba(44, 147, 255, 0.2); color: var(--accent); border: 1px solid var(--accent);">NEW POSITION (ODD)</div>`;
             if (isFlipped) {{
                 html += ` <div class="badge" style="background: rgba(255, 75, 75, 0.2); color: var(--danger); border: 1px solid var(--danger);">FLIPPED TOKEN</div>`;
             }}
        }} else if (!isTransferred && isTransferredOrig) {{
             html += `<div class="badge" style="background: rgba(255, 255, 255, 0.1); color: #aaa; border: 1px dashed #aaa;">SKIPPED POSITION</div>`;
        }}

        // TABLE 1: ODD FINAL PROBS
        if (batch.top_k_tokens && batch.top_k_tokens[tIdx] && batch.top_k_tokens[tIdx].length > 0) {{
            html += `<h3>ODD Distribution (Top 5)</h3>
            <table><thead><tr><th style="text-align:left">Token</th><th style="text-align:right">ODD P</th><th style="text-align:right">Std P</th></tr></thead><tbody>`;
            const tops = batch.top_k_tokens[tIdx];
            const probs = batch.top_k_probs[tIdx];
            const orig_probs = batch.top_k_probs_original ? batch.top_k_probs_original[tIdx] : [];

            tops.forEach((t, i) => {{
                const p = probs[i];
                const p_orig = orig_probs[i] !== undefined ? orig_probs[i] : 0;

                let p_style = "color:var(--accent)";
                if (p > p_orig + 0.05) p_style = "color: var(--success); font-weight:bold;";
                else if (p < p_orig - 0.05) p_style = "color: var(--danger)";

                html += `<tr>
                    <td>${{t}}</td>
                    <td class="val" style="${{p_style}}">${{p.toFixed(4)}}</td>
                    <td class="val" style="color:#666">${{p_orig.toFixed(4)}}</td>
                </tr>`;
            }});
            html += `</tbody></table>`;
        }}

        inspectorContent.innerHTML = html;
    }}

    slider.addEventListener('input', (e) => renderStep(e.target.value));

    speedSelect.addEventListener('change', () => {{
        if (isPlaying) {{
            clearInterval(playInterval);
            playInterval = setInterval(() => {{
                let v = parseInt(slider.value) + 1;
                if (v >= history.length) v = 0;
                slider.value = v;
                renderStep(v);
            }}, parseInt(speedSelect.value));
        }}
    }});

    playBtn.addEventListener('click', () => {{
        if (isPlaying) {{
            clearInterval(playInterval);
            playBtn.innerText = "Play";
            isPlaying = false;
        }} else {{
            playBtn.innerText = "Pause";
            isPlaying = true;
            playInterval = setInterval(() => {{
                let v = parseInt(slider.value) + 1;
                if (v >= history.length) v = 0;
                slider.value = v;
                renderStep(v);
            }}, parseInt(speedSelect.value));
        }}
    }});

    renderStep(0);
</script>
</body>
</html>
"""


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="ODD Demo")

    with st.spinner("Loading LLaDA Model..."):
        model, tokenizer, embedding_matrix, mask_token_id = get_model_resources()

    if "history_log" not in st.session_state:
        st.session_state.history_log = []

    st.title("Orthogonal Diverse Diffusion (ODD) Demo")

    with st.sidebar:
        st.header("1. Parameters")
        prompt_input = st.text_area("Prompt", "Write python code to compute the factorial of n", height=80)

        c1, c2 = st.columns(2)
        with c1:
            batch_size = st.number_input("Batch Size", 1, 64, 4)
            gen_len = st.number_input("Gen Length", 16, 128, 64)
        alpha = st.number_input("Alpha (Diversity Step Size)", 0.0, 1024.0, 16.0)
        with c2:
            steps = st.number_input("Steps", 10, 100, 32)
            temp = st.number_input("Temperature", 0.0, 5.0, 1.0)

        quality_scale = st.number_input("Quality scale", 0.0, 100.0, 1.0)

        st.divider()
        st.subheader("DPP Controls")
        strategy_name = st.selectbox("Strategy", ["odd", "dpp", "orthogonal_projection", "random_probe",])
        target = st.selectbox("Kernel Target", ["logits", "embeddings"])
        pool = st.selectbox("Pooling", ["max", "mean", "positional"])

        if st.button("Generate New Run", type="primary", use_container_width=True):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            feature_extractor = FeatureExtractor(embedding_matrix=embedding_matrix, kernel_target=target,
                                                 pooling_method=pool, top_k=0)
            dpp_strategy = get_strategy(strategy_name, alpha, quality_scale, feature_extractor)

            generator = AppGenerator(model, tokenizer, dpp_strategy, mask_token_id)

            data, _ = generator.generate(prompt_input, batch_size, steps, gen_len, temp)

            run_record = {
                "id": f"{timestamp} - {prompt_input[:20]}...",
                "timestamp": timestamp,
                "params": {"prompt": prompt_input, "batch": batch_size, "steps": steps, "alpha": alpha, "temp": temp,
                           "strategy": strategy_name},
                "data": data
            }
            st.session_state.history_log.insert(0, run_record)
            st.rerun()

        st.divider()
        st.header("2. History")
        run_options = [r["id"] for r in st.session_state.history_log]
        selected_run_id = st.selectbox("Select Run", run_options, index=0 if run_options else None)
        current_run = next((r for r in st.session_state.history_log if r["id"] == selected_run_id), None)

        if st.session_state.history_log:
            st.download_button("Download History JSON", json.dumps(st.session_state.history_log), "dpp_history.json",
                               "application/json")

        uploaded_file = st.file_uploader("Upload History JSON", type="json")
        if uploaded_file is not None:
            try:
                st.session_state.history_log = json.load(uploaded_file)
                st.success("History loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")

    if current_run:
        st.markdown(f"**Viewing Run:** `{current_run['id']}`")
        tabs = st.tabs(["Visualization", "Metrics (Charts)", "Final Output"])

        with tabs[0]:
            st.components.v1.html(generate_viz_html(current_run["data"]), height=600, scrolling=True)

        with tabs[1]:
            st.subheader("Force & Entropy Over Time")
            chart_data = [{"step": frame["step"], "batch": f"Batch {b_idx}",
                           "entropy": sum([e for e in batch["entropy"] if e > 0]) / (
                                       len([e for e in batch["entropy"] if e > 0]) or 1),
                           "force": sum([f for f in batch["force"] if f > 0]) / (
                                       len([f for f in batch["force"] if f > 0]) or 1)} for frame in current_run["data"]
                          for b_idx, batch in enumerate(frame["batches"])]
            df = pd.DataFrame(chart_data)
            if not df.empty:
                st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x='step', y='force', color='batch',
                                                                           tooltip=['step', 'batch',
                                                                                    'force']).properties(
                    title="Average Repulsion Force per Step", height=300), use_container_width=True)
                st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x='step', y='entropy', color='batch',
                                                                           tooltip=['step', 'batch',
                                                                                    'entropy']).properties(
                    title="Average Entropy per Step", height=300), use_container_width=True)
            else:
                st.info("No metric data available.")

        with tabs[2]:
            st.subheader("Clean Output Text")
            for i, batch in enumerate(current_run["data"][-1]["batches"]):
                with st.expander(f"Batch {i}", expanded=True):
                    st.code(
                        "".join([t for t, is_spec in zip(batch["tokens"], batch["is_special"]) if not is_spec]).replace(
                            "⏎", "\n"), language="text")
    else:
        st.info("No runs found. Generate a new run or upload a history file.")