"""
run_all_configs.py
Runs all four model variants across all tickers, lookback windows, and
sparsity lambdas. Outputs a timestamped results text file and an HTML
dashboard with per-date gate value lookup.

Configuration: 8 tickers x 4 lookbacks x 3 lambdas (for gated models)
             = 8 x 4 x (2 ungated + 6 gated) = 256 total runs
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

from data_pipeline import fetch_stock_ohlcv, build_tier0_features, standardize_train_only, make_windows
from model_a_lstm import LSTMBaseline
from model_b_vanilla_tcn import TCNAttention as VanillaTCN
from model_c_static_gate import TCNAttention as StaticGateTCN
from model_d_dynamic_gate import TCNAttention as DynamicGateTCN

# ---------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------

TICKERS = ["AAPL", "AMZN", "JPM", "XOM", "NEE", "GILD", "NVDA", "LUV"]
START = "2015-01-01"
END = "2026-01-01"
TRAIN_FRAC = 0.8
LOOKBACKS = [300, 600, 900, 1200]
LAMBDAS = [0, 0.001, 0.0001]       # sparsity penalties (0 = no penalty)
EPOCHS = 2
LR = 1e-3
BATCH_SIZE = 128
HIDDEN_DIM = 32
SEED = 42


# ---------------------------------------------------------------
# Training and evaluation (shared across all models)
# ---------------------------------------------------------------

def train_and_evaluate(model, train_loader, test_X, test_y,
                       epochs, lr, device, sparsity_lambda=0.0, gate_type="none"):
    """Train a model and return its best-epoch predictions, MSE, direction
    accuracy, and gate values (if applicable).

    gate_type controls how the forward pass and sparsity penalty work:
        "none"    — LSTM or vanilla TCN (no gate values returned)
        "static"  — static gate (sparsity uses sum of gate values)
        "dynamic" — dynamic gate (sparsity uses mean of gate values)
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # pre-convert test data to tensors
    test_X_t = torch.tensor(test_X, dtype=torch.float32, device=device)
    test_y_t = torch.tensor(test_y, dtype=torch.float32, device=device)

    best_mse = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # forward pass differs by model type
            if gate_type == "none":
                # LSTM returns a single tensor; TCN returns (pred, attn_weights)
                pred = model(xb) if isinstance(model, LSTMBaseline) else model(xb)[0]
                loss = loss_fn(pred, yb)
            else:
                # gated models return (pred, attn_weights, gate_values)
                pred, _, gate_values = model(xb)
                loss = loss_fn(pred, yb)
                # sparsity penalty: static uses sum (21 fixed values),
                # dynamic uses mean (varies per window)
                if sparsity_lambda > 0:
                    gate_penalty = gate_values.sum() if gate_type == "static" else gate_values.mean()
                    loss = loss + gate_penalty * sparsity_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # evaluate on test set
        model.eval()
        with torch.no_grad():
            if gate_type == "none":
                test_pred = model(test_X_t) if isinstance(model, LSTMBaseline) else model(test_X_t)[0]
            else:
                test_pred, _, _ = model(test_X_t)
            test_loss = loss_fn(test_pred, test_y_t).item()

        # track best model state
        if test_loss < best_mse:
            best_mse = test_loss
            best_epoch = epoch
            best_state = model.state_dict().copy()

        if epoch % 20 == 0 or epoch == 1:
            print(f"    ep {epoch:3d} | train {np.mean(train_losses):.6f} | test {test_loss:.6f}")

    # restore best-epoch weights and compute final predictions
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if gate_type == "none":
            preds = (model(test_X_t) if isinstance(model, LSTMBaseline) else model(test_X_t)[0]).cpu().numpy()
            gates = None
        else:
            pred_t, _, gates_t = model(test_X_t)
            preds = pred_t.cpu().numpy()
            gates = gates_t.cpu().numpy()

    return {
        "best_mse": best_mse,
        "best_epoch": best_epoch,
        "direction_acc": np.mean(np.sign(preds) == np.sign(test_y)),
        "preds": preds,
        "params": sum(p.numel() for p in model.parameters()),
        "all_gates": gates,
    }


# ---------------------------------------------------------------
# HTML dashboard generator
# ---------------------------------------------------------------

def build_dashboard(all_results, gate_data, feature_names, timestamp):
    """Generate a standalone HTML dashboard with embedded JSON data.
    Includes tabs for gate values, date lookup, performance, heatmap, and rankings."""

    # format average gate data for the dashboard
    gates_json = []
    for g in gate_data:
        gates_json.append({
            "ticker": g["ticker"], "lookback": g["lookback"], "model": g["model"],
            "lam": g["lam"], "features": feature_names,
            "gate_avg": g["gates_avg"].tolist(),
            "gate_std": g["gates_std"].tolist() if g["gate_type"] == "dynamic" else [0.0] * len(feature_names),
        })

    # format per-window gate data for date lookup (dynamic gate only)
    per_window_json = []
    for g in gate_data:
        if g["gate_type"] == "dynamic" and "per_window_gates" in g:
            per_window_json.append({
                "ticker": g["ticker"], "lookback": g["lookback"],
                "dates": g["test_dates"],
                "prices": [round(p, 2) for p in g["test_prices"]],
                "gates": g["per_window_gates"].tolist(),
            })

    # format results summary
    results_json = [{
        "ticker": r["ticker"], "lookback": r["lookback"], "model": r["model"],
        "lam": r["lambda"], "mse": round(r["mse"], 6),
        "dir_acc": round(r["dir_acc"] * 100, 2), "mae": round(r["mae"], 2),
        "mape": round(r["mape"], 2), "params": r["params"], "epoch": r["epoch"],
    } for r in all_results]

    # inject data into HTML template
    html = _get_html_template()
    html = html.replace("/*RESULTS_DATA*/", json.dumps(results_json))
    html = html.replace("/*GATES_DATA*/", json.dumps(gates_json))
    html = html.replace("/*PERWINDOW_DATA*/", json.dumps(per_window_json))
    html = html.replace("/*TIMESTAMP*/", timestamp)

    filename = f"dashboard_{timestamp}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Dashboard saved: {filename}")


def _get_html_template():
    """Return the full HTML/CSS/JS template for the dashboard.
    Data placeholders (/*RESULTS_DATA*/, /*GATES_DATA*/, etc.) are replaced
    at generation time with JSON-serialized experiment results."""
    return r"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Thesis Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f5f6fa;color:#2c3e50;padding:20px}
h1{font-size:22px;color:#1a252f;margin-bottom:4px}
h2{font-size:16px;margin:16px 0 8px;color:#1a252f;border-bottom:2px solid #3498db;padding-bottom:4px}
.sub{color:#7f8c8d;font-size:12px;margin-bottom:12px}
.ctrl{background:#fff;padding:12px;border-radius:8px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.1);display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.ctrl label{font-weight:600;font-size:12px;color:#555}
.ctrl select,.ctrl input[type=date]{padding:5px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px}
.ctrl button{padding:6px 14px;border:none;border-radius:4px;background:#1f3864;color:#fff;font-size:12px;font-weight:600;cursor:pointer}
.p{background:#fff;padding:12px;border-radius:8px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
table{border-collapse:collapse;width:100%;font-size:12px}
th{background:#1f3864;color:#fff;padding:6px 8px;text-align:center;font-weight:600}
td{padding:5px 8px;text-align:center;border-bottom:1px solid #eee}
tr:hover{background:#f0f4f8}
.bc{display:flex;align-items:center;gap:4px}.b{height:13px;border-radius:3px;min-width:2px}
.bs{background:#3498db}.bd{background:#9b59b6}.bl{font-size:11px;color:#555;min-width:34px;text-align:right}
.t{display:inline-block;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600}
.ta{background:#d5f5e3;color:#1e8449}.tsx{background:#fadbd8;color:#c0392b}
.tdy{background:#e8daef;color:#7d3c98}.tst{background:#eaecee;color:#5d6d7e}
.hc{width:100%;height:24px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;border-radius:3px}
.pg{color:#1e8449}.pb{color:#c0392b}
.tabs{display:flex;gap:3px;margin-bottom:12px}
.tab{padding:7px 14px;border-radius:6px 6px 0 0;cursor:pointer;font-size:12px;font-weight:600;background:#ddd;color:#555}
.tab.on{background:#1f3864;color:#fff}
.dr{margin-top:10px;padding:12px;border:2px solid #3498db;border-radius:8px;background:#f0f7ff}
</style></head><body>
<h1>Thesis Results Dashboard</h1><p class="sub">Generated /*TIMESTAMP*/</p>
<div class="ctrl">
<div><label>Ticker</label><br><select id="st" onchange="u()"></select></div>
<div><label>Lookback</label><br><select id="sl" onchange="u()"></select></div>
<div><label>Compare</label><br><select id="st2" onchange="u()"><option value="">--</option></select></div>
</div>
<div class="tabs">
<div class="tab on" onclick="sT(0)">Gate Values</div>
<div class="tab" onclick="sT(1)">Date Lookup</div>
<div class="tab" onclick="sT(2)">Performance</div>
<div class="tab" onclick="sT(3)">Heatmap</div>
<div class="tab" onclick="sT(4)">Rankings</div>
</div>
<div id="out"></div>
<script>
const R=/*RESULTS_DATA*/;
const G=/*GATES_DATA*/;
const PW=/*PERWINDOW_DATA*/;
const TK=[...new Set(R.map(r=>r.ticker))];
const LB=[...new Set(R.map(r=>r.lookback))].sort((a,b)=>a-b);
const F=G.length?G[0].features:[];
let TB=0;
function init(){const s=document.getElementById('st'),s2=document.getElementById('st2'),l=document.getElementById('sl');
TK.forEach(t=>{s.innerHTML+=`<option>${t}</option>`;s2.innerHTML+=`<option>${t}</option>`});
LB.forEach(v=>l.innerHTML+=`<option>${v}</option>`);u()}
function sT(i){TB=i;document.querySelectorAll('.tab').forEach((t,j)=>t.className=j===i?'tab on':'tab');u()}
function u(){const tk=document.getElementById('st').value,lb=+document.getElementById('sl').value,t2=document.getElementById('st2').value,o=document.getElementById('out');
if(TB===0)vG(tk,lb,t2,o);else if(TB===1)vD(tk,lb,t2,o);else if(TB===2)vP(tk,lb,t2,o);else if(TB===3)vH(tk,lb,t2,o);else vK(tk,lb,t2,o)}
function br(v,c){const p=Math.round(v*100);return `<div class="bc"><div class="b ${c}" style="width:${p}%"></div><span class="bl">${p}%</span></div>`}
function tg(v){return v>0.5?'<span class="t ta">ACTIVE</span>':'<span class="t tsx">SUPPRESSED</span>'}
function dt(s){return s>0.05?'<span class="t tdy">DYNAMIC</span>':'<span class="t tst">STABLE</span>'}

function vG(tk,lb,t2,o){let h='';
const dg=G.find(g=>g.ticker===tk&&g.lookback===lb&&g.model==='Dynamic Gate');
const sg=G.find(g=>g.ticker===tk&&g.lookback===lb&&g.model==='Static Gate');
if(dg){h+=`<div class="p"><h2>${tk} Dynamic Gate (LB=${lb})</h2>`;h+=gTbl(dg,'bd',1);
if(t2&&t2!==tk){const d2=G.find(g=>g.ticker===t2&&g.lookback===lb&&g.model==='Dynamic Gate');if(d2)h+=`<h2>${tk} vs ${t2}</h2>`+cTbl(dg,d2)}h+='</div>'}
if(sg){h+=`<div class="p"><h2>${tk} Static Gate (LB=${lb})</h2>`;h+=gTbl(sg,'bs',0);
if(t2&&t2!==tk){const s2=G.find(g=>g.ticker===t2&&g.lookback===lb&&g.model==='Static Gate');if(s2)h+=`<h2>${tk} vs ${t2}</h2>`+cTbl(sg,s2)}h+='</div>'}
if(!dg&&!sg)h='<div class="p">No gate data.</div>';o.innerHTML=h}

function gTbl(g,c,sd){const p=F.map((f,i)=>({n:f,a:g.gate_avg[i],s:g.gate_std[i]})).sort((a,b)=>b.a-a.a);
let h='<table><tr><th>Feature</th><th>Value</th><th style="width:35%">Bar</th>';if(sd)h+='<th>Std</th><th>Var</th>';h+='<th>Status</th></tr>';
for(const x of p){h+=`<tr><td style="text-align:left;font-weight:500">${x.n}</td><td>${x.a.toFixed(3)}</td><td>${br(x.a,c)}</td>`;
if(sd)h+=`<td>${x.s.toFixed(3)}</td><td>${dt(x.s)}</td>`;h+=`<td>${tg(x.a)}</td></tr>`}return h+'</table>'}

function cTbl(g1,g2){const p=F.map((f,i)=>({n:f,v1:g1.gate_avg[i],v2:g2.gate_avg[i],d:g2.gate_avg[i]-g1.gate_avg[i]})).sort((a,b)=>Math.abs(b.d)-Math.abs(a.d));
let h=`<table><tr><th>Feature</th><th>${g1.ticker}</th><th>${g2.ticker}</th><th>Diff</th></tr>`;
for(const x of p){const dc=Math.abs(x.d)>0.05?(x.d>0?'pg':'pb'):'';
h+=`<tr><td style="text-align:left;font-weight:500">${x.n}</td><td>${x.v1.toFixed(3)}</td><td>${x.v2.toFixed(3)}</td><td class="${dc}" style="font-weight:600">${x.d>0?'+':''}${x.d.toFixed(3)}</td></tr>`}return h+'</table>'}

function vD(tk,lb,t2,o){
const pw0=PW.find(p=>p.ticker===tk&&p.lookback===lb);
let h=`<div class="p"><h2>Date Lookup — Dynamic Gate Values</h2>
<p style="color:#7f8c8d;font-size:12px;margin-bottom:10px">Enter a date to see the exact gate values the dynamic gate assigned for ${tk}.</p>
<div class="ctrl"><div><label>Date</label><br><input type="date" id="ld"></div><button onclick="doL()">Look Up</button></div></div>`;
if(pw0)h+=`<div class="p"><p style="font-size:12px">Available: ${pw0.dates[0]} to ${pw0.dates[pw0.dates.length-1]} (${pw0.dates.length} windows)</p></div>`;
h+='<div id="lr"></div>';o.innerHTML=h}

function doL(){const tk=document.getElementById('st').value,lb=+document.getElementById('sl').value,
t2=document.getElementById('st2').value,ds=document.getElementById('ld').value,o=document.getElementById('lr');
if(!ds){o.innerHTML='<div class="p">Pick a date.</div>';return}
let h=lkup(tk,lb,ds);if(t2&&t2!==tk){h+=lkup(t2,lb,ds);h+=lkCmp(tk,t2,lb,ds)}o.innerHTML=h}

function lkup(tk,lb,ds){const pw0=PW.find(p=>p.ticker===tk&&p.lookback===lb);
if(!pw0)return `<div class="p">No data for ${tk} LB=${lb}.</div>`;
let bi=0,bd=Infinity;for(let i=0;i<pw0.dates.length;i++){const d=Math.abs(new Date(pw0.dates[i])-new Date(ds));if(d<bd){bd=d;bi=i}}
const ad=pw0.dates[bi],pr=pw0.prices[bi],gt=pw0.gates[bi],off=Math.round(bd/864e5);
let h=`<div class="dr"><h2>${tk} — ${ad}`;if(off>0)h+=` <span style="font-size:12px;color:#e67e22">(${off}d from ${ds})</span>`;
h+=`</h2><p style="font-size:13px;margin-bottom:8px">Close: <strong>$${pr.toFixed(2)}</strong> | LB: ${lb}</p>`;
h+='<table><tr><th>Feature</th><th>Gate</th><th style="width:40%">Bar</th><th>Status</th></tr>';
const ps=F.map((f,i)=>({n:f,v:gt[i]})).sort((a,b)=>b.v-a.v);
for(const p of ps)h+=`<tr><td style="text-align:left;font-weight:500">${p.n}</td><td>${p.v.toFixed(3)}</td><td>${br(p.v,'bd')}</td><td>${tg(p.v)}</td></tr>`;
return h+'</table></div>'}

function lkCmp(tk1,tk2,lb,ds){const p1=PW.find(p=>p.ticker===tk1&&p.lookback===lb),p2=PW.find(p=>p.ticker===tk2&&p.lookback===lb);
if(!p1||!p2)return '';let i1=0,i2=0,d1=Infinity,d2=Infinity;
for(let i=0;i<p1.dates.length;i++){const d=Math.abs(new Date(p1.dates[i])-new Date(ds));if(d<d1){d1=d;i1=i}}
for(let i=0;i<p2.dates.length;i++){const d=Math.abs(new Date(p2.dates[i])-new Date(ds));if(d<d2){d2=d;i2=i}}
const g1=p1.gates[i1],g2=p2.gates[i2];
const ps=F.map((f,i)=>({n:f,v1:g1[i],v2:g2[i],d:g2[i]-g1[i]})).sort((a,b)=>Math.abs(b.d)-Math.abs(a.d));
let h=`<div class="p"><h2>${tk1} vs ${tk2} on ${ds}</h2><table><tr><th>Feature</th><th>${tk1}</th><th>${tk2}</th><th>Diff</th></tr>`;
for(const p of ps){const c=Math.abs(p.d)>0.05?(p.d>0?'pg':'pb'):'';
h+=`<tr><td style="text-align:left;font-weight:500">${p.n}</td><td>${p.v1.toFixed(3)}</td><td>${p.v2.toFixed(3)}</td><td class="${c}" style="font-weight:600">${p.d>0?'+':''}${p.d.toFixed(3)}</td></tr>`}
return h+'</table></div>'}

function vP(tk,lb,t2,o){let h='';
function pt(t,l){const rs=R.filter(r=>r.ticker===t&&r.lookback===l);if(!rs.length)return '';
const b=rs.reduce((a,b)=>a.dir_acc>b.dir_acc?a:b);
let s=`<div class="p"><h2>${t} (LB=${l})</h2><table><tr><th>Model</th><th>MSE</th><th>Dir Acc</th><th>MAE</th><th>MAPE</th><th>Params</th><th>Epoch</th></tr>`;
for(const r of rs){const bg=r.dir_acc===b.dir_acc?'background:#d5f5e3':'';
s+=`<tr style="${bg}"><td style="font-weight:500">${r.model}</td><td>${r.mse.toFixed(6)}</td><td style="font-weight:700">${r.dir_acc.toFixed(2)}%</td><td>$${r.mae.toFixed(2)}</td><td>${r.mape.toFixed(2)}%</td><td>${r.params.toLocaleString()}</td><td>${r.epoch}</td></tr>`}
return s+'</table></div>'}
h+=pt(tk,lb);if(t2&&t2!==tk)h+=pt(t2,lb);
const al=R.filter(r=>r.ticker===tk&&r.model==='Dynamic Gate');
if(al.length>1){h+=`<div class="p"><h2>${tk} Dynamic Gate vs Lookback</h2><table><tr><th>LB</th><th>MSE</th><th>Dir Acc</th><th>MAE</th><th>Epoch</th></tr>`;
for(const r of al.sort((a,b)=>a.lookback-b.lookback)){const w=r.mse>0.001?' class="pb" style="font-weight:600"':'';
h+=`<tr><td>${r.lookback}</td><td${w}>${r.mse.toFixed(6)}</td><td style="font-weight:700">${r.dir_acc.toFixed(2)}%</td><td>$${r.mae.toFixed(2)}</td><td>${r.epoch}</td></tr>`}
h+='</table></div>'}o.innerHTML=h}

function vH(tk,lb,t2,o){const dg=G.filter(g=>g.lookback===lb&&g.model==='Dynamic Gate');
if(!dg.length){o.innerHTML='<div class="p">No data.</div>';return}
let h=`<div class="p"><h2>Cross-Stock Heatmap (LB=${lb})</h2><div style="overflow-x:auto"><table><tr><th>Feature</th>`;
for(const g of dg)h+=`<th>${g.ticker}</th>`;h+='</tr>';
for(let i=0;i<F.length;i++){h+=`<tr><td style="text-align:left;font-weight:500">${F[i]}</td>`;
for(const g of dg){const v=g.gate_avg[i];const bg=v>0.5?`rgba(39,174,96,${(v*.7).toFixed(2)})`:`rgba(192,57,43,${((1-v)*.5).toFixed(2)})`;
const fg=v>0.65||v<0.3?'#fff':'#333';h+=`<td><div class="hc" style="background:${bg};color:${fg}">${v.toFixed(2)}</div></td>`}h+='</tr>'}
h+='</table></div></div>';o.innerHTML=h}

function vK(tk,lb,t2,o){const dg=G.filter(g=>g.lookback===lb&&g.model==='Dynamic Gate');
if(!dg.length){o.innerHTML='<div class="p">No data.</div>';return}
const fv=F.map((f,i)=>{const vs=dg.map(g=>g.gate_avg[i]);const m=vs.reduce((a,b)=>a+b,0)/vs.length;
const s=Math.sqrt(vs.reduce((a,b)=>a+(b-m)**2,0)/vs.length);return {name:f,std:s,mean:m,min:Math.min(...vs),max:Math.max(...vs)}}).sort((a,b)=>b.std-a.std);
let h=`<div class="p"><h2>Cross-Stock Variation (LB=${lb})</h2><table><tr><th>#</th><th>Feature</th><th>Std</th><th>Mean</th><th>Min</th><th>Max</th><th>Range</th></tr>`;
fv.forEach((x,i)=>{const r=x.max-x.min;const c=r>0.3?'pb':r>0.15?'':'pg';
h+=`<tr><td>${i+1}</td><td style="text-align:left;font-weight:500">${x.name}</td><td>${x.std.toFixed(3)}</td><td>${x.mean.toFixed(3)}</td><td>${x.min.toFixed(3)}</td><td>${x.max.toFixed(3)}</td><td class="${c}" style="font-weight:600">${r.toFixed(3)}</td></tr>`});
h+='</table></div>';
const bm=[...fv].sort((a,b)=>b.mean-a.mean);
h+=`<div class="p"><h2>Universally Active vs Suppressed</h2><div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">`;
h+=`<div><h2>Most Active</h2><table><tr><th>Feature</th><th>Avg</th></tr>`;
for(let i=0;i<Math.min(5,bm.length);i++)h+=`<tr><td style="text-align:left">${bm[i].name}</td><td class="pg" style="font-weight:600">${bm[i].mean.toFixed(3)}</td></tr>`;
h+='</table></div><div><h2>Most Suppressed</h2><table><tr><th>Feature</th><th>Avg</th></tr>';
const rv=[...bm].reverse();for(let i=0;i<Math.min(5,rv.length);i++)h+=`<tr><td style="text-align:left">${rv[i].name}</td><td class="pb" style="font-weight:600">${rv[i].mean.toFixed(3)}</td></tr>`;
h+='</table></div></div></div>';o.innerHTML=h}

init();
</script></body></html>"""


# ---------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------

def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = []    # one entry per model/ticker/lookback/lambda run
    gate_data = []      # gate values for dashboard visualization
    feature_names = []  # populated on first ticker

    for ticker in TICKERS:
        print(f"\n{'#' * 60}\n  TICKER: {ticker}\n{'#' * 60}")

        # load and build features (shared across all lookbacks for this ticker)
        df = fetch_stock_ohlcv(ticker, START, END)
        X_df, y = build_tier0_features(df, start=START, end=END)
        feature_names = list(X_df.columns)
        all_dates = X_df.index
        all_prices = df["adj_close"].reindex(all_dates)
        print(f"  {len(X_df)} rows, {len(feature_names)} features")

        for lookback in LOOKBACKS:
            print(f"\n{'=' * 50}\n  {ticker} | LB={lookback}\n{'=' * 50}")

            # preprocess: standardize and create sliding windows
            train_end_idx = int(TRAIN_FRAC * len(X_df))
            X_scaled = standardize_train_only(X_df, train_end_idx)
            X_train, y_train, X_test, y_test = make_windows(
                X_scaled, y, lookback=lookback, train_end_idx=train_end_idx)

            # map test windows to calendar dates and prices
            T = len(X_df)
            end_positions = np.arange(lookback - 1, T)
            train_mask = end_positions < train_end_idx
            test_end_positions = end_positions[~train_mask]
            test_prices = np.array([all_prices.iloc[p] for p in test_end_positions])
            test_dates = [all_dates[p] for p in test_end_positions]
            print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

            # create data loader for training
            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                     torch.tensor(y_train, dtype=torch.float32))
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            input_dim = X_train.shape[2]

            # define all model configurations to run for this ticker/lookback
            # ungated models (LSTM, vanilla TCN) run once; gated models run per lambda
            configs = [
                {"name": "LSTM", "fn": lambda: LSTMBaseline(input_dim, HIDDEN_DIM), "gt": "none", "lam": 0.0},
                {"name": "Vanilla TCN", "fn": lambda: VanillaTCN(input_dim, HIDDEN_DIM), "gt": "none", "lam": 0.0},
            ]
            for lam in LAMBDAS:
                configs.append({"name": "Static Gate", "fn": lambda: StaticGateTCN(input_dim, HIDDEN_DIM), "gt": "static", "lam": lam})
                configs.append({"name": "Dynamic Gate", "fn": lambda: DynamicGateTCN(input_dim, HIDDEN_DIM), "gt": "dynamic", "lam": lam})

            # run each configuration
            for cfg in configs:
                print(f"\n  --- {ticker} | {cfg['name']} lam={cfg['lam']:.4f} | lb={lookback} ---")
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                model = cfg["fn"]()
                res = train_and_evaluate(model, train_loader, X_test, y_test,
                                         EPOCHS, LR, device, cfg["lam"], cfg["gt"])

                # compute dollar-level metrics from log return predictions
                pred_prices = test_prices * np.exp(res["preds"])
                actual_prices = test_prices * np.exp(y_test)
                price_errors = pred_prices - actual_prices
                mae = np.mean(np.abs(price_errors))
                mape = np.mean(np.abs(price_errors / actual_prices)) * 100
                print(f"    MSE:{res['best_mse']:.6f} ep:{res['best_epoch']} "
                      f"dir:{res['direction_acc'] * 100:.2f}% MAE:${mae:.2f}")

                # collect gate data for dashboard
                if res["all_gates"] is not None:
                    gates = res["all_gates"]
                    entry = {
                        "ticker": ticker, "lookback": lookback, "model": cfg["name"],
                        "lam": cfg["lam"], "gate_type": cfg["gt"],
                        "gates_avg": gates.mean(axis=0) if gates.ndim > 1 else gates,
                        "gates_std": gates.std(axis=0) if gates.ndim > 1 else np.zeros(gates.shape[-1]),
                    }
                    # store per-window gates for date lookup (dynamic gate only)
                    if cfg["gt"] == "dynamic":
                        entry["per_window_gates"] = gates
                        entry["test_dates"] = [str(d.date()) for d in test_dates]
                        entry["test_prices"] = test_prices.tolist()
                    gate_data.append(entry)

                # store results for text output and dashboard
                all_results.append({
                    "ticker": ticker, "lookback": lookback, "model": cfg["name"],
                    "lambda": cfg["lam"], "mse": res["best_mse"], "epoch": res["best_epoch"],
                    "dir_acc": res["direction_acc"], "mae": mae, "mape": mape, "params": res["params"],
                })

    # save results to timestamped text file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    results_file = f"results_{timestamp}.txt"
    with open(results_file, "w") as f:
        for r in all_results:
            f.write(f"{r['ticker']}  {r['model']:<20s}  lb={r['lookback']:4d}  "
                    f"lam={r['lambda']:.4f}  MSE={r['mse']:.6f}  "
                    f"dir={r['dir_acc'] * 100:.2f}%  MAE=${r['mae']:.2f}  MAPE={r['mape']:.2f}%\n")
    print(f"\n  Results saved: {results_file}")

    # print quick summary grouped by ticker
    print(f"\n{'=' * 50}\n  SUMMARY\n{'=' * 50}")
    for ticker in dict.fromkeys(r["ticker"] for r in all_results):
        ticker_results = [r for r in all_results if r["ticker"] == ticker]
        print(f"\n  {ticker}:")
        print(f"  {'Model':<25s}  {'MSE':>10s}  {'Dir Acc':>8s}")
        print(f"  {'-' * 25}  {'-' * 10}  {'-' * 8}")
        for r in ticker_results:
            lam_str = f" (l={r['lambda']:.4f})" if r["lambda"] > 0 else ""
            print(f"  {r['model'] + lam_str:<25s}  {r['mse']:10.6f}  {r['dir_acc'] * 100:7.2f}%")

    # generate HTML dashboard
    build_dashboard(all_results, gate_data, feature_names, timestamp)
    print(f"\n  Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()