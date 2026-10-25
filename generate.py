#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推荐器（小白版 · 抗刷分 · 实时）
====================================
你打一句话说想干啥，它直接回你「用哪个模型」。

背后用 LMArena 官方「人类盲测匿名对战」数据集，按任务拆成多个专业榜
（综合对话 / 代码生成 / 视觉理解）。每个榜都是真实用户盲测、模型匿名，
厂商无法预知题目或干预结果 —— 业界公认最刷不了分的评测方式。

交互：输入想做的事 → 自动判定任务类别 → 给出该类别专业榜的冠军模型。
每天 GitHub Actions 自动刷新，手机浏览器收藏即可。

零第三方依赖（仅标准库）。若 HuggingFace 要求鉴权，仓库 Secrets 配 HF_TOKEN。
"""

import urllib.request
import urllib.parse
import json
import ssl
import sys
import os
import datetime
import html
from collections import Counter

UA = {"User-Agent": "Mozilla/5.0 (model-recommender/1.0)"}
HF_BASE = "https://datasets-server.huggingface.co"

CANDIDATES = [
    "lmarena-ai/leaderboard-dataset",
    "lmsys/chatbot_arena_leaderboard",
]

# 专业榜：(config, 中文标签)。每个都是 LMArena 独立盲测竞技场
ARENAS = [
    ("text",   "综合对话"),
    ("webdev", "代码生成"),
    ("vision", "视觉理解"),
]

TOP_N = 50
VOTES_UNSTABLE = 100
ROWS_PER_PAGE = 100
MAX_PAGES = 40
SUB_PRIORITY = ["coding", "math", "hard", "english", "chinese", "multilingual",
                "instruction", "reasoning", "knowledge", "vision", "vqa",
                "long", "short", "french", "spanish", "japanese"]


def http_get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read().decode("utf-8", "replace")


def _json(url):
    return json.loads(http_get(url))


def _find(names, *keys):
    for n in names:
        if not n:
            continue
        if any(k in (n or "").lower() for k in keys):
            return n
    return None


def discover_configs(dataset):
    ds = urllib.parse.quote(dataset)
    try:
        s = _json(f"{HF_BASE}/splits?dataset={ds}")
        raw = [(x.get("config"), x.get("split") or "train")
               for x in s.get("splits", []) if x.get("config")]
    except Exception:
        raw = []
    best = {}
    for cfg, sp in raw:
        if cfg not in best or sp == "latest":
            best[cfg] = sp
    return list(best.items()) or [(None, "train")]


def fetch_one(dataset, config, split):
    ds = urllib.parse.quote(dataset)
    base = f"{HF_BASE}/rows?dataset={ds}"
    if config:
        base += "&config=" + urllib.parse.quote(str(config))
    base += f"&split={urllib.parse.quote(split)}"

    all_rows, features = [], None
    for p in range(MAX_PAGES):
        page = _json(f"{base}&offset={p * ROWS_PER_PAGE}&length={ROWS_PER_PAGE}")
        if features is None:
            features = page.get("features")
        rows = page.get("rows", [])
        if not rows:
            break
        all_rows.extend(r.get("row", {}) for r in rows)
        if len(rows) < ROWS_PER_PAGE:
            break

    if not all_rows:
        raise ValueError("未返回任何行")
    if not features:
        features = [{"name": k} for k in all_rows[0].keys()]
    names = [f.get("name") for f in features]

    col_model = _find(names, "model") or names[0]
    col_org = _find(names, "organ", "org")
    col_score = _find(names, "arena", "overall") or _find(names, "score", "rating", "elo")
    col_rank = _find(names, "rank")
    col_votes = _find(names, "vote")
    col_cat = _find(names, "categor")

    if not col_score:
        raise ValueError("找不到分数列")

    pivot = {}
    for row in all_rows:
        m = str(row.get(col_model) or "?")
        try:
            sc = float(row.get(col_score))
        except (TypeError, ValueError):
            continue
        cat = str(row.get(col_cat)) if col_cat else None
        if cat:
            pivot.setdefault(m, {})[cat] = sc

    if col_cat:
        board = [r for r in all_rows
                 if str(r.get(col_cat)) in ("overall", "Overall", "ALL", "all")]
        if not board:
            board = all_rows
    else:
        board = all_rows

    freq = Counter()
    for subs in pivot.values():
        for c in subs:
            if c and c.lower() != "overall":
                freq[c] += 1
    sub_order = sorted(freq.keys(),
                       key=lambda c: (SUB_PRIORITY.index(c.lower()) if c.lower() in SUB_PRIORITY else 99,
                                      -freq[c]))
    sub_cols = sub_order[:6]

    records = []
    for row in board:
        m = str(row.get(col_model) or "?")
        try:
            score = float(row.get(col_score))
        except (TypeError, ValueError):
            continue
        votes_raw = row.get(col_votes)
        try:
            votes = int(float(votes_raw)) if votes_raw not in (None, "") else 0
        except (TypeError, ValueError):
            votes = 0
        subs = {c: pivot.get(m, {}).get(c) for c in sub_cols}
        records.append({"model": m, "org": str(row.get(col_org) or ""),
                        "score": score, "votes": votes, "subs": subs})

    if len(records) < 3:
        raise ValueError(f"仅解析到 {len(records)} 条总榜记录")
    records.sort(key=lambda x: x["score"], reverse=True)
    mapping = {"model": col_model, "org": col_org, "score": col_score,
               "rank": col_rank, "votes": col_votes, "subs": sub_cols,
               "dataset": dataset, "config": config}
    return records, mapping


def fetch_arena(dataset, config):
    for split in ("latest", "train", "full"):
        try:
            recs, mp = fetch_one(dataset, config, split)
            return recs, mp, f"{config}/{split}"
        except Exception as e:
            last = f"{config}/{split}: {e}"
    raise RuntimeError(last)


def analyze(records):
    for i, r in enumerate(records, 1):
        r["rank"] = i
        flags = []
        if r["votes"] < VOTES_UNSTABLE:
            flags.append("样本不足·排名不稳定")
        valid_subs = {k: v for k, v in r["subs"].items() if v is not None}
        if valid_subs and r["rank"] <= 20:
            ratios = []
            for k, v in valid_subs.items():
                better = sum(1 for x in records if (x["subs"].get(k) or -1) >= v)
                ratios.append(better / max(1, len(records)))
            avg_ratio = sum(ratios) / len(ratios)
            if r["rank"] / max(1, len(records)) < 0.2 and avg_ratio > 0.5:
                flags.append("总分与分项背离·疑似异常")
        r["flags"] = flags
    return records


# ----------------------------------------------------------------------------
# 页面渲染
# ----------------------------------------------------------------------------
def render_board(records, mapping, tab_id):
    sub_headers = "".join(f"<th>{html.escape(s)}</th>" for s in mapping["subs"])
    rows_html = ""
    for r in records[:TOP_N]:
        flag_html = ""
        if r["flags"]:
            flag_html = ' <span class="warn">⚠ ' + "; ".join(html.escape(f) for f in r["flags"]) + "</span>"
        sub_cells = "".join(
            f"<td>{r['subs'].get(s) if r['subs'].get(s) is not None else '—'}</td>"
            for s in mapping["subs"])
        rank_cls = "top" if r["rank"] <= 3 else ""
        rows_html += (
            f"<tr><td class='{rank_cls}'>{r['rank']}</td>"
            f"<td><b>{html.escape(r['model'])}</b>{flag_html}"
            f"<div class='org'>{html.escape(r['org'])}</div></td>"
            f"<td>{r['score']:.1f}</td>"
            f"<td>{r['votes']:,}</td>{sub_cells}</tr>")
    return f"""
<div id="tab-{tab_id}" class="tabpane">
<table><thead><tr><th>#</th><th>模型</th><th>分数</th><th>票数</th>{sub_headers}</tr></thead>
<tbody>{rows_html}</tbody></table></div>"""


def build_html(boards, status, local=False):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 给前端推荐器用的精简数据（每榜前 10）
    rec = {}
    labels = {}
    for b in boards:
        labels[b["config"]] = b["label"]
        rec[b["config"]] = [{"model": r["model"], "org": r["org"],
                             "score": round(r["score"], 1), "rank": r["rank"]}
                            for r in b["records"][:10]]
    rec_json = json.dumps(rec, ensure_ascii=False)
    labels_json = json.dumps(labels, ensure_ascii=False)

    tabs_html, panes_html = "", ""
    for idx, b in enumerate(boards):
        active = " active" if idx == 0 else ""
        tabs_html += f'<button class="tab{active}" onclick="showTab({idx})">{html.escape(b["label"])}</button>'
        panes_html += render_board(b["records"], b["mapping"], idx)

    status_html = ""
    for s in status:
        cls = "ok" if s["ok"] else "bad"
        status_html += (f"<li><span class='{cls}'>{'✓' if s['ok'] else '✗'}</span> "
                        f"{html.escape(s['name'])}：{html.escape(str(s['detail']))}</li>")

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>模型推荐器 · 你说需求，它给模型</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0d0e10;color:#e6e6e6;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;padding:18px;max-width:680px;margin:0 auto}}
h1{{font-size:20px;margin:0 0 4px}}
.sub{{font-size:12px;color:#9aa0a6;margin:0 0 16px}}
.box{{background:#16181d;border:1px solid #2a2d33;border-radius:14px;padding:16px;margin-bottom:14px}}
input{{width:100%;padding:13px 14px;font-size:15px;border-radius:10px;border:1px solid #2f333a;background:#0d0e10;color:#fff;outline:none}}
input:focus{{border-color:#1f6feb}}
.row{{display:flex;gap:10px;margin-top:10px}}
button.go{{flex:1;background:#1f6feb;color:#fff;border:none;border-radius:10px;padding:13px;font-size:15px;font-weight:700;cursor:pointer}}
button.go:active{{transform:scale(.98)}}
.chips{{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}}
.chip{{background:#1a1c20;color:#c4c8cd;border:1px solid #2a2d33;border-radius:18px;padding:6px 14px;font-size:13px;cursor:pointer}}
.ans{{margin-top:14px;display:none}}
.ans.show{{display:block;animation:fade .3s}}
@keyframes fade{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:none}}}}
.ans h2{{font-size:15px;color:#9aa0a6;margin:0 0 6px;font-weight:600}}
.pick{{font-size:22px;font-weight:800;color:#ffd54a;line-height:1.3}}
.pick small{{font-size:13px;color:#9aa0a6;font-weight:400}}
.top3{{margin:10px 0 0;padding:0;list-style:none}}
.top3 li{{padding:7px 0;border-bottom:1px solid #222428;font-size:14px}}
.top3 li b{{color:#e6e6e6}}
.note{{font-size:12px;color:#c4c8cd;line-height:1.6;background:#16181d;border:1px solid #2a2d33;border-radius:12px;padding:12px 14px;margin-bottom:16px}}
.note b{{color:#8ab4f8}}
details{{margin-top:18px}}
summary{{cursor:pointer;font-size:14px;font-weight:600;color:#c4c8cd;padding:8px 0}}
.tabs{{display:flex;gap:8px;margin:10px 0}}
.tab{{background:#1a1c20;color:#c4c8cd;border:1px solid #2a2d33;border-radius:20px;padding:7px 16px;font-size:13px;font-weight:600;cursor:pointer}}
.tab.active{{background:#1f6feb;color:#fff;border-color:#1f6feb}}
.tabpane{{display:none}}
.tabpane:first-of-type{{display:block}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:9px 8px;border-bottom:1px solid #222428;white-space:nowrap}}
th{{color:#9aa0a6;font-weight:600;font-size:11px;text-transform:uppercase}}
td.top{{color:#ffd54a;font-weight:700}}
.org{{font-size:11px;color:#80868b;margin-top:2px}}
.warn{{color:#ff8a80;font-size:11px;font-weight:600;display:block;margin-top:3px}}
.status{{margin-top:16px;font-size:12px;color:#bdc1c6}}
.status li{{margin:4px 0}}
.ok{{color:#81c995;font-weight:700}}.bad{{color:#ff8a80;font-weight:700}}
.foot{{margin-top:16px;font-size:11px;color:#6b7075;line-height:1.6}}
</style>
</head>
<body>
<h1>模型推荐器</h1>
<div class="sub">更新于 {now} · 数据来自 LMArena 人类盲测榜（匿名对战，最难刷榜）</div>

<div class="note">
<b>怎么用：</b>下面一句话说你想干啥，点「推荐」就行。写代码会自动看<b>代码榜</b>、看图片看<b>视觉榜</b>、写文案看<b>综合榜</b>，
直接给你该类别最靠谱的模型。底下小按钮可手动切换类别。
</div>

<div class="box">
  <input id="q" placeholder="例如：帮我写个周报 / 写个爬虫代码 / 看看这张图里有什么" onkeydown="if(event.key==='Enter')reco()">
  <div class="row"><button class="go" onclick="reco()">推荐模型</button></div>
  <div class="chips">
    <span class="chip" onclick="pick('text')">💬 综合对话</span>
    <span class="chip" onclick="pick('webdev')">💻 代码生成</span>
    <span class="chip" onclick="pick('vision')">👁 视觉理解</span>
  </div>
  <div id="ans" class="ans">
    <h2 id="ansLabel"></h2>
    <div class="pick" id="ansPick"></div>
    <ol class="top3" id="ansTop3"></ol>
  </div>
</div>

<details>
  <summary>想看各榜单完整排名？</summary>
  <div class="tabs">{tabs_html}</div>
  {panes_html}
</details>

<div class="status"><b>数据源状态</b><ul>{status_html}</ul></div>
<div class="foot">本页由 GitHub Actions 每日自动生成 · 数据源 lmarena-ai/leaderboard-dataset（LMArena 官方每日更新）{' · 本地调试模式' if local else ''}</div>

<script>
const BOARDS = {rec_json};
const LABELS = {labels_json};
const TIPS = {{
  text: "综合对话：写文案、聊天、总结、翻译、润色等日常文字活儿",
  webdev: "代码生成：写程序、脚本、爬虫、修 bug、前后端开发",
  vision: "视觉理解：看图、识别物体、读截图/文档图片、OCR"
}};
function classify(t){{
  const kw={{webdev:["代码","编程","写代码","爬虫","程序","脚本","bug","调试","函数","api","前端","后端","算法","开发","代码生成","python","java","js"],
            vision:["图","图片","照片","视觉","识别","截图","ocr","画面","物体","这张"]}};
  for(const a in kw){{ if(kw[a].some(k=>t.includes(k))) return a; }}
  return "text";
}}
function render(arena){{
  const list=BOARDS[arena]||BOARDS["text"];
  const top=list[0];
  document.getElementById("ansLabel").textContent="「"+LABELS[arena]+"」专业榜冠军";
  document.getElementById("ansPick").innerHTML=
    top.model+" <small>("+top.org+" · 榜第"+top.rank+" · 分数"+top.score+")</small>";
  const ol=document.getElementById("ansTop3"); ol.innerHTML="";
  list.slice(0,3).forEach((m,i)=>{{
    const li=document.createElement("li");
    li.innerHTML="<b>"+(i+1)+". "+m.model+"</b> <span style='color:#80868b'>· "+m.org+"</span>";
    ol.appendChild(li);
  }});
  document.getElementById("ans").classList.add("show");
}}
function reco(){{ const t=document.getElementById("q").value.trim(); if(!t)return; render(classify(t)); }}
function pick(a){{ document.getElementById("q").value=""; render(a); }}
</script>
</body>
</html>"""
    return page


# ----------------------------------------------------------------------------
def main():
    local = "--local" in sys.argv
    status = []
    boards = []

    dataset = None
    for cand in CANDIDATES:
        try:
            discover_configs(cand)
            dataset = cand
            break
        except Exception:
            continue
    if not dataset:
        dataset = CANDIDATES[0]

    for config, label in ARENAS:
        try:
            recs, mp, src = fetch_arena(dataset, config)
            analyze(recs)
            boards.append({"config": config, "label": label,
                           "records": recs, "mapping": mp})
            status.append({"name": f"LMArena · {label}", "ok": True,
                           "detail": f"成功解析 {len(recs)} 个模型（{src}）"})
        except Exception as e:
            status.append({"name": f"LMArena · {label}", "ok": False,
                           "detail": f"拉取失败：{e}"})

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "index.html")
    page = build_html(boards, status, local=local)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"[OK] 已生成 {out_path}（榜单 {len(boards)} 个，状态 {sum(s['ok'] for s in status)}/{len(status)}）")
    for s in status:
        print(f"  {'✓' if s['ok'] else '✗'} {s['name']}: {s['detail']}")


if __name__ == "__main__":
    main()
