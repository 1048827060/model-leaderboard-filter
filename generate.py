#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推荐器（小白版 · 抗刷分 · 实时 · 带性价比）
================================================
你打一句话说想干啥，它直接回你「用哪个模型」，并列出前几名候选。

- 候选列表（不止冠军）、区分国产/国外、算出「性价比榜」。
- 性价比 = 该榜 LMArena 分数 ÷ 参考均价（输入/输出均价，每百万 token 美元）。
  价格未知或开源自部署的模型标「—」，不参与数值排序。
- 背后用 LMArena 官方「人类盲测匿名对战」数据集（综合/代码/视觉三竞技场）。
  真实用户盲测、模型匿名，厂商无法预知题目或干预结果，业界公认最刷不了分。

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

# ---------------------------------------------------------------------------
# 国产 / 国外 识别（按模型名 + 机构名关键字匹配，命中即「国产」）
# ---------------------------------------------------------------------------
DOMESTIC_KW = [
    "qwen", "alibaba", "aliyun", "alimo", "damo", "deepseek", "zhipu", "zhipuai",
    "glm", "chatglm", "internlm", "telechat", "tencent", "hunyuan", "bytedance",
    "doubao", "baidu", "ernie", "moonshot", "kimi", "stepfun", "step-", "minimax",
    "baichuan", "01-ai", "01.ai", "yi-", "iflytek", "spark", "sensetime", "sensechat",
    "shanghai ai lab", "ling", "skywork", "xverse", "wanka", "modelbest", "abab",
    "flagopen", "metax", "taiji", "pangu", "betty", "yuewen", "thudm", "visualglm",
]


def region_of(model, org):
    s = (str(model) + " " + str(org)).lower()
    if any(k in s for k in DOMESTIC_KW):
        return "国产"
    return "国外"


# ---------------------------------------------------------------------------
# 参考价表：模型关键字 -> (输入价, 输出价) 每百万 token 美元（参考值，可能过时）
# 仅用于性价比估算；未知/开源自部署标 None。
# ---------------------------------------------------------------------------
PRICE = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00), "gpt-4.1-nano": (0.10, 0.40), "gpt-5": (10.00, 30.00),
    "o1": (15.00, 60.00), "o3": (10.00, 40.00), "o4": (10.00, 40.00),
    "claude 3.5 sonnet": (3.00, 15.00), "claude 3.7 sonnet": (3.00, 15.00),
    "claude 3.5 haiku": (0.80, 4.00), "claude opus 4": (15.00, 75.00),
    "claude opus 5": (15.00, 75.00), "claude 4": (3.00, 15.00),
    "gemini 1.5 pro": (1.25, 5.00), "gemini 1.5 flash": (0.075, 0.30),
    "gemini 2.0 flash": (0.10, 0.40), "gemini 2.5 pro": (1.25, 10.00),
    "gemini 2.5 flash": (0.30, 2.50),
    "llama 3.1 405b": (2.70, 2.70), "llama 3.1 70b": (0.59, 0.79),
    "llama 3.3 70b": (0.59, 0.79), "llama 4": (0.50, 0.80),
    "mistral large": (2.00, 6.00), "mistral small": (0.20, 0.60),
    "deepseek-v3": (0.27, 1.10), "deepseek-r1": (0.55, 2.19), "deepseek-v2": (0.27, 1.10),
    "qwen2.5-7b": (0.07, 0.14), "qwen2.5-72b": (0.35, 0.70), "qwen-max": (1.60, 6.40),
    "qwen-plus": (0.40, 1.20), "qwen-turbo": (0.20, 0.60), "qwen3": (0.30, 0.90),
    "glm-4": (1.00, 1.00), "glm-4-plus": (1.40, 4.20), "glm-4-air": (0.10, 0.10),
    "glm-z1": (1.40, 4.20), "doubao": (0.07, 0.14), "doubao-pro": (0.14, 0.28),
    "doubao-1.5": (0.10, 0.40), "ernie": (0.50, 0.50), "ernie-4.5": (0.50, 0.50),
    "kimi": (1.00, 1.00), "moonshot": (1.00, 1.00), "minimax": (0.42, 1.05),
    "abab": (0.42, 1.05), "baichuan": (0.50, 0.50), "step": (0.50, 0.50),
    "yi-": (0.50, 0.50), "spark": (0.50, 0.50), "internlm": (0.30, 0.30),
    "phi-4": (0.10, 0.10), "phi-3": (0.10, 0.10), "command": (1.00, 3.00),
    "nova": (0.80, 3.20), "grok": (5.00, 15.00), "jamba": (0.50, 1.50),
}


def price_of(model):
    s = str(model).lower()
    for k, v in PRICE.items():
        if k in s:
            return v
    return None


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
        # —— 新增：国产/国外 + 价格 + 性价比 ——
        r["region"] = region_of(r["model"], r["org"])
        pr = price_of(r["model"])
        if pr:
            r["pin"], r["pout"] = pr
            avg = (pr[0] + pr[1]) / 2
            r["value"] = round(r["score"] / avg, 1) if avg > 0 else None
        else:
            r["pin"] = r["pout"] = None
            r["value"] = None
        r["flags"] = flags
    return records


# ----------------------------------------------------------------------------
# 页面渲染
# ----------------------------------------------------------------------------
def build_html(boards, status, local=False):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 给前端用的精简数据（每榜前 30，含区域/价格/性价比）
    rec = {}
    labels = {}
    for b in boards:
        labels[b["config"]] = b["label"]
        rec[b["config"]] = [{
            "model": r["model"], "org": r["org"], "region": r["region"],
            "score": round(r["score"], 1), "rank": r["rank"],
            "pin": r.get("pin"), "pout": r.get("pout"), "value": r.get("value"),
            "flags": r["flags"],
        } for r in b["records"][:30]]
    rec_json = json.dumps(rec, ensure_ascii=False)
    labels_json = json.dumps(labels, ensure_ascii=False)

    status_html = ""
    for s in status:
        cls = "ok" if s["ok"] else "bad"
        status_html += (f"<li><span class='{cls}'>{'✓' if s['ok'] else '✗'}</span> "
                        f"{html.escape(s['name'])}：{html.escape(str(s['detail']))}</li>")

    page = TEMPLATE
    page = (page.replace("__NOW__", now)
                .replace("__RECORDS__", rec_json)
                .replace("__LABELS__", labels_json)
                .replace("__LOCAL__", " · 本地调试模式" if local else ""))
    return page


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>模型推荐器 · 候选列表 + 性价比榜</title>
<style>
* { box-sizing: border-box }
body { margin:0; background:#0d0e10; color:#e6e6e6; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; padding:18px; max-width:720px; margin:0 auto }
h1 { font-size:20px; margin:0 0 4px }
.sub { font-size:12px; color:#9aa0a6; margin:0 0 14px }
.box { background:#16181d; border:1px solid #2a2d33; border-radius:14px; padding:16px; margin-bottom:14px }
input { width:100%; padding:13px 14px; font-size:15px; border-radius:10px; border:1px solid #2f333a; background:#0d0e10; color:#fff; outline:none }
input:focus { border-color:#1f6feb }
.row { display:flex; gap:10px; margin-top:10px }
button.go { flex:1; background:#1f6feb; color:#fff; border:none; border-radius:10px; padding:13px; font-size:15px; font-weight:700; cursor:pointer }
button.go:active { transform:scale(.98) }
.chips { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap }
.chip { background:#1a1c20; color:#c4c8cd; border:1px solid #2a2d33; border-radius:18px; padding:6px 14px; font-size:13px; cursor:pointer }
.note { font-size:12px; color:#c4c8cd; line-height:1.7; background:#16181d; border:1px solid #2a2d33; border-radius:12px; padding:12px 14px; margin-bottom:14px }
.note b { color:#8ab4f8 }
.ans { margin-top:14px; display:none }
.ans.show { display:block; animation:fade .3s }
@keyframes fade { from { opacity:0; transform:translateY(6px) } to { opacity:1; transform:none } }
.ans h2 { font-size:15px; color:#9aa0a6; margin:0 0 8px; font-weight:600 }
.cand { display:flex; align-items:center; gap:8px; flex-wrap:wrap; padding:10px 0; border-bottom:1px solid #222428 }
.cand .rk { width:20px; height:20px; border-radius:50%; background:#2a2d33; color:#fff; font-size:11px; display:flex; align-items:center; justify-content:center; flex:0 0 auto }
.badge { font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px }
.badge.cn { background:#0f3320; color:#81c995; border:1px solid #1f6f4a }
.badge.fr { background:#0f2740; color:#8ab4f8; border:1px solid #1f4f8f }
.cand b { font-size:15px }
.cand .org { font-size:12px; color:#80868b }
.cand .meta { font-size:12px; color:#9aa0a6; width:100%; padding-left:28px }
.star { color:#ffd54a; font-weight:700; font-size:12px }
.sech { border-top:1px solid #222428; padding-top:2px; margin-top:12px }
.sech:first-of-type { border-top:none; margin-top:6px }
.sect { font-size:13px; font-weight:700; color:#8ab4f8; margin:6px 0 2px; padding-left:2px }
.sect .cnt { color:#6b7075; font-weight:600; font-size:11px; margin-left:6px }
.empty { font-size:12px; color:#6b7075; padding:4px 0 10px }
details { margin-top:18px }
summary { cursor:pointer; font-size:14px; font-weight:600; color:#c4c8cd; padding:8px 0 }
.bar { display:flex; gap:8px; margin:10px 0; flex-wrap:wrap; align-items:center }
.tab { background:#1a1c20; color:#c4c8cd; border:1px solid #2a2d33; border-radius:20px; padding:7px 14px; font-size:13px; font-weight:600; cursor:pointer }
.tab.active { background:#1f6feb; color:#fff; border-color:#1f6feb }
select { background:#0d0e10; color:#e6e6e6; border:1px solid #2f333a; border-radius:8px; padding:7px 10px; font-size:13px }
table { width:100%; border-collapse:collapse; font-size:13px; margin-top:8px }
th,td { text-align:left; padding:9px 8px; border-bottom:1px solid #222428; white-space:nowrap }
th { color:#9aa0a6; font-weight:600; font-size:11px }
td.rk { color:#ffd54a; font-weight:700 }
td.warn { color:#ff8a80; font-size:11px; font-weight:600 }
td.topv { color:#ffd54a; font-weight:700 }
.status { margin-top:16px; font-size:12px; color:#bdc1c6 }
.status li { margin:4px 0 }
.ok { color:#81c995; font-weight:700 } .bad { color:#ff8a80; font-weight:700 }
.foot { margin-top:16px; font-size:11px; color:#6b7075; line-height:1.7 }
</style>
</head>
<body>
<h1>模型推荐器</h1>
<div class="sub">更新于 __NOW__ · LMArena 人类盲测榜（匿名对战，最难刷榜）__LOCAL__</div>

<div class="note">
<b>怎么用：</b>下面一句话说你想干啥，点「推荐模型」。它会按该类别给出<b>三组各前 5 名</b>：
<b>总榜</b>（综合最强）、<b>国外</b>（仅海外模型）、<b>国内</b>（仅国产模型），并标出
价格与<b>性价比</b>（★最划算）。怕判错可点下方类别按钮。<br>
<b>性价比</b> = 该榜分数 ÷ 参考均价（每百万 token 美元，价格取自内置参考表，可能过时；
价格未知/开源自部署标「—」）。想看完整排名和性价比榜，点下方「展开完整榜单」。
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
    <div id="ansList"></div>
  </div>
</div>

<details>
  <summary>展开完整榜单（候选列表 + 国产/国外筛选 + 性价比排序）</summary>
  <div class="bar" id="tabs"></div>
  <div class="bar">
    <span style="font-size:13px;color:#9aa0a6">区域：</span>
    <select id="region" onchange="renderBoard()">
      <option value="all">全部</option>
      <option value="国产">只看国产</option>
      <option value="国外">只看国外</option>
    </select>
    <span style="font-size:13px;color:#9aa0a6">排序：</span>
    <select id="sort" onchange="renderBoard()">
      <option value="score">按分数</option>
      <option value="value">按性价比</option>
    </select>
  </div>
  <table id="board"><thead></thead><tbody></tbody></table>
</details>

<div class="status"><b>数据源状态</b><ul>__STATUS__</ul></div>
<div class="foot">
本页由 GitHub Actions 每日自动生成 · 数据源 lmarena-ai/leaderboard-dataset（LMArena 官方每日更新）<br>
价格为内置参考价（USD/百万 token），仅供性价比估算，实际以厂商官网为准。
</div>

<script>
const BOARDS = __RECORDS__;
const LABELS = __LABELS__;
const TIPS = {
  text: "综合对话：写文案、聊天、总结、翻译、润色等日常文字活儿",
  webdev: "代码生成：写程序、脚本、爬虫、修 bug、前后端开发",
  vision: "视觉理解：看图、识别物体、读截图/文档图片、OCR"
};
const ARENA_TABS = [
  {key:"text", label:"💬 综合对话"},
  {key:"webdev", label:"💻 代码生成"},
  {key:"vision", label:"👁 视觉理解"},
  {key:"value", label:"💰 性价比"},
];
let curArena = "text";

function classify(t){
  const kw={webdev:["代码","编程","写代码","爬虫","程序","脚本","bug","调试","函数","api","前端","后端","算法","开发","代码生成","python","java","js"],
            vision:["图","图片","照片","视觉","识别","截图","ocr","画面","物体","这张"]};
  for(const a in kw){ if(kw[a].some(k=>t.includes(k))) return a; }
  return "text";
}
function priceStr(m){
  if(m.pin==null) return "价格未知/开源";
  return "$"+m.pin+"/"+m.pout+" 每M";
}
function candHtml(m,i,best){
  const badge = m.region==="国产"
    ? "<span class='badge cn'>国产</span>" : "<span class='badge fr'>国外</span>";
  const star = (best && m.model===best.model) ? " <span class='star'>★最划算</span>" : "";
  const flag = (m.flags&&m.flags.length) ? " <span class='star' style='color:#ff8a80'>⚠"+(m.flags[0])+"</span>" : "";
  const val = m.value!=null ? "性价比 "+m.value : "性价比 —";
  return "<div class='cand'><span class='rk'>"+(i+1)+"</span>"+badge
    + "<b>"+m.model+"</b> <span class='org'>"+m.org+"</span>"
    + "<span class='meta'>分数 "+m.score+" · "+priceStr(m)+" · "+val+star+flag+"</span></div>";
}
function section(title, list, best){
  let h="<div class='sech'><div class='sect'>"+title
    + "<span class='cnt'>共 "+list.length+" 个</span></div>";
  if(!list.length){ h+="<div class='empty'>（暂无可推荐模型）</div>"; }
  list.forEach((m,i)=>{ h+=candHtml(m,i,best); });
  h+="</div>";
  return h;
}
function renderReco(a){
  const all=(BOARDS[a]||BOARDS["text"]);
  const total=all.slice(0,5);
  const foreign=all.filter(m=>m.region==="国外").slice(0,5);
  const domestic=all.filter(m=>m.region==="国产").slice(0,5);
  let best=null;
  all.forEach(m=>{ if(m.value!=null && (best==null||m.value>best.value)) best=m; });
  document.getElementById("ansLabel").textContent="「"+LABELS[a]+"」推荐";
  let html="";
  html += section("总榜前 5（综合最强）", total, best);
  html += section("国外前 5（仅海外模型）", foreign, best);
  html += section("国内前 5（仅国产模型）", domestic, best);
  document.getElementById("ansList").innerHTML=html;
  document.getElementById("ans").classList.add("show");
}
function reco(){ const t=document.getElementById("q").value.trim(); if(!t)return; renderReco(classify(t)); }
function pick(a){ document.getElementById("q").value=""; renderReco(a); }

function renderTabs(){
  const el=document.getElementById("tabs"); el.innerHTML="";
  ARENA_TABS.forEach(t=>{
    const b=document.createElement("button");
    b.className="tab"+(t.key===curArena?" active":"");
    b.textContent=t.label;
    b.onclick=()=>{ curArena=t.key; renderTabs(); renderBoard(); };
    el.appendChild(b);
  });
}
function renderBoard(){
  let list;
  if(curArena==="value"){
    list=(BOARDS["text"]||[]).filter(m=>m.value!=null).slice().sort((a,b)=>b.value-a.value);
  } else {
    list=(BOARDS[curArena]||[]).slice();
    const region=document.getElementById("region").value;
    const sort=document.getElementById("sort").value;
    if(region!=="all") list=list.filter(m=>m.region===region);
    if(sort==="value") list=list.filter(m=>m.value!=null).sort((a,b)=>b.value-a.value);
    else list=list.sort((a,b)=>b.score-a.score);
  }
  const head=["#","模型","区域","分数","价格(输入/输出)","性价比"];
  document.querySelector("#board thead").innerHTML="<tr>"+head.map(h=>"<th>"+h+"</th>").join("")+"</tr>";
  let body="";
  list.slice(0,30).forEach((m,i)=>{
    const badge = m.region==="国产"
      ? "<span class='badge cn'>国产</span>" : "<span class='badge fr'>国外</span>";
    const price = m.pin==null ? "—" : "$"+m.pin+"/"+m.pout;
    const val = m.value!=null ? "<td class='topv'>"+m.value+"</td>" : "<td>—</td>";
    const flag = (m.flags&&m.flags.length) ? "<td class='warn'>⚠"+(m.flags[0])+"</td>" : "";
    body += "<tr><td class='rk'>"+(i+1)+"</td><td><b>"+m.model+"</b> <span class='org'>"+m.org+"</span></td>"
      + "<td>"+badge+"</td><td>"+m.score+"</td><td>"+price+"</td>"+val+flag+"</tr>";
  });
  document.querySelector("#board tbody").innerHTML=body;
}
renderTabs();
renderBoard();
</script>
</body>
</html>"""


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
