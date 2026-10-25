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
# 国产 / 国外 识别
# 分层判定：① 机构名(org)硬规则（最可靠，org 字段是厂商自己登记的）
#            ② 模型名 + 机构名关键词回退
# 曾经的事故：Meta 的模型系列叫 "Muse Spark"，被关键词 "spark"（讯飞星火）
# 误判成国产；小米 MiMo 没进关键词表被判成国外。故 org 硬规则优先。
# ---------------------------------------------------------------------------

# 机构级：已知国外厂商（org 命中即「国外」，优先级最高）
FOREIGN_ORGS = [
    "meta", "google", "deepmind", "alphabet", "anthropic", "openai", "xai",
    "x.ai", "mistral", "microsoft", "msft", "cohere", "ai21", "amazon",
    "aws", "bedrock", "nvidia", "databricks", "stability", "perplexity",
    "inflection", "ai2", "allen ai", "allenai", "eleuther", "bigcode",
    "huggingface", "together ai", "anyscale", "snowflake", "salesforce",
    "apple", "ibm", "intel", "amd", "reka", "writer", "liquid ai",
]

# 机构级：已知国产厂商（org 命中即「国产」）
DOMESTIC_ORGS = [
    "alibaba", "aliyun", "damo", "alimo", "tencent", "hunyuan", "baidu",
    "bytedance", "volcengine", "doubao", "zhipu", "zhipuai", "zai",
    "moonshot", "deepseek", "01.ai", "01-ai", "stepfun", "minimax",
    "iflytek", "sensetime", "shanghai ai lab", "shanghaiai", "openbmb",
    "modelbest", "internlm", "thudm", "skywork", "xverse", "wanka",
    "baichuan", "xiaomi", "mimo", "metax", "flagopen",
]

# 模型名关键词：只保留高置信度、不易误伤模型的词
# （已剔除 "spark"——会误伤 Meta Muse Spark；"ling"——过短易误伤）
DOMESTIC_KW = [
    "qwen", "alibaba", "aliyun", "alimo", "damo", "deepseek", "zhipu", "zhipuai",
    "glm", "chatglm", "z-ai", "zai", "internlm", "telechat", "tencent", "hunyuan",
    "bytedance", "doubao", "baidu", "ernie", "moonshot", "kimi", "stepfun",
    "minimax", "baichuan", "01-ai", "01.ai", "yi-", "iflytek", "星火", "讯飞",
    "sensetime", "sensechat", "shanghai ai lab", "skywork", "xverse", "wanka",
    "modelbest", "abab", "flagopen", "metax", "taiji", "pangu", "betty",
    "yuewen", "thudm", "visualglm", "xiaomi", "mimo", "volcengine", "openbmb",
]


def region_of(model, org):
    o = str(org or "").lower().strip()
    s = (str(model) + " " + str(org)).lower()
    # ① 机构硬规则：国外厂商优先（防止 Muse Spark 之类被模型名误伤）
    if o and any(k in o for k in FOREIGN_ORGS):
        return "国外"
    # ② 机构硬规则：国产厂商
    if o and any(k in o for k in DOMESTIC_ORGS):
        return "国产"
    # ③ 模型名关键词回退
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
    "abab": (0.42, 1.05),     "baichuan": (0.50, 0.50), "step": (0.50, 0.50),
    "yi-": (0.50, 0.50), "internlm": (0.30, 0.30),
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
                .replace("__STATUS__", status_html)
                .replace("__LOCAL__", " · 本地调试模式" if local else ""))
    return page


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>模型推荐器</title>
<style>
:root{
  --bg:#0a0b0d; --surface:#121317; --surface-2:#17181d; --surface-3:#23262c;
  --line:#22242a; --line-soft:#191b20;
  --text:#edeff2; --text-2:#9aa0a8; --text-3:#696f77;
  --accent:#4b8ef7; --accent-soft:rgba(75,142,247,.13); --accent-line:rgba(75,142,247,.34);
  --cn:#4ec98e; --cn-bg:rgba(78,201,142,.11);
  --fr:#7fa8ef; --fr-bg:rgba(127,168,239,.11);
  --gold:#f0c35c; --gold-bg:rgba(240,195,92,.11);
  --warn:#ef8d7e; --warn-bg:rgba(239,141,126,.11);
  --r-sm:8px; --r-md:12px; --r-lg:16px; --r-pill:999px;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--text);line-height:1.55;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased;padding:30px 18px 44px;max-width:660px;margin:0 auto}

header h1{font-size:21px;font-weight:700;letter-spacing:-.012em;margin:0}
.stamp{font-size:11.5px;color:var(--text-3);margin:7px 0 0;letter-spacing:.01em}

.hint{font-size:12.5px;color:var(--text-3);line-height:1.9;margin:22px 0 24px;padding-top:20px;border-top:1px solid var(--line-soft)}
.hint b{color:var(--text-2);font-weight:600}

.search{margin:0}
.field{display:flex;align-items:center;gap:10px;background:var(--surface);border:1px solid var(--line);
  border-radius:var(--r-lg);padding:5px 5px 5px 16px;transition:border-color .16s}
.field:focus-within{border-color:var(--accent-line)}
.field input{flex:1;min-width:0;background:transparent;border:none;outline:none;color:var(--text);
  font-size:15px;padding:11px 0;font-family:inherit}
.field input::placeholder{color:var(--text-3)}
button.go{flex:0 0 auto;background:var(--accent);color:#fff;border:none;border-radius:11px;
  padding:10px 17px;font-size:13.5px;font-weight:600;cursor:pointer;font-family:inherit;transition:opacity .15s}
button.go:hover{opacity:.9}
button.go:active{transform:scale(.97)}
.chips{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
.chip{background:var(--surface);color:var(--text-2);border:1px solid var(--line);border-radius:var(--r-pill);
  padding:7px 15px;font-size:12.5px;cursor:pointer;transition:.15s;user-select:none}
.chip:hover{color:var(--text)}
.chip.active{background:var(--accent-soft);color:#9cc3ff;border-color:var(--accent-line)}

.ans{margin-top:30px;display:none}
.ans.show{display:block;animation:fade .28s ease}
@keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.eyebrow{font-size:11.5px;color:var(--text-3);font-weight:600;letter-spacing:.08em;margin-bottom:14px}
.seg{display:inline-flex;background:var(--surface-2);border:1px solid var(--line);border-radius:var(--r-md);
  padding:3px;gap:2px;margin-bottom:20px}
.segbtn{background:transparent;border:none;color:var(--text-2);border-radius:9px;padding:7px 15px;
  font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit;transition:.15s;white-space:nowrap}
.segbtn:hover{color:var(--text)}
.segbtn.active{background:var(--surface-3);color:#fff;box-shadow:0 1px 3px rgba(0,0,0,.35)}

.sect{font-size:15px;font-weight:650;color:var(--text);margin:0 0 4px;letter-spacing:-.005em}
.sect .cnt{font-size:11px;color:var(--text-3);font-weight:500;margin-left:8px;letter-spacing:0}
.empty{font-size:12.5px;color:var(--text-3);padding:8px 0 4px}
.cand{display:grid;grid-template-columns:26px 1fr;gap:0 11px;padding:14px 0;border-bottom:1px solid var(--line-soft);align-items:start}
.cand:last-child{border-bottom:none}
.cand .rk{width:24px;height:24px;border-radius:50%;background:var(--surface-2);color:var(--text-3);
  font-size:11.5px;font-weight:600;display:flex;align-items:center;justify-content:center;
  font-variant-numeric:tabular-nums;margin-top:1px}
.cand .rk.top{color:var(--gold);background:var(--gold-bg)}
.cand .body{min-width:0}
.cand .line1{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.cand .line1 b{font-size:14.5px;font-weight:600;color:var(--text);letter-spacing:-.005em}
.cand .org{font-size:11.5px;color:var(--text-3)}
.cand .meta{margin-top:6px;font-size:12px;color:var(--text-2);display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.cand .meta i{color:var(--text-3);font-style:normal}
.badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:var(--r-pill);flex:0 0 auto;letter-spacing:.01em}
.badge.cn{color:var(--cn);background:var(--cn-bg)}
.badge.fr{color:var(--fr);background:var(--fr-bg)}
.star{color:var(--gold);background:var(--gold-bg);border-radius:var(--r-pill);padding:2px 8px;
  font-size:11px;font-weight:600;flex:0 0 auto}
.flag{color:var(--warn);background:var(--warn-bg);border-radius:var(--r-pill);padding:2px 8px;
  font-size:11px;font-weight:500}

details{margin-top:30px;border-top:1px solid var(--line-soft);padding-top:18px}
summary{cursor:pointer;font-size:13px;font-weight:600;color:var(--text-2);list-style:none;
  display:flex;align-items:center;gap:9px;user-select:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"";width:5px;height:5px;border-right:1.5px solid var(--text-3);
  border-bottom:1.5px solid var(--text-3);transform:rotate(-45deg);transition:transform .18s;flex:0 0 auto}
details[open] summary::before{transform:rotate(45deg)}
.bar{display:flex;gap:8px;margin:16px 0 6px;flex-wrap:wrap;align-items:center}
.bar .fl{font-size:11.5px;color:var(--text-3);margin-right:-3px}
.tab{background:var(--surface);color:var(--text-2);border:1px solid var(--line);border-radius:var(--r-pill);
  padding:7px 14px;font-size:12.5px;font-weight:600;cursor:pointer;font-family:inherit;transition:.15s}
.tab:hover{color:var(--text)}
.tab.active{background:var(--accent-soft);color:#9cc3ff;border-color:var(--accent-line)}
select{background:var(--surface);color:var(--text-2);border:1px solid var(--line);border-radius:var(--r-sm);
  padding:7px 10px;font-size:12.5px;outline:none;font-family:inherit;cursor:pointer}
select:focus{border-color:var(--accent-line)}

table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
th{font-size:10.5px;color:var(--text-3);font-weight:600;text-align:left;padding:10px 8px;
  border-bottom:1px solid var(--line);letter-spacing:.06em;white-space:nowrap}
td{padding:11px 8px;border-bottom:1px solid var(--line-soft);white-space:nowrap;vertical-align:middle}
tbody tr:last-child td{border-bottom:none}
td.rk{color:var(--text-3);font-weight:600;font-variant-numeric:tabular-nums}
td.rk.top{color:var(--gold)}
td.m b{font-weight:600}
td.m .org{font-size:11px;color:var(--text-3);margin-left:5px}
td.num{font-variant-numeric:tabular-nums}
td.dim{color:var(--text-3)}
td.topv{color:var(--gold);font-weight:600;font-variant-numeric:tabular-nums}
td.warn{color:var(--warn);font-size:11px;font-weight:500}

.status{margin-top:26px;padding-top:18px;border-top:1px solid var(--line-soft)}
.status-title{font-size:11px;color:var(--text-3);font-weight:600;letter-spacing:.08em}
.status ul{list-style:none;padding:0;margin:9px 0 0;font-size:11.5px;color:var(--text-3);line-height:1.9}
.ok{color:var(--cn);font-weight:700;margin-right:2px}
.bad{color:var(--warn);font-weight:700;margin-right:2px}
.foot{margin-top:20px;font-size:11px;color:#54595f;line-height:1.9}
</style>
</head>
<body>
<header>
  <h1>模型推荐器</h1>
  <p class="stamp">更新于 __NOW__ · LMArena 人类盲测榜（匿名对战，最难刷榜）__LOCAL__</p>
</header>

<div class="hint">
一句话说你想干啥，点<b>推荐</b>。结果按<b>全部 / 国内 / 国外</b>切换查看各前 5 名，标出价格与<b>性价比</b>（★ 最划算）。<br>
性价比 = 该榜分数 ÷ 参考均价（每百万 token 美元，内置参考价可能过时；未知或开源自部署标「—」）。
完整排名与性价比榜见下方「展开完整榜单」。
</div>

<div class="search">
  <div class="field">
    <input id="q" placeholder="想干啥？例如：帮我写个周报 / 写个爬虫 / 看看这张图" onkeydown="if(event.key==='Enter')reco()">
    <button class="go" onclick="reco()">推荐</button>
  </div>
  <div class="chips">
    <span class="chip" onclick="pick('text')">综合对话</span>
    <span class="chip" onclick="pick('webdev')">代码生成</span>
    <span class="chip" onclick="pick('vision')">视觉理解</span>
  </div>
</div>

<div id="ans" class="ans">
  <div class="eyebrow" id="ansLabel"></div>
  <div class="seg" id="recoSeg"></div>
  <div id="ansList"></div>
</div>

<details>
  <summary>展开完整榜单与性价比榜</summary>
  <div class="bar" id="tabs"></div>
  <div class="bar">
    <span class="fl">区域</span>
    <select id="region" onchange="renderBoard()">
      <option value="all">全部</option>
      <option value="国产">只看国产</option>
      <option value="国外">只看国外</option>
    </select>
    <span class="fl">排序</span>
    <select id="sort" onchange="renderBoard()">
      <option value="score">按分数</option>
      <option value="value">按性价比</option>
    </select>
  </div>
  <div class="table-wrap"><table id="board"><thead></thead><tbody></tbody></table></div>
</details>

<div class="status">
  <div class="status-title">数据源状态</div>
  <ul>__STATUS__</ul>
</div>
<div class="foot">
  GitHub Actions 每日自动生成 · 数据源 LMArena 官方盲测榜<br>
  价格为内置参考价（USD / 百万 token），仅供性价比估算，实际以厂商官网为准
</div>

<script>
const BOARDS = __RECORDS__;
const LABELS = __LABELS__;
const ARENA_TABS = [
  {key:"text", label:"综合对话"},
  {key:"webdev", label:"代码生成"},
  {key:"vision", label:"视觉理解"},
  {key:"value", label:"性价比"},
];
let curArena = "text";

function classify(t){
  const kw={webdev:["代码","编程","写代码","爬虫","程序","脚本","bug","调试","函数","api","前端","后端","算法","开发","python","java","js"],
            vision:["图","图片","照片","视觉","识别","截图","ocr","画面","物体","这张"]};
  for(const a in kw){ if(kw[a].some(k=>t.includes(k))) return a; }
  return "text";
}
function priceStr(m){
  if(m.pin==null) return "价格未知 / 开源";
  return "$"+m.pin+" / $"+m.pout+" 每M";
}
function candHtml(m,i,best){
  const badge = m.region==="国产"
    ? "<span class='badge cn'>国产</span>" : "<span class='badge fr'>国外</span>";
  const star = (best && m.model===best.model) ? "<span class='star'>★ 最划算</span>" : "";
  const flag = (m.flags&&m.flags.length) ? "<span class='flag'>⚠ "+m.flags[0]+"</span>" : "";
  const val = m.value!=null ? "性价比 "+m.value : "性价比 —";
  return "<div class='cand'><span class='rk"+(i===0?" top":"")+"'>"+(i+1)+"</span>"
    + "<div class='body'><div class='line1'>"+badge+"<b>"+m.model+"</b>"
    + "<span class='org'>"+m.org+"</span>"+star+"</div>"
    + "<div class='meta'><span>分数 "+m.score+"</span><i>·</i><span>"+priceStr(m)
    + "</span><i>·</i><span>"+val+"</span>"+flag+"</div></div></div>";
}
const RECO_SEGS=[
  {k:"all",      t:"全部 · 总榜", title:"总榜前 5（综合最强）"},
  {k:"domestic", t:"国内",        title:"国内前 5（国产模型）"},
  {k:"foreign",  t:"国外",        title:"国外前 5（海外模型）"},
];
let recoRegion="all", recoData={};
function renderSeg(){
  document.getElementById("recoSeg").innerHTML=RECO_SEGS.map(s=>
    "<span class='segbtn"+(s.k===recoRegion?" active":"")+"' onclick='setRecoRegion(\""+s.k+"\")'>"+s.t+"</span>"
  ).join("");
}
function setRecoRegion(k){ recoRegion=k; renderSeg(); renderRecoList(); }
function renderRecoList(){
  const seg=RECO_SEGS.find(s=>s.k===recoRegion)||RECO_SEGS[0];
  const list=recoData[recoRegion]||[];
  let best=null;
  list.forEach(m=>{ if(m.value!=null && (best==null||m.value>best.value)) best=m; });
  let html="<div class='sect'>"+seg.title+"<span class='cnt'>共 "+list.length+" 个</span></div>";
  if(!list.length){ html+="<div class='empty'>暂无可推荐模型</div>"; }
  list.forEach((m,i)=>{ html+=candHtml(m,i,best); });
  document.getElementById("ansList").innerHTML=html;
}
function renderReco(a){
  const all=(BOARDS[a]||BOARDS["text"]);
  recoData={
    all:      all.slice(0,5),
    foreign:  all.filter(m=>m.region==="国外").slice(0,5),
    domestic: all.filter(m=>m.region==="国产").slice(0,5),
  };
  recoRegion="all";
  document.getElementById("ansLabel").textContent="「"+LABELS[a]+"」推荐";
  renderSeg();
  renderRecoList();
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
  const head=["#","模型","区域","分数","价格(输入/输出)","性价比","备注"];
  document.querySelector("#board thead").innerHTML="<tr>"+head.map(h=>"<th>"+h+"</th>").join("")+"</tr>";
  let body="";
  list.slice(0,30).forEach((m,i)=>{
    const badge = m.region==="国产"
      ? "<span class='badge cn'>国产</span>" : "<span class='badge fr'>国外</span>";
    const price = m.pin==null ? "—" : "$"+m.pin+"/"+m.pout;
    const val = m.value!=null ? "<td class='topv'>"+m.value+"</td>" : "<td class='dim'>—</td>";
    const flag = (m.flags&&m.flags.length) ? "<td class='warn'>⚠ "+m.flags[0]+"</td>" : "<td class='dim'></td>";
    body += "<tr><td class='rk"+(i===0?" top":"")+"'>"+(i+1)+"</td>"
      + "<td class='m'><b>"+m.model+"</b><span class='org'>"+m.org+"</span></td>"
      + "<td>"+badge+"</td><td class='num'>"+m.score+"</td><td class='dim'>"+price+"</td>"+val+flag+"</tr>";
  });
  document.querySelector("#board tbody").innerHTML=body;
}
renderTabs();
renderBoard();
</script>
</body>
</html>
"""


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
