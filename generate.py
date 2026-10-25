#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型榜单筛选器生成器（抗刷榜 · 实时刷新）
================================================
拉取「人类盲测榜 LMArena」为主源，做多分项一致性检查，
自动标记刷榜嫌疑，生成可直接托管到 GitHub Pages 的静态页面。

零第三方依赖（仅用标准库），可在 GitHub Actions 上每日自动运行。

数据源：lmarena-ai/leaderboard-dataset（LMArena 官方每日更新的数据集）
  - 选 text arena 的 latest 一期、过滤 category==overall 得到总榜
  - 用各 category 的分值做「总分与分项背离」抗刷榜检查
若 HuggingFace datasets-server 要求鉴权，可在仓库 Secrets 配置 HF_TOKEN
（Settings → Secrets → Actions → New repository secret）。

用法：
    python3 generate.py            # 生成到 ./public/index.html
    python3 generate.py --local    # 同上加本地调试信息
"""

import urllib.request
import urllib.parse
import json
import ssl
import sys
import os
import datetime
import html

UA = {"User-Agent": "Mozilla/5.0 (model-leaderboard-tool/1.0)"}
HF_BASE = "https://datasets-server.huggingface.co"

# 候选数据源（按顺序尝试，第一个能解析出总榜的即用）
CANDIDATES = [
    "lmarena-ai/leaderboard-dataset",      # 官方每日更新数据集（推荐）
    "lmsys/chatbot_arena_leaderboard",     # 旧版数据集（兜底）
]

TOP_N = 40                 # 展示前 N 名
VOTES_UNSTABLE = 100       # 票数低于此值视为样本不足
ROWS_PER_PAGE = 100        # HF datasets-server 单页上限
MAX_PAGES = 40             # 最多抓 4000 行
# 分项优先级（用于挑出要展示/参与背离检查的能力维度）
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
        nl = n.lower()
        if any(k in nl for k in keys):
            return n
    return None


def discover_configs(dataset):
    """返回 [(config, split), ...]，优先 text arena 的 latest 一期。"""
    ds = urllib.parse.quote(dataset)
    try:
        s = _json(f"{HF_BASE}/splits?dataset={ds}")
        raw = [(x.get("config"), x.get("split") or "train")
               for x in s.get("splits", []) if x.get("config")]
    except Exception:
        raw = []
    # 同一 config 只保留 latest（若没有则保留原 split）
    best = {}
    for cfg, sp in raw:
        if cfg not in best or sp == "latest":
            best[cfg] = sp
    pairs = list(best.items())
    prio = {"text": 0, "text_style_control": 1, "vision": 2,
            "vision_style_control": 3, "webdev": 4, "search": 10,
            "document": 10, "agent": 20, "text_to_image": 30,
            "image_edit": 30, "text_to_video": 30, "image_to_video": 30,
            "video_edit": 30, "default": 9}
    pairs.sort(key=lambda p: prio.get((p[0] or "").lower(), 15))
    return pairs or [(None, "train")]


def fetch_one(dataset, config, split):
    """解析单个 (dataset, config, split)；成功返回 (records, mapping)。"""
    ds = urllib.parse.quote(dataset)
    base = f"{HF_BASE}/rows?dataset={ds}"
    if config:
        base += "&config=" + urllib.parse.quote(str(config))
    base += f"&split={urllib.parse.quote(split)}"

    # 分页抓取（HF 单页上限 100）
    all_rows = []
    features = None
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

    # 按 category 透视：model -> {category: score}
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

    # 选总榜行
    if col_cat:
        board = [r for r in all_rows
                 if str(r.get(col_cat)) in ("overall", "Overall", "ALL", "all")]
        if not board:
            board = all_rows
    else:
        board = all_rows

    # 确定要展示的分项维度（取频率最高的若干 category，排除 overall）
    from collections import Counter
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
        records.append({
            "model": m,
            "org": str(row.get(col_org) or ""),
            "score": score,
            "votes": votes,
            "subs": subs,
        })

    if len(records) < 3:
        raise ValueError(f"仅解析到 {len(records)} 条总榜记录")

    records.sort(key=lambda x: x["score"], reverse=True)
    mapping = {"model": col_model, "org": col_org, "score": col_score,
               "rank": col_rank, "votes": col_votes, "subs": sub_cols,
               "dataset": dataset, "config": config}
    return records, mapping


def fetch_lmarena():
    """依次尝试候选数据源，返回 (records, mapping, source_label)。"""
    last_err = None
    for dataset in CANDIDATES:
        try:
            pairs = discover_configs(dataset)
        except Exception as e:
            last_err = f"{dataset}: 结构探测失败 {e}"
            continue
        for config, split in pairs:
            try:
                recs, mp = fetch_one(dataset, config, split)
                label = dataset + (f" (config={config}, split={split})" if config else "")
                return recs, mp, label
            except Exception as e:
                last_err = f"{dataset}/{config}/{split}: {e}"
                continue
    raise RuntimeError(last_err or "所有候选数据源均不可用")


# ----------------------------------------------------------------------------
# 抗刷榜一致性检查
# ----------------------------------------------------------------------------
def analyze(records):
    """给每条记录计算排名与刷榜嫌疑标记。"""
    for i, r in enumerate(records, 1):
        r["rank"] = i
        flags = []

        if r["votes"] < VOTES_UNSTABLE:
            flags.append("样本不足·排名不稳定")

        # 分项背离：Overall 进前列，但各分项普遍显著靠后
        valid_subs = {k: v for k, v in r["subs"].items() if v is not None}
        if valid_subs and r["rank"] <= 20:
            ratios = []
            for k, v in valid_subs.items():
                better = sum(1 for x in records if (x["subs"].get(k) or -1) >= v)
                ratios.append(better / max(1, len(records)))
            avg_ratio = sum(ratios) / len(ratios)
            # ratio 越小=分项越靠前(好)，越大=越靠后(差)
            # overall 进前 20%，但分项平均仍落在后 50% 之后 → 背离
            if r["rank"] / max(1, len(records)) < 0.2 and avg_ratio > 0.5:
                flags.append("总分与分项背离·疑似异常")

        r["flags"] = flags

    return records


# ----------------------------------------------------------------------------
# 生成 HTML（移动端友好 · 深色）
# ----------------------------------------------------------------------------
def build_html(records, mapping, status, local=False):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sub_headers = "".join(f"<th>{html.escape(s)}</th>" for s in mapping["subs"])

    rows_html = ""
    for r in records[:TOP_N]:
        flag_html = ""
        if r["flags"]:
            flag_html = ' <span class="warn">⚠ ' + "; ".join(html.escape(f) for f in r["flags"]) + "</span>"
        sub_cells = ""
        for s in mapping["subs"]:
            v = r["subs"].get(s)
            sub_cells += f"<td>{v if v is not None else '—'}</td>"
        rank_cls = "top" if r["rank"] <= 3 else ""
        rows_html += (
            f"<tr><td class='{rank_cls}'>{r['rank']}</td>"
            f"<td><b>{html.escape(r['model'])}</b>{flag_html}"
            f"<div class='org'>{html.escape(r['org'])}</div></td>"
            f"<td>{r['score']:.1f}</td>"
            f"<td>{r['votes']:,}</td>"
            f"{sub_cells}</tr>"
        )

    status_html = ""
    for s in status:
        cls = "ok" if s["ok"] else "bad"
        detail = html.escape(str(s["detail"]))
        status_html += f"<li><span class='{cls}'>{'✓' if s['ok'] else '✗'}</span> {html.escape(s['name'])}：{detail}</li>"

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>模型榜单筛选器 · 抗刷榜实时榜</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#0d0e10;color:#e6e6e6;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;padding:16px;}}
h1{{font-size:19px;margin:0 0 4px}}
.sub{{font-size:12px;color:#9aa0a6;margin:0 0 16px}}
.note{{background:#16181d;border:1px solid #2a2d33;border-radius:12px;padding:12px 14px;font-size:12px;color:#c4c8cd;line-height:1.6;margin-bottom:16px}}
.note b{{color:#8ab4f8}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:9px 8px;border-bottom:1px solid #222428;white-space:nowrap}}
th{{color:#9aa0a6;font-weight:600;font-size:11px;text-transform:uppercase;position:sticky;top:0;background:#0d0e10}}
td.top{{color:#ffd54a;font-weight:700}}
.org{{font-size:11px;color:#80868b;margin-top:2px}}
.warn{{color:#ff8a80;font-size:11px;font-weight:600;display:block;margin-top:3px}}
.status{{margin-top:18px;font-size:12px;color:#bdc1c6}}
.status li{{margin:4px 0}}
.ok{{color:#81c995;font-weight:700}}
.bad{{color:#ff8a80;font-weight:700}}
.foot{{margin-top:18px;font-size:11px;color:#6b7075;line-height:1.6}}
</style>
</head>
<body>
<h1>模型榜单筛选器</h1>
<div class="sub">更新于 {now} · 主源 LMArena 人类盲测榜（匿名对战，最难刷榜）</div>

<div class="note">
<b>为什么信这个榜？</b> LMArena 由真实用户盲测对战、模型名匿名，厂商无法预知题目或干预结果，是公认的「人类偏好」基准，抗刷榜能力最强。<br>
<b>怎么看「⚠」标记？</b> 标红代表：①票数过少、统计不稳；②总分高但各分项普遍靠后（总分与分项背离，疑似异常）。遇到标红请谨慎对待。<br>
<b>数据源：</b> {html.escape(str(mapping.get('dataset', 'LMArena')))}（text arena · 总榜）。若页面提示鉴权失败，可在仓库 Settings → Secrets → Actions 添加 <code>HF_TOKEN</code>。
</div>

<table>
<thead><tr><th>#</th><th>模型</th><th>分数</th><th>票数</th>{sub_headers}</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>

<div class="status"><b>数据源状态</b><ul>{status_html}</ul></div>
<div class="foot">本页由 GitHub Actions 每日自动生成 · 字段映射：model={html.escape(str(mapping.get('model','?')))} / score={html.escape(str(mapping.get('score','?')))} / votes={html.escape(str(mapping.get('votes','?')))}{' · 本地调试模式' if local else ''}</div>
</body>
</html>"""
    return page


# ----------------------------------------------------------------------------
def main():
    local = "--local" in sys.argv
    status = []
    records = []
    mapping = {"model": "model", "org": "org", "score": "score",
               "rank": "rank", "votes": "votes", "subs": []}

    try:
        recs, mp, label = fetch_lmarena()
        records = recs
        mapping = mp
        status.append({"name": "LMArena 盲测榜", "ok": True,
                       "detail": f"成功解析 {len(records)} 个模型（源：{label}；分数列：{mp['score']}）"})
    except Exception as e:
        status.append({"name": "LMArena 盲测榜", "ok": False, "detail": f"拉取失败：{e}"})

    if records:
        analyze(records)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "index.html")
    page = build_html(records, mapping, status, local=local)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"[OK] 已生成 {out_path}（模型 {len(records)} 条，状态 {sum(s['ok'] for s in status)}/{len(status)}）")
    for s in status:
        print(f"  {'✓' if s['ok'] else '✗'} {s['name']}: {s['detail']}")


if __name__ == "__main__":
    main()
