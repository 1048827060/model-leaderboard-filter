# 模型榜单筛选器（抗刷榜 · 实时刷新）

一个**零服务器、不开发小程序**的模型榜单工具：用 GitHub Actions 每天自动拉取
**LMArena 人类盲测榜**（匿名对战、最难刷榜），做一致性检查并自动标红刷榜嫌疑，
结果发布成一个网页链接，**手机浏览器收藏即可随时看最新排名**。

## 它解决什么

- **不想一个个体验模型**（没指标性）→ 直接用权威盲测榜排名
- **怕榜单刷榜** → 主源用 LMArena 盲测榜（厂商无法干预）；页面自动标记
  「样本不足 / 总分与分项背离」的疑似异常模型
- **不想搭网站/小程序** → Python 跑在 GitHub 免费云，结果只是个静态网页链接
- **手机怎么看** → 手机浏览器打开 Pages 链接，每天自动刷新，无需在手机装 Python

## 三步上线（约 5 分钟）

1. **建仓库**：在 GitHub 新建一个**公开**仓库，把本目录所有文件 push 上去
   （`generate.py`、`.github/workflows/update.yml`、`README.md`）。
2. **开 Pages**：仓库 `Settings → Pages → Build and deployment → Source` 选
   **GitHub Actions**。
3. **跑一次**：`Actions` 标签页点 `更新模型榜单` → `Run workflow`。
   完成后页面顶部会显示你的链接：
   `https://<你的用户名>.github.io/<仓库名>/`

把这个链接**收藏到手机浏览器**，之后每天自动更新。

> 想立刻看效果？在 Actions 页手动 Run workflow 一次即可，不用等定时。

## 本地调试

需要本地能访问 `huggingface.co`：

```bash
python3 generate.py --local
# 生成 ./public/index.html，用浏览器打开查看
```

## 关于「抗刷榜」与数据边界（重要）

- **主源 LMArena**：真实用户盲测对战、模型匿名，厂商无法预知题目或操纵结果，
  是公认的「人类偏好」基准，抗刷榜能力最强。
- **当前为单主源 + 内部一致性检查**。脚本已预留第二独立源（如 OpenCompass /
  Artificial Analysis）的接入位——在 `generate.py` 的 `fetch_lmarena()` 之后
  追加一个返回相同结构 `records` 的源，并在 `main()` 里并入 `analyze()` 即可做
  **跨源交叉验证**（某模型只在一个榜靠前 → 自动标红）。多源交叉是更彻底的抗刷榜，
  但各源 API 端点需你按实际情况填入（见下方「接入第二源」）。
- **标红含义**：
  - `样本不足·排名不稳定`：票数低于阈值（默认 100），统计意义弱。
  - `总分与分项背离·疑似异常`：Overall 进前 20% 但各分项平均只在中游之后。

## 接入第二源（可选增强）

在 `generate.py` 增加类似函数，返回 `[{model, org, score, votes, subs}]` 列表，
于 `main()` 中与 LMArena 结果合并后交给 `analyze()` 做跨源排名一致性检查。
注意各源模型命名不统一，需要做名称归一化（如统一成 "GPT-4o" 这样的标准名）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `generate.py` | 拉取榜单、解析、一致性检查、生成 HTML（零第三方依赖） |
| `.github/workflows/update.yml` | 定时（UTC 0/12 点）+ 手动触发，部署到 Pages |
| `README.md` | 本文档 |
| `public/index.html` | 自动生成的页面（由 Actions 产出，无需手建） |
