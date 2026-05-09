# CiteWeave

**一个通过 OpenClaw 操作的本地引文智能系统。**

CiteWeave 会把学术 PDF 转换成可搜索的本地研究系统。OpenClaw 是入口：它帮助部署本地 stack，持续同步 Zotero library，并为用户提供自然语言界面，用于上传、诊断、查询和维护。CiteWeave 负责这个入口背后的本地基础设施：PDF 抽取、引文解析、embedding、Neo4j、Qdrant、GROBID 和研究查询内核。

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## 从这里开始

这份 README 是给人读的：它解释项目是什么、能做什么、各个部分如何配合。

如果你是 **OpenClaw**，或者你正在部署 CiteWeave，请阅读：

- [`docs/openclaw/README.md`](docs/openclaw/README.md) — OpenClaw 操作入口；
- [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md) — 本地部署步骤；
- [`docs/openclaw/PACKAGE_INTERFACE.md`](docs/openclaw/PACKAGE_INTERFACE.md) — actions、接口和查询路由逻辑。

这个拆分是故意的。README 解释产品；`docs/openclaw/` 解释如何操作它。

---

## CiteWeave 能做什么

CiteWeave 会从学术 PDF 构建本地研究数据库。部署完成后，OpenClaw 用户可以让它：

- 持续同步 Zotero library；
- 上传单篇 PDF 或批量导入 PDF 文件夹；
- 在导入前诊断 PDF 抽取质量；
- 构建并查询 Neo4j 引文图谱；
- 构建并查询 Qdrant 语义向量索引；
- 从图谱、向量、作者和 PDF 内容路径检索证据；
- 回答文献、作者、引用、论证和组合性研究问题；
- 检查健康状态、routes、导入进度、未解析引用和查询历史。

典型场景：

- 文献综述；
- 引用追踪；
- 理论来源梳理；
- 作者和论文比较；
- unresolved references 清理；
- “这个观点从哪里来的？”这类研究问题。

---

## 产品模型

CiteWeave 设计为 **OpenClaw Package**。

OpenClaw 负责：

- 面向用户的自然语言入口；
- 部署协同；
- 通过 Docker Compose 拉起本地基础设施；
- Zotero 定时导入自动化；
- 操作选择：upload、sync、diagnose、query、health、progress、telemetry；
- 面向 agent 的 CiteWeave 调用。

CiteWeave 负责：

- Zotero/PDF 导入；
- PDF 解析和引文抽取；
- Neo4j 图存储；
- Qdrant 向量存储；
- 本地或 OpenAI embedding；
- GROBID 元数据与结构抽取；
- 研究查询规划和回答综合。

边界很重要：

> OpenClaw 判断用户想执行什么操作。CiteWeave 判断如何检索和整合研究证据。

OpenClaw 不是数据库层。它是本地研究 stack 的入口和协调层。

---

## 架构

```text
Human researcher
    │
    ▼
OpenClaw
    │  natural-language entrypoint, deployment coordination,
    │  Zotero sync scheduling, operation selection
    ▼
CiteWeave OpenClaw adapter
    │  src/adapters/openclaw_facade.py
    ▼
CiteWeave kernel
    │  upload, diagnose, route, query, progress, telemetry
    ▼
Local research infrastructure
    ├── Docker Compose stack
    │   ├── Neo4j citation graph
    │   ├── Qdrant vector indexes
    │   └── GROBID PDF extraction
    ├── Zotero PDF source
    └── Embeddings
        ├── local SentenceTransformers   默认
        └── OpenAI Embeddings            可选
```

CLI 仍然存在，但它是运维 adapter。它适合验证和调试，不是产品中心。

---

## Zotero 作为持久数据源

正常部署从告诉 OpenClaw 用户的 Zotero library 在哪里开始。之后 CiteWeave 会把该 library 作为持续 PDF 数据源。

OpenClaw 会把路径持久化为：

```env
CITEWEAVE_ZOTERO_LIBRARY_DIR=/path/to/Zotero
```

OpenClaw 先通过 Docker Compose 拉起本地服务层，然后定时调用：

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --json
```

同步脚本会解析 Zotero `storage/`，递归发现 PDF，并委托给 CiteWeave 的可断点续传 batch uploader。这样本地研究数据库会随着 Zotero library 的变化持续增长。

具体部署步骤见 [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md)。

---

## 查询模型

对于普通研究问题，OpenClaw 不应该手动查询 Neo4j 或 Qdrant。它应该把用户的完整研究问题交给 CiteWeave：

```python
facade.query(question, confirmation="continue")
```

然后由 CiteWeave 决定内部使用哪些路径：

- 语义向量检索；
- 图谱引用遍历；
- 作者和论文查找；
- 已抽取 PDF 内容；
- unresolved citation tracking；
- 最终回答综合。

如果用户提出的是运维请求，OpenClaw 才调用对应 action：`upload_pdf`、`batch_upload`、`diagnose_pdf`、`progress`、`health`、`routes`、`query_history` 或 `list_pending_citations`。

完整接口文档见 [`docs/openclaw/PACKAGE_INTERFACE.md`](docs/openclaw/PACKAGE_INTERFACE.md)。

---

## Embeddings

CiteWeave 当前支持两套 embedding 方案：

| Provider | 默认 | Model | Vector size | API key |
|---|---:|---|---:|---|
| `local` | 是 | `all-MiniLM-L6-v2` | 384 | 不需要 |
| `openai` | 否 | `text-embedding-3-small` | 1536 | 需要 |

默认本地模式让安装更自包含。OpenAI embeddings 可以在用户准备迁移向量索引时启用。

切换 provider 会改变向量维度，因此已有 Qdrant collection 需要先重建或迁移，不能直接混用。数据库会记住自己的形状。烦，但合理。

---

## 文档地图

| 文档 | 读者 | 用途 |
|---|---|---|
| [`README.md`](README.md) | 英文读者 | 产品概览和架构 |
| [`README.zh.md`](README.zh.md) | 中文读者 | 中文概览 |
| [`docs/openclaw/README.md`](docs/openclaw/README.md) | OpenClaw / operator | 操作入口 |
| [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md) | OpenClaw / operator | 本地部署和 Zotero sync |
| [`docs/openclaw/PACKAGE_INTERFACE.md`](docs/openclaw/PACKAGE_INTERFACE.md) | OpenClaw / integrator | actions、接口、查询逻辑 |
| [`docs/KERNEL_AND_OPENCLAW.md`](docs/KERNEL_AND_OPENCLAW.md) | 开发者 | kernel / adapter 架构 |

---

## 开发

开发门禁：

```bash
.venv/bin/python -m ruff check src tests scripts/sync_zotero_pdfs.py
.venv/bin/python -m ruff check tests/manual --select F --ignore E501
python3 -m compileall -q src tests scripts/sync_zotero_pdfs.py
.venv/bin/python -m pytest -q
python3 scripts/repo_privacy_audit.py
```

期望隐私审计结果：

```text
PRIVACY_AUDIT_OK
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
