# CiteWeave

**把 Zotero 文献库变成本地引文智能系统。**

学术 PDF 里有论点、引用、方法、理论脉络和大量上下文。问题是，这些结构通常都被压在文件里面，只有人一篇篇读的时候才会浮出来。CiteWeave 做的事情就是把这些结构抽出来、索引起来，并让它们可以被查询：不是又一个聊天壳，而是一套本地研究 stack，包括引文图谱、语义向量索引、citation context 和面向学术问题的查询内核。

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## 这是什么

CiteWeave 会导入学术 PDF，并围绕它们构建一个本地研究数据库：

- 用 GROBID 和本地 PDF parser 做 **PDF 抽取**；
- 把正文引用和参考文献对应起来，做 **citation parsing**；
- 用 **Neo4j** 存 papers、citations、paragraphs、sentences 和关系；
- 用 **Qdrant** 为句子、段落、章节和引用文本建立语义向量索引；
- 支持本地 SentenceTransformers 或 API-based embedding；
- 用 research query kernel 组合图谱、向量、作者、引用和 PDF 内容路径。

你可以问它：

- “哪些论文继承了 Teece 1997？它们具体怎么用的？”
- “找一下 multimarket competition 的文献，并总结主要理论分歧。”
- “这篇论文讨论 competitive response 的时候引用了谁？”
- “我的文献库里有哪些 unresolved references 还缺 PDF？”

重点是：CiteWeave 不只是保存 PDF。它试图保留 PDF 里的学术结构。PDF 文件夹是一堆文件；带 citation context 的图谱和向量索引，才是研究工具。差别很大，虽然目录看起来都叫 `papers/`。

---

## OpenClaw 在这里扮演什么角色

CiteWeave 设计成通过 [OpenClaw](https://github.com/openclaw/openclaw) 操作的本地研究 package。

OpenClaw 负责对话入口和运维循环：部署检查、Zotero 同步调度、上传请求、进度监控、诊断，以及用户提出的研究问题。

CiteWeave 负责真正的研究机器：导入、抽取、引文分析、图数据库/向量数据库存储，以及最终回答综合。

一个好记的边界是：

> OpenClaw 判断用户想做什么；CiteWeave 判断研究证据应该怎么找。

对于普通研究问题，OpenClaw 应该把完整问题交给 CiteWeave，而不是自己手动去查 Neo4j 或 Qdrant。

```python
facade.query("Which papers discuss platform competition?", confirmation="continue")
```

---

## 它怎么工作

```text
Zotero library / PDF folder
        │
        ▼
PDF extraction + metadata
        │
        ▼
Citation and structure parsing
        │
        ├── Neo4j graph
        │     papers, citations, paragraphs, sentences, relationships
        │
        ├── Qdrant vector indexes
        │     sentences, paragraphs, sections, citations
        │
        └── Processed local artifacts
              metadata, JSONL, original PDFs, diagnostics
        │
        ▼
Research query kernel
        │
        ▼
OpenClaw conversational interface
```

CLI 仍然保留，用于维护和调试；但推荐的产品路径是 OpenClaw facade。

---

## 可以拿它做什么

### 持续索引 Zotero 文献库

把 CiteWeave 指向 Zotero library，它会递归发现 `storage/` 下的 PDF，处理它们，并跟踪可恢复进度。

```env
CITEWEAVE_ZOTERO_LIBRARY_DIR=/path/to/Zotero
```

```bash
.venv/bin/python scripts/sync_zotero_pdfs.py --source "$CITEWEAVE_ZOTERO_LIBRARY_DIR" --json
```

### 上传或诊断单篇 PDF

```bash
.venv/bin/citeweave upload ./papers/example.pdf
.venv/bin/citeweave diagnose ./papers/example.pdf
```

### 提研究问题

```bash
.venv/bin/citeweave query "Which papers discuss competitive dynamics and platform strategy?"
```

### 查看导入进度

```bash
.venv/bin/citeweave progress /path/to/Zotero/storage
```

---

## 本地基础设施

典型本地部署会用 Docker Compose 拉起：

- **Neo4j** — 引文图谱和结构化研究实体；
- **Qdrant** — 语义向量检索；
- **GROBID** — 学术 PDF 元数据和参考文献抽取。

具体部署见 [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md)。简版流程：

```bash
bash scripts/bootstrap_openclaw.sh
bash scripts/deploy_local_stack.sh
bash scripts/deployment_check.sh
```

除非你在排查问题，否则不要手动拼这些服务。未来的你已经够忙了，别再给他留谜题。

---

## Embeddings 与向量重建

CiteWeave 支持本地 SentenceTransformers embedding，也支持 OpenAI-compatible embedding provider。当前本地配置可以通过 `config/qdrant_config.json` 和环境变量选择，例如：

```env
CITEWEAVE_EMBEDDING_PROVIDER=local
CITEWEAVE_EMBEDDING_PROFILE=bge_large_en
CITEWEAVE_EMBEDDING_DEVICE=auto
```

重要规则：

> 只要更换基础 embedding model、provider 或 vector dimension，就必须重建向量索引并全量重跑语料。

不要在换 embedding 后继续沿用旧 progress 做 `--resume`。旧向量属于旧 embedding space。即使两个模型输出的维度碰巧一样，它们的距离也不能比较。把它们混进同一个 Qdrant collection，会让检索结果以一种很安静、很昂贵的方式坏掉。

正确流程：

1. 停止正在运行的 ingest。
2. 更新并确认 embedding 配置。
3. 重建或迁移 Qdrant collections。
4. 清空受影响数据源的 batch progress。
5. 从头全量 re-ingest。
6. 跑代表性查询，确认索引可信。

更多细节见 [`docs/openclaw/DEPLOYMENT.md#9-embedding-configuration`](docs/openclaw/DEPLOYMENT.md#9-embedding-configuration)。

---

## 文档

- [`docs/openclaw/README.md`](docs/openclaw/README.md) — 通过 OpenClaw 操作 CiteWeave；
- [`docs/openclaw/DEPLOYMENT.md`](docs/openclaw/DEPLOYMENT.md) — 本地部署、Zotero sync、健康检查、重建流程；
- [`docs/openclaw/PACKAGE_INTERFACE.md`](docs/openclaw/PACKAGE_INTERFACE.md) — facade methods、intent routing、输出约定；
- [`docs/KERNEL_AND_OPENCLAW.md`](docs/KERNEL_AND_OPENCLAW.md) — kernel / adapter 架构；
- [`docs/data_structures/README.md`](docs/data_structures/README.md) — 图谱和向量数据模型。

English overview: [`README.md`](README.md)。

---

## 开发

push 前常用门禁：

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
