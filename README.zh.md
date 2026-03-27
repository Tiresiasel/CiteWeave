# CiteWeave

**句子级引用图谱 + 语义检索，面向学术论文的 RAG 系统。**

将 PDF 解析为句子级引用关系，构建引用图谱，通过多智能体系统对论文库进行学术问答。专为社会科学研究者设计——追踪论点如何在文献间流动；其他领域亦可使用。

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## 5 分钟快速启动（先走 CLI / Docker）

```bash
# 1. 克隆并配置
git clone https://github.com/Tiresiasel/CiteWeave.git
cd CiteWeave

# 2. 一键完成本地 CLI 部署
bash scripts/bootstrap_local.sh

# 3. 使用 CLI
.venv/bin/python -m src.core.cli upload path/to/paper.pdf
.venv/bin/python -m src.core.cli query "哪些论文讨论了 X？"
.venv/bin/python -m src.core.cli routes
.venv/bin/python -m src.core.cli chat

# 4. 或切换到 OpenClaw 模式
bash scripts/bootstrap_openclaw.sh
# 然后验证：
.venv/bin/python -m src.core.cli routes
bash scripts/deployment_check.sh
# 再从 OpenClaw 会话里调用 CiteWeave。
```

---

## 第一部分：作为 CLI 应用使用 CiteWeave

这是最基础、也最重要的部署方式。即便你之后接入 OpenClaw，底层依然是
这套本地 CLI + Docker 服务。

### 当前代码分支上实际可用的命令

- `upload`
- `diagnose`
- `batch-upload`
- `progress`
- `chat`
- `query`
- `routes`

### 本地 CLI 模式（无需 OpenClaw）

在 `.env` 中设置：

```bash
CITEWEAVE_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...yourkey...
```

然后通过项目虚拟环境运行：

```bash
.venv/bin/python -m src.core.cli upload path/to/paper.pdf
.venv/bin/python -m src.core.cli diagnose path/to/paper.pdf
.venv/bin/python -m src.core.cli batch-upload ./papers --resume
.venv/bin/python -m src.core.cli query "哪些论文讨论了 X？"
.venv/bin/python -m src.core.cli routes
.venv/bin/python -m src.core.cli chat
```

> 为什么推荐 `.venv/bin/python`？
> 因为 CLI 在启动时就会 import 项目依赖。对一台全新的机器来说，
> 如果当前环境没有装齐依赖，直接 `python -m src.core.cli` 会失败。

---

## 依赖服务

通过 Docker Compose 一键启动三个依赖服务：

```bash
docker-compose up -d
```

| 服务 | 端口 | 用途 |
|------|------|------|
| **Neo4j** | 7474 / 7687 | 引用图谱存储（支持 Cypher 查询） |
| **Qdrant** | 6333 / 6334 | 语义向量索引（ANN 检索） |
| **GROBID** | 8070 | PDF 结构化解析（提取作者、标题、章节） |

验证部署健康状态：

```bash
bash scripts/deployment_check.sh
```

---

## CLI 命令参考

所有操作建议通过项目虚拟环境执行：

```
.venv/bin/python -m src.core.cli <命令> [选项]
```

### `upload <pdf_path>` — 上传并解析论文

```bash
.venv/bin/python -m src.core.cli upload path/to/paper.pdf
.venv/bin/python -m src.core.cli upload path/to/paper.pdf --diagnose
.venv/bin/python -m src.core.cli upload path/to/paper.pdf --force
```

### `diagnose <pdf_path>` — PDF 质量诊断

```bash
.venv/bin/python -m src.core.cli diagnose path/to/paper.pdf
```

### `batch-upload <目录>` — 批量上传

```bash
.venv/bin/python -m src.core.cli batch-upload path/to/papers/
.venv/bin/python -m src.core.cli batch-upload path/to/papers/ --resume
.venv/bin/python -m src.core.cli batch-upload path/to/papers/ --sequential
```

### `progress <目录>` — 查看/清理批量处理进度

```bash
.venv/bin/python -m src.core.cli progress path/to/papers/
.venv/bin/python -m src.core.cli progress path/to/papers/ --clear
```

### `chat` — 交互式多轮对话

```bash
.venv/bin/python -m src.core.cli chat
```

### `query "<问题>"` — 单轮查询入口

单轮进入 LangGraph research workflow。
当你希望控制信息摘要之后的流程时，可以使用 `--confirmation`。

```bash
.venv/bin/python -m src.core.cli query "哪些论文讨论了带宽与定价的关系？"
.venv/bin/python -m src.core.cli query "总结一下 Michael Porter 1980" --confirmation continue
```

### `routes` — 路由配置诊断

查看当前生效的 route 配置，包括 alias、priority 映射，以及 addon / env 覆盖。

```bash
.venv/bin/python -m src.core.cli routes
```

---

## 第二部分：将 CiteWeave 接入 OpenClaw

### OpenClaw 模式到底改变了什么

OpenClaw 不会替代 CiteWeave 的存储或解析层，它只是接管 CiteWeave 的
LLM 后端。

底层部署并没有变：

- Neo4j 仍然存 citation graph
- Qdrant 仍然存语义向量
- GROBID 仍然负责 PDF 解析
- CiteWeave 的 CLI / Python 代码仍然负责 ingestion 和 chat 逻辑

变化只是：CiteWeave 不再自己直连 OpenAI，而是把所有 LLM 调用改为发往
**本地 OpenClaw gateway**。

### 工作原理

```
OpenClaw Agent (Atlas)
    │
    │  CITEWEAVE_LLM_PROVIDER=openclaw
    │  所有 LLM 调用 → http://localhost:18789/v1
    │
    ├──→ .venv/bin/python -m src.core.cli chat
    │       │
    │       └── Neo4j + Qdrant + GROBID
    │
    └──（可选）直接 import Python API
            LangGraphResearchSystem()
```

### 具体接入流程

1. 先完成上面的 **本地 CLI / Docker 部署**，或者直接运行：

```bash
bash scripts/bootstrap_openclaw.sh
```

这个脚本会准备 `.env`、创建虚拟环境、安装依赖、启动 Docker 服务，并让项目保持在 OpenClaw 模式。

2. 如果你想手动设置，`.env` 至少应包含：

```bash
CITEWEAVE_LLM_PROVIDER=openclaw
CITEWEAVE_LLM_MODEL=openai-codex/gpt-5.4
CITEWEAVE_LLM_API_BASE=http://localhost:18789/v1
CITEWEAVE_NEO4J_PASSWORD=0xC1735
```

3. 确认本地 OpenClaw gateway 正在运行：

```bash
openclaw gateway status
```

4. 再跑一次部署检查：

```bash
bash scripts/deployment_check.sh
```

如果配置正确，你会看到 gateway 连通性检查通过。

5. 然后就可以在 OpenClaw 会话里调用 CiteWeave，例如：

```text
Atlas，帮我用 CiteWeave 上传这些 PDF，然后用 chat 模式带我检查 citation graph。
```

### 关键安全 / 行为说明

在 `openclaw` 模式下，CiteWeave **不会**把你真实的 OpenAI API key 传给
gateway。代码会把它替换成一个无害占位符，真正的认证由 OpenClaw 自己的
本地会话 / gateway 流程处理。

---

## 配置说明

### 环境变量（优先级高于 JSON 配置文件）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CITEWEAVE_LLM_PROVIDER` | `openai` | `openclaw` · `openai` · `ollama` |
| `CITEWEAVE_LLM_MODEL` | — | 模型名称 |
| `CITEWEAVE_LLM_API_BASE` | — | openclaw / ollama 模式的 API 地址 |
| `CITEWEAVE_LLM_API_KEY` | — | API Key（openclaw 模式可填任意值） |
| `CITEWEAVE_NEO4J_PASSWORD` | `0xC1735` | Neo4j 密码 |
| `CITEWEAVE_ENV` | `production` | `production` · `development`（详细日志） |

### Neo4j 默认密码

默认密码 `0xC1735` 是有意为之，方便本地开发记忆。
**生产部署前务必修改：**

```bash
CITEWEAVE_NEO4J_PASSWORD=your-secure-password
```

---

## 开发

### Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt          # 句子切分依赖
```

### 运行测试

```bash
python -m unittest discover -s tests
python -m unittest discover -s tests -p 'test_routing.py'
```

### 隐私审计（提交前必须通过）

```bash
python3 scripts/repo_privacy_audit.py
```

任何隐私审计失败都会阻止提交。检查项包括：
- 绝对本地路径（`/home/tiresias`、`.openclaw/workspace`）
- Token / 密钥写入 tracked 文件
- `data/` 或 `test_files/` 中的运行时数据被 tracked

---

## 引用类型分类

CiteWeave 对论文中的每个句子进行分类：

| 类型 | 说明 |
|------|------|
| `CLAIM_MAIN` | 核心论点 / 主要主张 |
| `CLAIM_SUPPORTING` | 次要支撑论点 |
| `EVIDENCE_EMPIRICAL` | 实证数据 / 结论 |
| `EVIDENCE_THEORETICAL` | 理论支撑 |
| `EVIDENCE_LITERATURE` | 引用支撑 |
| `COUNTERARGUMENT` | 反论点 / 假设 |
| `METHODOLOGY` | 方法描述 |
| `REBUTTAL` | 明确反驳 |
| `QUESTION_MOTIVATION` | 研究问题 / 动机 |
| `FUTURE_WORK` | 未来方向 |
| `NON_ARGUMENT` | 中立 / 过渡性文字 |

---

## License

Apache License 2.0 — 见 [LICENSE](LICENSE)。
