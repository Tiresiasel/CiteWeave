# CiteWeave

**句子级引用图谱 + 语义检索，面向学术论文的 RAG 系统。**

将 PDF 解析为句子级引用关系，构建引用图谱，通过多智能体系统对论文库进行学术问答。专为社会科学研究者设计——追踪论点如何在文献间流动；其他领域亦可使用。

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## 5 分钟快速启动

```bash
# 1. 克隆并配置
git clone https://github.com/Tiresiasel/CiteWeave.git
cd CiteWeave
cp .env_template .env          # 编辑 .env — 见下方「选择 LLM 模式」

# 2. 启动依赖服务（Neo4j + Qdrant + GROBID）
docker-compose up -d

# 3. 验证部署
bash scripts/deployment_check.sh

# 4. 上传论文并查询
python -m src.core.cli upload path/to/paper.pdf
python -m src.core.cli query "哪些论文讨论了 X？"
```

---

## 两种运行模式

### 模式 A — 本地 CLI（无需 OpenClaw）

```bash
# 在 .env 中设置
CITEWEAVE_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...yourkey...
```

### 模式 B — OpenClaw 集成（推荐给 OpenClaw 用户）

```bash
# 在 .env 中设置
CITEWEAVE_LLM_PROVIDER=openclaw
# CITEWEAVE_LLM_MODEL 默认: openai-codex/gpt-5.4
# CITEWEAVE_LLM_API_BASE 默认: http://localhost:18789/v1
```

设置为 `openclaw` 后，所有 LLM 调用（包括 `language_processor`、`query_analyzer`、
`response_generator` 等所有 Agent）都会自动路由到本地 OpenClaw gateway，
**无需单独配置 OpenAI API Key**。OpenClaw 通过会话认证，不需要额外的 API Key。

OpenClaw Agent 可直接通过 CLI 调用 CiteWeave：

```
Atlas，帮我上传这些 PDF，然后回答哪些论文讨论了 X。
```

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

所有操作均通过 `python -m src.core.cli`：

```
python -m src.core.cli <命令> [选项]
```

### `upload <pdf_path>` — 上传并解析论文

```bash
python -m src.core.cli upload path/to/paper.pdf        # 解析并入库
python -m src.core.cli upload path/to/paper.pdf --diagnose  # 仅质量诊断
python -m src.core.cli upload path/to/paper.pdf --force     # 强制重新处理
```

### `query "<问题>"` — 学术问答

```bash
python -m src.core.cli query "哪些论文讨论了带宽与定价的关系？"
```

### `chat` — 交互式多轮对话

```bash
python -m src.core.cli chat
```

### `batch-upload <目录>` — 批量上传

```bash
python -m src.core.cli batch-upload path/to/papers/           # 4 并行
python -m src.core.cli batch-upload path/to/papers/ --resume  # 跳过已完成
python -m src.core.cli batch-upload path/to/papers/ --sequential  # 顺序处理
```

### `diagnose <pdf_path>` — PDF 质量诊断

```bash
python -m src.core.cli diagnose path/to/paper.pdf
```

### `routes` — 路由配置诊断

打印当前生效的路由配置（addon 配置文件优先级、环境变量覆盖等）。

```bash
python -m src.core.cli routes
```

### `papers [--all | --limit N]` — 列出数据库中的论文

```bash
python -m src.core.cli papers --limit 20   # 前 20 篇
python -m src.core.cli papers --all          # 全部
```

---

## OpenClaw 集成

### 工作原理

```
OpenClaw Agent (Atlas)
    │
    │  CITEWEAVE_LLM_PROVIDER=openclaw
    │  所有 LLM 调用 → http://localhost:18789/v1 (OpenClaw gateway)
    │
    ├──→ CLI: python -m src.core.cli query "..."
    │       │
    │       └── Neo4j + Qdrant + GROBID（Docker 服务）
    │
    └──（可选）直接 import Python API
            from src.agents.multi_agent_research_system import LangGraphResearchSystem
```

### 配置步骤

1. 编辑 `.env`：

```bash
CITEWEAVE_LLM_PROVIDER=openclaw
CITEWEAVE_LLM_MODEL=openai-codex/gpt-5.4          # 可选
CITEWEAVE_LLM_API_BASE=http://localhost:18789/v1  # 默认值，可选
CITEWEAVE_NEO4J_PASSWORD=0xC1735                   # 生产环境请修改
```

2. 确认 OpenClaw gateway 正在运行：

```bash
openclaw gateway status
```

3. 验证 CiteWeave 正确检测到 gateway：

```bash
bash scripts/deployment_check.sh
```

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
