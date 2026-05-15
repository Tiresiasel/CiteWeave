# CiteWeave Agent 安装指南

这份文档是 `INSTALL.md` 的中文说明版，面向正在安装 CiteWeave 的 Research Agent。

CiteWeave 是一个本地个人文献检索库。Research Agent 负责安装、配置、同步和用户交互；CiteWeave 负责解析 PDF、构建引用图谱、建立向量索引，并为研究问题提供可验证的本地证据。

## 安装前提

宿主机需要具备：

- 一个能读取本仓库并执行本地命令的 Research Agent，例如 Codex、OpenClaw、Claude Code 或其他本地 Agent；
- Docker 与 Docker Compose，用于运行 Neo4j、Qdrant 和 GROBID；
- Python 3.9+，用于创建 `.venv` 并安装依赖；
- Bash-compatible shell。

可选项：

- Git，用于 clone 仓库；
- Embedding API key，仅在用户选择 API embedding 时需要；
- MinerU，仅在用户启用高质量 PDF-to-Markdown 解析时需要。

## Agent 安装流程

Agent 按顺序完成以下步骤：

1. 识别当前 Research Agent，并写入安装配置。
2. 询问文献来源：Zotero、Mendeley、EndNote、普通 PDF 文件夹或单篇 PDF 测试。
3. 询问 embedding 方案：本地模型或 API embedding。
4. 询问同步策略：每 5 分钟、每 30 分钟、每日或自定义。
5. 运行本地 bootstrap，创建 `.env`、`.venv`，安装依赖并启动服务。
6. 对文献来源执行 dry run，确认能发现 PDF。
7. 启动可恢复 ingestion。
8. 用 health、routes 和 progress 命令验证系统状态。

## 推荐命令

应用安装选择：

```bash
.venv/bin/python scripts/apply_install_choices.py \
  --research-agent Codex \
  --reference-manager zotero \
  --source-location-mode custom \
  --source-dir /path/to/Zotero \
  --embedding-mode local \
  --embedding-profile bge_large_en \
  --sync-schedule every_5_minutes \
  --processors 10 \
  --skip-failed
```

启动本地环境：

```bash
bash scripts/bootstrap_local.sh
```

验证文献来源：

```bash
.venv/bin/python scripts/sync_literature_pdfs.py \
  --source "$CITEWEAVE_LITERATURE_SOURCE_DIR" \
  --reference-manager "$CITEWEAVE_REFERENCE_MANAGER" \
  --dry-run \
  --json
```

开始导入：

```bash
.venv/bin/python scripts/sync_literature_pdfs.py \
  --source "$CITEWEAVE_LITERATURE_SOURCE_DIR" \
  --reference-manager "$CITEWEAVE_REFERENCE_MANAGER" \
  --json \
  --processors 10 \
  --skip-failed
```

验证运行状态：

```bash
.venv/bin/citeweave health --json
.venv/bin/citeweave routes --json
.venv/bin/citeweave progress "$CITEWEAVE_LITERATURE_SOURCE_DIR" --json
```

## 安全规则

- 外部 API 由用户显式选择后再启用。
- 持续同步使用可恢复 ingestion。
- 更换 embedding provider、model 或 vector dimension 后，执行完整向量重建和全量 re-ingest。
- 删除 Docker volumes、清空 Qdrant collections、清理 progress、force restart 等破坏性操作需要用户确认。
- Agent 检测候选路径后交给用户确认，再写入配置。
