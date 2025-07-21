# MinerU使用说明

## 概述
MinerU是一个可选的高质量PDF解析引擎，默认**禁用**。

## 何时使用MinerU
- 包含复杂表格的学术论文
- 有数学公式的技术文档
- 多栏布局的期刊文章
- 需要高质量结构保持的文档

## 启用MinerU

### 1. 安装MinerU
```bash
pip install magic-pdf[full]
```

### 2. 修改配置文件
编辑 `config/model_config.json`：
```json
{
  "pdf_processing": {
    "enable_mineru": true,
    "mineru_fallback": true
  }
}
```

### 3. 验证配置
```python
from src.document_processor import DocumentProcessor
processor = DocumentProcessor()
# 日志会显示 "MinerU-enhanced PDF processor (enabled via config)"
```

## 配置选项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_mineru` | `false` | 是否启用MinerU |
| `mineru_fallback` | `true` | MinerU失败时是否回退到传统方法 |
| `preferred_engine` | `"auto"` | 传统方法的首选引擎 |
| `mineru_config.timeout` | `300` | MinerU处理超时时间（秒） |
| `mineru_config.enable_formula` | `true` | 启用公式解析 |
| `mineru_config.enable_table` | `true` | 启用表格解析 |
| `mineru_config.device` | `"auto"` | 计算设备选择 |

## 注意事项
- MinerU需要较高的计算资源
- 首次使用会下载模型（约2GB）
- 处理速度比传统方法慢，但质量更高
- 如果MinerU不可用，系统会自动回退到传统PDF处理器 