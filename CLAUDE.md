# CLAUDE.md

本文件旨在为 Claude Code (claude.ai/code) 提供在 `momu-agent-framework` 仓库中工作的指导。

## 项目概述

`momu-agent-framework` 是基于 HelloAgents 框架构建的智能体框架。项目处于早期阶段，`momu_agent/` 是主包。

## 常用命令

本项目使用 `uv` 进行依赖和环境管理。

```bash
# 安装依赖
uv sync --frozen --all-extras

# 运行应用
uv run python main.py

# 运行所有测试
uv run pytest

# 运行单个测试
uv run pytest tests/test_dummy.py::test_true_is_true

# 检查代码风格（不修改）
uv run ruff check .
uv run ruff format --check .

# 自动修复
uv run ruff check --fix .
uv run ruff format .
```

## 代码风格

使用 Ruff，配置如下：
- 行长度：88
- 目标版本：Python 3.11+
- 启用规则：`E`（pycodestyle 错误）、`F`（pyflakes）、`I`（isort）
- 忽略 `E501`（行过长）
- 字符串使用双引号

## Git 提交规范

提交信息格式：`type: description`

常用 type：
- `feat`：新功能
- `fix`：缺陷修复
- `refactor`：重构（非功能变更、非缺陷修复）
- `test`：测试相关
- `chore`：构建、依赖、CI 等维护性改动
- `style`：代码格式调整（不影响逻辑）
- `docs`：文档变更

## CI

CI 在推送/PR 到 `main` 分支时触发，分两个任务：
- `lint`：检查 Ruff 格式和风格
- `test`：在 Python 3.11、3.12、3.13 上运行 pytest