# Chatting Agent — 带记忆功能的命令行聊天工具

这是一个学习驱动的项目，**每一行代码都带有教学目的**。它完整覆盖了你的学习计划中的所有知识点：

- Python 类型注解 + 异步编程（`async/await`）
- LLM API 调用（OpenAI 兼容接口）
- Token 计算与上下文窗口管理
- 提示词工程（System / User / Assistant 消息角色）
- 对话历史管理（滑动窗口、Token 截断策略）
- 结构化输出

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件（或直接在命令行中设置）：

```bash
# 使用 OpenAI 官方
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# 或者使用国内兼容模型（如硅基流动、阿里云等）
# OPENAI_API_KEY=your-key
# OPENAI_BASE_URL=https://api.siliconflow.cn/v1
# OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### 3. 运行

```bash
python main.py
```

---

## 📚 核心知识点速览

### 1. Token 机制与上下文窗口

LLM 不是按"字符"或"单词"来理解文本的，而是按 **Token**（一种文本的编码单元）来处理。一般来说：
- 1 个英文单词 ≈ 1.3 个 Token
- 1 个汉字 ≈ 1-2 个 Token

每个模型都有**上下文窗口（Context Window）**限制，例如：
- GPT-4o-mini: 128K tokens
- GPT-3.5-turbo: 16K tokens

如果输入超过窗口限制，模型会报错，或者最前面的内容会被遗忘。因此我们需要**主动管理**上下文长度。

### 2. 消息角色分工

现代 LLM API 使用消息列表（Message List）格式，每条消息都有一个角色：

| 角色 | 作用 | 类比 |
|------|------|------|
| `system` | 设定全局规则、人格、约束 | 系统提示 |
| `user` | 用户的输入/问题 | 人类提问 |
| `assistant` | AI 的回复 | AI 回答 |

`system` 消息通常放在对话最开头，只放一次。它告诉模型"你是谁、你要做什么、怎么回答"。

### 3. 提示词工程基础模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **Zero-shot** | 直接给指令，不给示例 | 简单、明确的任务 |
| **Few-shot** | 给几个输入→输出示例 | 格式转换、分类 |
| **CoT (Chain-of-Thought)** | 要求模型"一步一步思考" | 数学、推理题 |
| **ReAct (Reasoning + Acting)** | 让模型思考→行动→观察→循环 | 需要调用工具/外部知识 |

### 4. 异步编程（async/await）

为什么 LLM 调用要用异步？

因为 API 调用是**网络 I/O 操作**，等待 OpenAI 服务器返回时，程序什么都不做。用 `async/await` 可以让程序在等待期间去做其他事（比如同时处理多个用户的请求、保持 UI 响应等）。

```python
import asyncio

async def main():
    # await 表示："我先去等待这个操作完成，但线程可以先去做别的"
    result = await some_async_function()
```

### 5. 对话记忆管理策略

**问题**：如果对话无限进行，消息列表会越积越长，最终超过 Token 限制。

**解决方案**：
1. **滑动窗口**：只保留最近 N 条对话（简单但可能丢失重要信息）
2. **Token 截断**：计算总 Token 数，超过时删除最旧的消息（精准但实现复杂）
3. **摘要压缩**：定期将历史对话压缩为一段摘要，只保留摘要 + 最近几轮（最智能，但成本高）

本项目实现了 **策略 1 和 2 的组合**。

---

## 🗂️ 项目结构

```
chatting/
├── config.py          # 配置管理（环境变量、模型参数）
├── token_utils.py     # Token 计算与截断工具
├── memory.py          # 对话记忆管理（核心教学模块）
├── chat_engine.py     # 异步聊天引擎（LLM 调用逻辑）
├── main.py            # 命令行入口（用户交互）
├── requirements.txt   # 依赖列表
└── README.md          # 你正在读的这个文档
```

---

## 🔄 关键流程图

```
用户输入 → 加入记忆(memory) → 检查 Token 是否超限
                                    ↓
                              超限？截断旧消息
                                    ↓
                              组装消息列表 → 调用 LLM API
                                    ↓
                              收到回复 → 加入记忆 → 展示给用户
```

---

## 📝 学习建议

按顺序阅读以下文件，每个文件都有详细注释：

1. **`config.py`** → 理解环境变量管理和类型注解
2. **`token_utils.py`** → 理解 Token 计算和截断逻辑
3. **`memory.py`** → 理解对话历史管理和策略设计
4. **`chat_engine.py`** → 理解异步 API 调用和错误处理
5. **`main.py`** → 理解命令行交互循环

---

## ⚠️ 常见踩坑

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 超过上下文限制报错 | 消息太多/太长 | 开启 Token 截断，或减少 `max_history` |
| 模型"忘记"了之前的设定 | 没有 system 消息 / system 被截断 | 确保 system 消息始终保留，且排在第一位 |
| 回复慢 | 同步调用阻塞了 | 使用 async/await，考虑流式输出 |
| 格式混乱 | 没有给模型明确的输出格式要求 | 在 system 中指定格式规则 |

---

**Happy Learning!** 🎓
