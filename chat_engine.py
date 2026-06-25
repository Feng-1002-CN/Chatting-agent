"""
chat_engine.py — 异步聊天引擎

【核心概念：为什么聊天引擎需要异步？】

API 调用（HTTP 请求）是 I/O 密集型操作：
- 发送请求后，你需要等待服务器处理（可能要几百毫秒到几秒）
- 在等待期间，CPU 其实什么都没做，只是干等着网络返回

如果不用异步（同步代码）：
```python
response = requests.post(...)  # 阻塞！程序停在这里等 2 秒
print("这行代码 2 秒后才执行")
```

如果用异步：
```python
response = await client.chat.completions.create(...)  # 挂起，CPU 去做别的事
# 等服务器返回后，自动恢复到这里继续执行
```

在命令行工具中，异步的好处是：
- 可以同时处理多个用户输入（虽然 CLI 通常只有一个用户，但底层逻辑一致）
- 支持流式输出（打字机效果）
- 代码模式可以复用到 Web 应用（FastAPI 就是全异步的）

【知识点：OpenAI SDK 的 async 版本】
OpenAI 的 Python SDK 提供了 AsyncOpenAI 客户端，所有方法都是异步的：
- 创建客户端：AsyncOpenAI(api_key=..., base_url=...)
- 调用 API：await client.chat.completions.create(...)
- 关闭客户端：await client.close()

【知识点：async with（异步上下文管理器）】
```python
async with AsyncOpenAI(...) as client:
    # 这里使用 client
# 离开 with 块时，自动调用 client.aclose()
```
这样确保资源被正确释放，即使发生异常也会关闭连接。
"""

from typing import Optional, List, Dict, Any
import openai
from config import ChatConfig
from memory import ConversationMemory
from token_utils import estimate_remaining_tokens


class ChatEngine:
    """
    异步聊天引擎 —— 负责和 LLM API 交互。

    【设计模式：依赖注入（Dependency Injection）】
    ChatEngine 不自己创建配置和记忆，而是接收它们作为参数。
    好处：
    - 更容易测试（可以注入 mock 配置）
    - 更灵活（同一个 engine 可以用于不同配置）
    - 更符合"单一职责原则"（engine 只负责 API 调用，不管理配置）
    """

    def __init__(self, config: ChatConfig, memory: ConversationMemory):
        self.config = config
        self.memory = memory
        # 创建异步 OpenAI 客户端
        # base_url 支持任意兼容 OpenAI 的 API（如硅基流动、DeepSeek、阿里云等）
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def chat(self, user_input: str) -> str:
        """
        处理一次用户输入，返回 AI 回复。

        【完整流程】
        1. 将用户输入加入记忆
        2. 从记忆中获取当前上下文（已应用截断策略）
        3. 检查剩余 Token 空间
        4. 调用 LLM API
        5. 将 AI 回复加入记忆
        6. 返回回复内容

        【参数】
        - user_input: 用户输入的字符串

        【返回值】
        - AI 的回复字符串

        【异常处理】
        - 网络超时、API 限制、上下文过长等都会抛出异常
        - 调用端需要做好 try/except 处理
        """
        # 步骤 1: 加入用户输入
        self.memory.add_user_message(user_input)

        # 步骤 2: 获取上下文（已截断）
        messages = self.memory.get_messages_for_api()

        # 步骤 3: 检查剩余 Token 空间
        remaining = estimate_remaining_tokens(
            messages, self.config.max_context_tokens, self.config.model
        )
        print(f"  [Token 统计] 剩余可用回复空间: {remaining} tokens")
        if remaining < 100:
            print("  ⚠️ 警告：上下文接近上限，建议输入 /clear 清空记忆")

        # 步骤 4: 调用 LLM API
        # 【知识点：API 参数详解】
        # - model: 模型名称（如 gpt-4o-mini, Qwen/Qwen2.5-7B-Instruct）
        # - messages: 消息列表，符合 OpenAI 格式
        # - temperature: 创造性参数（0~2），越低越保守，越高越随机
        # - max_tokens: 模型回复的最大 Token 数（防止回复过长）
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore
                temperature=self.config.temperature,
                max_tokens=min(remaining, 2000),  # 回复不超过 2000 或剩余空间
            )
        except openai.APIError as e:
            # API 返回错误（如模型不存在、参数错误）
            raise ChatEngineError(f"API 错误: {e}") from e
        except openai.APITimeoutError as e:
            # 请求超时
            raise ChatEngineError(f"请求超时: {e}") from e
        except openai.RateLimitError as e:
            # 速率限制（请求太频繁）
            raise ChatEngineError(f"速率限制，请稍后重试: {e}") from e

        # 步骤 5: 提取回复内容
        # response.choices[0].message.content 是 AI 的回复文本
        assistant_message = response.choices[0].message.content
        if not assistant_message:
            raise ChatEngineError("AI 返回了空内容")

        # 步骤 6: 加入记忆
        self.memory.add_assistant_message(assistant_message)

        return assistant_message

    async def close(self) -> None:
        """关闭客户端连接，释放资源。"""
        await self.client.close()

    async def __aenter__(self):
        """
        【知识点：异步上下文管理器】
        async with ChatEngine(...) as engine:
            ...
        进入时返回 self，退出时自动调用 __aexit__ 关闭连接。
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False  # 不吞掉异常，如果有异常继续抛出


class ChatEngineError(Exception):
    """聊天引擎自定义异常。"""
    pass
