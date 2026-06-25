"""
main.py — 命令行入口

【知识点：异步程序的主入口】

Python 的异步函数不能直接被调用，必须通过事件循环（event loop）来运行。

正确的方式：
```python
async def main():
    ...

if __name__ == "__main__":
    asyncio.run(main())
```

asyncio.run() 会：
1. 创建一个新的事件循环
2. 运行 main() 直到完成
3. 关闭事件循环

【知识点：sys.stdin 读取】
- input() 是同步的，会阻塞等待用户输入
- 在异步程序中，input() 仍然会阻塞事件循环
- 但对于 CLI 工具来说，这是可以接受的（只有一个用户）
- 如果是 Web 应用，就不能用 input()，而要用 HTTP 请求

【知识点：信号处理（Ctrl+C）】
asyncio 的 add_signal_handler 可以优雅地处理中断信号，
确保程序退出时资源被正确释放。
"""

import asyncio
import sys
from config import ChatConfig
from memory import ConversationMemory
from chat_engine import ChatEngine, ChatEngineError


# 命令帮助信息
HELP_TEXT = """
🤖 Chatting Agent — 命令帮助

可用命令（以 / 开头）：
  /help     显示此帮助信息
  /clear    清空对话记忆
  /history  显示当前对话历史摘要
  /stats    显示 Token 使用统计
  /quit     退出程序

其他提示：
  - 直接输入文字即可与 AI 对话
  - 当记忆接近上限时，系统会自动遗忘旧消息
  - 建议上下文过长时主动使用 /clear
"""


async def main() -> None:
    """
    主程序 —— 命令行交互循环。

    【流程】
    1. 加载配置
    2. 初始化记忆和引擎
    3. 进入交互循环：读取输入 → 处理命令或调用 AI → 输出回复
    4. 退出时清理资源
    """
    print("=" * 50)
    print("🤖 Chatting Agent — 带记忆功能的命令行聊天工具")
    print("=" * 50)
    print("输入 /help 查看命令，输入 /quit 退出\n")

    # 步骤 1: 加载配置
    try:
        config = ChatConfig.from_env()
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)

    print(f"📦 配置加载成功: {config}\n")

    # 步骤 2: 初始化记忆和引擎
    # 使用 async with 确保引擎在使用完毕后被正确关闭
    async with ChatEngine(
        config=config,
        memory=ConversationMemory(
            system_prompt=config.system_prompt,
            max_history=config.max_history,
            max_context_tokens=config.max_context_tokens,
            model=config.model,
        ),
    ) as engine:

        # 步骤 3: 交互循环
        while True:
            try:
                # 读取用户输入（绿色提示符）
                # \033[32m 是 ANSI 转义码，设置文本颜色为绿色
                # \033[0m 重置颜色
                user_input = input("\033[32m你\033[0m: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if not user_input:
                continue

            # 处理命令（以 / 开头）
            if user_input.startswith("/"):
                if await handle_command(user_input, engine):
                    break
                continue

            # 调用 AI
            print("\033[36mAI\033[0m: ", end="", flush=True)
            try:
                reply = await engine.chat(user_input)
                print(reply)
            except ChatEngineError as e:
                print(f"❌ 错误: {e}")
            print()

    # 步骤 4: 退出时资源已由 async with 自动释放
    print("👋 程序已退出，感谢使用！")


async def handle_command(command: str, engine: ChatEngine) -> bool:
    """
    处理斜杠命令。

    【参数】
    - command: 用户输入的命令（如 "/help"）
    - engine: 聊天引擎实例

    【返回值】
    - True: 应该退出程序
    - False: 继续交互
    """
    cmd = command.lower()

    if cmd == "/quit" or cmd == "/exit":
        print("👋 再见！")
        return True

    elif cmd == "/help":
        print(HELP_TEXT)

    elif cmd == "/clear":
        engine.memory.clear()

    elif cmd == "/history":
        print(f"📜 当前对话历史: {engine.memory.get_conversation_length()} 轮")
        print(f"📜 总消息数: {len(engine.memory.messages)} 条（含 system）")
        for i, msg in enumerate(engine.memory.messages):
            role_icon = {"system": "⚙️", "user": "🧑", "assistant": "🤖"}.get(msg["role"], "❓")
            preview = msg["content"][:50].replace("\n", " ")
            print(f"  [{i}] {role_icon} {msg['role']}: {preview}...")

    elif cmd == "/stats":
        from token_utils import count_tokens
        tokens = count_tokens(engine.memory.messages, engine.config.model)
        print(f"📊 Token 统计:")
        print(f"  - 当前上下文: {tokens} tokens")
        print(f"  - 上限: {engine.config.max_context_tokens} tokens")
        print(f"  - 使用率: {tokens / engine.config.max_context_tokens * 100:.1f}%")
        print(f"  - 最大保留轮数: {engine.config.max_history}")

    else:
        print(f"❓ 未知命令: {command}，输入 /help 查看可用命令")

    return False


if __name__ == "__main__":
    """
    【知识点：if __name__ == "__main__"】
    
    这是 Python 的惯用法：
    - 当文件被直接运行时（python main.py），__name__ == "__main__"，执行下面的代码
    - 当文件被 import 时（如 from main import something），__name__ 是模块名，不执行
    
    好处：
    - 可以被其他模块导入而不触发主程序逻辑
    - 是 Python 单元测试和模块化的基础
    """
    asyncio.run(main())
