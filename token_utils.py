"""
token_utils.py — Token 计算与截断工具

【知识点：为什么需要 Token 计算？】

LLM 不是直接理解文本的，它先把文本切分成 Token（编码单元）。
例如 "Hello world" 可能被切成 ["Hello", " world"] 或 ["He", "llo", " w", "orld"]。

不同模型的 tokenizer 不同，所以：
- 不能简单地用 "len(text)" 来估算
- 必须调用对应模型的 tokenizer 来精确计算

OpenAI 提供了 tiktoken 库，可以精确计算 GPT 系列模型的 Token 数。

【知识点：截断策略】

当对话历史太长时，我们需要删除一些旧消息。策略选择：
1. 保留 system 消息（绝对不能丢，它定义了 AI 的人格）
2. 从第二老的消息开始删除（保留最近的对话）
3. 如果只剩 system + 最新一条 user 仍然超限，报错提示

【知识点：Protocol / duck typing】

这里 Message 类型用了 Protocol，而不是固定的 dict。
这是 Python 的"鸭子类型"：只要对象有 role 和 content 属性，就认为是 Message。
好处是解耦——token_utils 不依赖具体的数据结构，只要是"长得像"的就行。
"""

from typing import List, Protocol, Sequence
import tiktoken


class Message(Protocol):
    """
    【知识点：Protocol（协议类）】
    
    Protocol 是 Python 3.8+ 引入的，用于定义"结构化类型"：
    - 只要一个类实现了 Protocol 中定义的所有方法/属性，就认为它符合该 Protocol
    - 不需要显式继承（不像 Java 的 interface 必须 implements）
    - 这是"鸭子类型"在类型系统里的正式化："如果它走起路来像鸭子，叫起来像鸭子，那它就是鸭子"

    这里我们定义 Message 只需要有 role 和 content 两个属性，
    这样任何有这两个属性的对象都可以传给 token_utils 的函数。
    """
    role: str
    content: str


def count_tokens(messages: Sequence[Message], model: str = "gpt-4o-mini") -> int:
    """
    计算一组消息的总 Token 数。

    【知识点：tiktoken 库】
    - tiktoken 是 OpenAI 官方开源的 tokenizer，可以精确计算 GPT 模型的 Token
    - 不同模型用不同的编码器（encoding），例如 "cl100k_base" 用于 GPT-4/GPT-3.5-turbo
    - 编码方式：文本 → 整数列表（token IDs），每个整数就是一个 token

    【参数】
    - messages: 消息列表，每个消息有 role 和 content
    - model: 模型名称，用于选择正确的 tokenizer

    【返回值】
    - 总 Token 数（整数）
    """
    try:
        # 尝试获取对应模型的编码器
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 如果模型不在 tiktoken 已知列表中，使用默认的 cl100k_base
        # 这适用于大多数兼容 OpenAI 的国产模型（如 Qwen、DeepSeek 等）
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0

    for message in messages:
        # 每条消息在 OpenAI 格式中有固定开销（约 4 个 token 用于字段名和格式）
        # 然后加上 role 和 content 本身的 token 数
        total_tokens += 4  # 消息格式开销（<|im_start|>role\n 等）
        total_tokens += len(encoding.encode(message.role))
        total_tokens += len(encoding.encode(message.content))

    # 对话结束时还有额外 2 个 token 开销（<|im_end|>）
    total_tokens += 2

    return total_tokens


def count_single_text(text: str, model: str = "gpt-4o-mini") -> int:
    """
    计算单段文本的 Token 数。
    
    用于快速估算用户输入的 token 数量，以便决定是否接受该输入。
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def truncate_messages(
    messages: List[Message],
    max_tokens: int,
    model: str = "gpt-4o-mini",
) -> List[Message]:
    """
    截断消息列表，使其总 Token 数不超过 max_tokens。

    【核心策略】
    1. 始终保留 system 消息（第一条）
    2. 从第二条（最旧的非系统消息）开始删除
    3. 保留最近的对话，直到 Token 数在限制内

    【为什么从第二老开始删，而不是最新的？】
    - 用户最关心的是"最近的对话"
    - 太久远的对话被遗忘是可以接受的
    - 如果删最新的，用户刚说的话就被忽略了，体验很差

    【知识点：List vs Sequence】
    - List 是可变的序列，可以 append、pop、insert
    - Sequence 是只读的序列协议（包括 tuple、list、range 等）
    - 函数参数用 Sequence 表示"只读访问"，用 List 表示"可能修改"
    - 这里返回 List 是因为我们返回的是一个新的列表（可能被调用者继续修改）
    """
    if not messages:
        return []

    # 如果只有一条消息（通常是 system），直接返回
    if len(messages) == 1:
        return messages[:]

    # 从完整列表的副本开始
    # 【知识点：浅拷贝】messages[:] 创建列表的浅拷贝，避免修改原列表
    trimmed = messages[:]

    # 循环检查 Token 数，如果超过限制就删除第二条（最旧的非系统消息）
    while len(trimmed) > 1 and count_tokens(trimmed, model) > max_tokens:
        # 删除 index 1（第二条消息），保留 index 0（system）
        # 【知识点：pop(index)】从列表中删除并返回指定位置的元素
        removed = trimmed.pop(1)
        # 打印日志让用户知道哪些消息被遗忘了（教学目的）
        print(f"  [记忆截断] 遗忘旧消息: [{removed.role}] {removed.content[:30]}...")

    return trimmed


def estimate_remaining_tokens(
    messages: List[Message],
    max_tokens: int,
    model: str = "gpt-4o-mini",
) -> int:
    """
    计算剩余可用 Token 数（留给模型回复的空间）。

    【为什么需要这个？】
    上下文窗口 = 输入 + 输出。如果我们把窗口全部占满给输入，
    模型就没有空间生成回复了。通常需要预留几百到几千 token 给输出。

    【返回值】
    - 剩余可用 Token 数（如果输入已经超过限制，返回负数）
    """
    used = count_tokens(messages, model)
    return max_tokens - used
