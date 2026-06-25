"""
memory.py — 对话记忆管理模块

【核心概念：为什么 LLM 需要"记忆"？】

LLM 本身是无状态的（stateless）。每一次 API 调用都是独立的：
你发送一组消息，模型返回一个回复，然后它就"忘记"了这一切。

要让它"记得"之前的对话，你必须在每次请求时把完整的对话历史重新发送给它。
这就需要我们（开发者）在程序中维护一个**消息列表**，这就是"记忆"。

【记忆管理的挑战】
1. 消息越积越多 → 超过 Token 限制 → API 报错
2. 无差别截断 → 丢失重要的 system 设定
3. 需要智能地保留最有价值的信息

本模块实现了三种策略的组合：
- 轮数限制（滑动窗口）：最多保留 N 轮对话
- Token 限制（精准截断）：超过总 Token 限制时，删除旧消息
- 角色保护：system 消息始终保留，永远不被截断
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from token_utils import count_tokens, truncate_messages


@dataclass
class ConversationMemory:
    """
    对话记忆类 —— 管理整个对话历史。

    【知识点：dataclass + field(default_factory=list)】
    - 如果直接用 messages: list = []，Python 的默认参数会在函数定义时求值，
      导致所有实例共享同一个列表！这是经典的 Python 坑。
    - field(default_factory=list) 会在每个实例创建时生成新的列表，避免共享。

    【属性】
    - system_prompt: 系统提示，定义 AI 的行为规则
    - max_history: 最大保留的对话轮数（一轮 = user + assistant）
    - max_context_tokens: 上下文 Token 上限
    - model: 模型名称，用于计算 token
    - messages: 完整的消息列表，格式符合 OpenAI API 要求
    """

    system_prompt: str
    max_history: int = 10
    max_context_tokens: int = 4000
    model: str = "gpt-4o-mini"
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """
        【知识点：__post_init__】
        dataclass 自动生成 __init__ 后调用的方法。
        用于做一些初始化后的额外设置（如初始化 system 消息）。
        """
        # 初始化时就把 system 消息放入列表
        # system 消息始终排第一位，告诉模型"你是谁、你要做什么"
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def add_user_message(self, content: str) -> None:
        """
        添加用户消息到记忆。

        【知识点：消息格式】
        OpenAI API 要求的消息格式：
        {"role": "user", "content": "用户说的话"}
        {"role": "assistant", "content": "AI 的回复"}
        {"role": "system", "content": "系统设定"}

        role 是字符串，必须是 "system" / "user" / "assistant" / "tool" 之一。
        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """添加 AI 回复到记忆。"""
        self.messages.append({"role": "assistant", "content": content})

    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """
        获取准备发送给 API 的消息列表，同时应用记忆管理策略。

        【核心逻辑：三步筛选】

        步骤 1: 轮数限制（滑动窗口）
        如果对话超过 max_history 轮（一轮 = user + assistant），
        只保留最近的 max_history 轮 + system 消息。
        这是"快速但粗略"的截断方式。

        步骤 2: Token 限制（精准截断）
        即使轮数没超，如果总 Token 数超过了 max_context_tokens，
        从第二条消息开始删除旧消息，直到 Token 数在限制内。
        这是"精准但慢"的截断方式。

        步骤 3: 返回结果
        返回一个"干净"的消息列表，可以直接传给 API。

        【为什么用两种策略？】
        - 轮数限制是 O(1) 的快操作，先过滤掉明显不需要的老消息
        - Token 计算需要遍历所有文本，是 O(n) 的慢操作，只在前者之后执行
        - 两者结合：效率 + 精度
        """
        # --- 步骤 1: 轮数限制 ---
        # system 消息不算在轮数内，单独保留
        system_msg = [self.messages[0]]  # 第一条永远是 system
        chat_messages = self.messages[1:]  # 剩余的是 user/assistant 对话

        # 一轮 = 2 条消息（user + assistant），但最后一轮可能只有 user（未回复）
        # 所以计算轮数：总对话消息数 // 2（向下取整）
        # 保留最近的 max_history * 2 条消息
        max_chat_messages = self.max_history * 2
        if len(chat_messages) > max_chat_messages:
            # 截断旧消息，保留最新的
            chat_messages = chat_messages[-max_chat_messages:]
            print(f"  [记忆滑动] 保留最近 {self.max_history} 轮对话，遗忘更早的消息")

        # 组合：system + 截断后的对话
        candidate = system_msg + chat_messages

        # --- 步骤 2: Token 限制 ---
        # 使用 token_utils 的 truncate_messages 进行精准截断
        # 它会从第二条消息开始删除，直到总 Token 数在限制内
        final_messages = truncate_messages(candidate, self.max_context_tokens, self.model)

        # 打印调试信息（教学目的）
        total_tokens = count_tokens(final_messages, self.model)
        print(f"  [Token 统计] 当前上下文: {total_tokens} tokens / {self.max_context_tokens} 上限")

        return final_messages

    def get_conversation_length(self) -> int:
        """返回对话轮数（不计 system）。"""
        # 减 1 是因为第一条是 system 消息，不算对话
        return max(0, len(self.messages) - 1) // 2

    def clear(self) -> None:
        """清空对话历史，只保留 system 消息。"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        print("🧹 记忆已清空，重新开始对话。")

    def __repr__(self) -> str:
        """返回记忆的摘要信息。"""
        return (
            f"ConversationMemory(rounds={self.get_conversation_length()}, "
            f"total_messages={len(self.messages)}, "
            f"system_prompt={self.system_prompt[:30]!r}...)"
        )
