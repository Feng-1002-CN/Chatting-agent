"""
config.py — 配置管理模块

【知识点：Python 类型注解 + dataclass + 环境变量】

为什么要用类型注解？
  - 在大型项目中，类型标注让代码更容易理解（尤其是团队协同时）
  - IDE 可以据此给出智能提示和自动补全
  - 可以用 mypy 等工具做静态类型检查，在运行前发现类型错误

为什么要用 dataclass？
  - 它自动为你生成 __init__、__repr__、__eq__ 等方法
  - 省去了写样板代码的时间，代码更简洁

为什么要用 @classmethod + from_env()？
  - 工厂模式：创建一个"从环境变量构造对象"的方法
  - 把配置解析逻辑和对象创建封装在一起，调用端更简洁
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatConfig:
    """
    聊天配置类 —— 所有和 LLM 相关的参数集中在这里。

    【类型注解说明】
    - str: 字符串类型
    - int: 整数类型
    - float: 浮点数类型
    - Optional[str]: 可选字符串，可以是 str 或 None

    【成员说明】
    - api_key: 你的 API 密钥，从环境变量读取
    - base_url: API 的基础地址，支持 OpenAI 兼容接口
    - model: 模型名称
    - max_history: 最大保留对话轮数（滑动窗口策略）
    - max_context_tokens: 上下文窗口的 Token 上限（超过就截断）
    - temperature: 创造性参数，0~2，越低越保守，越高越发散
    - system_prompt: 系统提示，设定 AI 的人格和行为规则
    """

    api_key: str
    base_url: str
    model: str
    max_history: int = 10             # 保留最近 10 轮对话（一轮 = user + assistant）
    max_context_tokens: int = 4000      # 上下文限制（不包括系统提示）
    temperature: float = 0.7
    system_prompt: str = (
        "你是一个 helpful 的 AI 助手。请用中文回答用户的问题。"
        "如果用户的问题需要推理，请一步一步思考后再给出答案。"
    )

    @classmethod
    def from_env(cls) -> "ChatConfig":
        """
        从环境变量创建配置实例。

        【知识点：os.environ.get()】
        - os.environ.get("KEY") 读取环境变量，如果不存在返回 None
        - os.environ.get("KEY", "默认值") 可以指定默认值
        - os.environ["KEY"] 如果不存在会抛出 KeyError，所以更安全的方式是用 get()

        【知识点：为什么用 classmethod 而不是 staticmethod？】
        - classmethod 接收 cls 作为第一个参数，可以返回 cls 的实例
        - staticmethod 不接收 cls，适合不依赖类状态的纯函数
        - 这里需要返回 ChatConfig 实例，所以用 classmethod
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ 环境变量 OPENAI_API_KEY 未设置！\n"
                "请执行：export OPENAI_API_KEY=your-key (Linux/Mac)\n"
                "或：set OPENAI_API_KEY=your-key (Windows CMD)\n"
                "或在当前目录创建 .env 文件"
            )

        return cls(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            max_history=int(os.environ.get("MAX_HISTORY", "10")),
            max_context_tokens=int(os.environ.get("MAX_CONTEXT_TOKENS", "4000")),
            temperature=float(os.environ.get("TEMPERATURE", "0.7")),
        )

    def __repr__(self) -> str:
        """
        【知识点：__repr__ 魔术方法】
        - 返回对象的"官方"字符串表示
        - dataclass 会自动生成这个，但这里我们覆盖它以隐藏 api_key（安全！）
        """
        return (
            f"ChatConfig(model={self.model!r}, base_url={self.base_url!r}, "
            f"max_history={self.max_history}, max_context_tokens={self.max_context_tokens}, "
            f"temperature={self.temperature})"
        )
