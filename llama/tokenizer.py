# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe


logger = getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256 # 保留特殊字符个数

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501
    # Llama 3 分词器正则表达式解析
    # 这个正则表达式(pat_str)是Llama 3模型的分词器(tokenizer)中使用的模式，用于将文本拆分成基本单元(tokens)。让我逐部分解释:

    # (?i:'s|'t|'re|'ve|'m|'ll|'d) - 匹配英语中常见的缩写形式(不区分大小写)：

    # 's (所有格或is的缩写)
    # 't (not的缩写)
    # 're (are的缩写)
    # 've (have的缩写)
    # 'm (am的缩写)
    # 'll (will的缩写)
    # 'd (would/had的缩写)
    # |[^\r\n\p{L}\p{N}]?\p{L}+ - 匹配:

    # 可选的非字母数字字符(不包括换行)
    # 后跟一个或多个字母(\p{L}+表示任何Unicode字母)
    # |\p{N}{1,3} - 匹配1到3位数字

    # | ?[^\s\p{L}\p{N}]+[\r\n]* - 匹配:

    # 可选空格
    # 一个或多个非空白、非字母、非数字的字符(基本上是标点符号)
    # 可选的换行符
    # |\s*[\r\n]+ - 匹配任意空白后跟一个或多个换行符

    # |\s+(?!\S) - 匹配一个或多个空白字符，后面没有非空白字符(即行尾空白)

    # |\s+ - 匹配一个或多个空白字符

    # 这个复杂的正则表达式允许分词器智能地将文本拆分成有意义的单元，例如：

    # 完整单词
    # 标点符号
    # 缩写形式
    # 空白和换行
    # 数字
    # 这种拆分方式使模型能更好地理解文本的语义结构，从而提高处理自然语言的能力。
    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        # 在 tokenizer.py 中，mergeable_ranks 是一个从 tiktoken 模型文件加载的数据结构，代表 BPE（字节对编码）算法中的合并规则和它们的优先级。具体来说：

        # 定义与来源：

        # 通过 load_tiktoken_bpe(model_path) 函数从模型文件中加载
        # 是一个字典结构，将字节对（token对）映射到它们的合并优先级（rank）
        # 作用：

        # 它定义了分词器将文本拆分成子词时使用的规则
        # 包含了模型词表中所有基本词元（tokens）及其排序优先级
        # 决定了哪些字符序列应该被合并成一个词元
        # 在代码中的使用：

        # 用于初始化 tiktoken 的 Encoding 对象
        # 是构建词表（vocabulary）的基础
        # num_base_tokens = len(mergeable_ranks) 用来计算模型的基本词表大小
        # 本质：

        # 这是 BPE 算法的核心组件，包含了训练好的合并规则
        # 每个"规则"表示哪些字符序列应该被视为单个词元
        # 规则的"rank"（排名）决定了应用合并的优先顺序
        # 简单来说，mergeable_ranks 是 Llama 3 模型的词表和分词规则的核心，决定了文本如何被切分成词元（tokens）。
        mergeable_ranks = load_tiktoken_bpe(model_path) # 加载tiktoken模型
        num_base_tokens = len(mergeable_ranks) # 基本令牌数
        # 从模型中加载特殊令牌
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        } # 生成特殊令牌的字典
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks, # 合并等级，用于构建词表
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"] # 开始文本符号
        self.eos_id: int = self.special_tokens["<|end_of_text|>"] # 结束文本符号
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000 # 最大编码字符数

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000 # 最大非空格字符数

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        ) # 生成器表达式，目的是将长文本分割成短文本，以便于分词，方法是根据空格或非空格字符进行分割
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos: # 如果需要添加开始符号
            t.insert(0, self.bos_id)
        if eos: # 如果需要添加结束符号
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False # 判断第一个字符是否为空格
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space: # 异或运算，判断是否是空格
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


class ChatFormat:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]: # 编码头部，即角色
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]: # 编码消息
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"]) # 编码完一个消息之后，添加结束符号
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens
# ChatFormat 类的作用是将对话消息编码为模型输入的格式。它包含了以下方法：
# encode_header(message: Message)：编码消息头部，即角色（role）。
# encode_message(message: Message)：编码消息内容。
# encode_dialog_prompt(dialog: Dialog)：编码对话消息，包括角色和内容。
# 这些方法将对话消息转换为模型输入的格式，以便模型能够理解和处理。
# 最终编码的格式是 <|begin_of_text|> + 消息角色 + <|end_header_id|> + 换行符 + 消息内容 + <|eot_id|> 
# + <|begin_of_text|> + 消息角色 + <|end_header_id|> + 换行符 + 消息内容 + <|eot_id|> 
# + ... + <|begin_of_text|> + assistant + <|end_header_id|> + 换行符。
# 最后一个 assistant 消息是模型的提示，用于生成对话的下一部分。