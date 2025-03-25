from typing import Iterator
import time
import random
from llama_cpp import Llama

class Generator:
    def __init__(self) -> None:
        self.llm = Llama.from_pretrained(
            repo_id="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            filename="DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf",
        )

    def generate(self, prompt: str) -> Iterator[str]:
        stream = self.llm(prompt, max_tokens=80, stream=True)

    # 생성된 텍스트를 하나씩 반환합니다.
        for chunk in stream:
            yield chunk['choices'][0]['text']
