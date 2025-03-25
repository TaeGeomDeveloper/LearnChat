from typing import Iterator
from llama_cpp import Llama

class Generator:
    def __init__(self) -> None:
        # 딥시크 모델을 초기화합니다.
        self.llm = Llama.from_pretrained(
            repo_id="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            filename="DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf",
        )

    def generate(self, prompt: str) -> Iterator[str]:
        # 프롬프트를 기반으로 텍스트를 생성합니다.
        stream = self.llm(prompt, max_tokens=80, stream=True)

        # 생성된 텍스트를 하나씩 반환합니다.
        for chunk in stream:
            yield chunk['choices'][0]['text']

# 테스트용 코드 추가
if __name__ == "__main__":
    generator = Generator()
    
    test_prompts = [
        "I want to be a doctor because",
    ]
    
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        print("Generated response:", end=' ')
        stream = generator.generate(prompt)
        for chunk in stream:
            print(chunk, flush=True, end='')
        print("\n" + "-"*30)