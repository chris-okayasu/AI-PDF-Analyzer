import httpx
from typing import List


class OllamaAdapter:
    """
    Unified adapter for Ollama.
    Handles both text generation and embeddings.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        temperature: float = 0.2,
        max_tokens: int = 300,
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.embed_model = embed_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["response"]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []

        async with httpx.AsyncClient() as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embed_model,
                        "prompt": text,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                vectors.append(response.json()["embedding"])

        return vectors