import aiohttp
import asyncio
import json

class LLMClient:
    def __init__(self, default_model: str):
        self.default_model = default_model
        self.base_url = "http://localhost:11434/api/generate"

    async def complete(self, prompt: str, model: str = None, temperature: float = 0.7) -> str:
        model = model or self.default_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Ollama错误 {resp.status}: {text}")
                data = await resp.json()
                return data["response"]