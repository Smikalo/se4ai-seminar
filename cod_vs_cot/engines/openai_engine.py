# engines/openai_engine.py   (works with openai>=1.0)
import os, time, asyncio
from openai import AsyncOpenAI

MODEL = os.getenv("MODEL", "gpt-3.5-turbo-0125")
_client = AsyncOpenAI()                  # uses OPENAI_API_KEY env-var

async def call_chat(prompt, *, n=1, temperature=0.7, max_tokens=256):
    t0 = time.perf_counter()
    resp = await _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - t0
    texts = [c.message.content.strip() for c in resp.choices]
    return texts, latency
