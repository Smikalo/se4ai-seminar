# cod_vs_cot/engines/openai_engine.py
# Requires openai >= 1.0
import os, time
from openai import AsyncOpenAI

# small ≈ cheap   · large ≈ best-reasoning
MODEL_COD = os.getenv("MODEL_COD", "gpt-4o-mini")#"gpt-3.5-turbo-0125")
MODEL_COT = os.getenv("MODEL_COT", "gpt-4o-mini")

_client = AsyncOpenAI()                # OPENAI_API_KEY picked up from env

def _elapsed(t0): return time.perf_counter() - t0

async def call_chat(prompt: str,
                    *,
                    mode: str = "cot",     # "cod" or "cot"
                    temperature: float = 0.7,
                    max_tokens: int = 256):
    """
    CoD → one concise chain (≤5 words/step) on a *small* model.
    CoT → one verbose chain on a *large* model.
    Returns ([text], latency_seconds)
    """
    if mode not in {"cod", "cot"}:
        raise ValueError("mode must be 'cod' or 'cot'")

    system = ("Think step by step, but only keep a minimum draft for each "
              "step, with 5-10 words at most. Return the answer after '####'."
              if mode == "cod"
              else
              "Think step by step to solve the problem. "
              "Return the answer after '####'.")

    model = MODEL_COD if mode == "cod" else MODEL_COT

    t0 = time.perf_counter()
    resp = await _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = _elapsed(t0)
    txt     = resp.choices[0].message.content.strip()
    return [txt], latency
