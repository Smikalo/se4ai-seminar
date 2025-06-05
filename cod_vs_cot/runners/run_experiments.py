"""
run_experiments.py – run CoD-vs-CoT benchmarks.

Usage
-----
python -m cod_vs_cot.runners.run_experiments \
       --tasks mmlu gsm8k hellaswag arc strategyqa \
       --limit 100 --cod_samples 3
"""

import argparse, asyncio, json, os, pathlib
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv

load_dotenv()          # picks up OPENAI_API_KEY from .env

from tqdm import tqdm

from cod_vs_cot.benchmarks.loaders import get_dataset, TASKS
from cod_vs_cot.engines.openai_engine import call_chat

LETTER = "ABCD"        # helper for multiple choice

# ---------------------------------------------------------------------------
# prompt helpers
# ---------------------------------------------------------------------------

def join_choices(opts: List[str]) -> str:
    return "\n".join(f"{LETTER[i]}. {o}" for i, o in enumerate(opts))

def build_prompt(task: str, template: str, ex: dict):
    """
    Returns (prompt, gold_answer) for one record.
    Template must contain '{question}' placeholder.
    """
    # ---------- MMLU -------------------------------------------------------
    if task == "mmlu":
        stem  = ex["question"]
        opts  = ex["choices"]
        prompt_q = f"{stem}\n\n{join_choices(opts)}"
        gold     = ex["answer"]          # already 'A' .. 'D'

    # ---------- ARC-Challenge ---------------------------------------------
    elif task == "arc":
        stem = ex["question"]

        # new schema (list of dicts) or old (dict of lists)
        if isinstance(ex["choices"], list):
            opts = [c["text"] for c in ex["choices"]]
        else:
            opts = ex["choices"]["text"]

        prompt_q = f"{stem}\n\n{join_choices(opts)}"
        gold     = ex["answerKey"]       # 'A' .. 'D'

    # ---------- HellaSwag --------------------------------------------------
    elif task == "hellaswag":
        stem = ex["ctx"].strip()
        opts = ex["endings"]
        prompt_q = (
            f"{stem}\n\nChoose the best ending:\n{join_choices(opts)}\n"
            "Respond with A, B, C or D."
        )
        # label is '0'..'3' **as a string**  → convert to int → letter
        gold = LETTER[int(ex["label"])]

    # ---------- GSM-8K -----------------------------------------------------
    elif task == "gsm8k":
        prompt_q = ex["question"]
        gold     = ex["answer"]

    # ---------- StrategyQA -------------------------------------------------
    elif task == "strategyqa":
        prompt_q = f"{ex['question']}\n\nAnswer yes or no."
        gold     = ex["answer"]          # bool

    else:
        raise ValueError(f"Unknown task '{task}'")

    return template.format(question=prompt_q), gold

# ---------------------------------------------------------------------------
# core runner
# ---------------------------------------------------------------------------

async def run_task(task, template_path, kind, samples, limit):
    ds        = get_dataset(task, limit=limit)
    template  = pathlib.Path(template_path).read_text()
    records   = []

    for ex in tqdm(ds, desc=task):
        prompt, gold = build_prompt(task, template, ex)

        # answers, latency = await call_chat(
        #     prompt,
        #     n=samples,
        #     temperature=0 if samples == 1 else 0.7,
        # )
        answers, latency = await call_chat(
            prompt,
            mode="cod" if kind == "cod" else "cot",  # ← NEW
            temperature=0.7,
        )

        records.append(
            {
                "id":       ex.get("id"),
                "prompt":   prompt,
                "answers":  answers,
                "gold":     gold,
                "latency":  latency,
            }
        )
    return records


async def main(args):
    ts = datetime.now(timezone.utc).isoformat()
    os.makedirs(args.outdir, exist_ok=True)

    for kind, tpl in [("cod", "cod_vs_cot/prompts/cod.txt"),
                      ("cot", "cod_vs_cot/prompts/cot.txt")]:
        for task in args.tasks:
            recs = await run_task(
                task,
                tpl,
                kind,
                args.cod_samples if kind == "cod" else 1,
                args.limit,
            )
            out = pathlib.Path(args.outdir) / f"{task}__{kind}__{ts}.json"
            out.write_text(json.dumps(recs, indent=2))
            print(f"Wrote {out}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS.keys()))
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cod_samples", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    asyncio.run(main(ap.parse_args()))
