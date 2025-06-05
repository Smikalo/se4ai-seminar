from datasets import load_dataset

# repo, subset
TASKS = {
    "mmlu":       ("cais/mmlu", "all"),           # four-way MC
    "arc":        ("ai2_arc",  "ARC-Challenge"),  # four-way MC
    "gsm8k":      ("gsm8k",    "main"),           # free-form
    "hellaswag":  ("hellaswag","plain_text"),     # four-way MC
    "strategyqa": ("wics/strategy-qa", "strategyQA"),  # yes / no
}

# which split to use
SPLIT = {
    "mmlu":       "validation",
    "arc":        "validation",
    "gsm8k":      "test",
    "hellaswag":  "validation",
    "strategyqa": "test",
}

def get_dataset(name, limit=None):
    repo, subset = TASKS[name]
    ds = load_dataset(repo, subset, split=SPLIT[name], trust_remote_code=True)
    return ds.select(range(limit)) if limit else ds