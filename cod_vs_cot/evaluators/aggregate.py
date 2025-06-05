"""
Roll up many result JSONs into a CSV for plotting.

python -m evaluators.aggregate results/*.json
"""
import argparse, pandas as pd, glob
from .accuracy import score_file

def main(args):
    rows = [score_file(f) for f in args.files]
    df = pd.DataFrame(rows)
    df.to_csv("results/summary.csv", index=False)
    print(df)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    main(p.parse_args())
