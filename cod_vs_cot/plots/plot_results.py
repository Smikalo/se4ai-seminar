"""
Plot accuracy & latency side-by-side.

python -m plots.plot_results results/summary.csv
"""
import pandas as pd, argparse, matplotlib.pyplot as plt, pathlib

def plot(df):
    tasks = df["task"].str.extract(r"^(.*)__")[0]
    df["task_clean"] = tasks
    figure, ax1 = plt.subplots(figsize=(10,6))
    width = 0.35
    idx = range(len(df)//2)
    cod = df[df["task"].str.contains("__cod")]
    cot = df[df["task"].str.contains("__cot")]

    ax1.bar([i - width/2 for i in idx], cod["accuracy"], width, label="CoD accuracy", alpha=0.5)
    ax1.bar([i + width/2 for i in idx], cot["accuracy"], width, label="CoT accuracy", alpha=0.5)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(idx, cod["task_clean"])
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(idx, cod["latency_avg"], marker="o", linestyle="--", label="CoD latency(s)")
    ax2.plot(idx, cot["latency_avg"], marker="x", linestyle="--", label="CoT latency(s)")
    ax2.set_ylabel("Avg latency (s)")
    ax2.legend(loc="upper right")

    plt.title("CoD vs CoT: Accuracy & Latency by Task")
    plt.tight_layout()
    outfile = pathlib.Path("results/comparison.png")
    plt.savefig(outfile, dpi=180)
    print(f"Plot saved to {outfile}")

def main(csv):
    df = pd.read_csv(csv)
    plot(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    main(parser.parse_args().csv)
