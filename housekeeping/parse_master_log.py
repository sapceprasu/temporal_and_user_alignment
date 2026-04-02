import argparse
import json
import os

import pandas as pd
import matplotlib.pyplot as plt


FACTORS = ["harmless", "helpful", "style"]
MAIN_METRIC = "policy_pref_accuracy_norm"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_final_table(records, metric=MAIN_METRIC):
    rows = []
    for rec in records:
        order = rec["order_name"]
        final_stage = f"stage_3_{rec['order'][-1]}"
        eval_metrics = rec["stages"][final_stage]["eval_metrics"]

        vals = {factor: eval_metrics[factor][metric] for factor in FACTORS}
        row = {
            "order": order,
            **vals,
            "avg": sum(vals.values()) / len(vals),
            "worst": min(vals.values()),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("avg", ascending=False)
    return df


def extract_stage_table(records, metric=MAIN_METRIC):
    rows = []
    for rec in records:
        order = rec["order_name"]
        for stage_name, stage_data in rec["stages"].items():
            if stage_name == "stage_0_base_eval":
                eval_metrics = stage_data
                stage_idx = 0
            else:
                eval_metrics = stage_data["eval_metrics"]
                stage_idx = int(stage_name.split("_")[1])

            for factor in FACTORS:
                rows.append({
                    "order": order,
                    "stage": stage_name,
                    "stage_idx": stage_idx,
                    "factor": factor,
                    "value": eval_metrics[factor][metric],
                })
    return pd.DataFrame(rows)


def plot_final_heatmap(df, output_path):
    plot_df = df.set_index("order")[["harmless", "helpful", "style", "avg", "worst"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(plot_df.values, aspect="auto")

    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            ax.text(j, i, f"{plot_df.iloc[i, j]:.3f}", ha="center", va="center")

    ax.set_title("Final performance across training orders")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_stage_trajectories(stage_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for order, sub in stage_df.groupby("order"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for factor in FACTORS:
            s = sub[sub["factor"] == factor].sort_values("stage_idx")
            ax.plot(s["stage_idx"], s["value"], marker="o", label=factor)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["base", "stage1", "stage2", "stage3"])
        ax.set_ylabel(MAIN_METRIC)
        ax.set_title(order)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{order}_trajectory.png"), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_log", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = read_jsonl(args.master_log)

    final_df = extract_final_table(records)
    final_csv = os.path.join(args.output_dir, "final_order_comparison.csv")
    final_df.to_csv(final_csv, index=False)

    stage_df = extract_stage_table(records)
    stage_csv = os.path.join(args.output_dir, "stagewise_table.csv")
    stage_df.to_csv(stage_csv, index=False)

    plot_final_heatmap(final_df, os.path.join(args.output_dir, "final_order_heatmap.png"))
    plot_stage_trajectories(stage_df, os.path.join(args.output_dir, "trajectories"))

    print("Saved:")
    print(final_csv)
    print(stage_csv)
    print(os.path.join(args.output_dir, "final_order_heatmap.png"))
    print(os.path.join(args.output_dir, "trajectories"))


if __name__ == "__main__":
    main()