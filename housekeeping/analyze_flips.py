import argparse
import json
import os
from typing import Dict, List


FACTORS = ["harmless", "helpful", "style"]


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_stage_dirs(order_dir: str):
    entries = sorted(os.listdir(order_dir))
    stage_dirs = []
    for name in entries:
        full = os.path.join(order_dir, name)
        if os.path.isdir(full) and name.startswith("stage_"):
            stage_dirs.append((name, full))
    return stage_dirs


def load_per_example(order_dir: str):
    stage_dirs = get_stage_dirs(order_dir)
    loaded = {}

    for stage_name, stage_dir in stage_dirs:
        if stage_name == "stage_0_base_eval":
            eval_dir = stage_dir
        else:
            eval_dir = os.path.join(stage_dir, "eval_all")

        factor_data = {}
        for factor in FACTORS:
            path = os.path.join(eval_dir, f"{factor}_per_example.jsonl")
            if os.path.exists(path):
                factor_data[factor] = read_jsonl(path)

        summary_path = os.path.join(eval_dir, "eval_results.json")
        summary = read_json(summary_path) if os.path.exists(summary_path) else {}

        loaded[stage_name] = {
            "summary": summary,
            "per_example": factor_data,
        }

    return loaded


def summarize_stage_metrics(loaded: Dict):
    out = {}
    for stage_name, stage_data in loaded.items():
        out[stage_name] = {}
        for factor in FACTORS:
            if factor in stage_data["summary"]:
                s = stage_data["summary"][factor]
                out[stage_name][factor] = {
                    "policy_pref_accuracy_norm": s.get("policy_pref_accuracy_norm"),
                    "policy_pref_margin_norm": s.get("policy_pref_margin_norm"),
                    "policy_pref_accuracy": s.get("policy_pref_accuracy"),
                    "policy_pref_margin": s.get("policy_pref_margin"),
                    "relative_pref_accuracy": s.get("relative_pref_accuracy"),
                    "reward_margin": s.get("reward_margin"),
                }
    return out


def compare_two_stages(rows_a: List[Dict], rows_b: List[Dict], metric="policy_accuracy_norm"):
    assert len(rows_a) == len(rows_b), "Stage files have different number of examples"

    flips = {
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
        "ties_or_other": 0,
    }

    detailed = []

    for a, b in zip(rows_a, rows_b):
        assert a["example_id"] == b["example_id"], "Example IDs do not align"

        va = a[metric]
        vb = b[metric]

        if va == 0 and vb == 1:
            flips["wrong_to_correct"] += 1
            flip_type = "wrong_to_correct"
        elif va == 1 and vb == 0:
            flips["correct_to_wrong"] += 1
            flip_type = "correct_to_wrong"
        elif va == 1 and vb == 1:
            flips["unchanged_correct"] += 1
            flip_type = "unchanged_correct"
        elif va == 0 and vb == 0:
            flips["unchanged_wrong"] += 1
            flip_type = "unchanged_wrong"
        else:
            flips["ties_or_other"] += 1
            flip_type = "ties_or_other"

        detailed.append({
            "example_id": a["example_id"],
            "factor": a["factor"],
            "prompt": a["prompt"],
            "chosen": a["chosen"],
            "rejected": a["rejected"],
            "from_metric": va,
            "to_metric": vb,
            "from_margin_norm": a["policy_margin_norm"],
            "to_margin_norm": b["policy_margin_norm"],
            "margin_norm_delta": b["policy_margin_norm"] - a["policy_margin_norm"],
            "flip_type": flip_type,
        })

    return flips, detailed


def save_top_examples(path: str, rows: List[Dict], top_k: int = 20):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows[:top_k]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(args):
    loaded = load_per_example(args.order_dir)
    ensure_dir(args.output_dir)

    # 1. save compact stage summary
    stage_summary = summarize_stage_metrics(loaded)
    stage_summary_path = os.path.join(args.output_dir, "stage_metric_summary.json")
    with open(stage_summary_path, "w", encoding="utf-8") as f:
        json.dump(stage_summary, f, indent=2, ensure_ascii=False)

    stage_names = list(loaded.keys())

    # 2. compare consecutive stages for each factor
    all_flip_summary = {}

    for factor in FACTORS:
        all_flip_summary[factor] = {}

        for i in range(len(stage_names) - 1):
            s1 = stage_names[i]
            s2 = stage_names[i + 1]

            if factor not in loaded[s1]["per_example"] or factor not in loaded[s2]["per_example"]:
                continue

            flips, detailed = compare_two_stages(
                loaded[s1]["per_example"][factor],
                loaded[s2]["per_example"][factor],
                metric=args.metric,
            )

            key = f"{s1}__TO__{s2}"
            all_flip_summary[factor][key] = flips

            # save top positive and negative margin changes
            pos_sorted = sorted(detailed, key=lambda x: x["margin_norm_delta"], reverse=True)
            neg_sorted = sorted(detailed, key=lambda x: x["margin_norm_delta"])

            save_top_examples(
                os.path.join(args.output_dir, f"{factor}__{key}__top_improved.jsonl"),
                pos_sorted,
                top_k=args.top_k,
            )
            save_top_examples(
                os.path.join(args.output_dir, f"{factor}__{key}__top_worsened.jsonl"),
                neg_sorted,
                top_k=args.top_k,
            )

            # save all detailed flips if requested
            if args.save_all_detailed:
                with open(
                    os.path.join(args.output_dir, f"{factor}__{key}__all_detailed.jsonl"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for row in detailed:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    flip_summary_path = os.path.join(args.output_dir, "flip_summary.json")
    with open(flip_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_flip_summary, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(f"- {stage_summary_path}")
    print(f"- {flip_summary_path}")
    print(f"- top improved / worsened example files in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="policy_accuracy_norm",
        choices=["policy_accuracy", "policy_accuracy_norm"],
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save_all_detailed", action="store_true")
    args = parser.parse_args()
    main(args)