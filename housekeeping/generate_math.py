import os
import json
import pandas as pd

def calculate_mechanistic_metrics(root_dir="results"):
    rows = []
    # Identify our three experiment arms
    runs = {"phase1_run1": "Harm->Help->Style", 
            "phase1_run2": "Help->Harm->Style", 
            "phase1_run3": "Style->Help->Harm"}
    
    for run_id, run_name in runs.items():
        run_path = os.path.join(root_dir, run_id)
        if not os.path.exists(run_path): continue
        
        # Load Stage 0 (Base), Stage 1, Stage 2, Stage 3
        data_by_stage = {}
        for s in range(4):
            # Locate the json file (handling the nested 'eval_all' folder)
            stage_dir = f"stage_{s}_base_eval" if s == 0 else f"stage_{s}_{run_name.split('->')[s-1].lower().replace('.','')}"
            # Fuzzy match stage directory names
            actual_dir = [d for d in os.listdir(run_path) if d.startswith(f"stage_{s}")]
            if not actual_dir: continue
            
            p1 = os.path.join(run_path, actual_dir[0], "eval_results.json")
            p2 = os.path.join(run_path, actual_dir[0], "eval_all", "eval_results.json")
            eval_path = p1 if os.path.exists(p1) else p2
            
            if os.path.exists(eval_path):
                with open(eval_path, 'r') as f:
                    data_by_stage[s] = json.load(f)

        # Compute Metrics
        for factor in ['harmless', 'helpful', 'style']:
            # 1. Peak Performance (The best the model ever did on this factor)
            peak_acc = max([data_by_stage[s][factor]['pref_accuracy'] for s in data_by_stage])
            # 2. Final Performance (At the end of the sequence)
            final_acc = data_by_stage[3][factor]['pref_accuracy']
            # 3. Forgetting/Drift (Absolute drop)
            drift = peak_acc - final_acc
            # 4. Retention Ratio (Percentage of peak preserved)
            retention = (final_acc / peak_acc) if peak_acc > 0 else 0

            rows.append({
                "Factor": factor,
                "Run_Order": run_name,
                "Peak_Accuracy": round(peak_acc, 4),
                "Final_Accuracy": round(final_acc, 4),
                "Net_Drift": round(drift, 4),
                "Retention_Ratio": round(retention, 4)
            })

    df = pd.DataFrame(rows)
    
    # Generate the Text Report
    with open(os.path.join(root_dir, "alignment_math_report.txt"), "w") as f:
        f.write("=== PHENOMENON VALIDATION: ANISOTROPIC DRIFT REPORT ===\n\n")
        
        f.write("1. INTRINSIC VOLATILITY (By Factor)\n")
        f.write("------------------------------------\n")
        f.write("This shows how much each factor decays on average, regardless of order.\n")
        summary = df.groupby("Factor")[["Net_Drift", "Retention_Ratio"]].mean()
        f.write(summary.to_string())
        f.write("\n\n")
        
        f.write("2. ORDER SENSITIVITY (Cross-Run Consistency)\n")
        f.write("--------------------------------------------\n")
        f.write("If drift is similar across runs, the anisotropy is INTRINSIC to the factor.\n")
        pivot = df.pivot(index="Factor", columns="Run_Order", values="Net_Drift")
        f.write(pivot.to_string())
        f.write("\n\n")
        
        f.write("3. CONCLUSION FOR REVIEWERS\n")
        f.write("---------------------------\n")
        max_drift_factor = summary['Net_Drift'].idxmax()
        min_drift_factor = summary['Net_Drift'].idxmin()
        f.write(f"- Most Volatile Factor (Low Gamma Candidate): {max_drift_factor.upper()}\n")
        f.write(f"- Most Stable Factor (High Gamma Candidate): {min_drift_factor.upper()}\n")
        f.write(f"- Evidence of Anisotropy: The variance in Net_Drift is {summary['Net_Drift'].var():.6f}\n")

    # Save as JSON for programmatic use
    df.to_json(os.path.join(root_dir, "alignment_metrics_summary.json"), orient="records", indent=2)
    print("Math report generated: results/alignment_math_report.txt")

if __name__ == "__main__":
    calculate_mechanistic_metrics()