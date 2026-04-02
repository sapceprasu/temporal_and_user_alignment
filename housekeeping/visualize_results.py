import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_plot_style():
    sns.set_theme(style="white", palette="muted")
    # Fallback font logic for Linux servers
    plt.rcParams.update({
        "font.family": "DejaVu Serif", # Standard on most Linux distros
        "axes.labelweight": "bold",
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

def load_data(root_dir="results"):
    rows = []
    runs = {"phase1_run1": "Harm.->Help.->Style", 
            "phase1_run2": "Help.->Harm.->Style", 
            "phase1_run3": "Style->Help.->Harm."}
    
    for run_id, run_name in runs.items():
        run_path = os.path.join(root_dir, run_id)
        if not os.path.exists(run_path): continue
        
        stages = sorted([d for d in os.listdir(run_path) if d.startswith("stage_")])
        for stage in stages:
            eval_file = "eval_results.json"
            stage_path = os.path.join(run_path, stage)
            
            # Check for nested or direct eval results
            potential_paths = [
                os.path.join(stage_path, "eval_all", eval_file),
                os.path.join(stage_path, eval_file)
            ]
            
            eval_path = next((p for p in potential_paths if os.path.exists(p)), None)

            if eval_path:
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
                    stage_num = int(stage.split('_')[1])
                    for factor in ['harmless', 'helpful', 'style']:
                        rows.append({
                            "Run_ID": run_id,
                            "Order": run_name,
                            "Stage": f"Stage {stage_num}",
                            "Stage_Int": stage_num,
                            "Factor": factor.capitalize(),
                            "Accuracy": eval_data[factor]["pref_accuracy"] * 100,
                            "Margin": eval_data[factor]["reward_margin"],
                            "Loss": eval_data[factor]["loss"]
                        })
    return pd.DataFrame(rows)

def plot_comprehensive_matrix(df, output_dir):
    """Figure 1: Perfectly Uniform Faceted Heatmaps"""
    orders = df['Order'].unique()
    # Create a figure with a dedicated space for the shared colorbar at the bottom
    fig, axes = plt.subplots(1, len(orders), figsize=(18, 6), sharey=True)
    
    # Define a shared color range (min/max) so the colors are comparable across plots
    vmin = df['Accuracy'].min()
    vmax = df['Accuracy'].max()

    for i, order in enumerate(orders):
        data = df[df['Order'] == order].pivot_table(index="Stage", columns="Factor", values="Accuracy")
        
        # We set cbar=False for individual plots and use a shared one later
        sns.heatmap(data, annot=True, fmt=".1f", cmap="magma", 
                    ax=axes[i], cbar=False, vmin=vmin, vmax=vmax,
                    annot_kws={"size": 12, "weight": "bold"})
        
        axes[i].set_title(order, pad=20, fontsize=14, fontweight='bold')
        axes[i].set_xlabel("") # Clean up X labels
        if i == 0:
            axes[i].set_ylabel("Sequential Training Stage", fontsize=12)
        else:
            axes[i].set_ylabel("")

    # Create a single colorbar for all plots at the bottom
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.03]) 
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Preference Accuracy (%)')

    plt.suptitle("Anisotropic Retention Landscape: Cross-Order Comparison", y=1.02, fontsize=18)
    # Adjust layout to make room for titles and colorbar
    plt.subplots_adjust(bottom=0.2, wspace=0.1) 
    
    plt.savefig(os.path.join(output_dir, "fig1_landscape_matrix_uniform.png"), bbox_inches='tight')
    plt.close()

def plot_forgetting_deltas(df, output_dir):
    """Figure 2: The 'Net Drift' Chart - Simple and Powerful"""
    # Calculate the change from Stage 1 (after initial training) to Stage 3 (final)
    # This shows the 'Survivability' of the factor
    
    summary_data = []
    for run in df['Order'].unique():
        run_df = df[df['Order'] == run]
        for factor in ['Harmless', 'Helpful', 'Style']:
            # Accuracy after it was first trained
            # (Finding the first stage where it wasn't the base 50%)
            initial_acc = run_df[run_df['Accuracy'] > 51].groupby('Factor').get_group(factor)['Accuracy'].iloc[0]
            # Final accuracy at Stage 3
            final_acc = run_df[run_df['Stage'] == 'Stage 3'].groupby('Factor').get_group(factor)['Accuracy'].iloc[0]
            
            summary_data.append({
                "Order": run,
                "Factor": factor,
                "Net_Drift": final_acc - initial_acc
            })
            
    drift_df = pd.DataFrame(summary_data)

    plt.figure(figsize=(10, 6))
    # Plotting the Average Net Drift across ALL runs
    sns.barplot(data=drift_df, x="Factor", y="Net_Drift", palette="coolwarm", capsize=.1)
    
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.title("Intrinsic Factor Volatility (Aggregated Across All Orders)", fontsize=14, fontweight='bold')
    plt.ylabel("Net Change in Accuracy (%) after further training", fontsize=12)
    plt.xlabel("Alignment Factor", fontsize=12)
    
    # Adding a text box explanation for the reviewer
    plt.text(0.5, drift_df['Net_Drift'].min() - 2, 
             "Negative values indicate 'Forgetting' / Interference", 
             ha='center', color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(output_dir, "fig2_simple_drift.png"), bbox_inches='tight')
    plt.close()

def plot_confidence_radar(df, output_dir):
    """Figure 3: Radar Chart for Confidence (Margin)"""
    # Group by stage across all runs to see the confidence 'shape'
    radar_df = df.groupby(['Stage_Int', 'Factor'])['Margin'].mean().unstack()
    labels = radar_df.columns
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    for i, (idx, row) in enumerate(radar_df.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=f"Stage {idx}")
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title("Alignment Confidence Geometry: Reward Margin Radar", y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(output_dir, "fig3_confidence_radar.png"), bbox_inches='tight')
    plt.close()

def main():
    setup_plot_style()
    output_dir = "results/visuals_comprehensive"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data()
    if df.empty:
        print("No data found. Check your 'results' directory structure.")
        return

    plot_comprehensive_matrix(df, output_dir)
    plot_forgetting_deltas(df, output_dir)
    plot_confidence_radar(df, output_dir)
    
    print(f"Comprehensive suite generated in {output_dir}")

if __name__ == "__main__":
    main()