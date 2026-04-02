import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_per_example_data(root_results_dir: str, run_id: str, factor_name: str) -> pd.DataFrame:
    run_path = Path(root_results_dir) / run_id
    records = []
    
    # Locate and sort stage directories (e.g., stage_0, stage_1)
    stage_dirs = sorted(run_path.glob("stage_*"), key=lambda p: p.name)
    
    for stage_path in stage_dirs:
        # Check primary and nested evaluation paths
        file_name = f"{factor_name}_per_example.jsonl"
        file_path = stage_path / file_name
        
        if not file_path.exists():
            file_path = stage_path / "eval_all" / file_name
            
        if not file_path.exists():
            continue

        stage_label = stage_path.name.replace("stage_", "Stage ")
        
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                records.append({
                    "Stage": stage_label,
                    "Margin": data.get("reward_margin", 0),
                    "Factor": factor_name
                })
                
    return pd.DataFrame(records)

def plot_margin_ridge(df: pd.DataFrame, factor_name: str, output_path: str):
    """
    Generates a ridge plot (joyplot) showing the evolution of reward margins.
    """
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Create the aesthetic palette
    palette = sns.cubehelix_palette(df["Stage"].nunique(), rot=-.25, light=.7)
    
    # Initialize the FacetGrid
    grid = sns.FacetGrid(
        df, 
        row="Stage", 
        hue="Stage", 
        aspect=9, 
        height=1.2, 
        palette=palette
    )
    
    # Map KDE plots to create the 'ridge' effect
    grid.map(sns.kdeplot, "Margin", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    grid.map(sns.kdeplot, "Margin", clip_on=False, color="white", lw=2, bw_adjust=.5)
    
    # Reference line at y=0
    grid.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Polish visual styling
    grid.set_titles("")
    grid.set(yticks=[], ylabel="")
    grid.despine(bottom=True, left=True)
    
    plt.suptitle(
        f"Evolution of {factor_name.title()} Preference Distribution", 
        y=0.98, 
        fontsize=14
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    pass