import json 
import csv 
from pathlib import Path

class Logger: 
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.run_dir / 'metrics.jsonl'
        self.loss_file = self.run_dir / 'loss_curves.csv'

        with open(self.loss_file,"w") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])

    def log_metrics(self, metrics: dict):

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def log_loss(self, step: int, loss: float):
        with open(self.loss_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([step, loss])
    