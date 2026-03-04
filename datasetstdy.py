from datasets import load_dataset
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd



prompt = f'Here is the context: {context} \n\n And the question is :Question: {question} \n\n Answer:'

dataset = load_dataset("rajpurkar/squad")
# 
data = pd.DataFrame(dataset['train'][:5])
# print(data)

for a_data in data.itertuples():
    
    
