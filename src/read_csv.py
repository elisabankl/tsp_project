import pandas as pd


#df = pd.read_csv("comparison_data_FlowShop_100runs_20250727_145740.csv")
#print(df.columns)
#print(df[["p","Cheapest Insertion_avg_reward", "Cheapest Insertion_std_dev"]])

import numpy as np

def std_to_95ci(std_dev, sample_size):
    return 1.962 * std_dev / np.sqrt(sample_size)

# Use with your dataframe
sample_size = 100
print(std_to_95ci(0.598876957304356 , sample_size))