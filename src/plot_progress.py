import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file, skipping metadata lines
df = pd.read_csv('masked_ppo_eval_non_normalized_monitor.csv', header=None, names=['r', 'l', 't'], comment='#')

# Convert columns to numeric types
df['r'] = pd.to_numeric(df['r'], errors='coerce')
df['l'] = pd.to_numeric(df['l'], errors='coerce')
df['t'] = pd.to_numeric(df['t'], errors='coerce')

# Drop rows with NaN values (which were non-numeric)
df.dropna(inplace=True)

# Calculate the rolling average
window_size = 5000  # You can adjust the window size as needed
df['rolling_avg'] = df['r'].rolling(window=window_size).mean()

# Plot the results
plt.figure(figsize=(10, 6))
#plt.plot(df['r'], label='Original Rewards')
plt.plot(df['rolling_avg'], label=f'Rolling Average (window={window_size})', color='orange')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rolling Average of Rewards During Evaluation (Every 50,000 Training Steps, 5,000 Evaluation Steps)')
plt.legend()
plt.show()