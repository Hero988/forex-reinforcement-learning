import os
import zipfile
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import os
import matplotlib.dates as mdates

"""## Collect and Extract the github"""

# Define the directory and file paths
data_dir = "forex_data"
zip_file_path = "forex_data.zip"
zip_url = "https://github.com/Hero988/forex-reinforcement-learning/blob/main/forex_data_pair_per_folder.zip?raw=true"

# Check if the data directory already exists
if not os.path.exists(data_dir):
  print("Data directory not found. Downloading and extracting data...")

  # Download the ZIP file
  response = requests.get(zip_url)
  if response.status_code == 200:
      with open(zip_file_path, "wb") as zip_file:
          zip_file.write(response.content)
      print("Download complete.")
  else:
      print("Failed to download the ZIP file. Please check the URL.")
      exit()

  # Extract the ZIP file
  with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
      zip_ref.extractall(data_dir)
  print(f"Data extracted to {data_dir}.")

  # Clean up the ZIP file
  os.remove(zip_file_path)
  print("ZIP file removed.")
else:
  print("Data directory already exists. Skipping download and extraction.")

class ForexTradingEnv(gym.Env):
    """
    Custom Forex Trading Environment
    """
    def __init__(self, df, initial_balance=10000):
        super(ForexTradingEnv, self).__init__()

        # Market data
        self.df = df
        self.current_step = 0

        # Initial settings
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = 0  # Amount of currency held
        self.net_worth = initial_balance

        # Action space: [Hold, Buy, Sell]
        self.action_space = spaces.Discrete(3)

        # Observation space: [Open, High, Low, Close, Volume, Balance, Positions]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        # Seeding for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset environment state
        self.balance = self.initial_balance
        self.positions = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        self.net_worths = [self.net_worth]  # Reset net worth tracking
        self.rewards = []  # Reset rewards tracking
        self.steps = [0]  # Track step numbers

        # Return the initial observation and an empty info dictionary
        return self._next_observation(), {}

    def _next_observation(self):
        """
        Get the next observation from the environment.
        """
        data = self.df.iloc[self.current_step]
        obs = np.array([
            data['open'],
            data['high'],
            data['low'],
            data['close'],
            data['tick_volume'],
            self.balance
        ])
        return obs

    def step(self, action):
        """
        Perform an action and advance the environment by one step.
        """
        # Get current market data
        data = self.df.iloc[self.current_step]
        current_price = data['close']

        # Calculate reward
        reward = 0
        if action == 1:  # Buy
            self.positions += self.balance / current_price
            self.balance = 0
        elif action == 2:  # Sell
            self.balance += self.positions * current_price
            self.positions = 0
        self.net_worth = self.balance + self.positions * current_price
        reward = self.net_worth - self.net_worths[-1]  # Reward is the change in net worth

        # Track performance
        self.net_worths.append(self.net_worth)
        self.rewards.append(reward)
        self.steps.append(self.current_step)

        # Get the current time
        current_time = data['time']  # Ensure the 'time' column exists in your DataFrame

        # Move to the next step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # Add a placeholder for the truncated flag

        # Return observation, reward, terminated, truncated, and info
        return (
            self._next_observation(),
            reward,
            terminated,
            truncated,
            {"net_worth": self.net_worth, "time": current_time},
        )

def plot_performance(times, net_worths, rewards, output_dir):
  """
  Plot net worth and rewards over time.
  """
  # Convert times to datetime if not already
  if isinstance(times[0], str):
      times = pd.to_datetime(times)

  plt.figure(figsize=(12, 8))

  # Plot net worth over time
  plt.subplot(2, 1, 1)
  plt.plot(times, net_worths, label="Net Worth", color="blue")
  plt.title("Net Worth Over Time")
  plt.xlabel("Time")
  plt.ylabel("Net Worth")
  plt.legend()
  plt.grid()
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
  plt.xticks(rotation=45)

  # Plot rewards over time
  plt.subplot(2, 1, 2)
  plt.plot(times, rewards, label="Rewards", color="green")
  plt.title("Rewards Over Time")
  plt.xlabel("Time")
  plt.ylabel("Reward")
  plt.legend()
  plt.grid()
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
  plt.xticks(rotation=45)

  plt.tight_layout()

  # Save the plot to the output directory
  plt.savefig(os.path.join(output_dir, "equity_plot.png"))
  plt.close()

def train_evaluate_save(data_dir, output_dir, timesteps=800000):
  os.makedirs(output_dir, exist_ok=True)  # Ensure the main output directory exists
  for pair_folder in os.listdir(data_dir):
      pair_path = os.path.join(data_dir, pair_folder)
      if not os.path.isdir(pair_path):
          continue

      # Create a subfolder inside output_dir for the pair_folder
      pair_output_dir = os.path.join(output_dir, pair_folder)
      os.makedirs(pair_output_dir, exist_ok=True)

      # Load Data
      five_years_data_path = os.path.join(pair_path, f"{pair_folder}_5_years.csv")
      one_year_data_path = os.path.join(pair_path, f"{pair_folder}_2024_present.csv")
      if not os.path.exists(five_years_data_path) or not os.path.exists(one_year_data_path):
          print(f"Data files missing for {pair_folder}")
          continue

      df_train = pd.read_csv(five_years_data_path)
      df_eval = pd.read_csv(one_year_data_path)

      # Initialize Environment
      env_training = DummyVecEnv([lambda: ForexTradingEnv(df_train)])
      env_evaluating = DummyVecEnv([lambda: ForexTradingEnv(df_eval)])

      # Train Model
      model = DQN("MlpPolicy", env_training, verbose=1)
      model.learn(total_timesteps=timesteps)

      # Save Model in pair's subfolder
      model_save_path = os.path.join(pair_output_dir, f"{pair_folder}_dqn_model")
      model.save(model_save_path)

      # Evaluate Model
      obs = env_evaluating.reset()
      total_reward, net_worths, rewards, times = 0, [], [], []
      while True:
          action, _ = model.predict(obs, deterministic=True)
          obs, reward, done, info = env_evaluating.step(action)  # Only 4 values are unpacked

          # Extract values from vectorized environment
          reward = reward[0]
          done = done[0]
          info = info[0]

          # Track performance
          net_worths.append(info["net_worth"])
          rewards.append(reward)
          times.append(info["time"])
          total_reward += reward

          # Check termination
          if done:
              break

      # Save Trade Diary in pair's subfolder
      trade_diary_path = os.path.join(pair_output_dir, f"{pair_folder}_trade_diary.csv")
      trade_diary = pd.DataFrame({"Time": times, "Net Worth": net_worths, "Reward": rewards})
      trade_diary.to_csv(trade_diary_path, index=False)

      # Save Equity Plot in pair's subfolder
      plot_performance(times, net_worths, rewards, pair_output_dir)

      print(f"Completed training and evaluation for {pair_folder}")

data_directory = "forex_data_pair_per_folder"
output_directory = "forex_results"
train_evaluate_save(data_directory, output_directory)
