import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pandas as pd
import zipfile
import requests

class ForexTradingEnv(gym.Env):
    """
    Custom Environment for Forex trading using the Gymnasium framework.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, output_dir, initial_balance=1000, window_size=12, lot_size=10000, risk_factor = 0.1):
        super(ForexTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.lot_size = lot_size
        self.output_dir = output_dir
        self.risk_factor = risk_factor  # Define the risk factor

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Actions: 0 = hold, 1 = buy, 2 = sell

        # Flattened observation size: window_size * number_of_features
        obs_size = self.window_size * len(self.df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Initialize variables (unchanged)
        self.current_step = 0
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.position = 0
        self.entry_price = 0
        self.times = []
        self.net_worths = [initial_balance]
        self.unrealized_highs = []
        self.unrealized_lows = []
        self.net_worth = self.balance
        self.max_net_worth = initial_balance  # Track the highest net worth so far
        self.max_drawdown = 0  # Track the largest drawdown

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.times = []
        self.net_worths = [self.initial_balance]  # Initialize with the starting net worth
        self.unrealized_highs = []
        self.unrealized_lows = []
        self.net_worth = self.balance
        self.max_net_worth = self.initial_balance  # Track the highest net worth so far
        self.max_drawdown = 0  # Track the largest drawdown
        observation = self._get_observation()
        info = {}  # Optional dictionary for additional information
        return observation, info

    def step(self, action):
        done = False
        reward = 0

        # Get the current price
        current_price = self.df.loc[self.current_step, 'close']

        # Extract components from the new columns
        current_year = self.df.loc[self.current_step, 'year']
        current_month = self.df.loc[self.current_step, 'month']
        current_day = self.df.loc[self.current_step, 'day']
        current_hour = self.df.loc[self.current_step, 'hour']
        current_minute = self.df.loc[self.current_step, 'minute']
        current_second = self.df.loc[self.current_step, 'second']

        # Combine the components into a single string or datetime object
        current_time = f"{current_year:04d}-{current_month:02d}-{current_day:02d} {current_hour:02d}:{current_minute:02d}:{current_second:02d}"

        current_high_price = self.df.loc[self.current_step, 'high']
        current_low_price = self.df.loc[self.current_step, 'low']
        
        profit = 0

        # Execute action
        if action == 1:  # Buy
            if self.position == 0:  # Open a new Buy position
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:  # Close a Sell position
                profit = (self.entry_price - current_price) * self.lot_size
                self.net_worth += profit
                self.position = 1
                self.entry_price = current_price
        elif action == 2:  # Sell
            if self.position == 0:  # Open a new Sell position
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:  # Close a Buy position
                profit = (current_price - self.entry_price) * self.lot_size
                self.net_worth += profit
                self.position = -1
                self.entry_price = current_price
        elif action == 0:  # Hold
            pass  # No manual reward adjustment here

        unrealized_high = 0
        unrealized_low = 0
        current_drawdown = 0

        if self.position == 1:  # Long position
            unrealized_high = (current_high_price - self.entry_price) * self.lot_size
            unrealized_low = (current_low_price - self.entry_price) * self.lot_size
            self.max_net_worth = max(self.max_net_worth, self.net_worth + unrealized_high)  # Use unrealized high
        elif self.position == -1:  # Short position
            unrealized_high = (self.entry_price - current_low_price) * self.lot_size
            unrealized_low = (self.entry_price - current_high_price) * self.lot_size
            self.max_net_worth = max(self.max_net_worth, self.net_worth + unrealized_low)  # Use unrealized low
        else:
            # For no position, update max_net_worth based only on the current net worth
            self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Calculate drawdown
        if self.position != 0:  # Consider drawdown only if a position is open
            # Use the maximum (less negative) of unrealized_high and unrealized_low
            worst_unrealized = max(unrealized_high, unrealized_low)
            potential_net_worth = self.net_worth + worst_unrealized
            current_drawdown = (self.max_net_worth - potential_net_worth) / self.max_net_worth
        else:
            # When no position is open, calculate drawdown based on the current net worth
            current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth

        # Update the maximum drawdown
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        reward = (self.net_worth - self.net_worths[-1]) - (self.max_drawdown * self.risk_factor)

        # Append time and net worth
        self.times.append(current_time)
        self.net_worths.append(self.net_worth)

        self.unrealized_highs.append(unrealized_high)
        self.unrealized_lows.append(unrealized_low)

        # Check if done
        if self.net_worth <= 0 or self.current_step >= len(self.df) - 1:
            done = True

        self.current_step += 1
        obs = self._get_observation()

        # Add additional info
        info = {"current_time": current_time, "current_step": self.current_step}

        return obs, reward, done, info

    def _get_observation(self):
        """
        Get the current window of price data as the observation.
        """
        start = self.current_step - self.window_size
        end = self.current_step
        obs = self.df.iloc[start:end].values  # Extract data as a NumPy array

        # Adding current profit
        current_profit = self.net_worth - self.initial_balance
        if self.max_drawdown != 0:
            # Adding current drawdown
            current_drawdown = (self.max_drawdown-self.net_worth) / self.max_drawdown
        else:
            current_drawdown = 0

        # Append all new features
        obs = np.append(obs, [self.position, current_drawdown, current_profit, self.net_worth])

        # Flatten the observation
        obs = np.array(obs, dtype=np.float32).flatten()
        return obs


    def render(self, mode='human'):
        """
        Render the environment.
        """
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Net Worth: {self.net_worth:.2f}")
    
    def plot_graph(self):
        plt.figure(figsize=(12, 6))

        # Define the full path for the file
        file_path = os.path.join(self.output_dir, "trade_diary_plot.png")

        # Plot net worth, high, and low
        plt.plot(self.times, self.net_worths[1:], label="Net Worth", linewidth=2)
        # plt.plot(self.times, self.net_worths_high, label="Net Worth High", linestyle="--", color="green")
        # plt.plot(self.times, self.net_worths_low, label="Net Worth Low", linestyle="--", color="red")

        # Add labels and title
        plt.xlabel("Time Steps")
        plt.ylabel("Net Worth")
        plt.title("Net Worth Over Time")
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(file_path)
        plt.close()  # Close the plot to free memory

    def generate_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, "trade_diary.csv")

        # Create a DataFrame from the data
        data = {
            "Time": self.times,
            "Net Worth": self.net_worths[1:],
            "Net Worth High": self.unrealized_highs,
            "Net Worth Low": self.unrealized_lows,
        }
        df = pd.DataFrame(data)

        # Save the DataFrame to CSV
        df.to_csv(file_path, index=False)

    def close(self):
        pass

# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def select_action(self, state, step):
        # Epsilon-greedy policy
        epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(-1. * step / self.epsilon_decay)
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore
        else:
            # Flatten the state
            state = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()  # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        # Flatten state and next_state
        state_flat = state.flatten()
        next_state_flat = next_state.flatten()
        
        self.memory.append((state_flat, action, reward, next_state_flat, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists to NumPy arrays first, then create tensors
        states = torch.tensor(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(np.array(actions, dtype=np.int64)).to(self.device)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32)).to(self.device)
        next_states = torch.tensor(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.tensor(np.array(dones, dtype=np.float32)).to(self.device)


        # Q-values of current states
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Q-values of next states
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        shared_output = self.shared_layers(state)
        action_logits = self.actor(shared_output)
        state_value = self.critic(shared_output)
        return action_logits, state_value

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, clip_epsilon=0.2, epochs=10, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=10000)

    def select_action(self, state):
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
        action_logits, _ = self.policy_net(state_tensor)
        temperature = 1.0  # Adjust this to control randomness (e.g., higher for more exploration)
        action_probs = torch.softmax(action_logits / temperature, dim=-1).cpu().detach().numpy()
        action = np.random.choice(self.action_dim, p=action_probs[0])
        return action, action_probs[0]
        
    def store_transition(self, state, action, action_prob, reward, next_state, done):
        self.memory.append((state, action, action_prob, reward, next_state, done))

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        discounted_sum = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return returns

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, action_probs, rewards, next_states, dones = zip(*self.memory)

        # Convert lists to NumPy arrays first, then create tensors
        states = torch.tensor(np.array(states, dtype=np.float32), device=self.device)
        actions = torch.tensor(np.array(actions, dtype=np.int64), device=self.device)
        action_probs = torch.tensor(np.array(action_probs, dtype=np.float32), device=self.device)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32), device=self.device)
        next_states = torch.tensor(np.array(next_states, dtype=np.float32), device=self.device)
        dones = torch.tensor(np.array(dones, dtype=np.float32), device=self.device)

        _, next_values = self.policy_net(next_states)
        returns = self.compute_returns(rewards.cpu().numpy(), dones.cpu().numpy(), next_values[-1].item())
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                batch_indices = slice(i, i + self.batch_size)
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_action_probs = action_probs[batch_indices]
                batch_returns = returns[batch_indices]

                action_logits, values = self.policy_net(batch_states)
                dist = torch.distributions.Categorical(logits=action_logits)
                new_action_probs = dist.log_prob(batch_actions)  # Log probabilities for selected actions

                # Calculate the ratio for policy loss
                ratios = torch.exp(new_action_probs - torch.log(batch_action_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1) + 1e-10))

                # Clipped surrogate loss
                advantages = batch_returns - values.squeeze()
                clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                # Optimize the policy network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear the memory after training
        self.memory.clear()

def train_ppo(env, agent, save_dir, num_episodes=500):
    rewards_per_episode = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, action_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, action_prob, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Train the agent after each episode
        agent.train()

        rewards_per_episode.append(total_reward)
        last_net_worth = env.net_worths[-1]
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}, Latest Net Worth: {last_net_worth}")

    file_path = os.path.join(save_dir, "ppo_model.pth")
    torch.save(agent.policy_net.state_dict(), file_path)

    return rewards_per_episode

def evaluate_ppo(agent, save_dir, env, num_episodes=1):
    """
    Evaluate a trained PPO agent on the given environment.

    Args:
        agent: Trained PPO agent.
        env: Environment to evaluate the agent on.
        num_episodes: Number of episodes to run for evaluation.

    Returns:
        total_rewards: List of total rewards for each episode.
    """
    total_rewards = []

    # Load the trained model weights with weights_only=True
    file_path = os.path.join(save_dir, "ppo_model.pth")
    agent.policy_net.load_state_dict(torch.load(file_path, weights_only=True))
    agent.policy_net.eval()

    for episode in range(num_episodes):
        state, info = env.reset()  # Unpack the tuple from env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = agent.select_action(state)  # PPO action selection
            state, reward, done, info = env.step(action)  # Ensure state is updated correctly
            episode_reward += reward
            total_rewards.append(reward)  # Append total reward for this episode

        # Optionally render or save data
        env.plot_graph()
        env.generate_csv()

    return total_rewards

def train_dqn(env, agent, save_dir, num_episodes=500, update_target_steps=10):
    rewards_per_episode = []
    steps = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, steps)
            next_state, reward, done, info = env.step(action)

            # Store transition in replay memory
            agent.store_transition(state, action, reward, next_state, done)

            # Train the agent
            agent.train()

            state = next_state
            total_reward += reward
            steps += 1

        # Update target network every few episodes
        if episode % update_target_steps == 0:
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        last_net_worth = env.net_worths[-1]
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}, Latest Net Worth: {last_net_worth}")

    file_path = os.path.join(save_dir, "dqn_model.pth")
    torch.save(agent.policy_net.state_dict(), file_path)

    return rewards_per_episode

def evaluate_dqn(agent, save_dir, env, num_episodes=1):
    """
    Evaluate a trained DQN agent on the given environment.

    Args:
        agent: Trained DQN agent.
        env: Environment to evaluate the agent on.
        num_episodes: Number of episodes to run for evaluation.

    Returns:
        total_rewards: List of total rewards for each episode.
    """
    total_rewards = []

    # Load the trained model weights
    file_path = os.path.join(save_dir, "dqn_model.pth")
    agent.policy_net.load_state_dict(torch.load(file_path, weights_only=True))
    agent.policy_net.eval()

    for episode in range(num_episodes):
        state, info = env.reset()  # Unpack the tuple
        done = False
        episode_reward = 0

        while not done:
            # Convert the state to tensor and use DQN policy for action selection
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            action = agent.policy_net(state_tensor).argmax(dim=1).item()
            
            # Take the action and get the next state, reward, and done flag
            state, reward, done, info = env.step(action)
            episode_reward += reward

            total_rewards.append(reward)  # Append the total reward for this episode

        # Optionally render or save data
        env.plot_graph()
        env.generate_csv()

    return total_rewards

def get_data():
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

def add_time_components_and_remove_time(df, time_column='time'):
    """
    Extracts year, month, day, hour, minute, and second from a datetime column, 
    adds them as new columns, and removes the original time column.
    
    Args:
        df (pd.DataFrame): The dataframe containing the datetime column.
        time_column (str): The name of the datetime column to process.

    Returns:
        pd.DataFrame: The dataframe with new time component columns added and the original time column removed.
    """
    df_new = df.copy()
    # Convert the column to datetime if not already in datetime format
    df_new[time_column] = pd.to_datetime(df_new[time_column])

    # Extract components and add them as new columns
    df_new['year'] = df_new[time_column].dt.year
    df_new['month'] = df_new[time_column].dt.month
    df_new['day'] = df_new[time_column].dt.day
    df_new['hour'] = df_new[time_column].dt.hour
    df_new['minute'] = df_new[time_column].dt.minute
    df_new['second'] = df_new[time_column].dt.second

    # Remove the original time column
    df_new = df_new.drop(columns=[time_column])

    return df_new

def plot_rewards(rewards, save_dir, file_name):
    """
    Plot rewards over episodes.

    Args:
        rewards: List of rewards for each episode.
        save_dir: Directory to save the plot.
        file_name: File name for the saved plot.
    """
    file_path = os.path.join(save_dir, file_name)
    plt.figure(figsize=(12, 6))

    # Generate x values dynamically based on rewards length
    episodes = range(1, len(rewards) + 1)

    # Plot rewards
    plt.plot(episodes, rewards, label="Rewards", linewidth=2)

    # Add labels, title, and legend
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards Over Episodes")
    plt.legend()

    # Show grid for better readability
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(file_path)
    plt.close()

def train_evaluate_save(data_dir, output_dir, models):
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

        df_train_old = pd.read_csv(five_years_data_path)
        df_eval_old = pd.read_csv(one_year_data_path)

        # Convert 'time' column to numeric format (e.g., UNIX timestamp)
        if 'time' in df_train_old.columns:
            df_train = add_time_components_and_remove_time(df_train_old, time_column='time')
            df_eval = add_time_components_and_remove_time(df_eval_old, time_column='time')

        num_episodes = 25

        for model_name in models:
            # Construct the full path for the subfolder inside pair_output_dir
            model_save_path_folder = os.path.join(pair_output_dir, model_name)
            os.makedirs(model_save_path_folder, exist_ok=True)

            # Initialize the training environment
            env_training = ForexTradingEnv(df_train, model_save_path_folder, initial_balance=10000, window_size=12, lot_size=10000)

            # Initialize the evaluating environment
            env_evaluating = ForexTradingEnv(df_eval, model_save_path_folder, initial_balance=10000, window_size=12, lot_size=10000)

            state_dim = env_training.observation_space.shape[0] + 4  # Add 4 for new features
            action_dim = env_training.action_space.n

            if model_name == 'PPO':
                ppo_agent = PPOAgent(state_dim, action_dim)
                rewards_ppo = train_ppo(env=env_training, save_dir=model_save_path_folder, agent=ppo_agent, num_episodes=num_episodes)
                plot_rewards(rewards=rewards_ppo, save_dir=model_save_path_folder,file_name="rewards_for_training.png")
                ppo_rewards_evaluation = evaluate_ppo(ppo_agent, model_save_path_folder, env=env_evaluating)
                plot_rewards(rewards=ppo_rewards_evaluation, save_dir=model_save_path_folder, file_name="rewards_for_evaluation.png")
                
            elif model_name == 'DQN':
                dqn_agent = DQNAgent(state_dim, action_dim)
                rewards_dqn = train_dqn(env=env_training, save_dir=model_save_path_folder, agent=dqn_agent,num_episodes=num_episodes )
                plot_rewards(rewards=rewards_dqn, save_dir=model_save_path_folder,file_name="rewards_for_training.png")
                dqn_rewards_evaluation = evaluate_dqn(dqn_agent, model_save_path_folder, env_evaluating)
                plot_rewards(rewards=dqn_rewards_evaluation, save_dir=model_save_path_folder, file_name="rewards_for_evaluation.png")

        print(f"Completed training and evaluation for {pair_folder}")

get_data()
data_directory = "forex_data_pair_per_folder"
output_directory = "forex_results_reinforcement_learning_manual"
models = ['PPO', 'DQN']
train_evaluate_save(data_directory, output_directory, models)
