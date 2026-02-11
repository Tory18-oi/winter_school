import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
ENV_NAME = "FrozenLake-v1"
MAP_SIZE = "4x4" 
          
SLIPPERY = False           

TRAIN_EPISODES = 4000
MAX_STEPS = 100
LEARNING_RATE = 0.8        
DISCOUNT_RATE = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.002      
EPSILON_MIN = 0.01

def save_plots(rewards):
    """
    Generates and saves a graph of the training progress.
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(rewards, color='cyan', alpha=0.3, label='Raw Reward')
    
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, color='blue', linewidth=2, label='Moving Avg (50 eps)')

    plt.title(f"Agent Learning Progress ({ENV_NAME} | slippery={SLIPPERY})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    folder_path = "rl-methods/experiments"
    os.makedirs(folder_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder_path}/frozenlake_q-learning_{MAP_SIZE}_{'no_slip' if not SLIPPERY else 'slip'}_{timestamp}.png"
    
    plt.savefig(filename)
    print(f"📊 Metrics saved to {filename}")
    plt.close()

def watch_agent(qtable=None, delay=0.4):
    """
    Runs one episode visually.
    """
    env = gym.make(ENV_NAME, map_name=MAP_SIZE, is_slippery=SLIPPERY, render_mode="human")
    state, info = env.reset()
    done = False
    total_reward = 0
    
    print("\n🎬 Simulation Started...")
    
    for step in range(MAX_STEPS):
        if qtable is None:
            action = env.action_space.sample()           # Random
        else:
            action = np.argmax(qtable[state, :])         # Greedy
            
        new_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = new_state
        
        time.sleep(delay)
        
        if done:
            break
            
    if total_reward > 0.9:
        print(f"🏆 Успіх! Total Score: {total_reward}")
    else:
        print(f"Episode finished. Total Score: {total_reward}")
    
    env.close()

def train_agent():
    """
    Trains the agent and tracks rewards.
    """
    env = gym.make(ENV_NAME, map_name=MAP_SIZE, is_slippery=SLIPPERY, render_mode=None)
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))
    
    epsilon = EPSILON_START
    rewards_history = []

    print(f"🔄 Training {ENV_NAME} ({MAP_SIZE}, slippery={SLIPPERY}) for {TRAIN_EPISODES} episodes...")
    
    for ep in tqdm(range(TRAIN_EPISODES)):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        for _ in range(MAX_STEPS):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)
            
            # Q-Learning update
            current_q = qtable[state, action]
            max_future_q = np.max(qtable[new_state, :])
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q - current_q)
            qtable[state, action] = new_q
            
            state = new_state
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards_history.append(episode_reward)
        epsilon = max(EPSILON_MIN, epsilon - EPSILON_DECAY)
        
    env.close()
    
    save_plots(rewards_history)
    
    return qtable

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"❄️ Q-Learning on FrozenLake ({MAP_SIZE}, slippery={SLIPPERY})")
    
    # 1. Подивись на випадкового агента
    input("\n❌ Натисни [Enter] щоб подивитись НЕНАВЧЕНОГО агента...")
    watch_agent(qtable=None, delay=0.4)
    
    # 2. Навчай + графік
    input("💪 Натисни [Enter] щоб НАВЧИТИ агента та побудувати графік...")
    trained_qtable = train_agent()
    print("✅ Навчання завершено! Дивись збережену картинку.")

    # 3. Подивись на навченого
    while True:
        resp = input("\n🏆 Натисни [Enter] щоб подивитись НАВЧЕНОГО агента (або 'q' для виходу)... ")
        if resp.lower() in ['q', 'quit', 'вихід']:
            break
        watch_agent(qtable=trained_qtable, delay=0.4)