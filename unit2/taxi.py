import numpy as np
import gymnasium as gym
import random
import imageio
import tqdm

# Create environment
env = gym.make("Taxi-v3", render_mode="rgb_array")
def init_q_table(s_space, a_space):
    return np.zeros((s_space, a_space))

def greedy_action(q_table, state):
    return np.argmax(q_table[state])

def epsilon_greedy_action(q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return greedy_action(q_table, state)

def train(num_train_episodes, min_epsilon, max_epsilon, decay_rate, env, 
                                                        lr, max_steps, q_table):
    for episode in tqdm.tqdm(range(int(num_train_episodes))):
        # reduce epsilon
        epsilon = (min_epsilon + (max_epsilon - min_epsilon) * 
                                            np.exp(-decay_rate * episode))
        # reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False
        
        # Loop for each step of the episode
        for step in range(int(max_steps)):
            # Choose an action
            action = epsilon_greedy_action(q_table, state, epsilon)
            
            # Take the action and observe the outcome state and reward
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Update Q-table using TD(Temporal Difference) learning
            q_table[state, action] = q_table[state, action] + lr * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            
            # Transition to the next state
            state = next_state
            
            if terminated or truncated:
                break
    return q_table

def evaluate(q_table, num_eval_episodes, max_steps, env, seed=None):
    eval_reward = []
    for episode in tqdm.tqdm(range(int(num_eval_episodes))):
        if seed:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()
        step = 0
        terminated = False
        truncated = False
        total_rewards = 0
        
        for step in range(int(max_steps)):
            action = greedy_action(q_table, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_rewards += reward
            state = next_state
            
            if terminated or truncated:
                break
        eval_reward.append(total_rewards)
    mean_reward = np.mean(eval_reward)
    std_reward = np.std(eval_reward)
    return mean_reward, std_reward

def record_video(env, Qtable, out_directory, fps=1):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not terminated or truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action = greedy_action(Qtable, state)
        state, reward, terminated, truncated, info = env.step(
            action
        )  # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for  img in images], fps=fps)

# Traning parameters
num_train_episodes = 3e4
lr = 0.7

# Evaluation parameters
num_eval_episodes = 100

# Environment parameters
env_name = "FrozenLake-v1"
max_steps = 100
gamma = 0.95
eval_seed = []

# Exploration parameters
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005


s_space = env.observation_space.n
a_space = env.action_space.n

# Initialize Q-table
q_table = init_q_table(s_space, a_space)
q_table = train(num_train_episodes, min_epsilon, max_epsilon, decay_rate, env, lr, max_steps, q_table)
mean_reward, std_reward = evaluate(q_table, num_eval_episodes, max_steps, env, eval_seed)
record_video(env, q_table, "unit2/taxi.mp4", fps=8)

print("Training finished.\n")
print("Q-table:")
print(q_table)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")