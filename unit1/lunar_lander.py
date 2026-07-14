import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Instantiate the agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.98,
    ent_coef = 0.01,
    verbose=0,
    device="cuda"
)

# Train the agent
model.learn(total_timesteps=int(1e4))

model_name = "PPO_lunar_lander_V3"
model.save(model_name)

eval_env = Monitor(gym.make("LunarLander-v3", render_mode="rgb_array"))

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
