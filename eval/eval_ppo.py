from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def Eval_PPO(
    env: ImgObsWrapper,
    model: PPO,
    eps: int,
    graph_label: str,
    color: str
) -> None:
    rewards = [0]
    total_rewards = 0
    for i in range(eps):
        steps = 0
        obs, _ = env.reset()
        while True:
            action, _state = model.predict(obs)
            obs, _reward, terminated, trunctated, info = env.step(action)
            steps += 1
            if terminated or trunctated:
                total_rewards += _reward
                rewards.append(total_rewards)
                print(f'eval_ep: {i+1}/{eps} | current_reward: {_reward} | total_rewards: {total_rewards}', end='\r')
                obs = env.reset()
                break
    print(f'\nfinished evaluating {eps} eps. total_rewards: {total_rewards}')
    plt.figure(figsize=(15,5))
    plt.xlim(0, eps)
    plt.plot(rewards, linestyle="-", color=color)
    plt.title(graph_label, fontname='sans-serif', fontsize=14, fontstyle='italic')
    plt.xlabel("Episode", fontname='sans-serif', fontweight="semibold")
    plt.ylabel("Reward", fontname='sans-serif', fontweight="semibold")
    plt.savefig(f'./results/{graph_label.replace(" ", "_")}_eval_rewards.png')