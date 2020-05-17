import argparse
import torch
import gym
import time

from a3c import ActorCritic

# Parse training arguments.
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--env_name', type=str, default='CartPole-v0',
                    help='name of the gym environment with version')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--verbose', type=bool, default=False,
                    help='log execution details')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--model_path', type=str, default='./models/',
                    help='path to load trained model')


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Initialize actor-critic model here. The parameters initialized here
    # will be shared across all the sub-processes for both policy and value
    # network.
    env = gym.make(args.env_name)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    model = ActorCritic(state_space, action_space)
    model.load_state_dict(torch.load(args.model_path+f"/{1}"))

    episode_rewards = []
    for episode in range(100):
        # Fetch current state
        state = env.reset()
        state = torch.as_tensor(state, dtype=torch.float64)
        done = False
        rewards = 0
        while done is False:
            if args.render:
                time.sleep(0.1)
                env.render()

            # Get action from the current policy
            state_value, action_proba, log_proba = model.forward(state)

            # Randomly sample an action from the probability distribution
            action = torch.multinomial(action_proba, num_samples=1).item()

            # Take action on the environment and get reward, next_state
            next_state, reward, done, _ = env.step(action)
            state = torch.as_tensor(next_state, dtype=torch.float64)
            rewards += reward
        episode_rewards.append(rewards)

    avg_rewards = sum(episode_rewards)/len(episode_rewards)
    print(f"Average Reward: {avg_rewards}")
