import argparse
import torch
import torch.multiprocessing as mp
import gym
import matplotlib.pyplot as plt

from a3c import ActorCritic
from utils import log_parameter_metrics

# Parse training arguments.
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--num_processes', type=int, default=2,
                    help='number of subprocess to spawn default(2)')
parser.add_argument('--env_name', type=str, default='CartPole-v0',
                    help='name of the gym environment with version')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--verbose', type=bool, default=False,
                    help='log execution details')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--max_step', type=int, default=200,
                    help='maximum number of steps before termination')
parser.add_argument('--max_episode', type=int, default=50000,
                    help='maximum number of episodes before termination')
parser.add_argument('--model_path', type=str, default='./models/',
                    help='path to save trained model')

def train(pid, shared_model, args):

    torch.manual_seed(args.seed+pid)
    # Create environment with a random seed
    env = gym.make(args.env_name)
    env.seed(args.seed+pid)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    # Initialize a local copy of actor-critic shared_model.
    local_model = ActorCritic(state_space, action_space)

    # Instantiate optimization algorithm.
    optimizer = torch.optim.SGD(shared_model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum)

    episode = 0
    episode_rewards = []
    episode_policy_loss = []
    episode_value_loss = []
    while True:

        # Load the model parameters from global copy.
        local_model.load_state_dict(shared_model.state_dict())

        if args.verbose:
            log_parameter_metrics("local", local_model)
            log_parameter_metrics("shared", shared_model)

        # Fetch current state
        state = env.reset()
        state = torch.as_tensor(state, dtype=torch.float64)
        done = False

        # Run episode
        step = 1
        step_rewards = []
        step_log_probas = []
        step_state_values = []
        while not done:

            if args.render:
                env.render()

            # Get action from the current policy
            state_value, action_proba, log_proba = local_model.forward(state)

            # Randomly sample an action from the probability distribution
            action = torch.multinomial(action_proba, num_samples=1).item()

            # Take action on the environment and get reward, next_state
            next_state, reward, done, _ = env.step(action)
            state = torch.as_tensor(next_state, dtype=torch.float64)

            if done:
                reward = 0.0

            step += 1

            # TODO: Check for max episode length
            if step > args.max_step:
                done = True
                reward = state_value

            # Store data for loss computation
            step_rewards.append(reward)
            step_log_probas.append(log_proba[action])
            step_state_values.append(state_value)


        # Calculate loss over the trajectory
        R = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        for idx in reversed(range(len(step_rewards))):
            R = args.gamma * R + step_rewards[idx]
            advantage = R - step_state_values[idx]
            value_loss = value_loss + advantage.pow(2)
            policy_loss = policy_loss - step_log_probas[idx]*advantage

        # Reset gradient
        optimizer.zero_grad()

        # Calculate gradients by combining actor and critic loss.
        # This is happening on the local_model.
        loss = policy_loss + 0.5 * value_loss
        loss.backward()

        episode_rewards.append(sum(step_rewards[:-1]))
        episode_policy_loss.append(policy_loss)
        episode_value_loss.append(value_loss)

        # Clip gradients.
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 50)

        # Copy the gradients on local_model to the shared_model.
        for local_param, global_param in zip(local_model.parameters(),
                                             shared_model.parameters()):
            global_param.grad = local_param.grad

        # Backprop the gradients.
        optimizer.step()

        # Log metrics.
        if episode % 50 == 0:
            episode_rewards_ = episode_rewards[-100:]
            episode_policy_loss = episode_policy_loss[-100:]
            episode_value_loss = episode_value_loss[-100:]
            avg_reward = sum(episode_rewards_)/len(episode_rewards_)
            avg_policy_loss = sum(episode_policy_loss)/len(episode_policy_loss)
            avg_value_loss = sum(episode_value_loss)/len(episode_value_loss)
            print(f"Episode: {episode}")
            print(f"{pid} - Average Reward: {avg_reward}")
            print(f"{pid} - Loss: {avg_policy_loss} {avg_value_loss}")

        episode += 1
        # Save metrics on completion.
        if (episode > args.max_episode) or (avg_reward >= 195):
            plt.figure()
            plt.title("Average reward")
            plt.plot(range(0, episode), episode_rewards)
            plt.xlabel("episode")
            plt.ylabel("Average Reward per 100 episode")
            plt.savefig(args.model_path+"/1.png")
            break

    return 0

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Set the method to start a subprocess.
    # Using `spawn` here because it handles error propagation across multiple
    # sub-processes. When any of the subprocess fails it raises exception on
    # join.
    mp.set_start_method('spawn')

    # Initialize actor-critic model here. The parameters initialized here
    # will be shared across all the sub-processes for both policy and value
    # network.
    env = gym.make(args.env_name)
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    model = ActorCritic(state_space, action_space)
    model.share_memory()

    processes = []
    for pid in range(args.num_processes):

        # Start process
        p = mp.Process(target=train, args=(pid, model, args, ))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    torch.save(model.state_dict(), args.model_path+f"/{1}")
