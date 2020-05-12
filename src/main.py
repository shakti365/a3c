import argparse
import torch
import torch.multiprocessing as mp
import gym

from a3c import ActorCritic

# Parse training arguments.
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--num_processes', type=int, default=2,
                    help='number of subprocess to spawn default(2)')
parser.add_argument('--env_name', type=str, default='CartPole-v0',
                    help='name of the gym environment with version')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--max_step', type=int, default=100,
                    help='maximum number of steps before termination')

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
    while True:
        # Load the model parameters from global copy.
        local_model.load_state_dict(shared_model.state_dict())


        # Fetch current state
        state = env.reset()
        state = torch.as_tensor(state, dtype=torch.float64)
        done = False

        # Run episode
        step = 0
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

            if done:
                reward = 0.0

            # Store data for loss computation
            step_rewards.append(reward)
            step_log_probas.append(log_proba[action])
            step_state_values.append(state_value)

            step += 1

            # TODO: Check for max episode length
            if step > args.max_step:
                done = True
                reward = state_value

        # Calculate loss over the trajectory
        R = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        for idx in reversed(range(len(step_rewards))):
            R = args.gamma * R + step_rewards[idx]
            advantage = R - step_state_values[idx]
            value_loss += advantage.pow(2)
            policy_loss -= step_log_probas[idx]*advantage
        episode_rewards.append(sum(step_rewards))


        optimizer.zero_grad()
        loss = policy_loss + 0.5 * value_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 50)

        for local_param, global_param in zip(local_model.parameters(),
                                             shared_model.parameters()):
            global_param.grad = local_param.grad

        optimizer.step()

        if episode % 50 == 0:
            last_n = episode_rewards[-100:]
            print(f"{pid} - Average Reward: {sum(last_n)/len(last_n)}")
            print(f"{pid} - Loss: {policy_loss.item()} {value_loss.item()}")

        episode += 1

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
