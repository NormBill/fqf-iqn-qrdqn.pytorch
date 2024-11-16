import os
import yaml
import wandb
import argparse
from datetime import datetime

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import QRDQNAgent
from fqf_iqn_qrdqn.agent import NQ_RDQNAgent


def run(args):
    # Initialize WandB logging
    wandb.init(project='qrdqn', config=args, name=args.env_id, dir='logs')

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Log the configuration to WandB
    wandb.config.update(config)

    # Create environments.
    env = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent and run.
    if args.retain_quantiles.lower() == 'true':
        print("Predict q value distribution as continuous distribution")
        agent = QRDQNAgent(
            env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
            cuda=args.cuda, **config)
    else:
        print("Predict q value distribution as discrete distribution")
        agent = NQ_RDQNAgent(
            env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
            cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'qrdqn.yaml'))
    parser.add_argument('--retain_quantiles', type=str, default='True')
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
