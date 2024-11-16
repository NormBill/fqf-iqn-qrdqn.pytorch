import torch
from torch.optim import Adam
import wandb

from fqf_iqn_qrdqn.model.c51 import CatDQN
from fqf_iqn_qrdqn.utils import calculate_quantile_huber_loss, disable_gradients, evaluate_quantile_at_action, update_params

from .base_agent import BaseAgent
from geomloss import SamplesLoss
import geomloss
class NQ_RDQNAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, N=200, kappa=1.0, lr=5e-5, memory_size=10**6,
                 gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000,
                 epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0):
        super(NQ_RDQNAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, dueling_net, noisy_net, use_per, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed)

        # # Online network.
        # self.online_net = QRDQN(
        #     num_channels=env.observation_space.shape[0],
        #     num_actions=self.num_actions, N=N, dueling_net=dueling_net,
        #     noisy_net=noisy_net).to(self.device)
        # # Target network.
        # self.target_net = QRDQN(
        #     num_channels=env.observation_space.shape[0],
        #     num_actions=self.num_actions, N=N, dueling_net=dueling_net,
        #     noisy_net=noisy_net).to(self.device).to(self.device)
        # Online network.
        self.online_net = CatDQN(
            n=self.num_actions, num_channels=env.observation_space.shape[0]).to(self.device)
        # Target network.
        self.target_net = CatDQN(
            n=self.num_actions, num_channels=env.observation_space.shape[0]).to(self.device).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.optim = Adam(
            self.online_net.parameters(),
            lr=lr, eps=1e-2/batch_size)

        # Fixed fractions.
        taus = torch.arange(
            0, N+1, device=self.device, dtype=torch.float32) / N
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, N)

        self.N = N
        self.kappa = kappa
        self.gamma = gamma

    def learn(self):
        self.learning_steps += 1
        # self.online_net.sample_noise()
        # self.target_net.sample_noise()

        if self.use_per:
            (states, actions, rewards, next_states, dones), weights =\
                self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones =\
                self.memory.sample(self.batch_size)
            weights = None

        quantile_loss, mean_q, errors = self.calculate_loss(
            states, actions, rewards, next_states, dones, weights)
        # assert errors.shape == (self.batch_size, 1)

        update_params(
            self.optim, quantile_loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if self.use_per:
            self.memory.update_priority(errors)

        if 4*self.steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/quantile_loss', quantile_loss.detach().item(),
                4*self.steps)
            self.writer.add_scalar('stats/mean_Q', mean_q, 4*self.steps)

    def calculate_loss(self, states, actions, rewards, next_states, dones,
                       weights):

        # Calculate target pmf of current states.
        next_action, next_pmfs = self.target_net.get_action(torch.Tensor(next_states).to(self.device))
        next_action = actions.cpu().numpy()
        next_atoms = rewards + self.gamma * self.target_net.atoms * (1 - dones)
        # projection
        delta_z = self.target_net.atoms[1] - self.target_net.atoms[0]
        tz = next_atoms.clamp(self.target_net.v_min, self.target_net.v_max)

        b = (tz - self.target_net.v_min) / delta_z
        l = b.floor().clamp(0, self.target_net.n_atoms - 1)
        u = b.ceil().clamp(0, self.target_net.n_atoms - 1)
        # (l == u).float() handles the case where bj is exactly an integer
        # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
        d_m_l = (u + (l == u).float() - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs
        target_pmfs = torch.zeros_like(next_pmfs)
        for i in range(target_pmfs.size(0)):
            target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
            target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        #Calculate old pmfs
        _, old_pmfs = self.online_net.get_action(states, actions.flatten())
        # C51的衡量两个分布差异的方法
        # loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

        # 改法2：更换计算分布差异的方法的同时更换估计分布的方式，也就是使用pmf来对分布做估计
        # 因为使用了geomloss更换了update_param的方式
        # Define a Sinkhorn (~Wasserstein) loss between sampled measures
        p = 1
        entreg = .1
        gemloss_computation = SamplesLoss(loss="sinkhorn", p=1, cost=geomloss.utils.distances, blur=entreg ** (1 / p))
        gemloss_loss = gemloss_computation(old_pmfs.requires_grad_(),
                                           target_pmfs.requires_grad_()).mean()

        return gemloss_loss, next_action.mean().item(), \
            gemloss_loss.abs().sum().mean()

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.
        with torch.no_grad():
            action, _ = self.online_net.get_action(state)
            # print(action)
        return action

    def train_episode(self):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:
            # NOTE: Noises can be sampled only after self.learn(). However, I
            # sample noises before every action, which seems to lead better
            # performances.
            # self.online_net.sample_noise()

            if self.is_random(eval=False):
                action = self.explore()
            else:
                action = self.exploit(state)

            next_state, reward, done, _ = self.env.step(action)

            # To calculate efficiently, I just set priority=max_priority here.
            self.memory.append(state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.train_step_interval()

        # We log running mean of stats.
        self.train_return.append(episode_return)

        # We log evaluation results along with training frames = 4 * steps.
        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'return/train', self.train_return.get(), 4 * self.steps)

        print(f'Episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'return: {episode_return:<5.1f}')

        # wandb.log({
        #     'episode': self.episodes,
        #     'episode_steps': episode_steps,
        #     'episode_return': episode_return,  # Immediate reward of the current episode
        #     'training_return': self.train_return.get(),  # Running average reward
        #     'steps': self.steps
        # })