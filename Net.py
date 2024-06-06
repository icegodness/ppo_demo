import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.indicators = []
        self.loss = []
        self.ave_AoI_each_epoch = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.indicators[:]
        del self.loss[:]
        del self.ave_AoI_each_epoch[:]

def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Apply orthogonal initialization to actor and critic
        self.apply_orthogonal_init()

    def apply_orthogonal_init(self):
        for m in self.modules():
            orthogonal_init(m)

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask):
        
        # 生成一个动作概率分布
        action_probs = self.actor(state)
        # 确保 mask 在与 action_probs 相同的设备上
        mask = mask.to(action_probs.device)

        action_probs = action_probs * mask + 1e-18
        # 生成一个分布
        dist = Categorical(action_probs)
        # action index
        action = dist.sample()
       
        # 采取这个动作的概率的对数
        action_logprob = dist.log_prob(action)
        
        state_val = self.critic(state)
            
        return action.detach(), action_logprob.detach(), state_val.detach()
    # 评估需要回传的参数
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.buffer = RolloutBuffer()

        self.action_dim = action_dim
        # 两个网络，一个是当前的网络，一个是旧的网络
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
    # 随机选择动作
    def random_action(self):
        action = np.random.choice(self.action_dim)
        return action
    # agent选择动作
    def select_action(self, state):
        mask = torch.zeros(self.action_dim)
        if state.istran != 0:
                #print("mask action")
                #print(self.action_dim - 1)
                mask[self.action_dim - 1] = 1
                
        with torch.no_grad():
            state = state.to_ndarray_normalized()
            state = torch.FloatTensor(state).to(self.device)
            
            action, action_logprob, state_val = self.policy_old.act(state, mask)
            #print('action_inact:', action)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)


        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        # print('self.buffer.states:', self.buffer.states)
        for reward in reversed(self.buffer.rewards):
            # if is_terminal:
            #     discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # print('rewards:', rewards)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        #print('old_state_values:', old_state_values)
        #print('rewards:', rewards)
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Normalizing the advantages (new added)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        actor_loss_epochs = 0
        critic_loss_epochs = 0
        entropy_epochs = 0
        # Optimize policy for K epochs
        for i in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
           
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = 0.5 * self.MseLoss(state_values, rewards)
            entropy_loss = - 0.05 * dist_entropy
            loss = actor_loss + critic_loss + entropy_loss
            
            
            actor_loss_epochs += (actor_loss.detach().cpu().numpy().mean())
            critic_loss_epochs += (critic_loss.detach().cpu().numpy().mean())  

            

            entropy_epochs += (dist_entropy.detach().cpu().numpy().mean())
            
            #打印学习率lr
            actor_lr = self.optimizer.param_groups[0]['lr']
            critic_lr = self.optimizer.param_groups[1]['lr']

            # 保存loss
            
            if (i == self.K_epochs - 1):
                #print('loss:', loss)
                self.buffer.loss.append(loss)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

       
        return actor_loss_epochs/self.K_epochs, critic_loss_epochs/self.K_epochs, entropy_epochs/self.K_epochs, actor_lr, critic_lr

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))





