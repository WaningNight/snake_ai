import pygame
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class snake:
    def __init__(self, width, heigh, speed, render):
        self.width = width
        self.heigh = heigh
        self.render = render
        self.snake_head = [50, 50]
        self.snake_body = [[50, 50], [60, 50], [70, 50]]
        self.food_pos = [random.randrange(0, self.width - 10)//10*10, random.randrange(0, self.heigh - 10)//10*10]
        self.snake_speed = speed
        self.growth_size = 1
        self.steps = 0
        self.step_limit = (self.width + self.heigh) // 10
        self.old_pos = 0  # 0右1上2左3下
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((width, heigh))
            pygame.display.set_caption('贪吃蛇游戏')
            self.clock = pygame.time.Clock()

    def reset(self):
        self.snake_head = [50, 50]
        self.snake_body = [[50, 50], [60, 50], [70, 50]]
        self.food_pos = [random.randrange(0, self.width - 10)//10*10, random.randrange(0, self.heigh - 10)//10*10]
        self.growth_size = 1
        self.steps = 0
        self.old_pos = 0
        if self.render:
            self.draw()
        return self.get_state()

    def draw(self):
        self.screen.fill((0, 0, 0))
        pygame.event.pump()
        pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))
        for pos in self.snake_body:
            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.display.flip()
        self.clock.tick(self.snake_speed)

    def step(self, key):
        reward = 0.0

        old_food_dist = abs(self.snake_head[0] - self.food_pos[0]) + abs(self.snake_head[1] - self.food_pos[1])

        # 0直走1左转2右转
        if key == 1:
            self.old_pos = (self.old_pos + 1) % 4
        elif key == 2:
            self.old_pos = (self.old_pos + 3) % 4

        if self.old_pos == 0:
            self.snake_head[0] += 10
        elif self.old_pos == 1:
            self.snake_head[1] -= 10
        elif self.old_pos == 2:
            self.snake_head[0] -= 10
        elif self.old_pos == 3:
            self.snake_head[1] += 10
        # 撞墙
        if self.snake_head[0] < 0 or self.snake_head[0] >= self.width or self.snake_head[1] < 0 or self.snake_head[
            1] >= self.heigh:
            return self.get_state(), -10 * (1 - self.growth_size * 0.1), True, len(self.snake_body)-3
        # 撞到身体
        for x, y in self.snake_body:
            if self.snake_head[0] == x and self.snake_head[1] == y:
                return self.get_state(), -10 * (1 - self.growth_size * 0.1), True, len(self.snake_body)-3
        
        new_food_dist = abs(self.snake_head[0] - self.food_pos[0]) + abs(self.snake_head[1] - self.food_pos[1])
        # 如果靠近食物加1分，否则减1分
        reward += 1 if old_food_dist > new_food_dist else -1
        self.snake_body.insert(0, list(self.snake_head))
        # 吃到食物加分并重置步数，蛇越长加的越多
        if self.snake_head[0] == self.food_pos[0] and self.snake_head[1] == self.food_pos[1]:
            reward += 5 * (1 + self.growth_size * 0.1)
            self.growth_size += 1
            self.food_pos = [random.randrange(1, (self.width // 10)) * 10, random.randrange(1, (self.heigh // 10)) * 10]
        else:
            self.snake_body.pop()
        if self.render:
            self.draw()
        self.steps += 1
        # 吃到食物前的步数超过step_limit直接结束，扣10分
        if self.steps >= self.step_limit:
            return self.get_state(), -10, True, len(self.snake_body)-3
        # 返回的是状态、奖励、游戏是否结束、得分
        return self.get_state(), reward, False, len(self.snake_body)-3

    def get_state(self):
        avail_up = 1
        avail_down = 1
        avail_left = 1
        avail_right = 1
        pos_right = self.old_pos == 0
        pos_up = self.old_pos == 1
        pos_left = self.old_pos == 2
        pos_down = self.old_pos == 3
        for x, y in self.snake_body:
            if self.snake_head[0] + 10 == x and self.snake_head[1] == y:
                avail_right = 0
            if self.snake_head[0] - 10 == x and self.snake_head[1] == y:
                avail_left = 0
            if self.snake_head[0] == x and self.snake_head[1] + 10 == y:
                avail_down = 0
            if self.snake_head[0] == x and self.snake_head[1] - 10 == y:
                avail_up = 0
        wall_dist_x = min(self.snake_head[0], self.width - 10 - self.snake_head[0])
        wall_dist_y = min(self.snake_head[1], self.heigh - 10 - self.snake_head[1])
        food_pos_up = self.food_pos[1] < self.snake_head[1]
        food_pos_down = self.food_pos[1] > self.snake_head[1]
        food_pos_left = self.food_pos[0] < self.snake_head[0]
        food_pos_right = self.food_pos[0] > self.snake_head[0]
        list = [food_pos_up, food_pos_down, food_pos_left, food_pos_right, pos_up, pos_down, pos_left, pos_right, avail_up,
                avail_down, avail_left, avail_right, wall_dist_x, wall_dist_y]
        return np.array(list)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, 1)  # V是critic要学习的结果(R-V)
        )

    def forward(self, x):
        action_probabilities = self.action_layer(x)
        return action_probabilities

    # 在actor神经网络中进行一次前向传播并从输出的动作概率分布中按概率大小随机采样一个
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)  # 表示一个离散的概率分布
        action = dist.sample()  # 根据概率分布进行采样
        if np.random.rand() < 0.1:
            dist = Categorical(torch.tensor([1 / action_dim] * action_dim).to(device))

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 初始化成一样的策略

        self.MseLoss = nn.MSELoss()  # 均方误差

    def update(self, memory):
        # 计算折扣累积奖励
        rewards = []
        discounted_reward = 0
        # 由于是反向迭代，所以使用reversed从轨迹的最后一个时间步开始（reversed返回一个反向迭代器）
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 把折扣累积奖励标准化
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 转为tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        for _ in range(self.K_epochs):
            # 使用新策略（policy）评估旧策略
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算重要性采样的比例因子（新策略/旧策略）
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # A=R-V
            advantages = rewards - state_values.detach()
            # J(θ) = E[min(surr1, surr2)]
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


env = snake(400, 300, 200, False)
state_dim = 14
action_dim = 3  # 行动空间的维度
solved_reward = 50  # 当累积奖励达到此值时，任务被认为已解决
log_interval = 1  # 训练日志输出的频率（每多少次迭代输出一次日志）
max_episodes = 50000  # 最大训练回合数
n_latent_var = 128  # 神经网络中用于表示潜在变量的维度
update_timestep = 2000  # 多少时间步之后对策略进行一次更新
lr = 0.001  # 学习率
betas = (0.9, 0.999)  # Adam 优化器的 beta 参数，分别表示一阶和二阶矩的衰减率
gamma = 0.99  # 折扣因子，用于计算未来奖励的折现值
K_epochs = 4  # 在每次策略更新中使用前一轮的值迭代次数（重要性采样）
eps_clip = 0.2  # PPO 中的重要性比例裁剪阈值(1+ε,1-ε)

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

running_reward = 0  # 存储每个训练周期的累计奖励
avg_length = 0  # 当前训练过程中的平均步数
timestep = 0  # 训练过程中的时间步数

for i_episode in range(1, max_episodes + 1):
    state = env.reset()  # 初始化环境，重新开始新的训练周期
    t = 0
    scores = 0
    while True:
        timestep += 1

        # 在PPO更新中，我们首先用旧策略（policy_old）从环境中采集数据，然后使用新策略（policy）评估这些数据的概率、状态值等信息
        action = ppo.policy_old.act(state, memory)

        # 与环境交互，获取新状态、奖励、是否终止和额外信息
        state, reward, done, score = env.step(action)
        scores += score
        t += 1
        # 保存奖励和是否终止信息到记忆中
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # 如果到达策略更新的时间点，则执行策略更新
        if timestep % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            timestep = 0

        # 更新累计奖励
        running_reward += reward

        # 如果当前训练周期结束，则退出内层循环
        if done:
            break

    # 更新平均步数
    avg_length += t

    # 每隔一定的训练周期输出训练日志
    if i_episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        running_reward = int((running_reward / log_interval))
        scores = int((scores / log_interval))

        print(f'Episode:{i_episode} avg length:{avg_length} reward:{running_reward} score:{scores}')
        running_reward = 0
        avg_length = 0
        torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format("snake"))
