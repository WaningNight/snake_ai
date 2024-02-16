import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import pygame
import random
import math

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
        reward += 1 if old_food_dist > new_food_dist else -1
        self.snake_body.insert(0, list(self.snake_head))
        if self.snake_head[0] == self.food_pos[0] and self.snake_head[1] == self.food_pos[1]:
            reward += 5 * (1 + self.growth_size * 0.1)
            self.growth_size += 1
            self.food_pos = [random.randrange(1, (self.width // 10)) * 10, random.randrange(1, (self.heigh // 10)) * 10]
        else:
            self.snake_body.pop()
        if self.render:
            self.draw()
        self.steps += 1
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

class Dqn:
    def __init__(self):
        self.eval_net, self.target_net = self.Net(), self.Net()
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # memory存的内容是state, action ,reward, next state
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)
        self.loss = nn.MSELoss()

        self.fig, self.ax = plt.subplots()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()  # 调用父类nn.Module的构造函数，得到了父类的各个成员
            self.fc1 = nn.Linear(NUM_STATES, 128)
            self.fc1.weight.data.normal_(0, 0.1)
            self.fc2 = nn.Linear(128, NUM_ACTIONS)
            self.fc2.weight.data.normal_(0, 0.1)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    def store_trans(self, state, action, reward, next_state):
        if self.memory_counter % 500 == 0:
            print(f"经验池收集了{self.memory_counter}次经验")
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, [action], [reward], next_state))  # 记录一条数据
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 神经网络输入的是一个批次，这个相当于构建了只含一个状态的批次
        if np.random.randn() <= EPSILON:  # greedy-ε
            action_value = self.eval_net.forward(state)  # 通过评估网络前向传播得到各个动作的得分
            action = torch.max(action_value, 1)[1].item()  # torch.max返回一个元组，第一个元素是每个批次中每个维度最大值的列表，第二个元素是这些最大值对应的索引列表
        else:
            action = np.random.randint(0, NUM_ACTIONS)
        return action

    def learn(self):
        # 每学习100次，目标网络更新一次
        if self.learn_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的权重加载到目标网络
        self.learn_counter += 1

        # 从经验池中随机选择一个批次的数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])  # 提取状态
        # 注意动作必须是整数类型
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))  # 提取动作
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1: NUM_STATES + 2])  # 提取奖励
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])  # 提取下一个状态

        # 计算当前状态下的Q值
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        # 使用目标网络计算下一个状态的最大Q值
        q_next = self.target_net(batch_next_state).detach()
        # 计算目标Q值，使用公式：当前奖励 + 折扣因子 * 下一个状态的最大Q值
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        # 计算损失函数，目标是使预测的Q值和目标Q值之间的差异最小
        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


EPSILON = 0.9
GAMMA = 0.9
LR = 0.001
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 64

EPISODES = 10000
env = snake(400, 300, 20, False)
NUM_STATES = 14
NUM_ACTIONS = 3


net = Dqn()
print("开始训练...")

max_score = 5

for episode in range(EPISODES):
    state = env.reset()
    step_counter = 0
    rewards = 0
    while True:
        step_counter += 1
        action = net.choose_action(state)
        next_state, reward, done, score = env.step(action)

        if score > max_score:
            max_score = score
            torch.save(net.eval_net.state_dict(), './DQN_{}.pth'.format("snake"))

        rewards += reward
        # 将[st,at,rt,s(t+1)]存储到经验池中
        net.store_trans(state, action, reward, next_state)

        # 如果经验池中的数据量达到预设的容量
        if net.memory_counter >= MEMORY_CAPACITY:
            # 使用经验池中的数据来训练DQN
            net.learn()
            if done:
                print(f"episode:{episode} reward:{round(rewards/step_counter, 3)} score:{score}")
        if done:
            break
        state = next_state

