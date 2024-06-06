import argparse
import datetime
import os
import platform
import time
from random import random
import csv

from matplotlib import pyplot as plt

from Net import PPO

from env import mm_env,env_init,env_param
from utilities import set_seed

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

# 训练函数
def training(random_policy, num_episodes, seed):
    
    for i_episode in range(1, num_episodes + 1):

        # 初始化环境
        state = env.reset()
        done = False
        episode_reward = []
        ep_start_time = time.time()
        for slot in range(1, slots + 1):
            # 选择动作
            #print("----start slot----")
            #print('istran',state.istran)
            action = agent.select_action(state)
            #print('action_inslot:', action)
            # 更新环境
            next_state, reward, done = env.step(action, slot)
            # 存储经验
            agent.buffer.rewards.append(reward)
            episode_reward.append(reward)
            # 更新状态
            state = next_state
            #writer.add_scalar(f'reward{slot}', reward, slot)
            # 更新网络
            if slot % update_timestep == 0:
                # print('update')
                 # 生成时间戳
                

                # # 将buffer信息保存到csv中,buffer里的每一项保存在csv里面的一列
                # if not os.path.exists('./result'):
                #     os.makedirs('./result')
                # buffer_filename = f'./result/buffer.csv'
                # # 写入CSV文件
                # with open(buffer_filename, 'w', newline='') as buffer_csvfile:
                #     csvwriter = csv.writer(buffer_csvfile)

                #     # 写入标题行
                #     csvwriter.writerow(['State', 'Action', 'Log Probability', 'Reward'])

                #     # 将buffer数据写入CSV文件
                #     for states, actions, log_probs, rewards in zip(agent.buffer.states, agent.buffer.actions, agent.buffer.logprobs, agent.buffer.rewards):
                #         csvwriter.writerow([str(states), str(actions), str(log_probs), str(rewards)])

    
                a_loss, c_loss, entropy, actor_lr, critic_lr = agent.update()
                print('slot:{}, a_loss:{}, c_loss:{}, entropy:{}, actor_lr:{}, critic_loss:{}'.format(slot, a_loss, c_loss, entropy, actor_lr, critic_lr))

                # 写入CSV文件 - training data
                training_filename = f'./result/training_data.csv'
                with open(training_filename, 'w', newline='') as training_data_csvfile:
                    csvwriter = csv.writer(training_data_csvfile)
                    csvwriter.writerow(['Slot', 'A_loss', 'C_loss', 'Entropy', 'Actor_lr', 'Critic_lr'])
                    csvwriter.writerow([slot, a_loss, c_loss, entropy, actor_lr, critic_lr])



            # 记录每个回合的return
            if done:
                return_list.append(np.sum(env.state.AoI * env.state.weight))
                # 打印本episode的平均reward
                print('Episode [{}/{}], Average Reward: {:.2f}, time: {}'
                    .format(i_episode, num_episodes, np.mean(episode_reward), time.time() - ep_start_time))
                # 使用tensorboard记录
                writer.add_scalar('episode_return', np.mean(episode_reward), i_episode)
                break

            # 打印训练信息
            # if slot % 1 == 0:
            #     print('Episode [{}/{}], Slot [{}/{}], Reward: {:.2f}'
            #         .format(i_episode, num_episodes, slot, slots, reward))
        # 保存模型
        if i_episode % saving_interval == 0:
            directory = './models/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            agent.save(directory + f'model{n_hiddens}_(' + str(i_episode) + f'|{num_episodes})' + f'reward({np.mean(episode_reward)})_' + f'seed{seed}_' + f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}' + '.pth')
            #创建文件夹
            if not os.path.exists('./result'):
                os.makedirs('./result')
            # 创建文件
            np.save('./result/return_list.txt', return_list)
            print('Model saved after {} episodes'.format(i_episode))

   


    # 使用循环调度策略RR


    
    # 使用juventas的代码

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
    print('-------------------end traning-----------------')



    



if __name__ == '__main__':
    # 训练设备
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')
    #device = torch.device('cpu')

    print('\r\n------------------------------------')
    print('Enviroment Sumamry')
    print('------------------------------------')
    print('PyTorch ' + str(torch.__version__))
    print('Running with Python ' + str(platform.python_version()))
    print('Running on Device: ' + str(device))

    # 设置随机种子
    # seed =246810
    # seed = 41
    # set_seed(seed)



    # 训练参数
    #num_episodes = 1  # 总迭代次数
    gamma = 0.99  # 折扣因子
    actor_lr = 1e-3  # 策略网络的学习率
    critic_lr = 1e-2  # 价值网络的学习率
    n_hiddens = 128  # 隐含层神经元个数
    # n_states =   # 状态数
    # n_actions =  # 动作数
    n_UE = 10  # UE数目
    return_list = []  # 保存每个回合的return
    
    # update_timestep = 5  # 更新时间步长，每隔多少个时间步更新一次网络
    saving_interval = 15  # 保存模型的间隔, 每隔多少个回合保存一次模型


    slots = 10240  # 时隙数

    # 开始训练
    print("-------------------start traning-----------------")

    # 开始训练计时
    start_time = time.time()
    # 创建环境
    env = mm_env(env_init(n_UE))
    print(env.state.weight)
    print(env.state.distance)

    # 固定死环境
    env.state.weight = [0.7188461,  0.15161322, 0.68094836, 0.20270096, 0.2966249,  0.57962901,
 0.72017765, 0.83438789, 0.76947061, 0.21534061]
    env.state.distance = [10.23679039, 34.25003444, 62.18063476, 47.14164498, 40.87047495, 40.28956408,
 60.27275688, 47.76657904, 52.14803098, 58.06569597]



    n_states = env.state.num_states  # 状态数
    print('n_states', n_states)
    n_actions = env.action.num_actions  # 动作数
    print('n_actions', n_actions)

    # ----------------------------------------- #
    # 模型构建
    # ----------------------------------------- #

    agent = PPO(state_dim=n_states,  # 状态数
                action_dim=n_actions,  # 动作数
                hidden_dim=n_hiddens,  # 隐含层数
                lr_actor=actor_lr,  # 策略网络学习率
                lr_critic=critic_lr,  # 价值网络学习率
                # lmbda=0.95,  # 优势函数的缩放因子
                # lmbda=0.99,
                gamma=gamma,  # 折扣因子
                K_epochs=1,  # 一组序列训练的轮次
                eps_clip=0.2,  # PPO中截断范围的参数
                device=device
                )

    # 创建tensorboard writer
    writer = SummaryWriter(
    'runs/{}_PPO'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))


    # ----------------------------------------- #
    # 训练--回合更新 on_policy
    # ----------------------------------------- #
    
    # 选择随机策略
    #random_policy = 1


   
    update_timestep = 10  # 更新时间步长
    #seed_list = [41, 13579, 246810, 1234, 2234, 3234]
    seed_list = [41]
    for seed in seed_list:
        # 设置随机种子
        set_seed(seed)
        print('seed:', seed)
        training(random_policy=0, num_episodes = 500, seed = seed)


    
    
    