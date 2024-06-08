import math
import sys

import numpy as np
from matplotlib import pyplot as plt
import pdb

from state_action import State, Action
from math import atan2, degrees
from utilities import *
# #参数配置类
# class env_param():
#     def __init__(self):
#         pass

# #设置无线网络参数
# def env_init(num):
#     # 定义区域大小
#     area_size = 100  # 区域大小为100*100米

#     # 生成基站的坐标
#     base_station = np.array([area_size / 2, area_size / 2])

#     UE_num = num    #节点数量
#     # 生成节点的坐标
#     nodes_x = np.random.uniform(0, area_size, UE_num)
#     nodes_y = np.random.uniform(0, area_size, UE_num)
#     nodes = np.vstack((nodes_x, nodes_y)).T

#     # 计算每个节点到基站的距离和角度（0度到180度）
#     radius = np.linalg.norm(nodes - base_station, axis=1)
#     angle = np.array(
#         [degrees(atan2(node[1] - base_station[1], node[0] - base_station[0])) % 180 for node in nodes])

#     # -------------------------------------
#     # Antenna setting at BS and UE
#     # -------------------------------------
#     # sector数量
#     N_SSW_BS_vec = 72
#     #最小波束宽度
#     BeamWidth_TX_vec = 360. / N_SSW_BS_vec;  # Tx antenna beamwidth


#     # -------------------------------------
#     # Antenna gain between BS and UEs
#     # -------------------------------------
#     B = 2160 * 1e6  # Hz
#     fc = 60  # GHz
#     c = 3e8  # m/s
#     l = c / (fc * 1e9)  # 波长
#     pathloss = (l / (4 * np.pi)) ** 2  # 路径损耗
#     z = 0.05
#     alpha = 3.3
#     Gr_vec = 11
#     Ptx_BS_dBm = 30
#     N0 = -110  # dBm
#     packet_length = 100 * 1e6 * 8

#     # 可视化节点分布和通信范围
#     plt.figure(figsize=(10, 10))
#     plt.scatter(nodes[:, 0], nodes[:, 1], label='Nodes')
#     plt.scatter(base_station[0], base_station[1], color='red', label='Base Station')

#     # 在每个节点头顶显示距离、SNR和角度
#     for i in range(UE_num):
#         plt.text(nodes[i, 0], nodes[i, 1], f'd={radius[i]:.2f}, Angle={angle[i]:.2f}°', fontsize=8,
#                  ha='center', va='bottom')

#     plt.xlim(0, area_size)
#     plt.ylim(0, area_size)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Node Distribution and Communication Range')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     plt.close()


#     # Plot the sectors
#     plt.figure(figsize=(10, 10))
#     plt.scatter(nodes[:, 0], nodes[:, 1], label='Nodes')
#     plt.scatter(base_station[0], base_station[1], color='red', label='Base Station')

#     # Plot lines from the base station to the nodes representing sector boundaries
#     for i in range(N_SSW_BS_vec):
#         start_angle = i * BeamWidth_TX_vec
#         end_angle = (i + 1) * BeamWidth_TX_vec
#         plt.plot([base_station[0], base_station[0] + area_size * np.cos(np.radians(start_angle))],
#                  [base_station[1], base_station[1] + area_size * np.sin(np.radians(start_angle))], color='gray')
#         plt.plot([base_station[0], base_station[0] + area_size * np.cos(np.radians(end_angle))],
#                  [base_station[1], base_station[1] + area_size * np.sin(np.radians(end_angle))], color='gray')

#     plt.xlim(0, area_size)
#     plt.ylim(0, area_size)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Node Distribution and Sector Boundaries')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     plt.close()

#     # 保存参数
#     # BS UE 参数
#     env_parameter = env_param()  # 创建参数类
#     env_parameter.UE_num = UE_num
#     env_parameter.radius = radius
#     env_parameter.angle = angle

#     env_parameter.N_SSW_BS_vec = N_SSW_BS_vec
#     env_parameter.BeamWidth_TX_vec = BeamWidth_TX_vec
#     env_parameter.base_station = base_station
#     env_parameter.nodes = nodes

#     env_parameter.alpha = alpha
#     env_parameter.Gr_vec = Gr_vec
#     env_parameter.N0 = N0
#     env_parameter.B = B
#     env_parameter.package_size = packet_length
#     env_parameter.pathloss = pathloss
#     env_parameter.Ptx_BS_dBm = Ptx_BS_dBm
#     env_parameter.z = z
#     env_parameter.l = l

#     return env_parameter

# class mm_env:
#     def __init__(self, env_param):
#         self.num_nodes = env_param.UE_num           #节点数量
#         # self.max_beamwidth = env_param.UE_num / 2   #最大波束宽度
#         self.state = State(env_param)          #状态
#         self.action = Action(env_param)        #动作
#         self.env_param = env_param

#         #设置一个标志位，表示当前是否有正在传输的
#         self.indicator = 0
#         self.slot_length = 50 #单位ms

#         # 记录传输的开始
#         self.start_slot = -1
#         # 记录上次覆盖的节点
#         self.cover = []

#     def printEnvParam(self):
#         env_param = self.env_param
#         print('UE_num:', env_param.UE_num)
#         print('radius:', env_param.radius)
#         print('angle:', env_param.angle)
#         print('N_SSW_BS_vec:', env_param.N_SSW_BS_vec)
#         print('BeamWidth_TX_vec:', env_param.BeamWidth_TX_vec)
#         print('base_station:', env_param.base_station)
#         print('nodes:', env_param.nodes)

#     def check_action_sandity(self, action_index):
#         one_hot = np.zeros(self.action.num_actions)
#         one_hot[action_index] = 1
#         # print('action_index:', action_index)
#         action = self.action.one_hot_to_action(one_hot)
#         beam_width = action[0]
#         start_position = action[1]
#         beam_distance = action[2]
#         min_snr_node, min_snr, covered, data_rate = find_min_snr_node(self.env_param.nodes, self.env_param.base_station,
#                                                         start_position + beam_width / 2, beam_width, beam_distance, self.env_param.alpha,
#                                                     self.env_param.Gr_vec, self.env_param.N0, self.env_param.B,
#                                                     self.env_param.z, self.env_param.pathloss, self.env_param.Ptx_BS_dBm)
#         if min_snr_node is None:
#             return False
#         else:
#             return True

#     def reset(self):
#         # 重置环境状态
#         # 环境参数， 节点位置，
#         self.state.AoI = np.ones(self.num_nodes)  # 覆盖情况，1代表被覆盖，0代表没被覆盖
#         self.indicator = 0
#         # 记录传输的开始
#         self.start_slot = -1
#         # 记录上次覆盖的节点
#         self.cover = []
#         return self.state

class env_param():
    def __init__(self):
        pass

#设置无线网络参数
def env_init(num):
    # 定义区域大小
    area_size = 100  # 区域大小为100*100米

    # 生成基站的坐标
    base_station = np.array([area_size / 2, area_size / 2])

    UE_num = num    #节点数量
    # 生成节点的坐标
    nodes_x = np.random.uniform(0, area_size, UE_num)
    nodes_y = np.random.uniform(0, area_size, UE_num)
    nodes = np.vstack((nodes_x, nodes_y)).T

    # 计算每个节点到基站的距离和角度（0度到180度）
    radius = np.linalg.norm(nodes - base_station, axis=1)
    angle = np.array(
        [degrees(atan2(node[1] - base_station[1], node[0] - base_station[0])) % 180 for node in nodes])

    # -------------------------------------
    # Antenna setting at BS and UE
    # -------------------------------------
    # sector数量
    N_SSW_BS_vec = 72
    #最小波束宽度
    BeamWidth_TX_vec = 360. / N_SSW_BS_vec;  # Tx antenna beamwidth


    # -------------------------------------
    # Antenna gain between BS and UEs
    # -------------------------------------
    B = 2160 * 1e6  # Hz
    fc = 60  # GHz
    c = 3e8  # m/s
    l = c / (fc * 1e9)  # 波长
    pathloss = (l / (4 * np.pi)) ** 2  # 路径损耗
    z = 0.05
    alpha = 3.3
    Gr_vec = 11
    Ptx_BS_dBm = 30
    N0 = -110  # dBm
    packet_length = 100 * 1e6 * 8

    # 可视化节点分布和通信范围
    plt.figure(figsize=(10, 10))
    plt.scatter(nodes[:, 0], nodes[:, 1], label='Nodes')
    plt.scatter(base_station[0], base_station[1], color='red', label='Base Station')

    # 在每个节点头顶显示距离、SNR和角度
    for i in range(UE_num):
        plt.text(nodes[i, 0], nodes[i, 1], f'd={radius[i]:.2f}, Angle={angle[i]:.2f}°', fontsize=8,
                 ha='center', va='bottom')

    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Node Distribution and Communication Range')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()


    # Plot the sectors
    plt.figure(figsize=(10, 10))
    plt.scatter(nodes[:, 0], nodes[:, 1], label='Nodes')
    plt.scatter(base_station[0], base_station[1], color='red', label='Base Station')

    # Plot lines from the base station to the nodes representing sector boundaries
    for i in range(N_SSW_BS_vec):
        start_angle = i * BeamWidth_TX_vec
        end_angle = (i + 1) * BeamWidth_TX_vec
        plt.plot([base_station[0], base_station[0] + area_size * np.cos(np.radians(start_angle))],
                 [base_station[1], base_station[1] + area_size * np.sin(np.radians(start_angle))], color='gray')
        plt.plot([base_station[0], base_station[0] + area_size * np.cos(np.radians(end_angle))],
                 [base_station[1], base_station[1] + area_size * np.sin(np.radians(end_angle))], color='gray')

    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Node Distribution and Sector Boundaries')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

    # 保存参数
    # BS UE 参数
    env_parameter = env_param()  # 创建参数类
    env_parameter.UE_num = UE_num
    env_parameter.radius = radius
    env_parameter.angle = angle

    env_parameter.N_SSW_BS_vec = N_SSW_BS_vec
    env_parameter.BeamWidth_TX_vec = BeamWidth_TX_vec
    env_parameter.base_station = base_station
    env_parameter.nodes = nodes

    env_parameter.alpha = alpha
    env_parameter.Gr_vec = Gr_vec
    env_parameter.N0 = N0
    env_parameter.B = B
    env_parameter.package_size = packet_length
    env_parameter.pathloss = pathloss
    env_parameter.Ptx_BS_dBm = Ptx_BS_dBm
    env_parameter.z = z
    env_parameter.l = l

    return env_parameter

class mm_env:
    def __init__(self, env_param):
        self.num_nodes = env_param.UE_num           #节点数量
        self.max_beamwidth = env_param.UE_num / 2   #最大波束宽度
        self.state = State(env_param)          #状态
        self.action = Action(env_param)        #动作
        self.env_param = env_param
        self.slot = 0

        self.rending = False

        #设置一个标志位，表示当前是否有正在传输的
        self.indicator = 0
        self.slot_length = 50 #单位ms

        # 记录传输的开始
        self.start_slot = -1
        # 记录上次覆盖的节点
        self.cover = []

    def printEnvParam(self):
        env_param = self.env_param
        print('UE_num:', env_param.UE_num)
        print('radius:', env_param.radius)
        print('angle:', env_param.angle)
        print('N_SSW_BS_vec:', env_param.N_SSW_BS_vec)
        print('BeamWidth_TX_vec:', env_param.BeamWidth_TX_vec)
        print('base_station:', env_param.base_station)
        print('nodes:', env_param.nodes)

    def check_action_sandity(self, action_index):
        one_hot = np.zeros(self.action.num_actions)
        one_hot[action_index] = 1
        # print('action_index:', action_index)
        action = self.action.one_hot_to_action(one_hot)
        beam_width = action[0]
        start_position = action[1]
        beam_distance = action[2]
        min_snr_node, min_snr, covered, data_rate = find_min_snr_node(self.env_param.nodes, self.env_param.base_station,
                                                        start_position + beam_width / 2, beam_width, beam_distance, self.env_param.alpha,
                                                    self.env_param.Gr_vec, self.env_param.N0, self.env_param.B,
                                                    self.env_param.z, self.env_param.pathloss, self.env_param.Ptx_BS_dBm)
        if min_snr_node is None:
            return False
        else:
            return True




    def reset(self):
        # 重置环境状态
        # 环境参数， 节点位置，
        self.state.AoI = np.ones(self.num_nodes)  # 覆盖情况，1代表被覆盖，0代表没被覆盖
        
        # 记录传输的开始
        self.start_slot = -1
        # 记录上次覆盖的节点
        self.cover = []
        self.state.istran = 0
        print('reset weight: ', self.state.weight)
        print('reset distance: ', self.state.distance)
        return self.state

    def step(self, action_index, slot):
        if self.rending:
            print(f'-----------------enter step at slot: {slot}-----------------')
            print('AoI:', self.state.AoI)
            print('is_tran:', self.state.istran)
        done = False
        if slot == self.slot:
            done = True
        #解析动作
        # 如果是最后一个动作,这时候is_tran肯定不为0
        if action_index == self.action.num_actions-1:
            #终止动作, 不选择任何动作
            if self.rending: print('action: take no beams')
            # 此时有两种情况，一种是上次传输完，一种是上次没有传输完
            if self.start_slot != -1 and self.cover != []:
                if self.rending: print('tran not finished')
                # 这个时隙传输完
                if self.state.istran == 1:
                    if self.rending: print('==========tran finished===============')

                    # 计算AoI
                    self.state.AoI[self.cover] = slot - self.start_slot
                    self.start_slot = -1
                    self.cover = []
                    self.state.istran -= 1
                else:
                    self.state.AoI += 1
                    self.state.istran -= 1
            else:
                if self.rending: print('no tran')
                self.state.AoI += 1             
                #self.state.istran += 1          # 表示没有传输完

            next_state = self.state
            reward = -np.sum(self.state.AoI * self.state.weight) 
            return next_state, reward, done
        else:#否则就是选择了某个波束
            if self.rending: print('action: take beams')
            one_hot = np.zeros(self.action.num_actions)
            one_hot[action_index] = 1
            if self.rending: print('action_index:', action_index)
            action = self.action.one_hot_to_action(one_hot)
            beam_width = action[0]
            start_position = action[1]
            beam_distance = action[2]
        
        min_snr_node, min_snr, covered, data_rate = find_min_snr_node(self.env_param.nodes, self.env_param.base_station,
                                                    start_position + beam_width / 2, beam_width, beam_distance, self.env_param.alpha,
                                                  self.env_param.Gr_vec, self.env_param.N0, self.env_param.B,
                                                  self.env_param.z, self.env_param.pathloss, self.env_param.Ptx_BS_dBm)

        

        # 如果有调度成功的节点
        if min_snr_node is not None:
            #传输要多少时隙
            trans_time = self.env_param.package_size / data_rate
            if self.rending: print('time: ', trans_time)
            self.state.istran = math.ceil(trans_time * 1000 / self.slot_length) #转换成ms
            if self.rending: print('trans time: ', self.state.istran)
            #从这个时隙开始传输
            self.start_slot = slot
            self.cover = covered
            self.state.AoI += 1
        else:# 如果是选择了波束，但是并没有覆盖节点
            self.state.AoI += 1


        next_state = self.state

        # 检查是否AoI有问题
        if next_state.AoI.min() < 0:
            print('AoI:', next_state.AoI)
            print('slot:', slot)
            print('is_tran:', self.state.istran)
            pdb.set_trace()
            sys.exit('AoI is negative!')
        
        
        reward = -np.sum(self.state.AoI * self.state.weight) 


        return next_state, reward, done



# #调试
# if __name__ == '__main__':
#     env = mm_env(env_init(5))
#     env.printEnvParam()
#     # 模拟10000个时隙，使用最大
#     state = env.reset()
#     print('state:', state.AoI)
#     print('beam_width', env.action.beam_width)
#     print('len_beam_width', len(env.action.beam_width))
#     print('start_position', env.action.start_position)
#     print('len_start_position', len(env.action.start_position))
#     print('distance', env.action.distance)
#     print('len_distance', len(env.action.distance))
#     # 从node里面选取一个节点
#
#     next_state, reward, done = env.step(action, slot)  # 环境更新









#     print(env.state.coverage)
#     print(env.state.distance)
#     print(env.state.snr)
#     action = Action()
#     # 随机从action.distance里面选取一个值
#     a = np.random.choice(env.state.distance)
#     action_array = [120, 12, a]
#     state, reward, done =env.step(action_array)
#     print('action:',action_array)
#     print('state:',state.coverage)
#     print('reward:',reward)
#     print('done:',done)
# if __name__ == '__main__':
#     env = mm_env(env_init(512))
#     print(env.state.distance)
#     print(env.state.coverage)
#     print(env.state.snr)



