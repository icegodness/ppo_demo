import numpy as np
import torch




# 定义状态类
class State:
    def __init__(self, env_param):
        self.distance = env_param.radius
        self.num_states = env_param.UE_num + 1   # 状态空间大小
        self.AoI = np.ones(env_param.UE_num)  # AoI
        self.weight = np.random.uniform(0, 1, env_param.UE_num)  # 权重
        print("weight:", self.weight)
        self.istran = 0 # 是否是传输状态，0代表没有传输，1代表传输

    def to_ndarray_normalized(self):
        # 归一化处理
        AoI = (self.AoI - self.AoI.mean()) / (self.AoI.std() + 1e-8)
        #AoI = self.AoI / np.max(self.AoI)
        is_tran = np.array([self.istran], dtype=np.float32)
        # 返回拼接的向量，保证是浮点型
        return np.concatenate([AoI, is_tran]).astype(np.float32)


# 定义动作类
class Action:
    def __init__(self, env_param):
        #有最低波束宽度是5，最大波束宽度是180，生成beam_width
        self.beam_width = [i for i in range(5, 180 + 5, 5)]
        self.start_position = [i for i in range(0, 360, 5)]
        # self.no_tran = [0, 1]


        self.distance = env_param.radius
        self.num_type_of_action = 3  # 动作类型数量
        self.num_action_per_type = np.zeros(self.num_type_of_action, dtype=int)
        self.num_action_per_type[0] = len(self.beam_width)
        self.num_action_per_type[1] = len(self.start_position)
        self.num_action_per_type[2] = len(self.distance)
        # self.num_action_per_type[3] = len(self.no_tran)
        #动作空间大小
        self.num_actions = np.prod(self.num_action_per_type) + 1  # 动作空间大小

    def action_to_one_hot(self, action):
        beam_index = self.beam_width.index(action[0])
        start_index = self.start_position.index(action[1])
        distance_index = self.distance.index(action[2])
        # no_tran_index = self.no_tran.index(action[3])

        # 计算总索引
        # total_index = beam_index * len(self.start_position) * len(self.distance) * len(self.no_tran) + \
        #               start_index * len(self.distance) * len(self.no_tran) + \
        #               distance_index * len(self.no_tran) + no_tran_index
        total_index = beam_index * len(self.start_position) * len(self.distance)  + \
                      start_index * len(self.distance)  + \
                      distance_index 
        # 创建  one-hot 编码
        one_hot_vector = np.zeros(self.num_actions)
        one_hot_vector[total_index] = 1

        # 在最后增加一个终止动作
        one_hot_vector = np.append(one_hot_vector, 0)

        return one_hot_vector

    def one_hot_to_action(self, one_hot_vector):
        # 去除终止动作
        one_hot_vector = one_hot_vector[:-1]
        
        total_index = np.argmax(one_hot_vector)

        # 恢复各部分索引
        beam_index = total_index // (len(self.start_position) * len(self.distance))
        start_index = (total_index % (len(self.start_position) * len(self.distance))) // len(self.distance)
        distance_index = total_index % len(self.distance)

        # 查找原始值
        action = [self.beam_width[beam_index], self.start_position[start_index], self.distance[distance_index], 0]
        return action

    # def one_hot_to_action(self, one_hot_vector):
    #     total_index = np.argmax(one_hot_vector)

    #     # 恢复各部分索引
    #     beam_index = total_index // (len(self.start_position) * len(self.distance) * len(self.no_tran))
    #     remaining_index = total_index % (len(self.start_position) * len(self.distance) * len(self.no_tran))
        
    #     start_index = remaining_index // (len(self.distance) * len(self.no_tran))
    #     remaining_index = remaining_index % (len(self.distance) * len(self.no_tran))
        
    #     distance_index = remaining_index // len(self.no_tran)
    #     no_tran_index = remaining_index % len(self.no_tran)

    #     # 查找原始值
    #     action = [self.beam_width[beam_index], self.start_position[start_index], self.distance[distance_index], self.no_tran[no_tran_index]]
    #     return action


# 测试函数
def test_action_class():
    class EnvParam:
        radius = [10, 20, 30]  # Example distances

    env_param = EnvParam()
    action_instance = Action(env_param)

    # 定义一个测试动作
    test_action = [15, 30, 20, 0]

    # 测试 action_to_one_hot 和 one_hot_to_action
    one_hot = action_instance.action_to_one_hot(test_action)
    recovered_action = action_instance.one_hot_to_action(one_hot)

    assert test_action == recovered_action, f"Test failed: {test_action} != {recovered_action}"

    print("Test passed!")




# 执行测试函数
if __name__ == "__main__":

    test_action_class()


