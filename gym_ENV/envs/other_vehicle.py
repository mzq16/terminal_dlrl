from .lstm import LSTM
from collections import deque
import torch
import numpy as np
import random
import pandas as pd
from . import utils
import os

def init_lstm_model(input_dim, hidden_dim, output_dim, num_layers, model_path='model_50.pth'):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_directory, model_path)
        model = LSTM(input_dim, hidden_dim, output_dim, num_layers)
        model.load_state_dict(torch.load(model_path ,map_location=torch.device('cpu')))  # 替换'best_model.pth'为您的模型文件路径
        model.eval() 
        return model 

def init_other_vehicles(number_v, name_list, route_df):
        route_keys = list(route_df.keys())
        selected_ids = random.sample(name_list, number_v)
        other_vehicles_list = [other_vehicle() for x in range(number_v)]
        for id, vehicle in zip(selected_ids, other_vehicles_list):
            vehicle.v_id = id
            # 选择出的数据
            init_df = route_df[route_df[route_keys[0]] == id]
            init_df = init_df.iloc[:, [2, 3]]
            init_df = init_df.iloc[-50:, :]
            init_df = pd.DataFrame(init_df)

            # 归一化处理
            std = init_df.iloc[:, :].std()
            mean = init_df.iloc[:, :].mean()
            # std不为0
            if std[0] == 0:
                 std[0] = 1
            if std[1] == 0:
                 std[1] = 1
            init_df = (init_df - mean) / std
            vehicle.mean = mean.values
            vehicle.std = std.values

            # 填充到历史队列里面
            input_x = init_df.values
            for input_xy_coord in input_x:
                vehicle.recent_history.append(input_xy_coord)
            
        return other_vehicles_list

class other_vehicle(object):
    def __init__(self):
        self.v_id = None
        self.recent_history = deque(maxlen=50)
        self.history_point = deque(maxlen=2)
        self.mean = None
        self.std = None
        self.current_position = None
        self.curr_id = None

    def __str__(self):
        return f"Point({self.v_id})"

    def step(self, model):
        recent_history = np.array((list(self.recent_history)))
        input_x = torch.from_numpy(recent_history).type(torch.Tensor).unsqueeze(0)
        with torch.no_grad():
            pred_output = model(input_x)
        pred_output = pred_output.cpu().numpy()
        self.current_position = pred_output[0]
        self.recent_history.append(self.current_position)
        return None
    
    def reset(self):
        pass

    def destroy(self):
        self.recent_history.clear()

