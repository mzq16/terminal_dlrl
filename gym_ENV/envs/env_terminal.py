from typing import List, Optional, Union
import gymnasium as gym
from collections import defaultdict
import numpy as np
import math
import random
from . import utils
import cv2
import pygame
import copy
import torch
import os
from torch import nn
from collections import deque
import pandas as pd
from .ego_vehicle import ego_vehicle
from .other_vehicle import other_vehicle
from .other_vehicle import init_other_vehicles 
from .other_vehicle import init_lstm_model
from .reward import reward
import matplotlib.pyplot as plt


'''
action:
x_offsets = [0, 0, 0, -1, 1]
y_offsets = [0, -1, 1, 0, 0]
0:stop, 1:down, 2:up, 3:left, 4:right
'''


class Terminal_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 3}

    def __init__(self, num_vehicle:int = 10, map_size:list = [1200, 720], 
                 render_mode = 'rgb_array', seed = 24, text_width = 400,
                 model_arg = {'input_dim': 2, 'hidden_dim': 64, 'output_dim': 2, 'num_layers': 2,},
                 ) -> None:
        super().__init__()
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.number_v = num_vehicle
        self.text_width = text_width
        self.obs = None
        self.info = None
        self.map_size = map_size
        self.prev_action = np.array([0])
        # 用plt画图的话，map每次都需要画，所以在render中画即可
        # self.map_render = utils.create_background_map_regular(self.map_size[0], self.map_size[1]) # np.array((width+,heigh,3))
        plot_file_path = os.path.join(self.current_directory, 'edges.csv')
        self.plot_data = pd.read_csv(plot_file_path)

        # TODO background vehicle control init
        self.model = init_lstm_model(**model_arg)
        self.init_fig()
        self.init_ov()

        start_point_id = 1
        target_point_id = 455
        self.ev_handle = ego_vehicle(G=self.G, plot_xy2id=self.plot_xy2id, id2plot_xy=self.id2plot_xy, 
                                     start_point_id=start_point_id, target_point_id=target_point_id)
        self.reward_handle = reward(G=self.G, id2plot_xy=self.id2plot_xy, plot_xy2id=self.plot_xy2id, 
                                    start_id=start_point_id, des_id=target_point_id)
        
        """
        The following is action space and observation space
        只有4个动作，要不要考虑5个，第5个就是停止，
        如果需要的话在ego vehicle中要修改一下aligned option，aligned option + [curr_point_id]
        """
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict(
            {
            "ev_curr_id": gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.int32),       
            "ev_prev_id": gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.int32),
            # "ev_direction": gym.spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32),  # 4 direction, no stop
            # "render_img": gym.spaces.Box(low=0, high=255, shape=(map_size[1], map_size[0] + text_width, 3), dtype=np.uint8),
            "ov_id": gym.spaces.Box(low=0, high=1000, shape=(self.number_v, ), dtype=np.int32),
            "des_id": gym.spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
            "map_topo": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.int32),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.seed = seed
        aligned_option = self.ev_handle.aligned_option
        self.map_topo = np.array([int(i is None) for i in aligned_option]).reshape(4,)
        self.done = False

    def init_fig(self):
        # plt的过程中统计了映射id2plot_xy, plot_xy2id，后期可以提取出来
        fig, ax1 = plt.subplots(figsize=(16, 10))
        ax1 = fig.add_subplot(111)  
        fig, ax, G, id2plot_xy, plot_xy2id = utils.plot_map(self.plot_data, ax1)
        self.ax = ax
        self.G = G
        self.plot_xy2id = plot_xy2id
        self.id2plot_xy = id2plot_xy
        self.new_route = [v for k, v in id2plot_xy.items()]   # get 

    def init_ov(self):
        csv_file_path = os.path.join(self.current_directory, 'processed_route.csv')
        route_df = pd.read_csv(csv_file_path)
        id_list = utils.get_vehicle_name(route_df)
        self.other_vehicles_list = init_other_vehicles(self.number_v, id_list, route_df)
        self.plot_xys = {}
        for i in range(len(self.other_vehicles_list)):
            self._get_plot_xy(self.other_vehicles_list[i])

    def setup_statistic(self):
        # only count current time
        self.road_traffic = defaultdict(int)
    
    def step(self, action: int):
        # 1. action: ego act, other act
        map_topo = self.ev_handle.step(action=action)
        self.map_topo = np.array(map_topo).reshape(4,)
        # TODO get output from LSTM, return as observation
        for i in range(self.number_v):
            tmp_vehicle = self.other_vehicles_list[i]
            tmp_vehicle.step(self.model)
            self._get_plot_xy(tmp_vehicle)      # find all plot xy coords and put into self.plot_xys

        # 3. get reward, terminal & get info
        total_reward, done = self.reward_handle.step(self.other_vehicles_list, self.ev_handle)
        self.done = done
        info = self._get_info()
        self.info = info

        # 2. get world state (obervation), include background vehicle movement
        obs = self._get_obs(action)
        self.obs = copy.deepcopy(obs)
        
        # get obs 一般在info前，紧接着action。不过这里obs包括了info reward，所以就放到后面了
        
        return obs, total_reward, done, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        map_topo = self.ev_handle.reset(seed=seed)
        self.map_topo = np.array(map_topo).reshape(4,)
        #self.map_topo = np.zeros((4,))

        # TODO background vehicle control reset
        for i in range(self.number_v):
            # 实际上因为该问题的特殊性，这里的ov reset是None，因为ov的路径是根据历史路径来的，所以就不reset历史路径了
            self.other_vehicles_list[i].reset()

        self.reward_handle.reset(self.ev_handle.start_id, self.ev_handle.target_id)
        obs = self._get_obs(0)
        self.obs = copy.deepcopy(obs)
        info = self._get_info()
        self.done = False
        return obs, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            width, height = np.array(self.map_size) 
            self.window = utils.init_render(700, 600)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        render_img = self.get_render_img(self.text_width)
    
        canvas = pygame.surfarray.make_surface(render_img.swapaxes(0, 1))
        scaled_canvas = pygame.transform.scale(canvas, (700, 600))
        # self.window.blit(canvas)
        # pygame.display.flip()

        if self.render_mode == "human":
        # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(scaled_canvas, (0,0))
            # self.window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _get_obs(self, action: int):
        obs = {}
        ev_curr_id, ev_prev_id = self.ev_handle._get_ev_loc_id()
        obs['ev_curr_id'] = np.array([ev_curr_id]) if ev_curr_id else np.array([-1])
        obs['ev_prev_id'] = np.array([ev_prev_id]) if ev_prev_id else np.array([-1])
        # obs['ev_direction'] = np.array(action).reshape(1,)      # 其实curr - prev就是direction
        obs['des_id'] = np.array([self.ev_handle.target_id])
        ov_id_list = []
        for i in range(self.number_v):
            if self.other_vehicles_list[i].curr_id is None:
                raise ValueError("ov curr id is None")
            ov_id_list.append(self.other_vehicles_list[i].curr_id)
        obs['ov_id'] = np.array(ov_id_list)
        obs['map_topo'] = self.map_topo
        return obs

    def _get_info(self):
        return {
            'ev_loc': self.ev_handle._get_ev_loc_id(),
            'reward_info': self.reward_handle.reward_info,
            }

    def _get_plot_xy(self, tmp_vehicle):
        xy_coord = utils.train_data2xy(train_data = tmp_vehicle.recent_history[-1], mean = tmp_vehicle.mean, std = tmp_vehicle.std)
        plot_xy, node_id = utils.xy2plot(coord = xy_coord, idx_blue_dot = self.plot_xy2id, blue_dot_dict = self.id2plot_xy, 
                                         G = self.G, new_route = self.new_route, p = tmp_vehicle)
        self.plot_xys[tmp_vehicle.v_id] = plot_xy
        tmp_vehicle.curr_id = node_id
        tmp_vehicle.history_point.append(node_id)
        return None

    def get_render_img(self, text_width=300):
        # plot map and get G (graph)
        fig, ax, G, id2plot_xy, plot_xy2id = utils.plot_map(self.plot_data, self.ax)
        self.ax = ax
        self.G = G
        self.plot_xy2id = plot_xy2id
        self.id2plot_xy = id2plot_xy
        self.new_route = [v for k, v in id2plot_xy.items()]   # get node xy

        # check if other vehcile
        if len(self.plot_xys) != 0:
            for k, xy in self.plot_xys.items():
                self.ax.plot(xy[0], xy[1], color = 'black', marker = 'o', markersize = 10)

        # check if ego-vehicle
        if self.ev_handle:
            target_pos = id2plot_xy[self.ev_handle.target_id]
            self.ax.plot(target_pos[0], target_pos[1], color = 'green', marker = 's', markersize = 10)
            ev_curr_id, ev_prev_id = self.ev_handle._get_ev_loc_id()
            if ev_curr_id:
                ev_curr_pos = id2plot_xy[ev_curr_id]
                self.ax.plot(ev_curr_pos[0], ev_curr_pos[1], color = 'red', marker = '*', markersize = 10)
            if ev_curr_id and ev_prev_id:
                ev_prev_pos = id2plot_xy[ev_prev_id]
                x, y = zip(ev_curr_pos, ev_prev_pos)
                self.ax.plot(x, y, color='red')

        # fig to array & add info
        render_img = utils.figure_to_array(fig, self.map_size)
        text_img = self._get_render_txt(render_img, text_width)
        height, width, _ = render_img.shape
        render_img_info = np.ones((height, width + text_width, 3), dtype=np.uint8) * 255
        render_img_info[:height, :width] = render_img
        render_img_info[:height, width:width + text_width] = text_img
        self.ax.clear()
        plt.close()
        '''
        上面是新的，用plt画的，不过还得转一下到数组，plotmap其实有一些繁琐，后期可以考虑简化
        这个是old version了，用cv2画的
        curr_render = self.scale * (np.array(ev_curr_loc) + 1)
        prev_render = self.scale * (np.array(ev_prev_loc) + 1)
        map_render = None
        map_render = utils.create_background_map(self.map_size[0], self.map_size[1]) # np.array((width,heigh,3))
        if map_render is None:
            map_render = self.map_render    # 是否更新，如果更新怎会摸出轨迹
        render_img = cv2.circle(map_render, (curr_render[0],curr_render[1]), 10, (0, 0, 255), -1)  # 蓝色圆点表示本车
        start_loc = self._CR2pixel(self.ev_handle.start_CR)
        des_loc = self._CR2pixel(self.ev_handle.des_CR)
        render_img = cv2.circle(map_render, start_loc, 10, (255, 0, 0), -1)  # 红色圆点表示起点
        render_img = cv2.circle(map_render, des_loc, 10, (0, 255, 0), -1)  # 绿色圆点表示终点
        render_img = cv2.line(render_img, (prev_render[0],prev_render[1]), (curr_render[0],curr_render[1]), (0, 0, 255), 2) # 画走过的路径
        text_img = self._get_render_txt(render_img, text_width)
        height, width, _ = render_img.shape
        final_render_img = np.ones((height, width + text_width, 3), dtype=np.uint8) * 255
        final_render_img[:height, :width] = render_img
        final_render_img[:height, width:width+text_width] = text_img
        '''
        return render_img_info

    def _get_render_txt(self, render_img:np.ndarray, text_width):
        ev_curr_id, ev_prev_id = self.ev_handle._get_ev_loc_id()
        start_id = self.ev_handle.start_id
        des_id = self.ev_handle.target_id
        ev_curr_id = -1 if ev_curr_id is None else ev_curr_id
        ev_prev_id = -1 if ev_prev_id is None else ev_prev_id
        height, width, _ = render_img.shape
        text_img = np.ones((height, text_width, 3), dtype=np.uint8) * 255
        if self.info is None:
            return text_img
        info = self.info['reward_info']
        txt_t = []
        txt_t1 = f'r_exc:{info["r_exc"]:5.2f}, r_spd:{info["r_spd"]:5.2f}'
        #txt_t2 = f'r_spd:{info["r_spd"]:5.2f}'
        txt_t2 = f'r_t:{info["r_t"]:5.2f}, r_total:{info["r_total"]:5.2f}'
        # txt_t4 = f'r_total:{info["r_total"]:5.2f}'
        txt_t3 = f'r_dir:{info["r_dir"]:5.2f}, r_arr:{info["r_arr"]:5.2f}'
        # txt_t6 = f'r_arr:{info["r_arr"]:5.2f}'
        txt_t4 = f'r_dis:{info["r_dis"]:5.2f}, done:{self.done}'
        txt_t5 = f"ev_curr_id:{ev_curr_id}, ev_prev_id:{ev_prev_id}"
        txt_t6 = f"start_id:{start_id}, des_id:{des_id}"


        txt_t = [txt_t1, txt_t2, txt_t3, txt_t4, txt_t5, txt_t6]
        for i in range(len(txt_t)):
            text_img = cv2.putText(text_img, txt_t[i], (0, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return text_img









