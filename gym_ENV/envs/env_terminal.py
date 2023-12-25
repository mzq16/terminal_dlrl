from typing import List, Optional, Union
from git import Tree
import gymnasium as gym
from collections import defaultdict
import numpy as np
import math
import random

from pyparsing import col
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
import networkx as nx

plt.ioff()

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
                 route_number = 1, route_length = 5, 
                 ) -> None:
        super().__init__()
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.number_v = num_vehicle
        self.text_width = text_width
        self.obs = None
        self.route_number = route_number
        self.route_length = route_length
        
        self.map_size = map_size
        self.prev_action = np.array([0])
        # 用plt画图的话，map每次都需要画，所以在render中画即可
        # self.map_render = utils.create_background_map_regular(self.map_size[0], self.map_size[1]) # np.array((width+,heigh,3))
        plot_file_path = os.path.join(self.current_directory, 'edges.csv')
        self.plot_data = pd.read_csv(plot_file_path)
        latlon_file_path = os.path.join(self.current_directory, 'point_latlon.npy')
        self.latlon = np.load(latlon_file_path)
        self.plot_args = utils.preprocess_plot_map_quick(self.latlon)
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
        self.info = {'reward_info': self.reward_handle.reward_info}
        """
        The following is action space and observation space
        只有4个动作，要不要考虑5个，第5个就是停止，
        如果需要的话在ego vehicle中要修改一下aligned option，aligned option + [curr_point_id]
        """
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict(
            {
            #"ev_curr_id": gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.int64),       
            #"ev_prev_id": gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.int64),
            "history_id": gym.spaces.Box(low=0, high=1000, shape=(5,), dtype=np.int64),       
            # "ev_direction": gym.spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32),  # 4 direction, no stop
            "render_img": gym.spaces.Box(low=0, high=255, shape=(map_size[1], map_size[0], 3), dtype=np.uint8),
            "ov_id": gym.spaces.Box(low=0, high=1000, shape=(self.number_v, ), dtype=np.int64),
            "des_id": gym.spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int64),
            "map_topo": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.int64),
            "routes": gym.spaces.Box(low=0, high=1000, shape=(route_number, route_length), dtype=np.int64),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.seed = seed
        aligned_option = self.ev_handle.aligned_option
        self.map_topo = np.array([int(i is None) for i in aligned_option]).reshape(4,)
        
    def init_fig(self):
        # plt的过程中统计了映射id2plot_xy, plot_xy2id，后期可以提取出来
        fig, ax = plt.subplots(figsize=(16, 10))
        # ax = fig.add_subplot(111)  
        ax.set_xlim(0.020, 0.065)
        ax.set_ylim(-0.004, 0.008)
        G, id2plot_xy, plot_xy2id = utils.plot_map(plot_data=self.plot_data, ax1=ax)
        self.fig = fig
        self.ax = ax
        self.G = G
        self.plot_xy2id = plot_xy2id
        self.id2plot_xy = id2plot_xy
        self.new_route = [v for k, v in id2plot_xy.items()]   # get 
        self._init_draw_tools()

    def _init_draw_tools(self):
        self._scatter_ego = self.ax.scatter([], [], color = 'red', marker = '*', s = 150, zorder=4)
        self._scatter_other = self.ax.scatter([], [], color = 'black', marker = 'o', s = 80, zorder=2)
        self._line_route, = self.ax.plot([], [], color='green', linewidth=3, zorder=3)
        self._line_history, = self.ax.plot([], [], color='red', linewidth=3, zorder=2)

    def init_ov(self):
        csv_file_path = os.path.join(self.current_directory, 'processed_route.csv')
        route_df = pd.read_csv(csv_file_path)
        id_list = utils.get_vehicle_name(route_df)
        self.other_vehicles_list = init_other_vehicles(self.number_v, id_list, route_df)
        self.plot_xys = {}
        for i in range(len(self.other_vehicles_list)):
            self._get_plot_xy(self.other_vehicles_list[i])

    def init_img_info(self):
        self.done = False
        self.action = None

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
        obs = self._get_obs(action)
        total_reward, done = self.reward_handle.step(self.other_vehicles_list, self.ev_handle, obs["routes"])
        self.done = done
        self.action = action
        info = self._get_info()
        self.info = info

        # 2. get world state (obervation), include background vehicle movement
        # 这下面的数据都有可能出现ev loc为None的情况，因此需要考虑
        
        self.obs = copy.deepcopy(obs)
        
        # get obs 一般在info前，紧接着action。不过这里obs包括了info reward，所以就放到后面了
        
        return obs, total_reward, done, False, info
    
    def reset(self, seed=None, options=None, OD=None):
        super().reset(seed=seed)
        self.init_img_info()
        map_topo = self.ev_handle.reset(seed=seed, OD=OD)
        self.map_topo = np.array(map_topo).reshape(4,)

        # TODO background vehicle control reset
        for i in range(self.number_v):
            # 实际上因为该问题的特殊性，这里的ov reset是None，因为ov的路径是根据历史路径来的，所以就不reset历史路径了
            self.other_vehicles_list[i].reset()
        obs = self._get_obs(0, True)
        self.reward_handle.reset(self.ev_handle.start_id, self.ev_handle.target_id, obs["routes"])
        self.obs = copy.deepcopy(obs)
        info = self._get_info()
        self.info = info
        
        return obs, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self, **info_args):
        if self.render_mode == "rgb_array":
            return self._render_frame(**info_args)
        else:
            self._render_frame(**info_args)
    
    def _render_frame(self, **info_args):
        if self.window is None and self.render_mode == "human":
            width, height = np.array(self.map_size) 
            self.window = utils.init_render(700, 600)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        render_img_with_info = self._concat_info_img(text_width=self.text_width, **info_args)
    
        canvas = pygame.surfarray.make_surface(render_img_with_info.swapaxes(0, 1))
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

    def _get_obs(self, action: int, reset_init=False):
        obs = {}
        history_point_id = self.ev_handle.history_point_id
        if None in history_point_id:
            assert history_point_id[-1] is None
            print("history has None item")
            history_point_id[-1] = -1
        obs['history_id'] = np.array(list(history_point_id))
        #obs['ev_prev_id'] = np.array([ev_prev_id]) if ev_prev_id else np.array([-1])
        # obs['ev_direction'] = np.array(action).reshape(1,)      # 其实curr - prev就是direction
        obs['des_id'] = np.array([self.ev_handle.target_id])
        ov_id_list = []
        for i in range(self.number_v):
            if self.other_vehicles_list[i].curr_id is None:
                raise ValueError("ov curr id is None")
            ov_id_list.append(self.other_vehicles_list[i].curr_id)
        obs['ov_id'] = np.array(ov_id_list)
        obs['map_topo'] = self.map_topo
        tmp_paths = self._get_routes(num_routes=self.route_number, len_routes=self.route_length)
        obs["routes"] = np.array(tmp_paths)
        img = self.get_render_img(routes=obs["routes"], reset_init=reset_init)
        obs["render_img"] = img
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

    def _get_routes(self, num_routes: int = 2, len_routes: int = 5):
        shortest_paths = []
        if self.ev_handle.current_id is None:
            for i in range(num_routes):
                shortest_paths.append([-1 for _ in range(len_routes)])
            return shortest_paths
        
        for path in nx.shortest_simple_paths(self.G, source=self.ev_handle.current_id, target=self.ev_handle.target_id):
            shortest_paths.append(path)
            if len(shortest_paths) == num_routes:
                break
        # padding
        while num_routes > len(shortest_paths):
            shortest_paths.append([-1 for _ in range(len_routes)])
        # padding
        for i in range(num_routes):
            if len(shortest_paths[i]) > len_routes:
                shortest_paths[i] = shortest_paths[i][:len_routes]
            else:
                while len(shortest_paths[i]) < len_routes:
                    shortest_paths[i].append(-1)
        return shortest_paths

    def get_render_img(self, reset_init:bool=False, **info_args):
        # plot map and get G (graph)
        if reset_init:
            self.ax.clear()
            self._init_draw_tools()
            self.ax.plot(self.plot_args[0][:, 0], self.plot_args[0][:, 1], color='gray')
            self.ax.plot(self.plot_args[1][:, 0], self.plot_args[1][:, 1], color='gray')
            self.ax.scatter(self.plot_args[2][:,0], self.plot_args[2][:,1], color='b')
            self.ax.scatter(self.plot_args[3][:,0], self.plot_args[3][:,1], color='b')
            assert self.ev_handle
            target_pos = self.id2plot_xy[self.ev_handle.target_id]
            self.ax.scatter(target_pos[0], target_pos[1], color = 'green', marker = 's', s = 80)

        # check if other vehcile
        if len(self.plot_xys) != 0:
            xy_list = []
            for k, xy in self.plot_xys.items():
                xy_list.append(xy)
            self._updata_other(scatter=self._scatter_other, other_v_xy=xy_list, color='black', size=80)
        else:
            self._updata_other(self._scatter_other)

        # check if ego-vehicle
        if self.ev_handle:
            ev_curr_id, ev_prev_id = self.ev_handle._get_ev_loc_id()
            history_id = self.ev_handle.history_point_id
            if ev_curr_id is not None:
                ev_curr_pos = self.id2plot_xy[ev_curr_id]
                self._updata_ego(scatter=self._scatter_ego, new_x=ev_curr_pos[0], new_y=ev_curr_pos[1], color='red', size=150)
            else:
                self._updata_ego(self._scatter_ego)
                #self.ax.scatter(ev_curr_pos[0], ev_curr_pos[1], color = 'red', marker = '*', s = 150)
        
            xy_list = []
            if not None in history_id:
                for i in range(len(history_id)):
                    tmp_pos = self.id2plot_xy[history_id[i]]
                    xy_list.append(tmp_pos)
                x, y = zip(*xy_list)
                self._update_line(line=self._line_history, new_x=x, new_y=y, color='red', line_width=3)
            else:
                self._update_line(self._line_history)
                #self.ax.plot(x, y, color='red', linewidth=2)
        
        # plot routes
        routes = info_args.get("routes")
        if routes is not None:
            curr_route = routes[0]
            xy_list = []
            for i in range(self.route_length):
                if curr_route[i] != -1:
                    xy_list.append(self.id2plot_xy[curr_route[i]])
            x, y = zip(*xy_list)
            self._update_line(line=self._line_route, new_x=x, new_y=y, color='green', line_width=3)
        else:
            self._update_line(self._line_route)
            #self.ax.plot(x, y, color='green', linewidth=2)
        
        # fig to array & add info
        render_img = utils.figure_to_array(self.fig, self.map_size)
        return render_img
        
    def _updata_ego(self, scatter, new_x=None, new_y=None, color='red', size=150):
        if new_x:
            scatter.set_offsets([[new_x, new_y]])
            scatter.set_color(color)
            scatter.set_sizes([size])
        else:
            scatter.set_offsets([[None,None]])

    def _updata_other(self, scatter, other_v_xy=None, color='black', size=80):
        if other_v_xy:
            scatter.set_offsets(other_v_xy)
            scatter.set_color(color)
            scatter.set_sizes([size])
        else:
            scatter.set_offsets([[None,None]])
        
    def _update_line(self, line, new_x=None, new_y=None, color='g', line_width=2):
        line.set_xdata(new_x)
        line.set_ydata(new_y)
        line.set_color(color)
        line.set_linewidth(line_width)
        
    def _concat_info_img(self, text_width=300, **info_args):
        render_img = self.obs.get("render_img")
        text_img = self._get_render_txt(render_img, text_width, **info_args)
        height, width, _ = render_img.shape
        render_img_info = np.ones((height, width + text_width, 3), dtype=np.uint8) * 255
        render_img_info[:height, :width] = render_img
        render_img_info[:height, width:width + text_width] = text_img
        return render_img_info

    def _get_render_txt(self, render_img:np.ndarray, text_width, **info_args):
        # ev_curr_id, ev_prev_id = self.ev_handle._get_ev_loc_id()
        start_id = self.ev_handle.start_id
        des_id = self.ev_handle.target_id
        #ev_curr_id = -1 if ev_curr_id is None else ev_curr_id
        #ev_prev_id = -1 if ev_prev_id is None else ev_prev_id
        history_id = self.ev_handle.history_point_id
        height, width, _ = render_img.shape
        text_img = np.ones((height, text_width, 3), dtype=np.uint8) * 255
        if self.info is None:
            return text_img
        info = self.info['reward_info']
        txt_t = []
        txt_t1 = f'r_exc:{info["r_exc"]:5.2f}, r_timeout:{info["r_timeout"]:5.2f}'
        txt_t2 = f'r_t:{info["r_t"]:5.2f}, r_total:{info["r_total"]:5.2f}'
        txt_t3 = f'r_dir:{info["r_dir"]:5.2f}, r_arr:{info["r_arr"]:5.2f}'
        txt_t4 = f'r_dis:{info["r_dis"]:5.2f}, r_path:{info["r_path"]:5.2f}'
        txt_t5 = f"done:{self.done}, action:{self.action}"
        txt_t6 = f"start_id:{start_id}, des_id:{des_id}"

        # history id
        history_len = len(history_id)
        txt_histroy_list = [None for i in range(math.ceil(history_len / 2.0))]
        for i in range(history_len):
            if i % 2 == 0:
                tmp_txt = f't-{history_len - i}: {history_id[i]},'
                txt_histroy_list[int(i // 2)] = copy.deepcopy(tmp_txt)
            else:
                tmp_txt = txt_histroy_list[int(i // 2)] + f't-{history_len - i}: {history_id[i]}'
                txt_histroy_list[int(i // 2)] = tmp_txt
        
        topo = info_args.get('topo')
        action_prob = info_args.get('action_prob')
        txt_t7 = f"topo:{topo} "
        if action_prob is None:
            txt_t8 = f"prev_a_prob:{action_prob}"
        else:
            txt_t8 = f"prev_a_prob: "
            action_prob = action_prob.reshape(-1,)
            for i in range(len(action_prob)):
                tmp_ = float(action_prob[i])
                txt_t8 += f"{tmp_:.2f} "
        txt_t = [txt_t1, txt_t2, txt_t3, txt_t4, txt_t5, txt_t6, txt_t7, txt_t8] + txt_histroy_list
        for i in range(len(txt_t)):
            text_img = cv2.putText(text_img, txt_t[i], (0, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return text_img









