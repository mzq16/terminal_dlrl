import numpy as np
import random
from collections import deque
import torch
import math
from . import utils

class ego_vehicle_old(object):
    def __init__(self, start_CR: np.ndarray, des_CR: np.ndarray, seed: int = None) -> None:
        self.start_CR = start_CR
        self.des_CR = des_CR
        self.prev_CR = start_CR
        self.curr_CR = start_CR
        self.prev_direction = np.array([0, 0])
        self.curr_direction = np.array([0, 0])
        self.histroy_route = []
        if seed is None:
            seed = np.random.randint(1e5)
        self.random_state = np.random.RandomState(seed)

    def step(self, action: np.ndarray):
        #assert isinstance(action, list) or isinstance(action, int), "error typevalue of action"
        #assert isinstance(action, int), "error typevalue of action"
        # a_dir, a_speed = action
        turn_around = 0
        if (action != 0).all():
            # if not stop, update direction
            # if stop keep the direction 
            self.prev_direction = self.curr_direction
            self.curr_direction = action
            turn_around = np.array(self.prev_direction) * np.array(self.curr_direction) < -0.5  # in case of cal acc

        '''
        编号是从上到下,这和img中的画法以及GUROBI的排序是一样的,
        不一样的是上下左右,例如在一个点是(7,2),意思是在第7列第2行x=7,y=2,这个点向下走应该是到(7,3)
        1: np.array([0, 1]),
        2: np.array([0, -1]),
        3: np.array([-1, 0]),
        4: np.array([1, 0]),
        0:stop, 1:up, 2:down, 3:left, 4:right
        所以左右没有问题，上下应该相反，但是这对于训练应该是没有影响的
        '''

        new_loc = self.curr_CR + action
        self.prev_CR = self.curr_CR 
        self.curr_CR = new_loc
        self.histroy_route.append(new_loc)
    
        return self.curr_CR, self.prev_CR, turn_around
    
    def reset(self, seed=None):
        self.destroy()
        
        self.prev_direction = np.array([0, 0])
        self.curr_direction = np.array([0, 0])
        if seed is None:
            seed = np.random.randint(1e5)
        self.random_state = np.random.RandomState(seed)
        self.get_new_start_des()

    def destroy(self):
        self.histroy_route = []
        self.random_state = None
        
    def get_new_start_des(self):
        # get new start point and destination point
        des_CR = [self.random_state.choice(12), self.random_state.choice(9)]
        start_CR = [self.random_state.choice(12), self.random_state.choice(9)]
        while des_CR == start_CR:
            des_CR = [self.random_state.choice(12), self.random_state.choice(9)]
            start_CR = [self.random_state.choice(12), self.random_state.choice(9)]
        self.start_CR = np.array(start_CR)
        self.des_CR = np.array(des_CR)
        self.prev_CR = self.start_CR
        self.curr_CR = self.start_CR

    def set_new_start_des(self, start_CR = np.array([0, 0]), des_CR = np.array([8, 9])):
        # set new start point and destination point
        self.start_CR = start_CR
        self.des_CR = des_CR
        self.prev_CR = start_CR
        self.curr_CR = start_CR
           
    def _get_ev_loc(self):
        return self.curr_CR, self.prev_CR
    
    def _get_start_loc(self):
        return self.start_CR

    def _get_des_loc(self):
        return self.des_CR

class ego_vehicle(object):
    def __init__(self, G, plot_xy2id, id2plot_xy, start_point_id = None, target_point_id = 453, history_len = 5):
        self.history_inputxy = deque(maxlen = 10)         # useless
        self._history_point_id = deque(maxlen = history_len)       # include curr_id and prev_ids
        self._histroy_action = deque(maxlen = history_len)        # include curr_dir and prev_dir
        self.history_len = history_len
        self.plot_xy2id = plot_xy2id
        self.id2plot_xy = id2plot_xy
        self.G = G

        
        self.init_param(start_point_id, target_point_id)
        self._action_to_direction = { 
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }

    def init_param(self, start_point_id, target_point_id):
        self._target_id = int(target_point_id)
        self.start_id = int(start_point_id) if start_point_id is not None \
            else random.choice(list(range(len(self.id2plot_xy))).remove(target_point_id))
        self._current_id = self.start_id
        self.current_position = self.id2plot_xy[self._current_id]
        self.history_inputxy.append(self.current_position)
        for _ in range(self.history_len):
            self._history_point_id.append(self.start_id)
            self._histroy_action.append(np.array([0, 0]))
        self._aligned_option = self.get_aligned_option()

    # 执行一步的操作
    def step(self, action, random_flag = False):
        if self._current_id is None:
            raise ValueError("step currid is none")
        #neighbour_ids = list(self.G.neighbors(self.current_id))
        #neighbour_id2plot_xys = {neighbour_id: self.id2plot_xy[neighbour_id] for neighbour_id in neighbour_ids}
        #curr_plot_xy = self.id2plot_xy[self.current_id]
        #aligned_option = self.align_dir_from_angle(neighbour_id_xys = neighbour_id2plot_xys, curr_xy=curr_plot_xy)
        #aligned_option = self.get_aligned_option()
        if not isinstance(action, int):
            action = int(action)
        assert action < 4, "error action in aligned options"
        next_point_id = self._aligned_option[action]
        
        if random_flag:
            # pick up correct option
            all_are_none = all(element is None for element in self._aligned_option)
            if all_are_none:
                raise ValueError(f"no ways of node {self._current_id}")
            while next_point_id is None:
                next_point_id = np.random.choice(self._aligned_option)
        self._current_id = next_point_id
        if next_point_id is None:
            print(self._history_point_id)
        self.current_position = self.id2plot_xy[next_point_id] if next_point_id is not None else None
        self.history_inputxy.append(self.current_position)
        self._history_point_id.append(next_point_id)
        self._histroy_action.append(action)
        self._aligned_option = self.get_aligned_option()
        return [int(i is None) for i in self._aligned_option]

    def reset(self, seed=None, OD:tuple=None):
        self.destroy()
        if OD is None:
            start, target = np.random.choice(len(self.id2plot_xy), 2, replace=False)
        else:
            start, target = OD
        self.init_param(start, target)
        return [int(i is None) for i in self._aligned_option]

    def destroy(self):
        self.history_inputxy.clear()
        self._history_point_id.clear()
        self._histroy_action.clear()
        self._current_id = None
        self.start_id = None
        self._target_id = None

    def align_dir_from_angle(self, neighbour_id_xys:dict, curr_xy):
        # 上下左右，90,-90,180,0
        adjacent_point_id = [None, None, None, None]
        x, y = curr_xy 
        for neighbour_id, neighbour_xy in neighbour_id_xys.items():
            neighbor_x, neighbor_y = neighbour_xy
            angle = math.atan2(neighbor_y - y, neighbor_x - x) / math.pi * 180   # (y, x)
            if 45 < angle <= 135:
                assert adjacent_point_id[0] is None, print(f"neighbour_id:{neighbour_id}")
                adjacent_point_id[0] = neighbour_id       # Up
            elif -45 > angle >= -135:
                assert adjacent_point_id[1] is None, print(f"neighbour_id:{neighbour_id}")
                adjacent_point_id[1] = neighbour_id       # Down
            elif -135 > angle >= -180 or 180 >= angle > 135:
                assert adjacent_point_id[2] is None, print(f"neighbour_id:{neighbour_id}")
                adjacent_point_id[2] = neighbour_id       # Left
            elif -45 <= angle <= 45:
                assert adjacent_point_id[3] is None, print(f"neighbour_id:{neighbour_id}")
                adjacent_point_id[3] = neighbour_id       # Right
            else:
                print(f"neighbour_id:{neighbour_id}")
        return adjacent_point_id

    def get_aligned_option(self):
        if self._current_id is None:
            return [None for _ in range(4)]
        try:
            neighbour_ids = list(self.G.neighbors(self._current_id))
        except:
            print(self._current_id)
        neighbour_id2plot_xys = {neighbour_id: self.id2plot_xy[neighbour_id] for neighbour_id in neighbour_ids}
        curr_plot_xy = self.id2plot_xy[self._current_id]
        aligned_option = self.align_dir_from_angle(neighbour_id_xys = neighbour_id2plot_xys, curr_xy=curr_plot_xy)
        return aligned_option

    def id2xy(self, id):
        # (1) id to plot xy
        plot_xy = self.id2plot_xy[id]
        # (2) plot xy to xy
        xy = utils.plot2xy(plot_xy)
        return xy

    def _get_ev_loc_id(self):
        return self._current_id, self._history_point_id[-2]
    
    @property
    def history_point_id(self):
        return self._history_point_id
    
    @property
    def history_action(self):
        return self._histroy_action

    @property
    def target_id(self):
        return self._target_id
    
    @property
    def aligned_option(self):
        return self._aligned_option
    
    @property
    def current_id(self):
        return self._current_id

