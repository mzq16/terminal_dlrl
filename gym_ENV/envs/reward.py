import numpy as np
import math
from . import utils
from typing import List, Tuple, Dict
from .other_vehicle import other_vehicle
from .ego_vehicle import ego_vehicle

class reward(object):
    def __init__(self, G, id2plot_xy, plot_xy2id, start_id: int, des_id: int) -> None:
        ''''''
        self.G = G
        self.id2plot_xy = id2plot_xy
        self.plot_xy2id = plot_xy2id
        self.start_id = start_id
        self.des_id = des_id
        self.start_xy = self.id2xy(start_id)    # np.ndarray
        self.des_xy = self.id2xy(des_id)        # np.ndarray
        self.init_info()

        # TODO col & row -> xy, then calculate distance

        self.total_dis = np.linalg.norm(self.start_xy - self.des_xy)
        self.prev_dis = self.total_dis

    def step(self, other_vehicle_list: List[other_vehicle], ego_vehicle: ego_vehicle):
        curr_id, prev_id = ego_vehicle._get_ev_loc_id()
        curr_xy = self.id2xy(curr_id)
        # 0. if exceed the map
        if curr_id:
            r_exc = 0
        else:
            r_exc = -99
            r_total = r_exc
            self.reward_info['r_exc'] = r_exc
            self.reward_info['r_total'] = r_total
            done = True
            return r_total, done
            
        # 1. direction reward: the ego should drive ahead, if turn around ego will get penalty
        prev_dir, curr_dir  = ego_vehicle.histroy_direction

        if sum(prev_dir * curr_dir) < -0.5:
            r_dir = -5
        else:
            r_dir = 0
 
        # 2. distance from des reward: more close more reward
        distance = self.get_dis(curr_xy = curr_xy)
        r_dis = (self.prev_dis - distance) / self.total_dis
        self.prev_dis = distance

        # 3. distance from other vehicle
        # TODO sum 
        

        # 4. time, need to be as soon as possible
        r_t = -0.1 

        # 5. speed should not be zero, if ego vehicle at some current time receive positive reward, 
        #   it could stop forever to acheive higher total reward
        # r_spd = -0.1 if curr_CR == prev_CR else 0

        # 6. arrived 
        if curr_id == ego_vehicle.target_id:
            r_arr = 10
            done = True
        else:
            r_arr = 0
            done = False
        r_total = r_dir + r_dis + r_t + r_arr
        self.reward_info['r_total'] = r_total
        self.reward_info['r_dir'] = r_dir
        self.reward_info['r_dis'] = r_dis
        self.reward_info['r_t'] = r_t
        self.reward_info['r_arr'] = r_arr
        return r_total, done

    def reset(self, start_id, des_id):
        self.destroy()
        self.init_info()
        self.start_id = start_id
        self.des_id = des_id
        self.start_xy = self.id2xy(start_id)    # np.ndarray
        self.des_xy = self.id2xy(des_id)        # np.ndarray
        self.total_dis = np.linalg.norm(self.start_xy - self.des_xy)
        self.prev_dis = self.total_dis
        
    def destroy(self):
        self.total_dis = None
        self.prev_dis = None
        self.reward_info = {}

    def init_info(self):
        info = {}
        info['r_exc'] = 0
        info['r_spd'] = 0
        info['r_arr'] = 0
        info['r_t'] = 0
        info['r_dir'] = 0
        info['r_dis'] = 0
        info['r_total'] = 0
        self.reward_info = info

    def get_dis(self, curr_xy: np.ndarray):
        return np.linalg.norm(curr_xy - self.des_xy)
    
    def get_dis_dijkstra(self, curr_CR: list):
        # TODO write dijkstra algorithm  
        pass

    def id2xy(self, id: int):
        # (1) id to plot xy
        plot_xy = self.id2plot_xy[id]

        # (2) plot xy to xy
        plot_xy = [float(x) for x in plot_xy]      # Decimal to float
        xy = utils.plot2xy(plot_xy)

        return xy

    
class reward_old_version(object):
    def __init__(self, map_size: list, map_arc: list, map_xy: list, start_id: int, des_id: id) -> None:
        '''
        map_size:   list[col, row]
        map_arc:    list[tuple], (node_ind, nextnode_ind) -> (1, 2) means node 1 and node 2
        map_xy:     list[node_xy], global x&y, map_xy[node_ind] = (x, y) 
        des:        list[des_x, des_y]
        '''
        self.map_size = map_size
        self.map_arc = map_arc
        self.map_xy = map_xy
        self.start_id = start_id
        self.des_id = des_id
        self.init_info()
        start_CR=(0,0)
        des_CR=(0,0)
        # TODO col & row -> xy, then calculate distance
        self.total_dis = np.linalg.norm(start_CR - des_CR)
        self.prev_dis = self.total_dis

    def step(self, curr_CR, prev_CR, turn_around):
        # 0. if exceed the map
        if 0 <= curr_CR[0] < self.map_size[0] and 0 <= curr_CR[1] < self.map_size[1]:
            r_exc = 0
        else:
            r_exc = -99
            r_total = r_exc
            self.reward_info['r_exc'] = r_exc
            self.reward_info['r_total'] = r_total
            done = True
            return r_total, done
            
        # 1. direction reward: the ego should drive ahead, if turn around ego will get penalty
        direction  = curr_CR - prev_CR
        if turn_around:
            r_dir = -5
        else:
            r_dir = 0
 
        # 2. distance from des reward: more close more reward
        distance = self.get_dis(curr_CR = curr_CR)
        r_dis = (self.prev_dis - distance) / self.total_dis
        self.prev_dis = distance

        # 3. distance from other vehicle
        # TODO sum 
        

        # 4. time, need to be as soon as possible
        r_t = -0.1 

        # 5. speed should not be zero, if ego vehicle at some current time receive positive reward, 
        #   it could stop forever to acheive higher total reward
        # r_spd = -0.1 if curr_CR == prev_CR else 0

        # 6. arrived 
        if distance < 0.5:
            r_arr = 10
            done = True
        else:
            r_arr = 0
            done = False
        r_total = r_dir + r_dis + r_t + r_arr
        self.reward_info['r_total'] = r_total
        self.reward_info['r_dir'] = r_dir
        self.reward_info['r_dis'] = r_dis
        self.reward_info['r_t'] = r_t
        self.reward_info['r_arr'] = r_arr
        return r_total, done

    def reset(self, start_id, des_id):
        self.destroy()
        self.init_info()
        self.start_id = start_id
        self.des_id = des_id
        # TODO CR -> xy
        start_CR=(0,0)
        self.des_CR=(0,0)
        self.total_dis = math.sqrt((start_CR[0] - self.des_CR[0]) ** 2 + (start_CR[1] - self.des_CR[1]) ** 2)
        self.prev_dis = self.total_dis
        
    def destroy(self):
        self.total_dis = None
        self.prev_dis = None
        self.reward_info = {}

    def init_info(self):
        info = {}
        info['r_exc'] = 0
        info['r_spd'] = 0
        info['r_arr'] = 0
        info['r_t'] = 0
        info['r_dir'] = 0
        info['r_dis'] = 0
        info['r_total'] = 0
        self.reward_info = info

    def get_dis(self, curr_CR: np.ndarray):
        curr_xy = curr_CR
        des_xy = self.des_CR
        # TODO col & row -> xy
        return np.linalg.norm(curr_CR - self.des_CR)
    
    def get_dis_dijkstra(self, curr_CR: list):
        # TODO write dijkstra algorithm  
        pass