import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import networkx as nx
import csv
from decimal import Decimal
import os
import torch
import torch.nn as nn
import pandas as pd
import random
import sys
from collections import deque
import math
import pickle as pkl
from PIL import Image
import cv2
import pygame
from PIL import Image


# draw basic map
def plot_map(plot_data: pd.DataFrame, ax1):
    G = nx.Graph()
    blue_dot_dict={}
    idx_blue_dot={}
    preci = 15
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax1 = fig.add_subplot(111)  
    data_arr = plot_data.values
    point_count=0
    for i in range(len(data_arr)):
        row = data_arr[i][1:] # 第一项是123456 size =（11,37）
        row_new = row.reshape(int(len(row) / 2), 2)
        
        for i in range(len(row_new) - 1):
            #画出虚线             
            ax1.plot([row_new[i][0], row_new[i + 1][0]], [row_new[i][1], row_new[i+1][1]], color='r', linewidth = 1, linestyle = ':')
            
            #画出散点图             
            if(i%2 == 0):
                flag =True
                cnt = split(flag, row_new[i][0],row_new[i+1][0])
            else:
                flag = False
                cnt = split(flag, row_new[i][1],row_new[i+1][1])
        
            new_x = list(np.linspace(row_new[i][0],row_new[i+1][0],cnt))
            new_x = [round(Decimal(num), preci) for num in new_x]
            new_y = list(np.linspace(row_new[i][1],row_new[i+1][1],cnt))
            new_y = [round(Decimal(num), preci) for num in new_y]
            
            ax1.scatter(new_x, new_y, color='b')    
            #先把这些点保存到graph里面 
            dot_id_list=[]
            for idx,(coord_x,coord_y) in enumerate(zip(new_x,new_y)):
                coord_str=str(coord_x) + str(coord_y)
                if coord_str not in idx_blue_dot:
                    dot_id=len(blue_dot_dict)
                    blue_dot_dict[dot_id]=[coord_x,coord_y]
                    idx_blue_dot[coord_str]=dot_id
                    dot_id_list.append(dot_id)
                else:
                    dot_id=idx_blue_dot[coord_str]
                    dot_id_list.append(dot_id)
            
            edge_tuples = list(zip(dot_id_list[:-1], dot_id_list[1:]))
            G.add_edges_from(edge_tuples)
            point_count+=len(new_x)
            
            if(i % 2 == 1 and i != 1):
                flag = False
                cnt = split(flag, row_new[i - 3][1],row_new[i][1])
                new_x = list(np.linspace(row_new[i - 3][0], row_new[i][0], cnt))
                new_x = [round(Decimal(num), preci) for num in new_x]
                new_y = list(np.linspace(row_new[i - 3][1], row_new[i][1], cnt))
                new_y = [round(Decimal(num), preci) for num in new_y]
                
                ax1.plot([row_new[i-3][0],row_new[i][0]], [row_new[i-3][1], row_new[i][1]], color='r', linewidth=1, linestyle=':')
                ax1.scatter(new_x, new_y, color='b')
                dot_id_list=[]
                for idx, (coord_x, coord_y) in enumerate(zip(new_x,new_y)):
                    coord_str = str(coord_x) + str(coord_y)
                    if coord_str not in idx_blue_dot:
                        dot_id = len(blue_dot_dict)
                        blue_dot_dict[dot_id] = [coord_x,coord_y]
                        idx_blue_dot[coord_str] = dot_id
                        dot_id_list.append(dot_id)
                    else:
                        dot_id = idx_blue_dot[coord_str]
                        dot_id_list.append(dot_id)
            
                edge_tuples = list(zip(dot_id_list[:-1], dot_id_list[1:]))
                G.add_edges_from(edge_tuples)
                point_count += len(new_x)
        
            if(i == len(row_new)-2):
                flag = False
                cnt = split(flag, row_new[-4][1],row_new[-1][1])
                new_x = list(np.linspace(row_new[-4][0],row_new[-1][0],cnt))
                new_x = [round(Decimal(num), preci) for num in new_x]
                new_y = list(np.linspace(row_new[-4][1],row_new[-1][1],cnt))
                new_y = [round(Decimal(num), preci) for num in new_y]
                ax1.plot([row_new[-4][0],row_new[-1][0]], [row_new[-4][1], row_new[-1][1]], color='r', linewidth=1, linestyle=':')
                ax1.scatter(new_x, new_y, color='b')
                dot_id_list=[]
                for idx, (coord_x, coord_y) in enumerate(zip(new_x, new_y)):
                    coord_str=str(coord_x) + str(coord_y)
                    if coord_str not in idx_blue_dot:
                        dot_id = len(blue_dot_dict)
                        blue_dot_dict[dot_id] = [coord_x,coord_y]
                        idx_blue_dot[coord_str] = dot_id
                        dot_id_list.append(dot_id)
                    else:
                        dot_id = idx_blue_dot[coord_str]
                        dot_id_list.append(dot_id)
            
                edge_tuples = list(zip(dot_id_list[:-1], dot_id_list[1:]))
                G.add_edges_from(edge_tuples)
                point_count += len(new_x)
    return fig, ax1, G, blue_dot_dict, idx_blue_dot

def split(flag, x0, x1):
    sx = 0.0005
    sy = 0.0002 
    cnt = 0
    if(flag):
        cnt = (x1-x0)//sx
    else:
        cnt = (x1-x0)//sy
    return int(abs(cnt))

def xy2plot(coord, idx_blue_dot, blue_dot_dict, G, new_route, p=None):
    coord[0] = coord[0] - 121.98
    coord[1] = coord[1] - 29.75
    # 坐标转换
    def coordinate_transform(x_prime, y_prime, theta= -42):
        theta_rad = math.radians(theta)
        x = x_prime * math.cos(theta_rad) - y_prime * math.sin(theta_rad)
        y = y_prime * math.cos(theta_rad) + x_prime * math.sin(theta_rad)
        return x, y

    # 找到最近的点
    def find_nearest_blue_point(x, y, points):
        min_distance = float('inf')  # 初始化为正无穷
        nearest_point = None
        nearest_dot = None
        for point in points:
            px, py = point
            coord_str=str(point[0]) + str(point[1])
            dot_id=idx_blue_dot[coord_str]
            px = float(px)
            py = float(py)
            distance = math.sqrt((x - px)**2 + (y - py)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
                nearest_dot=dot_id
        return nearest_point,nearest_dot
    
    def choose_next_waypoint(x,y,prev_dot_id,history_point):
        neighbors = list(G.neighbors(prev_dot_id))
        neighbors_points=[blue_dot_dict[dot_id] for dot_id in neighbors if dot_id not in history_point]
        return neighbors_points
    
    x, y=coordinate_transform(coord[0],coord[1])

    # 第一步时候不用选图上邻居
    if p.curr_id is None:
        try:
            (x, y), dot_id=find_nearest_blue_point(x, y, new_route)
        except:
            print(1231,x,y,coord,p.v_id, p.recent_history[-1], p.mean, p.std)
        return (x,y),dot_id

    # 第二步开始要选择邻居点
    else:
        neighbors_points=choose_next_waypoint(x, y,p.curr_id,p.history_point)
        (x, y), dot_id=find_nearest_blue_point(x, y, neighbors_points)
        return (x,y),dot_id

def plot2xy(plot_xy) -> np.ndarray:
    x, y = rotate(plot_xy[0], plot_xy[1], theta = 42)
    x += 121.98
    y += 29.75
    return np.array([x, y])

def rotate(x_prime, y_prime, theta = -42):
    theta_rad = math.radians(theta)
    x = x_prime * math.cos(theta_rad) - y_prime * math.sin(theta_rad)
    y = y_prime * math.cos(theta_rad) + x_prime * math.sin(theta_rad)
    return x, y

def figure_to_array(fig, size = (1200, 720)):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    image = cv2.resize(image, size)
    return image[:,:,:3]    # RGB

def train_data2xy(train_data:np.ndarray, mean:np.ndarray, std:np.ndarray) -> np.ndarray:
    # train_data是标准化的 
    return train_data * std + mean

# init process
def get_vehicle_name(route_df, threshold = 50):
    route_keys = list(route_df.keys())
    df_id = route_df[route_keys[0]]
    id_name = df_id.value_counts().keys() # 获取设备名称
    name_50 = []
    for i in range(len(id_name)):
        name = id_name[i]
        temp_df = route_df[route_df[route_keys[0]] == name]
        
        if(len(temp_df) < threshold):
            continue
        else:
            name_50.append(name)
    return name_50

  

# past version function regular
def create_background_map_regular(width = 12, height = 9, scale = 100):
    # 创建一个白色背景
    # width, height = 12, 9
    image = np.ones(((height + 2) * scale, (width + 2) * scale, 3), dtype=np.uint8) * 255

    # 绘制路口的交叉点
    for i in range(height + 1):
        y = i * scale + scale
        for j in range(width + 1):
            x = j * scale + scale
            cv2.circle(image, (x, y), 5, (0, 0, 0), -1)  # 黑色圆点表示路口

    # 绘制水平道路
    for i in range(height+1):
        y = i * scale + scale
        for j in range(width):
            x1 = j * scale + scale
            x2 = (j + 1) * scale + scale
            cv2.line(image, (x1, y), (x2, y), (0, 0, 0), 2)  # 黑色线表示水平道路

    # 绘制垂直道路
    for i in range(width + 1):
        x = i * scale + scale
        for j in range(height):
            y1 = j * scale + scale
            y2 = (j + 1) * scale + scale
            cv2.line(image, (x, y1), (x, y2), (0, 0, 0), 2)  # 黑色线表示垂直道路

    # 显示或保存生成的路网图像
    #cv2.imwrite('Road Network.jpg', image)

    return image

def init_render(width, height):
    pygame.init()
    pygame.font.init()
    pygame.display.init()
    screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    screen.fill((0, 0, 0))
    pygame.display.flip()
    return screen

def draw_biarcs() -> list:
    arcs = []
    
    # define offsets
    row_offsets = [-1, 1, 0, 0]
    col_offsets = [0, 0, -1, 1]
    
    # find abut nodes
    for i in range(9):
        for j in range(12):
            current_node = i * 12 + j  # current node id
            for k in range(4):
                # cal abut node id
                new_i, new_j = i + row_offsets[k], j + col_offsets[k]
                
                # check abut node legal
                if 0 <= new_i < 9 and 0 <= new_j < 12:
                    neighbor_node = new_i * 12 + new_j  # abut node
                    arcs.append((current_node, neighbor_node))
    return arcs

def bi2single(bi_arcs: list) -> list:
    single_arc = []
    for arc in bi_arcs:
        revised_arc = arc if arc[0]< arc[1] else (arc[1], arc[0])
        if revised_arc not in single_arc:
            single_arc.append(revised_arc)
    return single_arc
