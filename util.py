import cv2
import numpy as np
import pygame
from PIL import Image

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

# draw bi arcs
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

# 2wayto1way
def bi2single(bi_arcs: list) -> list:
    single_arc = []
    for arc in bi_arcs:
        revised_arc = arc if arc[0]< arc[1] else (arc[1], arc[0])
        if revised_arc not in single_arc:
            single_arc.append(revised_arc)
    return single_arc

def figure_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image