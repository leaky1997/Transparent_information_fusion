import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 创建一个空的有向图
G = nx.DiGraph()

# 添加节点和边
G.add_nodes_from(['A', 'B', 'C', 'D'])
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])

# 初始化起点和终点位置
start_pos = {'A': (0, 0)}
end_pos = {'D': (10, 0)}

# 绘制图形的初始状态
pos = nx.spring_layout(G, pos=start_pos, fixed=start_pos.keys())
nx.draw(G, pos, with_labels=True)

# 定义更新函数，每一帧更新节点位置
def update(frame):
    if frame > 0:
        # 更新节点位置，使包从起点移动到终点
        alpha = min(frame / 100, 1)  # 控制包的移动速度
        for node in G.nodes():
            pos[node] = tuple((1 - alpha) * start_pos[node][i] + alpha * end_pos[node][i] for i in range(2))

    # 清除之前的绘图，并重新绘制更新后的图形
    ax.clear()
    nx.draw(G, pos, with_labels=True)

# 创建动画对象
ani = FuncAnimation(fig, update, frames=100, interval=100)

# 显示动画
plt.show()