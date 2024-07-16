
from post.A1_plot_config import configure_matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
# 获取Topk个索引

def get_top_k_weights(idx, layer,k = 10,weight_flag = 'weight_connection'):
    print('layer:',idx)
    if weight_flag == 'weight_connection':
        weight = layer.weight_connection.weight # .detach().cpu().numpy()
    elif weight_flag == 'skip_connection':
        layer.skip_connection.weight.data = F.softmax((1.0 / 0.2) *  # 0.09 / 0.2
                                                    layer.skip_connection.weight.data, dim=0)
        weight = layer.skip_connection.weight
    elif weight_flag == 'clf':
        weight = layer.weight
        
    # 计算权重的绝对值
    abs_weight = torch.abs(weight)

    # 将权重矩阵扁平化
    flat_abs_weights = abs_weight.flatten()

    # 使用 topk 方法找到扁平化权重中最大的 k 个元素及其索引
    values, flat_indices = flat_abs_weights.topk(k, largest=True)

    # 如果需要，将扁平化后的索引转换回原始矩阵的二维索引
    row_indices, col_indices = np.unravel_index(flat_indices.cpu().numpy(), abs_weight.shape)
    result_list = []

    # 在循环中将每个元素的信息添加到列表中
    for i in range(k):
        result_list.append((row_indices[i], col_indices[i], values[i].item()))
    return result_list


import networkx as nx

def signal_weight(model,k=20,weight_flag = 'weight_connection'):
    layer_top_weight = {}
    for idx, layer in enumerate(model.signal_processing_layers):
        result_list = get_top_k_weights(idx, layer,k=k,weight_flag = weight_flag)
        layer_top_weight[f'layer:{idx}'] = result_list
        print(result_list)
    return layer_top_weight
        
def draw_network_structure(layer_top_weight,precision = 6,
                           save_path='./plot',
                           filter_flag = False,
                           name = 'network_structure2'):
    """
    绘制网络结构图。
    
    参数:
    - layer_top_weight: 包含网络层和权重数据的字典。
    - save_path: 图片保存路径，不包含文件后缀。
    """
    G = nx.DiGraph()

    # 添加边
    for idx, (layer, tuples) in enumerate(layer_top_weight.items()):
        for target_idx, source_idx, weight in tuples:
            if filter_flag:
                if weight < 1e-4:  # 裁剪掉权重小于1e-4的边
                    continue                
            source_node = f'$s^{idx}_{{{source_idx}}}$'
            target_node = f'$s^{idx+1}_{{{target_idx}}}$'
            weight_ = round(weight, precision)
            G.add_edge(source_node, target_node, weight=weight_)
            
    if filter_flag:
        G = remove_edges_below_threshold(G)
        G = remove_unconnected_nodes(G)
        
    # 为了美化图形，为每一层的节点分配垂直位置
    pos = {}
    layer_levels = {}
    for node in sorted(G.nodes): # TODO number order
        layer = parse_layer(node)
        if layer not in layer_levels:
            layer_levels[layer] = 0
        pos[node] = (layer_levels[layer], -layer)
        layer_levels[layer] += 1


###############################################


    # 绘图
    # plt.figure(figsize=(12, 8))
    # nx.draw(G, pos, with_labels=True, node_size=2600, node_color="cornflowerblue", font_size=40, font_weight="bold", 
    #         edge_color="gray", width=2, arrowstyle="->", arrowsize=10)
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='yellowgreen', font_size=15)

# # 绘图 改颜色
#     plt.figure(figsize=(12, 8))
#     edges = G.edges(data=True)
#     weights = [d['weight'] for (u, v, d) in edges]
#     edge_colors = [plt.cm.Blues(weight) for weight in weights]

#     nx.draw(G, pos, with_labels=True, node_size=2600, node_color="cornflowerblue", 
#             font_size=40, font_weight="bold", edge_color=edge_colors, width=2, 
#             arrowstyle="->", arrowsize=10, alpha=0.7)
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='yellowgreen', font_size=15)

## 改透明度
    # 绘图
    
    plt.figure(figsize=(12, 8))
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for _, _, d in edges]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 0.000001
    edge_alphas = [(weight - min_weight) / (max_weight - min_weight) for weight in edge_weights]

    edge_colors = [(0, 0, 0, alpha) for alpha in edge_alphas]

    nx.draw(G, pos, with_labels=True, node_size=2600, node_color="cornflowerblue", font_size=40, font_weight="bold", 
            edge_color=edge_colors, width=2, arrowstyle="->", arrowsize=10)
    
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='yellowgreen', font_size=15)
    

    plt.axis('off')  # 隐藏坐标轴
    plt.savefig(f'{save_path}/filter_flag_{filter_flag}{name}.png')
    plt.savefig(f'{save_path}/filter_flag_{filter_flag}{name}.svg')
    plt.show()
    
def parse_layer(node):
    node_plain = node.replace('$', '').replace('{', '').replace('}', '')   
    layer, idx = node_plain.split('^')[1].split('_')
    layer = int(layer)
    idx = int(idx)
    return layer

def remove_edges_below_threshold(G, threshold=1e-4):
    """移除权重低于阈值的边。"""
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
    G.remove_edges_from(edges_to_remove)
    return G

def remove_unconnected_nodes(G):
    """移除没有入度或出度的节点，对于第一层和最后一层节点有特殊处理。"""
    removed = True
    while removed:
        removed = False
        nodes_to_remove = [node for node in G.nodes if G.in_degree(node) == 0 and G.out_degree(node) == 0]
        
        if nodes_to_remove:
            G.remove_nodes_from(nodes_to_remove)
            removed = True

        # 对于中间层节点，检查并移除那些没有入度或出度的节点
        max_layer_idx = max([parse_layer(node) for node in G.nodes])
        for node in list(G.nodes):
            layer = parse_layer(node)
            if (G.in_degree(node) == 0 and layer != 0) or (G.out_degree(node) == 0 and layer != max_layer_idx):
                G.remove_node(node)
                removed = True
    return G    


import matplotlib.pyplot as plt
import seaborn as sns
import torch





def signal_vis_weight(model, path='./plot', name='', k=20, weight_flag='weight_connection'):
    """
    绘制模型信号处理层的权重热力图。
    
    参数:
    - model: 包含信号处理层的模型。
    - path: 图片保存路径。
    - name: 图片保存的基础名字。
    - k: 可视化的层的数量（如果模型层数超过此数，则只显示前k层）。
    - weight_flag: 权重属性的名称，默认为'weight_connection'。
    """
    for idx, layer in enumerate(model):
        if idx >= k:
            break  # 只处理前k层
        
        weight_tensor = getattr(layer, weight_flag).weight.abs()  # 获取权重张量
        weight_data = weight_tensor.detach().cpu().numpy()  # 转为NumPy数组
        
        if len(weight_data.shape) > 2:
            weight_data = weight_data.squeeze()  # 如果权重是多维的，尝试去掉单维度
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))  # 可以根据需要调整图形大小
        sns.heatmap(weight_data, cmap='BuPu', annot=False, fmt="f",
                    linewidths=.5, xticklabels='', yticklabels='')  # 设置色彩映射为PuBu
        plt.title(f'Layer {idx} Weight Heatmap')  # 标题可以包含层的索引
        plt.savefig(path + f'/{weight_flag}{name}_layer{idx}.pdf', dpi=256)  # 保存为PDF
        plt.savefig(path + f'/{weight_flag}{name}_layer{idx}.svg', dpi=256) 
        plt.show()  # 显示图形


def parse_attention(noisy_data,model):
    
    model = model.cuda()
    model(torch.tensor(noisy_data).float().cuda())

    model = model.network
    SP_attentions = []
    for layer in model.signal_processing_layers:
        SP_attentions.append(layer.channel_attention.gate.squeeze().detach().cpu().numpy())
    FE_attention = model.feature_extractor_layers.FEAttention.gate.squeeze().detach().cpu().numpy()
    return SP_attentions,FE_attention

def visualize_Attention(sparse_matrix, labels,channel_groups, path='./plot', name=''): 
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为新罗马

# 设置绘图风格
    sns.set_theme(style="whitegrid",font='Times New Roman',font_scale=1.4)
    sns.set_palette("Set2")
# sns.set(style="whitegrid", context="talk")

# 创建一个大的绘图区域，包含多个子图
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 第一个子图：热力图
    sns.heatmap(sparse_matrix, ax=axs[0, 0], cmap="GnBu", cbar_kws={'label': 'Weight'}, alpha=0.8) # jet cmap="jet",
    axs[0, 0].set_title('Heatmap of Sparse Matrix')
    axs[0, 0].set_xlabel('Channel')
    axs[0, 0].set_ylabel('Sample')

# 第二个子图：矩阵中的非零元素位置
    axs[0, 1].spy(sparse_matrix, markersize=5, color='r')
    axs[0, 1].set_title('Selected node')
    axs[0, 1].grid(True)
    axs[0, 1].set_xlabel('Channel')
    axs[0, 1].set_ylabel('Sample')

# 第三个子图：根据类别绘制累积的样本权重分布
    unique_categories = np.unique(labels)
    for category in unique_categories:
        category_mask = (labels == category)
        accumulated_weights = sparse_matrix[category_mask].sum(axis=0)
    # axs[1, 0].plot(accumulated_weights, marker='o', label=f'Category {category}')
        x_positions = range(len(accumulated_weights))  # 生成x轴位置
        axs[1, 0].bar(x_positions, accumulated_weights, label=f'Category {category}', width=0.8,alpha=0.5)  # 使用bar绘制柱状图
    axs[1, 0].set_title('Accumulated attention by sample category')
    axs[1, 0].set_xlabel('Channel')
    axs[1, 0].set_ylabel('Accumulated Weight')
    axs[1, 0].grid(True)
    axs[1, 0].set_xticks(x_positions)
    axs[1, 0].legend(loc='upper right', fontsize='small')

# 第四个子图：根据通道类别绘制累积的权重分布
    for group_idx, group in enumerate(channel_groups):
        accumulated_weights = sparse_matrix[:, group].sum(axis=1)
        axs[1, 1].plot(accumulated_weights, marker='x', label=f'Channel Group {group_idx+1}')
    axs[1, 1].set_title('Accumulated attention by channel group')
    axs[1, 1].set_xlabel('Sample')
    axs[1, 1].set_ylabel('Accumulated Weight')
    axs[1, 1].grid(True)
    axs[1, 1].legend(loc='upper right', fontsize='small')

# 调整布局，使子图不重叠
    plt.tight_layout()

# 显示图形
    plt.savefig(path + f'/Attention{name}.pdf', dpi=256)  # 保存为PDF
    plt.savefig(path + f'/Attention{name}.svg', dpi=256)
    plt.show()