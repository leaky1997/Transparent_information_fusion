from post.A1_plot_config import configure_matplotlib
import matplotlib.pyplot as plt
import numpy as np
configure_matplotlib(style='ieee', font_lang='en')
import matplotlib.pyplot as plt
import numpy as np
import os


def data_to_tsne_dict():
# 对数据进行 t-SNE 降维
tsne = TSNE(n_components=2, random_state=0,perplexity=10)
tsne_feature = tsne.fit_transform(feature.cpu().detach().numpy())

    pass

def dict_to_figs(tsne_dict, labels, num_classes, plot_dir='./plot'):
    # 确保输出目录存在
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # 计算子图的行数和列数
    num_models = len(tsne_dict)
    rows = int(np.sqrt(num_models))
    cols = np.ceil(num_models / rows).astype(int)

    # 创建大图和子图
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    # 如果只有一个模型，axs不是数组，我们把它变成数组以简化处理
    if num_models == 1:
        axs = np.array([axs])
    
    # 遍历降维后的特征字典并绘制散点图
    for ax, (model_name, features) in zip(axs.ravel(), tsne_dict.items()):
        plot_scatter(features, labels, ax, num_classes, name=model_name)

    # 隐藏空白子图
    for i in range(num_models, rows * cols):
        fig.delaxes(axs.flatten()[i])

    # 调整布局和保存图像
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/tsne_all_models.png')
    plt.close()



def plot_scatter(features, labels, num_per_class,name = 'default'):
    # 定义标记、颜色和线型
    markers = ['*', 'o', 'x', 'v', '^', 's']
    colors = ['darkviolet','blue','r','green', 'y', 'g', 'c', 'b', 'm']
    lines = ['dotted', '-', '--', ':', '-.']
    # # 遍历每个类别
    # for fault in range(n_classes):
    #     # 根据类别筛选出对应的特征点
    #     class_indices = np.where(labels == fault)[0]
    #     class_features = features[class_indices, :]

    #     # 绘制散点图
    #     plt.scatter(class_features[:, 0], class_features[:, 1], 
    #                 label=f'Class {fault+1}', 
    #                 marker=markers[fault % len(markers)], 
    #                 c=colors[fault % len(colors)], 
    #                 linestyle=lines[fault % len(lines)])
    # 假设有 n 个类别
    n = len(np.unique(labels))
    plt.figure(figsize=(10, 6))
    for fault in range(n):
        # 选择当前类别的数据点
        id_fault = labels == fault
        x = features[id_fault, 0]
        y = features[id_fault, 1]
        
        # 绘制散点图
        plt.scatter(x, y, marker=markers[fault % len(markers)],
                    color=colors[fault % len(colors)],
                    # linestyle=lines[fault % len(lines)],
                    label=f'Class {fault}')
    # 设置图例位置在右上角
    plt.legend(loc='best')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加虚线网格
    plt.xticks([])  # 取消横坐标
    plt.yticks([])  # 取消纵坐标
    plt.savefig(f'plot/{name}.pdf')
    plt.show()