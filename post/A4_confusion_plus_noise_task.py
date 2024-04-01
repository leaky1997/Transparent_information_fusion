import sys
sys.path.append('./')

from post.A1_plot_config import configure_matplotlib
configure_matplotlib(style='ieee', font_lang='en')

from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

import pandas as pd
###################### heatmap_confusion ######################


def heatmap_confusion(matrix, plot_dir='./', name='confusion_matrix', cmap='coolwarm', save_type='.pdf'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap=cmap, annot=True, fmt="d", linewidths=.5,annot_kws={"size": 16})
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{name}{save_type}'), transparent=True, dpi=512)
    plt.close()

def calculate_confusion_matrix(predictions, test_labels, args,):
    pred_labels = predictions.argmax(dim=1) if len(predictions.shape) > 1 else predictions
    true_labels = test_labels.argmax(dim=1) if len(test_labels.shape) > 1 else test_labels

# 计算混淆矩阵
    conf_mat = ConfusionMatrix(task="multiclass",num_classes=args.num_classes)
    matrix = conf_mat(pred_labels, true_labels)
    return matrix


###################### noise ######################

def add_noise(data, snr):
    """
    向数据添加高斯噪声。
    """
    snr = 10 ** (snr / 10.0)
    x_power = torch.sum(data ** 2) / data.numel()
    noise_power = x_power / snr
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise

def plot_accuracy_vs_snr(test_data, test_labels, model, snr_levels, plot_dir='./', file_name='model'):
    accuracies = []
    for snr in snr_levels:
        noisy_data = add_noise(test_data, snr)
        with torch.no_grad():
            preds = model(noisy_data).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()
        accuracies.append(acc)
    pd.DataFrame(accuracies, columns=['Accuracy']).to_csv(os.path.join(plot_dir, f'{file_name}_accuracy_list.csv'), index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, accuracies, marker='o', linestyle='-')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. SNR')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, file_name) + '.pdf', dpi=300)
    plt.close()

# # 设置噪声级别范围
# snr_levels = np.arange(-5, 16, 1)  # 示例：从-5dB到15dB

# # 进行噪声实验并绘制准确率与SNR的关系图
# plot_accuracy_vs_snr(test_data, true_labels, model, snr_levels, plot_dir='你的图表保存目录', file_name='accuracy_vs_snr')


# 绘制混淆矩阵
    



