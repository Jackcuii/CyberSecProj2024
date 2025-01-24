import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prune_and_evaluate(prune_rate):
    # 定义VGG16模型
    print(f"Prune {prune_rate * 100}% parameters:")
    model = vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 100)  # 修改最后一层以适应CIFAR100

    # 加载训练好的模型参数
    model.load_state_dict(torch.load('vgg16_cifar100.pth', map_location=torch.device('cpu')))

    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 获取全连接层的第一层参数矩阵
    fc1_weights = model.classifier[0].weight.data.cpu().numpy()
    #print(fc1_weights)  # 应该输出 (4096, 25088)

    # 将该矩阵分为16x98个256x256的方阵
    blocks = fc1_weights.reshape(16, 256, 98, 256).swapaxes(1, 2).reshape(16, 98, 256, 256)

    # 生成掩码并进行剪枝
    mask = np.random.rand(16, 98) < prune_rate
    #print(mask)
    blocks[mask] = 0

    # 将修改后的参数矩阵重新赋值给模型
    fc1_weights = blocks.reshape(16, 98, 256, 256).swapaxes(1, 2).reshape(16, 256, 98, 256).reshape(4096, 25088)
    #print(fc1_weights)
    model.classifier[0].weight.data = torch.from_numpy(fc1_weights).to(device)

    # 测试模型
    model.eval()

    correct = 0
    total = 0

    # 加载CIFAR100测试数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    testset = datasets.CIFAR100(root='./fuck', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print("Accuracy: ", accuracy)
    return accuracy



def rand_and_evaluate(rate):
    # 定义VGG16模型
    model = vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 100)  # 修改最后一层以适应CIFAR100

    # 加载训练好的模型参数
    model.load_state_dict(torch.load('vgg16_cifar200.pth', map_location=torch.device('cpu')))

    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 获取全连接层的第一层参数矩阵
    fc1_weights = model.classifier[0].weight.data.cpu().numpy()

    # 将该矩阵分为16x98个256x256的方阵
    blocks = fc1_weights.reshape(16, 256, 98, 256).swapaxes(1, 2).reshape(16, 98, 256, 256)

    # 生成掩码并进行随机化
    mask = np.random.rand(16, 98) < rate
    #print(mask)
    random_values = np.random.uniform(low=-0.1, high=0.1, size=blocks[mask].shape)
    blocks[mask] = random_values

    # 将修改后的参数矩阵重新赋值给模型
    fc1_weights = blocks.reshape(16, 98, 256, 256).swapaxes(1, 2).reshape(16, 256, 98, 256).reshape(4096, 25088)
    model.classifier[0].weight.data = torch.from_numpy(fc1_weights).to(device)

    # 测试模型
    model.eval()

    correct = 0
    total = 0

    # 加载CIFAR100测试数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    testset = datasets.CIFAR100(root='./fuck', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(accuracy)
    return accuracy


import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def exchange_and_evaluate(exchange_pairs_sum):
    # 定义VGG16模型
    model = vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 100)  # 修改最后一层以适应CIFAR100

    # 加载训练好的模型参数
    model.load_state_dict(torch.load('vgg16_cifar100.pth', map_location=torch.device('cpu')))

    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 获取全连接层的第一层参数矩阵
    fc1_weights = model.classifier[0].weight.data.cpu().numpy()

    # 将该矩阵分为16x98个256x256的方阵
    blocks = fc1_weights.reshape(16, 256, 98, 256).swapaxes(1, 2).reshape(16, 98, 256, 256)

    # 随机取出指定数量的块对并互换
    for _ in range(exchange_pairs_sum):
        idx1 = (np.random.randint(16), np.random.randint(98))
        idx2 = (np.random.randint(16), np.random.randint(98))
        blocks[idx1], blocks[idx2] = blocks[idx2].copy(), blocks[idx1].copy()

    # 将修改后的参数矩阵重新赋值给模型
    fc1_weights = blocks.reshape(16, 98, 256, 256).swapaxes(1, 2).reshape(16, 256, 98, 256).reshape(4096, 25088)
    model.classifier[0].weight.data = torch.from_numpy(fc1_weights).to(device)

    # 测试模型
    model.eval()

    correct = 0
    total = 0

    # 加载CIFAR100测试数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    testset = datasets.CIFAR100(root='./fuck', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(accuracy)
    return accuracy



def prune_and_exchange(prune_rate, exchange_pairs_sum):
    # 定义VGG16模型
    model = vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 100)  # 修改最后一层以适应CIFAR100

    # 加载训练好的模型参数
    model.load_state_dict(torch.load('vgg16_cifar100.pth', map_location=torch.device('cpu')))

    # 使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 获取全连接层的第一层参数矩阵
    fc1_weights = model.classifier[0].weight.data.cpu().numpy()

    # 将该矩阵分为16x98个256x256的方阵
    blocks = fc1_weights.reshape(16, 256, 98, 256).swapaxes(1, 2).reshape(16, 98, 256, 256)

    # 生成掩码并进行剪枝
    mask = np.random.rand(16, 98) < prune_rate
    blocks[mask] = 0

    # 获取未被删除的块的索引
    non_zero_indices = np.argwhere(~mask)

    # 随机取出指定数量的块对并互换
    for _ in range(exchange_pairs_sum):
        idx1 = tuple(non_zero_indices[np.random.randint(len(non_zero_indices))])
        idx2 = (np.random.randint(16), np.random.randint(98))
        while mask[idx2]:
            idx2 = (np.random.randint(16), np.random.randint(98))
        blocks[idx1], blocks[idx2] = blocks[idx2].copy(), blocks[idx1].copy()

    # 将修改后的参数矩阵重新赋值给模型
    fc1_weights = blocks.reshape(16, 98, 256, 256).swapaxes(1, 2).reshape(16, 256, 98, 256).reshape(4096, 25088)
    model.classifier[0].weight.data = torch.from_numpy(fc1_weights).to(device)

    # 测试模型
    model.eval()

    correct = 0
    total = 0

    # 加载CIFAR100测试数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    testset = datasets.CIFAR100(root='./fuck', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(accuracy)
    return accuracy



import os
import json

def main():
    results_file = 'prune_results.json'
    results = {}

    # 如果结果文件存在，读取已有结果
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)

    # 定义剪枝率范围
    prune_rates = list(np.arange(0.50, 0.89, 0.02)) + list(np.arange(0.90, 1.001, 0.001))
    print(prune_rates)
    
    for prune_rate in prune_rates:
        prune_rate_str = f'{prune_rate:.3f}'
        if prune_rate_str not in results:
            results[prune_rate_str] = []

        # 如果已经有3个结果，跳过
        if len(results[prune_rate_str]) >= 3:
            continue

        # 计算3次并保存结果
        for _ in range(3 - len(results[prune_rate_str])):
            accuracy = prune_and_evaluate(prune_rate)
            results[prune_rate_str].append(accuracy)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)

    # 计算平均值并保存到文件
    with open(results_file, 'w') as f:
        for prune_rate_str in results:
            avg_accuracy = sum(results[prune_rate_str]) / len(results[prune_rate_str])
            results[prune_rate_str].append(avg_accuracy)
        json.dump(results, f, indent=4)


import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def plot_prune_results_with_error_bars(results_file):
    # 读取结果文件
    with open(results_file, 'r') as f:
        results = json.load(f)

    prune_rates = []
    avg_accuracies = []
    std_devs = []

    # 提取剪枝率、平均准确率和标准差
    for prune_rate_str, accuracies in results.items():
        prune_rate = float(prune_rate_str)
        accuracies = accuracies[:-1]  # 去掉最后一个平均值
        avg_accuracy = np.mean(accuracies)
        std_dev = np.std(accuracies)
        prune_rates.append(prune_rate)
        avg_accuracies.append(avg_accuracy)
        std_devs.append(std_dev)

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.errorbar(prune_rates, avg_accuracies, yerr=std_devs, fmt='o', ecolor='r', capsize=5, linestyle='-', color='b')
    plt.xlabel('Prune Rate')
    plt.ylabel('Average Accuracy')
    plt.title('Prune Rate vs. Average Accuracy with Error Bars')
    plt.grid(True)
    plt.show()


def main2():
    results_file = 'prune_and_exchange_results.json'
    results = {}

    # 如果结果文件存在，读取已有结果
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)

    # 定义剪枝率和交换对数范围
    prune_rates = np.arange(0.85, 1.00, 0.01)
    exchange_pairs_sums = np.arange(0, 376, 25)

    for prune_rate in prune_rates:
        for exchange_pairs_sum in exchange_pairs_sums:
            key = f'{prune_rate:.2f}_{exchange_pairs_sum}'
            if key not in results:
                results[key] = {
                    'prune_rate': float(prune_rate),
                    'exchange_pairs_sum': int(exchange_pairs_sum),
                    'accuracies': []
                }

            # 如果已经有3个结果，跳过
            if len(results[key]['accuracies']) >= 3:
                continue

            # 计算3次并保存结果
            for _ in range(3 - len(results[key]['accuracies'])):
                accuracy = prune_and_exchange(prune_rate, exchange_pairs_sum)
                results[key]['accuracies'].append(accuracy)
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)

    # 计算平均值并保存到文件
    with open(results_file, 'w') as f:
        for key in results:
            accuracies = results[key]['accuracies']
            if len(accuracies) == 3:
                avg_accuracy = sum(accuracies) / len(accuracies)
                results[key]['average_accuracy'] = avg_accuracy
        json.dump(results, f, indent=4)
# 调用函数绘制图表
#plot_prune_results_with_error_bars('prune_results.json')


#prune_and_exchange(0.9, 375)

#if __name__ == '__main__':
    #main2()
    
def plot_prune_exchange_results(results_file):
    # 读取结果文件
    with open(results_file, 'r') as f:
        results = json.load(f)

    prune_rates = []
    exchange_pairs_sums = []
    avg_accuracies = []

    # 提取剪枝率、交换对数和平均准确率
    for key, value in results.items():
        prune_rate = value['prune_rate']
        exchange_pairs_sum = value['exchange_pairs_sum']
        accuracies = value['accuracies']
        if len(accuracies) == 3:
            avg_accuracy = sum(accuracies) / len(accuracies)
            prune_rates.append(prune_rate)
            exchange_pairs_sums.append(exchange_pairs_sum)
            avg_accuracies.append(avg_accuracy)

    if not prune_rates or not exchange_pairs_sums or not avg_accuracies:
        print("No data available to plot.")
        return

    prune_rates = np.array(prune_rates)
    exchange_pairs_sums = np.array(exchange_pairs_sums)
    avg_accuracies = np.array(avg_accuracies)

    # 创建网格数据
    grid_x, grid_y = np.meshgrid(np.linspace(prune_rates.min(), prune_rates.max(), 100),
                                 np.linspace(exchange_pairs_sums.min(), exchange_pairs_sums.max(), 100))
    grid_z = griddata((prune_rates, exchange_pairs_sums), avg_accuracies, (grid_x, grid_y), method='cubic')

    # 绘制三维图像
    fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, color='r', alpha=0.6)
    ax.scatter(prune_rates, exchange_pairs_sums, avg_accuracies, color='b')
    ax.set_xlabel('Prune Rate')
    ax.set_ylabel('Exchange Pairs Sum')
    ax.set_zlabel('Average Accuracy')
    ax.set_title('3D Surface Plot')

    # 绘制等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.75)
    contour_lines = ax2.contour(grid_x, grid_y, grid_z, colors='black')
    ax2.clabel(contour_lines, inline=True, fontsize=8)
    ax2.scatter(prune_rates, exchange_pairs_sums, color='red')
    ax2.set_xlabel('Prune Rate')
    ax2.set_ylabel('Exchange Pairs Sum')
    ax2.set_title('Contour Plot')

    plt.colorbar(contour, ax=ax2, label='Average Accuracy')
    plt.tight_layout()
    plt.show()


# 调用函数绘制图表
plot_prune_exchange_results('prune_and_exchange_results.json')





