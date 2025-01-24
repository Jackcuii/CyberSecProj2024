import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16
from torchinfo import summary

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# 加载CIFAR100测试数据集
testset = torchvision.datasets.CIFAR100(root='./fuck', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 定义VGG16模型
model = vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 100)  # 修改最后一层以适应CIFAR100

# 加载训练好的模型参数
model.load_state_dict(torch.load('vgg16_cifar100.pth', map_location=torch.device('cpu')))

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# 测试模型
model.eval()
correct = 0
total = 0
times = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # print right or wrong
        times+=1


print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')