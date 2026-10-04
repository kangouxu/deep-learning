import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import ResNet



def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    return test_dataloader

def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            #特征
            test_data_x = test_data_x.to(device)
            #标签
            test_data_y = test_data_y.to(device)
            #评估模式
            model.eval()
            #前向传播过程
            output= model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)




if __name__=="__main__":
    # 加载模型
    model = ResNet()
    model.load_state_dict(torch.load('databest_model.pth'))
    # 加载测试数据
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    test_model_process(model, test_dataloader)
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    # device = "cuda" if torch.cuda.is_available() else 'cpu'
    # model = model.to(device)

    # classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)

    #         # 设置模型为验证模型
    #         model.eval()
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output, dim=1)
    #         result = pre_lab.item()
    #         label = b_y.item()
    #         print("预测值：",  classes[result], "------", "真实值：", classes[label])
