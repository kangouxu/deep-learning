import torch
from torch import nn
from torchsummary import summary


#LeNet 5
# 第1层输入层： Input为28×28×1
# 第2层卷积层： Input为28×28×1 ，卷积核5×5 ×1× 6；stride = 1，
# padding=2。 output为28×28×6
# 第3层平均池化层： Input为28×28×6，池化感受野为2×2，stride = 2，
# output为14×14×6
# 第4层卷积层： Input为14×14×6，卷积核5×5×6 ×16，stride = 1，
# padding=0, output为10×10×16
# 第5层平均池化层： Input为10×10×16，池化感受野为2×2 ，stride = 2 ,
# output 为5×5×16，Flatten操作，通过展平得到400个数据与之后的全连
# 接层相连。
# 第6～8层全连接层： 第6~8层神经元个数分别为120，84，10。其中神
# 经网络中用sigmoid作为激活函数，最后一层全连接层用softmax输出10
# 个分类。

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2)#卷积1
        self.sig=nn.Sigmoid()#激活
        self.s2=nn.AvgPool2d(kernel_size=2,stride=2)#池化  不改变输入通道数
        self.c3=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0)
        self.s4=nn.AvgPool2d(kernel_size=2,stride=2)
        self.flatten=nn.Flatten()#展平 

        self.f5=nn.Linear(in_features=400,out_features=120)#全连接层
        self.f6=nn.Linear(in_features=120,out_features=84)
        self.f7=nn.Linear(in_features=84,out_features=10)
    def forward(self,x):#x是输入图像  前向传播
        x=self.sig(self.c1(x))#
        x=self.s2(x)
        x=self.sig(self.c3(x))
        x=self.s4(x)
        x = self.flatten(x)
        x=self.sig(self.f5(x))
        x=self.sig(self.f6(x))
        x=self.f7(x)
        return x

if __name__ =="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeNet().to(device) 
    print(summary(model,(1,28,28)))


    
