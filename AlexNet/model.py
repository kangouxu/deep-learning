import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
#AlexNet
# 第1层输入层： Input为227×227×3
# 第2层卷积层： Input为227×227×3，卷积核为11×11×3× 96 ；stride=4，
# output为55×55×96
# 第3层最大池化层： Input为55×55×96，池化感受野为3×3，stride=2，
# output为27×27×96
# 第4层卷积层： Input为27×27×96，卷积核为5×5×96× 256 ；stride=1，padding=2 ，
# output为27×27×256
# 第5层最大池化层： Input为27×27×256，池化感受野为3 ×3，stride=2，
# output为13×13×256
# 第6层卷积层： Input为13×13×256，卷积核为3×3×256 × 384，stride=1，padding=1，
# output 为13×13×384
# 第7层卷积层： Input为13×13×384，卷积核为3×3×384 × 384，stride=1，padding=1，
# output 为13×13×384。
# 第8层卷积层： Input 13×13×384，卷积核为3×3×384 × 256，stride=1，padding=1，
# output 为13×13×256
# 第9层最大池化层： Input为13×13×256，池化感受野为3 ×3，stride=2，
# output 为6×6×256，Flatten操作，通过展平得到9216个数据后与之后的全连接层相连。
# 第10～12层全连接层：第10~12层神经元个数分别为4096，4096，1000。其中前两层在
# 使用relu后还使用了Dropout对神经元随机失活，最后一层全连接层用softmax输出1000
# 个分类（这里要说的是，softmax输出1000个分类原因是当时该网络设计是为了参加
# ImageNet这个比赛，该比赛最后是对1000个物体进行分类）

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.c2=nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=1)
        self.RELU=nn.ReLU()#激活
        self.s3=nn.MaxPool2d(kernel_size=3,stride=2)#池化  不改变输入通道数
        self.c4=nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2)
        self.s5=nn.MaxPool2d(kernel_size=3,stride=2)
        self.c6=nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.c7=nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.c8=nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.s9=nn.MaxPool2d(kernel_size=3,stride=2)
        self.flatten=nn.Flatten()#展平 

        self.f10=nn.Linear(in_features=9216,out_features=4096)#全连接层
        self.f11=nn.Linear(in_features=4096,out_features=4096)
        self.f12=nn.Linear(in_features=4096,out_features=10)#数据集总共10个类别

    def forward(self,x):#x是输入图像  前向传播
        x=self.RELU(self.c2(x))#
        x=self.s3(x)
        x=self.RELU(self.c4(x))
        x=self.s5(x)
        x=self.RELU(self.c6(x))
        x=self.RELU(self.c7(x))
        x=self.RELU(self.c8(x))
        x=self.s9(x)
        x = self.flatten(x)
        x=self.RELU(self.f10(x))
        x=F.dropout(x,0.5)
        x=self.RELU(self.f11(x))
        x=F.dropout(x,0.5)
        x=self.f12(x)
        return x

if __name__ =="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet().to(device) 
    print(summary(model,(1,227,227)))


    
