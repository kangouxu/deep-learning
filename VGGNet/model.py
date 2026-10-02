import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

# 第1个vgg block层：
# （1）输入为224×224×3，卷积核数量为64个；卷
# 积核的尺寸大小为3×3×3；步幅为1（stride = 1），
# 填充为1（padding=1 ）；卷积后得到shape为
# 224×224×64的特征图输出。
# （2）输入为224×224×64，卷积核数量为64个；卷
# 积核的尺寸大小为3×3×64；步幅为1（stride = 1），
# 填充为1（padding=1 ）；卷积后得到shape为
# 224×224×64的特征图输出。
# （3）输入为224×224×64，池化核为2×2，步幅为
# 2（stride = 2）后得到尺寸为112×112×64的池化
# 层的特征图输出。
# 第2个vgg block层：
# （1）输入为112×112×64，卷积核数量为128个；
# 卷积核的尺寸大小为3×3×64；步幅为1（stride =
# 1），填充为1（padding=1）；卷积后得到shape
# 为112×112×128的特征图输出。
# （2）输入为112×112×128，卷积核数量为128个；
# 卷积核的尺寸大小为3×3×128；步幅为1（stride =
# 1），填充为1（padding=1）；卷积后得到shape
# 为112×112×128的特征图输出。
# （3）输入为112×112×128，池化核为2×2，步幅
# 为2（stride = 2）后得到尺寸为56×56×128的池化
# 层的特征图输出。
# 第3个vgg block层：
# （1）输入为56×56×128，卷积核数量为256个；卷积核
# 的尺寸大小为3×3×128；步幅为1（stride = 1），填充
# 为1（padding=1）；卷积后得到shape为56×56×256的
# 特征图输出。
# （2）输入为56×56×256，卷积核数量为256个；卷积核
# 的尺寸大小为3×3×256；步幅为1（stride = 1），填充
# 为1（padding=1）；卷积后得到shape为56×56×256的
# 特征图输出。
# （3）输入为56×56×256，卷积核数量为256个；卷积核
# 的尺寸大小为3×3×256；步幅为1（stride = 1），填充
# 为1（padding=1）；卷积后得到shape为56×56×256的
# 特征图输出。
# （4）输入为56×56×256，池化核为2×2，步幅为2
# （stride = 2）后得到尺寸为28×28×256的池化层的特征
# 图输出。

# 第4个vgg block层：
# （1）输入为28×28×256，卷积核数量为512个；卷积核
# 的尺寸大小为3×3×256；步幅为1（stride = 1），填充为
# 1（padding=1）；卷积后得到shape为28×28×512的特
# 征图输出。
# （2）输入为28×28×512，卷积核数量为512个；卷积核
# 的尺寸大小为3×3×512；步幅为1（stride = 1），填充为
# 1（padding=1）；卷积后得到shape为28×28×512的特
# 征图输出。
# （3）输入为28×28×512，卷积核数量为512个；卷积核
# 的尺寸大小为3×3×512；步幅为1（stride = 1），填充为
# 1（padding=1）；卷积后得到shape为28×28×512的特
# 征图输出。
# （4）输入为28×28×512，池化核为2×2，步幅为2
# （stride = 2）后得到尺寸为14×14×512的池化层的特征
# 图输出。

# 第5个vgg block层：
# （1）输入为14×14×512，卷积核数量为512个；卷积核的尺寸大
# 小为3×3×512；步幅为1（stride = 1），填充为1（padding=1）；
# 卷积后得到shape为14×14×512的特征图输出。
# （2）输入为14×14×512，卷积核数量为512个；卷积核的尺寸大
# 小为3×3×512；步幅为1（stride = 1），填充为1（padding=1）；
# 卷积后得到shape为14×14×512的特征图输出。
# （3）输入为14×14×512，卷积核数量为512个；卷积核的尺寸大
# 小为3×3×512；步幅为1（stride = 1），填充为1（padding=1）；
# 卷积后得到shape为14×14×512的特征图输出。
# （4）输入为14×14×512，池化核为2×2，步幅为2（stride = 2）
# 后得到尺寸为7×7×512的池化层的特征图输出。该层后面还隐藏
# 了flatten操作，通过展平得到7×7×512=25088个参数后与之后的
# 全连接层相连。
class VGGNet16(nn.Module):
    def __init__(self):
        super(VGGNet16,self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) 
        )#112×112×64
        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) 
        )#56×56×128
        self.block3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) 
        )#28×28×256
        self.block4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) 
        )#14×14×512
        self.block5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) 
        )#7×7×512
        self.block6=nn.Sequential(
            nn.Flatten(),#7*7*512=25088
            nn.Linear(25088,4096),
            nn.ReLU(),
            nn.Dropout(0.5),    
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),   
            nn.Linear(4096,1000)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        


    def forward(self,x):#x是输入图像  前向传播
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.block6(x)
        return x

if __name__ =="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGGNet16().to(device) 
    print(summary(model,(1,224,224)))


    
