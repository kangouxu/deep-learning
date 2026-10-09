import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,19))
X=torch.rand(2,20)
print(net(X))

x = torch.arange(4)
y = torch.zeros(4)
torch.save(x,'x-file')
x2=torch.load('x-file')
print(x2)
torch.save([x,y],'x-files')
x2,y2=torch.load('x-files')
print((x2,y2))