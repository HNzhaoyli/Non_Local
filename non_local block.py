
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

class non_local_block(nn.Module):
    def __init__(self,in_channels,channel_scale_factor,avg_kernel_size, down_sample=True):
        super(non_local_block, self).__init__()
        self.theta = nn.Conv2d(in_channels,in_channels//channel_scale_factor,1,stride=1,padding=0)
        self.phi = nn.Conv2d(in_channels,in_channels//channel_scale_factor,1,stride=1,padding=0)
        self.gi = nn.Conv2d(in_channels,in_channels//channel_scale_factor,1,stride=1,padding=0)
        self.Wz = nn.Conv2d(in_channels//channel_scale_factor,in_channels,1,stride=1,padding=0)
        self.down_sample = down_sample
        self.spatial_scale_factor = avg_kernel_size
        self.avgpool = nn.MaxPool2d(kernel_size=avg_kernel_size,stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        ox = x
        theata = self.theta(x)
        if self.down_sample:
            x = self.avgpool(x)
        phi = self.phi(x)
        gi = self.gi(x)

        theata1 = theata.view(theata.size(0),theata.size(1),-1)
        print('theata',theata1.size())
        pi1 = phi.view(phi.size(0),phi.size(1),-1)
        print('pi1',pi1.size())
        theata1 = theata1.transpose(1,2)
        gi1 = gi.view(gi.size(0),gi.size(1),-1)
        print(gi1.size())
        gi1=gi1.transpose(1,2)

        m1 = torch.matmul(theata1,pi1)
        print('m1',m1.size())
        m1 = torch.softmax(m1,dim=2)
        m2 = torch.matmul(m1,gi1)

        m3 = m2.view(m2.size(0),m2.size(2),ox.size(2),-1)

        m4 = self.Wz(m3)



        z = m4 + ox

        return z


if __name__ == '__main__':
    inputs = Variable(torch.randn(16,1024,32,32))
    non_local = non_local_block(1024,2,2)
    outputs = non_local(inputs)
    print(outputs.shape)





