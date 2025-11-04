import torch
from torch import nn 
from torch.nn import Module
class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()
 
        self.h = h
        self.w = w
 
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
 
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
 
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
 
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
 
        return out
class VGG_Block(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,padding,stride,drop_out=0.3):
        super().__init__()
        if padding !=0 :
            padding_mode = 'reflect'
            self.block = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size,padding = padding,padding_mode = padding_mode,stride = stride,bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Dropout(drop_out),
        )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel),
                nn.Dropout(drop_out),
        )
    def forward(self,x):
        return self.block(x)

class Linear_(nn.Module):
    def __init__(self,length,out_features):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(length,out_features)
        )
    def forward(self,x):
        return self.linear(x)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 =VGG_Block(3,32,3,1,1) 
        self.ca1 = CA_Block(32,206,206,reduction=16)
        self.b2 = VGG_Block(32,64,4,1,2) 
        self.b3 = VGG_Block(64,64,3,1,2)
        self.b4 = VGG_Block(64,64,3,1,1)
        self.b5 = VGG_Block(64,128,3,1,1)
        self.b6 = VGG_Block(128,128,3,1,1)
        self.b7 = VGG_Block(128,256,2,0,2)
        self.b8 = VGG_Block(256,256,3,1,1)
        self.b9 = VGG_Block(256,512,2,0,2)
        self.b10 = VGG_Block(512,512,3,1,1)
        self.b11 = VGG_Block(512,1024,3,1,2)
        self.b12 = VGG_Block(1024,1024,3,1,1)
        self.ca2 = CA_Block(1024,7,7,reduction=16)
        self.linear = Linear_(50176,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        o1 = self.ca1(self.relu(self.b1(x)))
        o2 = self.relu(self.b2(o1))
        o3 = self.relu(self.b3(o2))
        o4 = self.relu(self.b4(o3))
        o5 = self.relu(self.b5(o4))
        o6 = self.relu(self.b6(o5))
        o7 = self.relu(self.b7(o6))
        o8 = self.relu(self.b8(o7))
        o9 = self.relu(self.b9(o8))
        o10 = self.relu(self.b10(o9))
        o11 = self.relu(self.b11(o10))
        o12 = self.relu(self.b12(o11))
        o13 = self.ca2(o12)
        out = self.linear(o13)
        return out 
    def modules(self):
        model_name = 'AlexNet_deep'
        return model_name
