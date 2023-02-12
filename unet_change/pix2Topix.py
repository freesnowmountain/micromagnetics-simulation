import torch.nn as nn
from torchsummary import summary
import torch
from collections import OrderedDict
import torch, gc
import os
import GPUtil


# 定义降采样部分
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.down(x)


# 定义上采样部分
class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=False):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5) if drop_out else nn.Identity()
        )
        
    def forward(self, x):
        
        return self.up(x)


# ---------------------------------------------------------------------------------
# 定义pix_G  =>input 128*128
class pix2pixG_128(nn.Module):
    def __init__(self):
        super(pix2pixG_128, self).__init__()
        # down sample
        self.down_1 = nn.Conv2d(3, 64, 4, 2, 1)  # [batch,3,128,128]=>[batch,64,64,64]
        for i in range(7):
            if i == 0:
                self.down_2 = downsample(64, 128)  # [batch,64,64,64]=>[batch,128,32,32]
                self.down_3 = downsample(128, 256)  # [batch,128,32,32]=>[batch,256,16,16]
                self.down_4 = downsample(256, 512)  # [batch,256,16,16]=>[batch,512,8,8]
                self.down_5 = downsample(512, 512)  # [batch,512,8,8]=>[batch,512,4,4]
                self.down_6 = downsample(512, 512)  # [batch,512,4,4]=>[batch,512,2,2]
                self.down_7 = downsample(512, 512)  # [batch,512,2,2]=>[batch,512,1,1]

        # up_sample
        for i in range(7):
            if i == 0:
                self.up_1 = upsample(512, 512)  # [batch,512,1,1]=>[batch,512,2,2]
                self.up_2 = upsample(1024, 512, drop_out=True)  # [batch,1024,2,2]=>[batch,512,4,4]
                self.up_3 = upsample(1024, 512, drop_out=True)  # [batch,1024,4,4]=>[batch,512,8,8]
                self.up_4 = upsample(1024, 256)  # [batch,1024,8,8]=>[batch,256,16,16]
                self.up_5 = upsample(512, 128)  # [batch,512,16,16]=>[batch,128,32,32]
                self.up_6 = upsample(256, 64)  # [batch,256,32,32]=>[batch,64,64,64]

        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.init_weight()

    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

    def forward(self, x):
        # down
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        down_6 = self.down_6(down_5)
        down_7 = self.down_7(down_6)
        # up
        up_1 = self.up_1(down_7)
        up_2 = self.up_2(torch.cat([up_1, down_6], dim=1))
        up_3 = self.up_3(torch.cat([up_2, down_5], dim=1))
        up_4 = self.up_4(torch.cat([up_3, down_4], dim=1))
        up_5 = self.up_5(torch.cat([up_4, down_3], dim=1))
        up_6 = self.up_6(torch.cat([up_5, down_2], dim=1))
        out = self.last_Conv(torch.cat([up_6, down_1], dim=1))
        return out



# 32*32
class pix2pixG_32(nn.Module):
    def __init__(self):
        super(pix2pixG_32, self).__init__()
        # down sample
        self.down_1 = nn.Conv2d(3, 16, 2, 2)  # [batch,3,32,32]=>[batch,16,16,16]
        for i in range(7):
            if i == 0:
                self.down_2 = downsample(16, 32)  # [batch,16,16,16]=>[batch,32,8,8]
                self.down_3 = downsample(32, 64)  # [batch,32,8,8]=>[batch,64,4,4]
                self.down_4 = downsample(64, 128)  # [batch,64,4,4]=>[batch,128,2,2]
                self.down_8 = downsample(128, 128)  # [batch,128,2,2]=>[batch,128,1,1]

        # up_sample
        for i in range(7):
            if i == 0:
                self.up_1 = upsample(128, 128)  # [batch,128,1,1]=>[batch,128,2,2]
                self.up_2 = upsample(256, 64, drop_out=True)  # [batch,256,2,2]=>[batch,64,4,4]
                self.up_3 = upsample(128, 32, drop_out=True)  # [batch,128,4,4]=>[batch,32,8,8]
                self.up_4 = upsample(64, 16)  # [batch,32,8,8]=>[batch,16,16,16]


        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=2, stride=2),
            nn.Tanh()
        )

        self.init_weight()

    def init_weight(self):
        for w in self.modules():
            #判断层并且传参
            if isinstance(w, nn.Conv2d):
                #权重初始化
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

    def forward(self, x):   #x(batch,3,32,32)
        # down

        down_1 = self.down_1(x)#(16,16,16)
        down_2 = self.down_2(down_1)#(32,8,8)
        down_3 = self.down_3(down_2)#(64,4,4)
        down_4 = self.down_4(down_3)#(128,2,2)
        down_8 = self.down_8(down_4)#（128，1，1）

        #print("当前行为为：下采样")
        #GPUtil.showUtilization()
        # up
        up_1 = self.up_1(down_8)#（128，2，2）
        up_2 = self.up_2(torch.cat([up_1, down_4], dim=1))#(64,4,4)
        up_3 = self.up_3(torch.cat([up_2, down_3], dim=1))#(32,8,8)
        up_4 = self.up_4(torch.cat([up_3, down_2], dim=1))#(16,16,16)

        #print("当前行为为：上采样")
        #GPUtil.showUtilization()
        out = self.last_Conv(torch.cat([up_4, down_1], dim=1))
        
        return out

'''
# ---------------------------------------------------------------------------------
# 32*32
class pix2pixG_32(nn.Module):
    def __init__(self):
        super(pix2pixG_32, self).__init__()
        # down sample
        self.down_1 = nn.Conv2d(3, 64, 4, 2, 1)  # [batch,3,32,32]=>[batch,64,16,16]
        for i in range(7):
            if i == 0:
                self.down_2 = downsample(64, 128)  # [batch,64,16,16]=>[batch,128,8,8]
                self.down_3 = downsample(128, 256)  # [batch,128,8,8]=>[batch,256,4,4]
                self.down_4 = downsample(256, 512)  # [batch,256,4,4]=>[batch,512,2,2]
                self.down_8 = downsample(512, 512)  # [batch,512,2,2]=>[batch,512,1,1]

        # up_sample
        for i in range(7):
            if i == 0:
                self.up_1 = upsample(512, 512)  # [batch,512,1,1]=>[batch,512,2,2]
                self.up_2 = upsample(1024, 256, drop_out=True)  # [batch,1024,2,2]=>[batch,256,4,4]
                self.up_3 = upsample(512, 128, drop_out=True)  # [batch,512,4,4]=>[batch,128,8,8]
                self.up_4 = upsample(256, 64)  # [batch,256,8,8]=>[batch,64,16,16]


        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.init_weight()

    def init_weight(self):
        for w in self.modules():
            #判断层并且传参
            if isinstance(w, nn.Conv2d):
                #权重初始化
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

    def forward(self, x):   #x(batch,32,32,3)
        # down
        down_1 = self.down_1(x)#(64,16,16)
        down_2 = self.down_2(down_1)#(128,8,8)
        down_3 = self.down_3(down_2)#(256,4,4)
        down_4 = self.down_4(down_3)#(512,2,2)
        down_8 = self.down_8(down_4)#（512，1，1）

        #print("当前行为为：下采样")
        #GPUtil.showUtilization()
        # up
        up_1 = self.up_1(down_8)#（512，2，2）
        up_2 = self.up_2(torch.cat([up_1, down_4], dim=1))#(256,4,4)
        up_3 = self.up_3(torch.cat([up_2, down_3], dim=1))#(128,8,8)
        up_4 = self.up_4(torch.cat([up_3, down_2], dim=1))#(64,16,16)

        #print("当前行为为：上采样")
        #GPUtil.showUtilization()
        out = self.last_Conv(torch.cat([up_4, down_1], dim=1))
        
        return out

'''

#*************备份***********************************
class pix2pixG_256(nn.Module):
    def __init__(self):
        super(pix2pixG_256, self).__init__()
        # down sample
        self.down_1 = nn.Conv2d(3, 64, 4, 2, 1)  # [batch,3,256,256]=>[batch,64,128,128]
        for i in range(7):
            if i == 0:
                self.down_2 = downsample(64, 128)  # [batch,64,128,128]=>[batch,128,64,64]
                self.down_3 = downsample(128, 256)  # [batch,128,64,64]=>[batch,256,32,32]
                self.down_4 = downsample(256, 512)  # [batch,256,32,32]=>[batch,512,16,16]
                self.down_5 = downsample(512, 512)  # [batch,512,16,16]=>[batch,512,8,8]
                self.down_6 = downsample(512, 512)  # [batch,512,8,8]=>[batch,512,4,4]
                self.down_7 = downsample(512, 512)  # [batch,512,4,4]=>[batch,512,2,2]
                self.down_8 = downsample(512, 512)  # [batch,512,2,2]=>[batch,512,1,1]

        # up_sample
        for i in range(7):
            if i == 0:
                self.up_1 = upsample(512, 512)  # [batch,512,1,1]=>[batch,512,2,2]
                self.up_2 = upsample(1024, 512, drop_out=True)  # [batch,1024,2,2]=>[batch,512,4,4]
                self.up_3 = upsample(1024, 512, drop_out=True)  # [batch,1024,4,4]=>[batch,512,8,8]
                self.up_4 = upsample(1024, 512)  # [batch,1024,8,8]=>[batch,512,16,16]
                self.up_5 = upsample(1024, 256)  # [batch,1024,16,16]=>[batch,256,32,32]
                self.up_6 = upsample(512, 128)  # [batch,512,32,32]=>[batch,128,64,64]
                self.up_7 = upsample(256, 64)  # [batch,256,64,64]=>[batch,64,128,128]

        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.init_weight()

    def init_weight(self):
        for w in self.modules():
            #判断层并且传参
            if isinstance(w, nn.Conv2d):
                #权重初始化
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

    def forward(self, x):   #x(batch,256,256,3)
        # down
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        down_6 = self.down_6(down_5)
        down_7 = self.down_7(down_6)
        down_8 = self.down_8(down_7)

        #print("当前行为为：下采样")
        #GPUtil.showUtilization()
        # up
        up_1 = self.up_1(down_8)
        up_2 = self.up_2(torch.cat([up_1, down_7], dim=1))
        up_3 = self.up_3(torch.cat([up_2, down_6], dim=1))
        up_4 = self.up_4(torch.cat([up_3, down_5], dim=1))
        up_5 = self.up_5(torch.cat([up_4, down_4], dim=1))
        up_6 = self.up_6(torch.cat([up_5, down_3], dim=1))
        up_7 = self.up_7(torch.cat([up_6, down_2], dim=1))
        #print("当前行为为：上采样")
        #GPUtil.showUtilization()
        out = self.last_Conv(torch.cat([up_7, down_1], dim=1))
        
        return out





if __name__ == '__main__':
    G = pix2pixG_32().to('cuda')
    summary(G, (3, 256, 256))