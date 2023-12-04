import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def make_model(args, parent=False):
    return Transmission()

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=1.0, rgb_mean=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return F.interpolate(out, scale_factor=2)

class Transmission(nn.Module):
    def __init__(self):
        super(Transmission, self).__init__()
        self.sub_mean = MeanShift()
        haze_class = models.densenet121(pretrained=True)

        ######### Image Size 256 x 256  ##########
        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        ######### Image Size 64 x 64  ##########
        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        ######### Image Size 32 x 32  ##########
        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        ######### Image Size 16 x 16  ##########
        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        ######### Image Size 8 x 8  ##########
        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        ######### Image Size 16 x 16  ##########
        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        ######### Image Size 32 x 32  ##########
        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)

        ######### Image Size 64 x 64  ##########
        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        ######### Image Size 128 x 128  ##########
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        ######### Image Size 256 x 256  ##########
        self.conv_refin=nn.Conv2d(19,20,3,1,1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(20+4, 1, kernel_size=3,stride=1,padding=1)
        #self.tanh = nn.Tanh()

        self.upsample = F.interpolate

        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.sub_mean(x) ##  256 X 256
        x0=self.pool0(self.relu0(self.norm0(self.conv0(x)))) ## 64 X 64

        x1=self.trans_block1(self.dense_block1(x0)) ###  32 x 32
        x2=self.trans_block2(self.dense_block2(x1)) ### 16 X 16
        x3=self.trans_block3(self.dense_block3(x2)) ## 8 X 8

        x4=self.trans_block4(self.dense_block4(x3)) ### 16 X 16
        x42=torch.cat([x4,x2],1) 

        x5=self.trans_block5(self.dense_block5(x42)) ###  32 x 32
        x52=torch.cat([x5,x1],1) 

        x6=self.trans_block6(self.dense_block6(x52)) ##  64 X 64
        x7=self.trans_block7(self.dense_block7(x6)) ##  128 X 128
        x8=self.trans_block8(self.dense_block8(x7)) ##  256 X 256

        x8=torch.cat([x8,x],1)
        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = torch.sigmoid(self.refine3(dehaze))

        return dehaze

