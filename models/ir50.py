from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb


##################################  Original Arcface Model #############################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


# i = 0

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))
        i = 0

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        # print(shortcut.shape)
        # print('---s---')
        res = self.res_layer(x)
        # print(res.shape)
        # print('---r---')
        # i = i + 50
        # print(i)
        # print('50')
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    # print('50')


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3),
            # get_block(in_channel=64, depth=128, num_units=4),
            # get_block(in_channel=128, depth=256, num_units=14),
            # get_block(in_channel=256, depth=512, num_units=3)
        ]
        blocks2 = [
            # get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            # get_block(in_channel=128, depth=256, num_units=14),
            # get_block(in_channel=256, depth=512, num_units=3)
        ]
        blocks3 = [
            # get_block(in_channel=64, depth=64, num_units=3),
            # get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            # get_block(in_channel=256, depth=512, num_units=3)
        ]

    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks1, blocks2, blocks3


class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()
        # assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks1, blocks2, blocks3 = get_blocks(num_layers)
        # blocks2 = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules1 = []
        for block in blocks1:
            for bottleneck in block:
                modules1.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        modules2 = []
        for block in blocks2:
            for bottleneck in block:
                modules2.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        modules3 = []
        for block in blocks3:
            for bottleneck in block:
                modules3.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        # modules4 = []
        # for block in blocks4:
        #     for bottleneck in block:
        #         modules4.append(
        #             unit_module(bottleneck.in_channel,
        #                         bottleneck.depth,
        #                         bottleneck.stride))
        self.body1 = Sequential(*modules1)
        self.body2 = Sequential(*modules2)
        self.body3 = Sequential(*modules3)
        # self.body4 = Sequential(*modules4)  
    
    def forward(self, x):
        x = F.interpolate(x, size=112)
        x = self.input_layer(x)
        x1 = self.body1(x)
        x2 = self.body2(x1)
        x3 = self.body3(x2)

        # x = self.output_layer(x)
        # return l2_norm(x)

        return x1, x2, x3

'''class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        in_planes=num_layers
        ratio=drop_ratio
        kernel_size=7
        super(Backbone, self).__init__()
        # assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks1, blocks2, blocks3 = get_blocks(num_layers)
        # blocks2 = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules1 = []
        for block in blocks1:
            for bottleneck in block:
                modules1.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        modules2 = []
        for block in blocks2:
            for bottleneck in block:
                modules2.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        modules3 = []
        for block in blocks3:
            for bottleneck in block:
                modules3.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        # modules4 = []
        # for block in blocks4:
        #     for bottleneck in block:
        #         modules4.append(
        #             unit_module(bottleneck.in_channel,
        #                         bottleneck.depth,
        #                         bottleneck.stride))
        self.body1 = Sequential(*modules1)
        self.body2 = Sequential(*modules2)
        self.body3 = Sequential(*modules3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.body4 = Sequential(*modules4)  
    
    def forward(self, x):
        x = F.interpolate(x, size=112)
        x = self.input_layer(x)
        print(x.shape)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        print(out.shape)
        x = self.sigmoid(out)
        print("Channel Attention: ",x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x= self.sigmoid(x)
        print("Spatial Attention: ",x.shape)
        x1 = self.body1(x)
        x2 = self.body2(x1)
        x3 = self.body3(x2)

        # x = self.output_layer(x)
        # return l2_norm(x)

        return x1, x2, x3'''


class ChannelAttention(Module):
    def __init__(self,input_nc,rotio = 16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.share_mlp = nn.Sequential(nn.Conv2d(in_channels=input_nc,out_channels=input_nc//rotio,kernel_size=1,bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=input_nc//rotio,out_channels=input_nc,kernel_size=1,bias=False))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        print("Channel Attention Input Shape: ",x.shape)
        avg_x = self.share_mlp(self.avgpool(x))
        max_x = self.share_mlp(self.maxpool(x))
        print("Channel Attention Output Shape: ",x.shape)
        return self.sigmoid(avg_x+max_x)
'''
    def forward(self, x):
        
        avg_out = self.fc(self.avg_pool(x))
        
        max_out = self.fc(self.max_pool(x))
        
        out = avg_out + max_out
        return self.sigmoid(out)'''

class SpatialAttention(Module):
    def __init__(self,kernel=7):
        super().__init__()
        assert kernel in (3,7),"Kernel size must be in 3 or 7"
        padding = 1 if kernel == 3 else 3
        self.cnn = nn.Conv2d(2,1,kernel_size=kernel,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        print("Spatial Attention Input Shape: ",x.shape)
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True).values
        x = torch.cat([avg_out,max_out],dim=1)
        x=torch.transpose(x, 0, 1)
        x = self.cnn(x)
        x=torch.transpose(x, 0, 1)
        print("Spatial Attention Output Shape:",x.shape)
        return self.sigmoid(x)
        
class cbam(Module):
    def __init__(self,input_nc):
        super().__init__()
        self.ca = ChannelAttention(input_nc)
        self.sa = SpatialAttention()
        
    def forward(self,x):
        print(x.shape)
        x = self.ca(x) * x
        print(x.shape)
        x = self.sa(x) * x
        return x
    
    def get_spatial(self,x):
        return self.sa(x)
def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for i, (k, v) in enumerate(state_dict.items()):
        # print(i)

        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():

            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            # print(k)
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model

# model = Backbone(50, 0.0, 'ir')
# ir_checkpoint = torch.load(r'C:\Users\86187\Desktop\project\mixfacial\models\pretrain\new_ir50.pth')
# print('hello')
# i1, i2, i3 = 0, 0, 0
# ir_checkpoint = torch.load(r'C:\Users\86187\Desktop\project\mixfacial\models\pretrain\ir50.pth', map_location=lambda storage, loc: storage)
# for (k1, v1), (k2, v2) in zip(model.state_dict().items(), ir_checkpoint.items()):
#     print(f'k1:{k1}, k2:{k2}')
#     model.state_dict()[k1] = v2

# torch.save(model.state_dict(), r'C:\Users\86187\Desktop\project\mixfacial\models\pretrain\new_ir50.pth')
#     print(k)
#     if k.startswith('body1'):
#         i1+=1
#     if k.startswith('body2'):
#         i2+=1
#     if k.startswith('body3'):
#         i3+=1
# print(f'i1:{i1}, i2:{i2}, i3:{i3}')

# print('-'*100)
# ir_checkpoint = torch.load(r'C:\Users\86187\Desktop\project\mixfacial\models\pretrain\ir50.pth', map_location=lambda storage, loc: storage)
# le = 0
# for k, v in ir_checkpoint.items():
#     # print(k)
#     if k.startswith('body'):
#         if le < i1:
#             le += 1
#             key = k.split('.')[0] + str(1) + k.split('.')[1:]
#             print(key)
# # ir_checkpoint = ir_checkpoint["model"]
# model = load_pretrained_weights(model, ir_checkpoint)
# img = torch.rand(size=(2,3,224,224))
# out1, out2, out3 = model(img)
# print(out1.shape, out2.shape, out3.shape)
