# -- coding: utf-8 --
'''
    model define 
'''

from mxnet import gluon
from mxnet.gluon import nn, model_zoo
from mxnet import ndarray as nd
import mxnet as mx

from mxnet.ndarray.contrib import MultiBoxPrior,MultiBoxTarget

from commom import multibox_layer

### define bockbone
def conv_act_layer(in_channels,prefix,num,kernel_size,padding,stride=1,use_bn=False):
    '''
        prefix = 'conv'+ str(layer num)+'_'
    '''
    net = nn.HybridSequential()
    net.add(nn.Conv2D(channels=in_channels,kernel_size=kernel_size,strides=stride,
                      padding=padding,prefix='{}{}_'.format(prefix,num)))
    if use_bn:
        net.add(nn.BatchNorm(prefix='{}{}_'.format(prefix,num)))
    net.add(nn.Activation('relu',prefix='{}relu_{}_'.format(prefix,num)))
    return net

def vgg16_backbone1(prefix='my_net_'):
    '''
        define network that can be used to ssd layer, which include 7 layer of output
        return *net
    '''
    net1 = nn.HybridSequential(prefix=prefix)
    #group 1
    net1.add(nn.Conv2D(channels=64,kernel_size=3,padding=1,prefix='conv1_1_'),
           nn.BatchNorm(prefix='conv1_1_'),
           nn.Activation('relu',prefix='conv1_1_relu_'))
    net1.add(nn.Conv2D(channels=64,kernel_size=3,padding=1,prefix='conv1_2_'),
       nn.BatchNorm(prefix='conv1_2_'),
       nn.Activation('relu',prefix='conv1_2_relu_'))
    net1.add(nn.MaxPool2D(pool_size=2,strides=2,prefix='pool1_'))
    #group2
    net1.add(nn.Conv2D(channels=128,kernel_size=3,padding=1,prefix='conv2_1_'),
           nn.BatchNorm(prefix='conv2_1_'),
           nn.Activation('relu',prefix='conv2_1_relu_'))
    net1.add(nn.Conv2D(channels=128,kernel_size=3,padding=1,prefix='conv2_2_'),
       nn.BatchNorm(prefix='conv2_2_'),
       nn.Activation('relu',prefix='conv2_2_relu_'))
    net1.add(nn.MaxPool2D(pool_size=2,strides=2,prefix='pool2_'))
    # group 3
    net1.add(nn.Conv2D(channels=256,kernel_size=3,padding=1,prefix='conv3_1_'),
           nn.BatchNorm(prefix='conv3_1_'),
           nn.Activation('relu',prefix='conv3_1_relu_'))
    net1.add(nn.Conv2D(channels=256,kernel_size=3,padding=1,prefix='conv3_2_'),
       nn.BatchNorm(prefix='conv3_2_'),
       nn.Activation('relu',prefix='conv3_2_relu_'))
    net1.add(nn.Conv2D(channels=256,kernel_size=3,padding=1,prefix='conv3_3_'),
       nn.BatchNorm(prefix='conv3_3_'),
       nn.Activation('relu',prefix='conv3_3_relu_'))  
    net1.add(nn.MaxPool2D(pool_size=2,strides=2,prefix='pool3_')) 
    #group4 
    net1.add(nn.Conv2D(channels=512,kernel_size=3,padding=1,prefix='conv4_1_'),
           nn.BatchNorm(prefix='conv4_1_'),
           nn.Activation('relu',prefix='conv4_1_relu_'))
    net1.add(nn.Conv2D(channels=512,kernel_size=3,padding=1,prefix='conv4_2_'),
       nn.BatchNorm(prefix='conv4_2_'),
       nn.Activation('relu',prefix='conv4_2_relu_'))
    net1.add(nn.Conv2D(channels=512,kernel_size=3,padding=1,prefix='conv4_3_'),
       nn.BatchNorm(prefix='conv4_3_'),
       nn.Activation('relu',prefix='conv4_3_relu_'))

    ### layer 2
    net2 = nn.HybridSequential(prefix=prefix)
    net2.add(nn.MaxPool2D(pool_size=2,strides=2,prefix='pool4_'))   
    #group5
    net2.add(nn.Conv2D(channels=512,kernel_size=3,padding=1,prefix='conv5_1_'),
           nn.BatchNorm(prefix='conv5_1_'),
           nn.Activation('relu',prefix='conv5_1_relu_'))
    net2.add(nn.Conv2D(channels=512,kernel_size=3,padding=1,prefix='conv5_2_'),
       nn.BatchNorm(prefix='conv5_2_'),
       nn.Activation('relu',prefix='conv5_2_relu_'))
    net2.add(nn.Conv2D(channels=512,kernel_size=3,padding=1,prefix='conv5_3_'),
       nn.BatchNorm(prefix='conv5_3_'),
       nn.Activation('relu',prefix='conv5_3_relu_'))    
    net2.add(nn.MaxPool2D(pool_size=3,strides=1,padding=1,prefix='pool5_'))     
    #group6  
    net2.add(nn.Conv2D(channels=1024,kernel_size=3,padding=6,dilation=6,prefix='conv6_1_'),
           nn.BatchNorm(prefix='conv6_1_'),
           nn.Activation('relu',prefix='conv6_1_relu_'))
    #group7
    net2.add(nn.Conv2D(channels=1024,kernel_size=1,prefix='conv7_1_'),
           nn.BatchNorm(prefix='conv7_1_'),
           nn.Activation('relu',prefix='conv7_1_relu_'))    
    # ssd extra layers
    ### layer 3
    net3 = nn.HybridSequential(prefix=prefix)
    net3.add(conv_act_layer(in_channels=256,prefix='conv8_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=512,prefix='conv8_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 4
    net4 = nn.HybridSequential(prefix=prefix)
    net4.add(conv_act_layer(in_channels=128,prefix='conv9_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv9_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 5
    net5 = nn.HybridSequential(prefix=prefix)
    net5.add(conv_act_layer(in_channels=128,prefix='conv10_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv10_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 6
    net6 = nn.HybridSequential(prefix=prefix)
    net6.add(conv_act_layer(in_channels=128,prefix='conv11_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv11_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 7
    net7 = nn.HybridSequential(prefix=prefix)
    net7.add(conv_act_layer(in_channels=128,prefix='conv12_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv12_',num=2,kernel_size=4,padding=1,stride=1))
    return [net1,net2,net3,net4,net5,net6,net7]

## fineturning 
def get_vgg11bn_conv(prefix,ctx=mx.gpu(0)):
    vgg16net = model_zoo.vision.vgg11_bn(pretrained=True,ctx=ctx)
    #  layer 1 
    net1 = nn.HybridSequential(prefix)
    net1.add(*(vgg16net.features[:21]))
    # layer 2
    net2 = nn.HybridSequential(prefix)
    net2.add(*(vgg16net.features[21:28]))
    return net1,net2

def vgg11_backbone(prefix='my_net_'):
    '''
        define network that can be used to ssd layer, which include 7 layer of output
        return *net
    '''
    net1,net2 = get_vgg11bn_conv(prefix)
    net2.add(nn.MaxPool2D(pool_size=3,strides=1,padding=1,prefix='pool5_'))     
    #group6  
    net2.add(nn.Conv2D(channels=1024,kernel_size=3,padding=6,dilation=6,prefix='conv6_1_'),
           nn.BatchNorm(prefix='conv6_1_'),
           nn.Activation('relu',prefix='conv6_1_relu_'))
    #group7
    net2.add(nn.Conv2D(channels=1024,kernel_size=1,prefix='conv7_1_'),
           nn.BatchNorm(prefix='conv7_1_'),
           nn.Activation('relu',prefix='conv7_1_relu_')) 
    # ssd extra layers
    ### layer 3
    net3 = nn.HybridSequential(prefix=prefix)
    net3.add(conv_act_layer(in_channels=256,prefix='conv8_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=512,prefix='conv8_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 4
    net4 = nn.HybridSequential(prefix=prefix)
    net4.add(conv_act_layer(in_channels=128,prefix='conv9_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv9_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 5
    net5 = nn.HybridSequential(prefix=prefix)
    net5.add(conv_act_layer(in_channels=128,prefix='conv10_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv10_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 6
    net6 = nn.HybridSequential(prefix=prefix)
    net6.add(conv_act_layer(in_channels=128,prefix='conv11_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv11_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 7
    net7 = nn.HybridSequential(prefix=prefix)
    net7.add(conv_act_layer(in_channels=128,prefix='conv12_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv12_',num=2,kernel_size=4,padding=1,stride=1))
    return [net1,net2,net3,net4,net5,net6,net7]

def get_vgg16bn_conv(prefix,ctx=mx.gpu(0)):
    vgg16net = model_zoo.vision.vgg16_bn(pretrained=True,ctx=ctx)
    #  layer 1 
    net1 = nn.HybridSequential(prefix)
    net1.add(*(vgg16net.features[:33]))
    # layer 2
    net2 = nn.HybridSequential(prefix)
    net2.add(*(vgg16net.features[33:43]))
    return net1,net2

def vgg16_backbone(prefix='my_net_'):
    '''
        define network that can be used to ssd layer, which include 7 layer of output
        return *net
    '''
    net1,net2 = get_vgg16bn_conv(prefix)
    net2.add(nn.MaxPool2D(pool_size=3,strides=1,padding=1,prefix='pool5_'))     
    #group6  
    net2.add(nn.Conv2D(channels=1024,kernel_size=3,padding=6,dilation=6,prefix='conv6_1_'),
           nn.BatchNorm(prefix='conv6_1_'),
           nn.Activation('relu',prefix='conv6_1_relu_'))
    #group7
    net2.add(nn.Conv2D(channels=1024,kernel_size=1,prefix='conv7_1_'),
           nn.BatchNorm(prefix='conv7_1_'),
           nn.Activation('relu',prefix='conv7_1_relu_')) 
    # ssd extra layers
    ### layer 3
    net3 = nn.HybridSequential(prefix=prefix)
    net3.add(conv_act_layer(in_channels=256,prefix='conv8_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=512,prefix='conv8_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 4
    net4 = nn.HybridSequential(prefix=prefix)
    net4.add(conv_act_layer(in_channels=128,prefix='conv9_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv9_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 5
    net5 = nn.HybridSequential(prefix=prefix)
    net5.add(conv_act_layer(in_channels=128,prefix='conv10_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv10_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 6
    net6 = nn.HybridSequential(prefix=prefix)
    net6.add(conv_act_layer(in_channels=128,prefix='conv11_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv11_',num=2,kernel_size=3,padding=1,stride=2))
    ### layer 7
    net7 = nn.HybridSequential(prefix=prefix)
    net7.add(conv_act_layer(in_channels=128,prefix='conv12_',num=1,kernel_size=1,padding=0),
            conv_act_layer(in_channels=256,prefix='conv12_',num=2,kernel_size=4,padding=1,stride=1))
    return [net1,net2,net3,net4,net5,net6,net7]

sizes = [[.07, .1025], [.15,.2121], [.3, .3674], [.45, .5196], [.6, .6708], \
        [.75, .8216], [.9, .9721]]
ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
        [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
normalizations = [10, -1, -1, -1, -1, -1, -1]

class RefineDet(nn.HybridBlock):
    def __init__(self,num_classes,sizes,ratios,normalizations,verbose=False,ctx=mx.gpu(0),prefix='RefineDet_',**kwargs):
        super(RefineDet,self).__init__(**kwargs)
        self.num_classes = num_classes
        self.verbose = verbose
        self.sizes = sizes
        self.ratios = ratios
        self.ctx = ctx
        self.normalizations = normalizations
        with self.name_scope():
            net_layers = vgg16_backbone(prefix)
            # net_layers = vgg11_backbone(prefix)
            self.net = nn.HybridSequential(prefix=prefix)
            self.net.add(*net_layers)
            self.net.initialize(ctx=ctx,init=mx.init.Xavier(),force_reinit=False)
            # self.net.hybridize()
            net_layers.clear()
            print('----- init ----')

    def hybrid_forward(self,F,x):
        ssd_layers = []
        out = x
        for layer in self.net:
            out = layer(out)
            ssd_layers.append(out)
            if self.verbose:
                print(out.shape)
        
        return ssd_layers



if __name__ == "__main__":
    net = RefineDet(num_classes=1,sizes=sizes,ratios=ratios,normalizations=normalizations,verbose=True,prefix='refineDet_')
    # print(net.collect_params())
    print(net)
    # net.hybridize()
    # x = nd.random_normal(shape=(1,3,512,512),ctx=mx.gpu(0))
    # y = net(x)
    # net.export('./test') # 前向计算中ssd_layers为list参与，因此不能进行保存net，后续修改逻辑
    
