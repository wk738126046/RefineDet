# -- coding: utf-8 --
'''
    define ARM ,TCB and ODM 
'''

from mxnet import gluon
from mxnet.gluon import nn, model_zoo
from mxnet import ndarray as nd
import mxnet as mx

from mxnet.ndarray.contrib import MultiBoxPrior,MultiBoxTarget

## latest TCB 
def conv_tcb_layer(out_channels,level,num,kernel_size,padding,stride=1,use_bn=False):
    '''
        function : bacis conv of TCB ( conv + bn + relu)   
        conv prefix = 'tcb_'+ level '_' + num
        level:  [-2,-1,0,1,2,3,4] \
             it is consistent with ssd_layers, the latest layer correspond to level[4]
    '''
    net = nn.HybridSequential(prefix='conv_tcb_layer_')
    net.add(nn.Conv2D(channels=out_channels,kernel_size=kernel_size,strides=stride,
                      padding=padding,prefix='tcb_{}_{}'.format(level,num)))
    if use_bn:
        net.add(nn.BatchNorm(prefix='tcb_{}_{}'.format(level,num)))
    net.add(nn.Activation('relu',prefix='tcb_{}_relu_{}_'.format(level,num)))
    return net

def tcb_module_last(ssd_layer, out_channels = 256, level = 1):
    '''
        function: latest full TCB
        ssd_layer: the output of ARM (ssd_layers)
        out_channels: number of filter (all out_channels of TCB are 256)
        level:  [-2,-1,0,1,2,3,4] \
             it is consistent with ssd_layers, the latest layer correspond to level[4]
        
        return: out, shape = (batch,256,1,1) if original size is 512*512 
    '''
    tcb_last = nn.HybridSequential(prefix='tcb_module_last_')
    tcb_last.add(conv_tcb_layer(out_channels=out_channels,level=level,num=1,kernel_size=3,padding=1),
                conv_tcb_layer(out_channels=out_channels,level=level,num=2,kernel_size=3,padding=1),
                conv_tcb_layer(out_channels=out_channels,level=level,num=3,kernel_size=3,padding=1))
    tcb_last.initialize(init=mx.init.Xavier(),ctx=mx.gpu(0))
    # tcb_last.hybridize() # symbol compile
    out = tcb_last(ssd_layer)
    return out
# init deconv
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
        (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)

## TCB contain deconv
def deconv_tcb_layer(out_channels,level,kernel_size=2,padding=0,stride=2,use_bn=False):
    '''
        function : bacis deconv of TCB ( deconv + bn + relu)   
        level:  [-2,-1,0,1,2,3,4] \
             it is consistent with ssd_layers, the latest layer correspond to level[4]
    '''
    net = nn.HybridSequential(prefix='deconv_tcb_layer_')
    net.add(nn.Conv2DTranspose(channels=out_channels,kernel_size=kernel_size,strides=stride,
                      padding=padding,prefix='deconv_tcb_{}_'.format(level)))
    if use_bn:
        net.add(nn.BatchNorm(prefix='deconv_tcb_{}_'.format(level)))
    net.add(nn.Activation('relu',prefix='deconv_tcb_{}_relu_'.format(level)))
    return net

def tcb_module(ssd_layer, deconv_layer,out_channels = 256, level = 1):
    '''
        function: full TCB contain deconv
        ssd_layer: the output of ARM (ssd_layers)
        deconv_layer: the output of next layer 
    '''
    tcb1 = nn.HybridSequential('tcb_module_conv_')
    tcb1.add(conv_tcb_layer(out_channels=out_channels,level=level,num=1,kernel_size=3,padding=1),
            conv_tcb_layer(out_channels=out_channels,level=level,num=2,kernel_size=3,padding=1))
    tcb1.initialize(init=mx.init.Xavier(),ctx=mx.gpu(0))
    # tcb1.hybridize() # symbol compile
    out1 = tcb1(ssd_layer)
#     tcb2 = nn.HybridSequential('tcb_module_deconv_')
#     tcb2.add(deconv_tcb_layer(out_channels=256,level=level))
    # print('----tcb module  11111 ---')
    if level == 3 and ssd_layer.shape[2] == 5: ## input 320 and deconv 3*3 to 5*5
        tcb2 = deconv_tcb_layer(out_channels=256,level=level,kernel_size=3,padding=1)
        tcb2.initialize(init=mx.init.Constant(bilinear_kernel(ssd_layer.shape[1],256,3)),ctx=mx.gpu(0))    
    else:
        tcb2 = deconv_tcb_layer(out_channels=256,level=level)
        tcb2.initialize(init=mx.init.Constant(bilinear_kernel(ssd_layer.shape[1],256,2)),ctx=mx.gpu(0))    
    ## TODO: init use bilinear_kernel
    # tcb2.initialize(init=mx.init.Xavier(),ctx=mx.gpu(0))
    out2 = tcb2(deconv_layer)
    # print('----tcb module  22222 ---')
    out3 = nd.ElementWiseSum(*[out1,out2])
    # print('----tcb module  33333 ---')
    tcb3 = nn.HybridSequential('tcb_module_EltwSum_')
    tcb3.add(nn.Activation('relu',prefix='tcb_{}_relu_3'.format(level)),
            conv_tcb_layer(out_channels=out_channels,level=level,num=3,kernel_size=3,padding=1))
    tcb3.initialize(init=mx.init.Xavier(),ctx=mx.gpu(0))
    out = tcb3(out3)
    # print('----tcb module  out ---')
    return out

# odm output 
def construct_refineDet(ssd_layers):
    '''
        RefineDet network for forward
        ssd_layers: the output of Network ;
        tcb_module_last: the latest transfer connection block ;
        tcb_module: others transfer connection block, include deconv used to concat next layer of network 
        
        return : all outputs of odm layers  
    '''
    out_layers = []
    layers = ssd_layers[::-1] # layers reverse TODO: use ndarry function
    # layers = nd.reverse(data=ssd_layers,axis=0)
    for k, ssd_layer in enumerate(layers):
        if k == 0:
            out_layer = tcb_module_last(layers[k],256,level=4-k)
            # print("odm_layer_6 : {}".format(out_layer.shape))
        else:
            out_layer = tcb_module(layers[k],out_layer,256,level=4-k)
            # print("odm_layer_{} : {}".format(6-k,out_layer.shape))
        out_layers.append(out_layer)
    return out_layers[::-1]
    # return nd.reverse(data=out_layers,axis=0)

# ssd_layers = [relu4_3, relu7, relu8_2, relu9_2, relu10_2, relu11_2, relu12_2]
# sizes = [[.07, .1025], [.15,.2121], [.3, .3674], [.45, .5196], [.6, .6708], \
#         [.75, .8216], [.9, .9721]]
# ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
#         [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
# normalizations = [20, -1, -1, -1, -1, -1, -1]
num_channels=[512] # channel of relu_4_3 , it is used to product the same shape with L2_Norm scale  
def multibox_layer(ssd_layers, num_classes,sizes=[.2, .95],ratios=[1], normalizations=-1,verbose=False,num_channels=[512]):
    '''
    the basic aggregation module for SSD detection. Takes in multiple layers,
    generate multiple object detection targets by customized layers
    num_channels: used to product the same shape with L2_Norm scale, which can Eltw product
    '''
    assert len(ssd_layers) > 0 , 'ssd layers must not be empty list'
    assert num_classes > 0 , 'num_classes {} must be larger than 0'.format(num_classes)
    assert len(ratios) > 0 , 'aspect ratios must not be empty list'
    assert len(sizes) > 0 , 'size must not be empty list'
    if not isinstance(ratios[0],list):
        ratios = [ratios] * len(ssd_layers) # provided only one ratio list to all ssd layers
    assert len(ratios) == len(ssd_layers) , 'ratios and ssd layers must be same length'
    
    # TODO: this fucntion need to consult original ssd anchor size (symbol_factory/get_scales)
    if len(sizes) == 2 and not isinstance(sizes[0],list):
        # if sizes are default,we need to provided size range,\
        #  compute the sizes for each layer  used to product anchors to every pixel
        assert sizes[0] > 0 and sizes[0] < 1 , 'default min size are error'
        assert sizes[1] > sizes[0] and sizes[1] < 1, 'default max size are error'
        import numpy as np
        start_offset = 0.07
        tmp = np.linspace(sizes[0],sizes[1], num=(len(ssd_layers)-1))
        min_sizes = [start_offset] + tmp.tolist
        max_sizes = tmp.tolist + [tmp(-1)+start_offset]
        sizes = [ size for size in zip(min_sizes,max_sizes)] # [(),(),...]
    assert len(sizes) == len(ssd_layers) , 'sizes and ssd layers must have same length'
    
    if not isinstance(normalizations,list):
        normalizations = [normalizations]*len(ssd_layers)
    assert len(normalizations) ==  len(ssd_layers) , 'L2_Norm scales and ssd layers must have same length'
    assert sum(x > 0 for x in normalizations) <= len(num_channels),'must provide number of channels for each normalized layer'

    # for k, ssd_layer in enumerate(ssd_layers):
    #     # normalize
    #     # print('ssd layer : {}'.format(ssd_layer))
    #     if normalizations[k] > 0:
    #         ssd_layer = nd.L2Normalization(ssd_layer,mode='channel',name='L2_Norm_layers_{}'.format(k))
    #         scale = nd.ones(shape=(1,num_channels[k],1,1)) * normalizations[k] ## ?? instead Variable ??
    #         print(scale)
    #         ssd_layer = nd.broadcast_mul(scale,ssd_layer)
    #         ssd_layers[k] = ssd_layer # update data
    # get odm output
    odm_layers = construct_refineDet(ssd_layers) 
    # According to calc conv, we can predict results of ssd layer ,which include ARM and ODM 
    arm_loc, arm_cls, arm_anchor_boxes = getpred(ssd_layers, 1, sizes, ratios, mode='arm',verbose=verbose)

    odm_loc, odm_cls = getpred(odm_layers, num_classes, sizes, ratios, mode='odm',verbose=verbose)
    return [arm_loc, arm_cls, arm_anchor_boxes, odm_loc, odm_cls]


def getpred(from_layers, num_classes, sizes, ratios, mode='arm',verbose=False):
    '''
        function: According to calc conv, we can predict results of ssd layer ,which include ARM and ODM .

        layers: outputs of network (ssd_layers and odm_layers) 
    '''
    loc_layers = []
    cls_layers = []
    anchor_layers = []
    num_classes += 1 # add background 
    for k , from_layer in enumerate(from_layers):
        # TODO: Add intermediate layers if it is necessary
        # estimate number of anchors per location
        size = sizes[k]
        assert len(size) > 0 , 'must provide at least one size'
        ratio = ratios[k]
        assert len(ratio) > 0 , 'must provide at least one ratio'
        size_str = '(' +','.join([str(x) for x in ratio]) + ')' # (ratio)
        num_anchors = len(size) + len(ratio) - 1 # layers[0]: 4, layers[:]: 6  per pixel

        # 1. create location prediction for layers
        num_loc_pred = num_anchors * 4 # bbox param
        loc_pred = box_predictor(from_layer,num_loc_pred,k,verbose=verbose)
        ## transpose(0,2,3,1).flatten()
        ## TODO: Memory clear ?
        loc_pred = nd.transpose(loc_pred,axes=(0,2,3,1))
        loc_pred = nd.flatten(loc_pred) # (batch,h*w*num_loc_pred)
        loc_layers.append(loc_pred)

        # 2. create class prediction layer
        num_cls_pred = num_anchors * num_classes
        cls_pred = class_predictor(from_layer,num_cls_pred,k,verbose=verbose)
        cls_pred = nd.transpose(cls_pred,axes=(0,2,3,1))
        cls_pred = nd.flatten(cls_pred) # (batch,h*w*num_cls_pred)
        cls_layers.append(cls_pred)

        # 3. generate anchors
        if mode == 'arm':
            # anchor shape: (1,h*w*num_anchors,4)
            anchors = MultiBoxPrior(from_layer,size,ratio)
            if verbose:
                print('arm_layer_{}_anchors: {}'.format(k,anchors.shape))
            anchor_layers.append(anchors) # Be calfull anchors' shape
        
    # 4. concat all param
    loc_preds = nd.concat(*loc_layers,dim=1) #(batch,h*w*num_loc_pred[:layers])
    cls_preds = nd.concat(*cls_layers,dim=1).reshape((0,-1,num_classes)) #(batch, h*w*num_anchors[:layers], num_classes)
    cls_preds = nd.transpose(cls_preds,axes=(0,2,1)) #(batch, num_classes, h*w*num_anchors[:layers] )
    if mode == 'arm':
        anchor_boxes = nd.concat(*anchor_layers,dim=1) # (1,h*w*num_anchors[:layers],4)
        # print('arm_loc_preds : {}, arm_cls_pred : {}, arm_anchors {}'.format(loc_preds.shape,cls_preds.shape,anchor_boxes.shape))
        return [loc_preds.as_in_context(mx.gpu(0)),cls_preds.as_in_context(mx.gpu(0)),anchor_boxes.as_in_context(mx.gpu(0))]
    else:
        # print('odm_loc_preds : {}, odm_cls_pred : {} '.format(loc_preds.shape,cls_preds.shape))
        return [loc_preds.as_in_context(mx.gpu(0)),cls_preds.as_in_context(mx.gpu(0))]
    
#classify: num_cls_pred = anchors*(num_classes+1)
def class_predictor(from_layer,num_cls_pred,k,verbose=True):
    net = nn.HybridSequential(prefix='class_predictor_layer_{}'.format(k))
    net.add(nn.Conv2D(num_cls_pred,kernel_size=3,strides=1,padding=1))
    net.initialize(ctx=mx.gpu(0),init=mx.init.Xavier())
    out = net(from_layer)
    if verbose:
        print('class_predictor_layer_{}: '.format(k), out.shape)
    return out

#regression: num_loc_pred = anchors*4
### predict: x_center,y_center,w,h
def box_predictor(from_layer,num_loc_pred,k,verbose=True):
    net = nn.HybridSequential(prefix='class_predictor_layer_{}'.format(k))
    net.add(nn.Conv2D(num_loc_pred,kernel_size=3,strides=1,padding=1))
    net.initialize(ctx=mx.gpu(0),init=mx.init.Xavier())
    out = net(from_layer)
    if verbose:
        print('box_predictor_layer_{}: '.format(k), out.shape)
    return out









