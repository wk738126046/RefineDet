# -- coding: utf-8 --
'''
    generate refine anchors   to ODM
'''

from mxnet import gluon
from mxnet.gluon import nn, model_zoo
from mxnet import ndarray as nd
import mxnet as mx

from mxnet.ndarray.contrib import MultiBoxPrior,MultiBoxTarget

def refine_anchor_generator(arm_anchor_boxes,arm_loc_preds):
    '''
        function: 
        input:
            arm_anchor_boxes: shape (1,h*w*num_anchors[:layers],4)
            arm_loc_preds: shape (batch,h*w*num_loc_pred[:layers])
    '''
    batch_size = arm_loc_preds.shape[0]
    arm_anchor_boxes = nd.concat(*[arm_anchor_boxes]*batch_size,dim=0) #(batch,h*w*num_anchors[:layers],4)
    arm_anchor_boxes_bs = nd.split(data=arm_anchor_boxes,axis=2,num_outputs=4)#(batch,all_anchors,1)*4
    
    al = arm_anchor_boxes_bs[0] # left top x
    at = arm_anchor_boxes_bs[1] # left top y
    ar = arm_anchor_boxes_bs[2] # right below x
    ab = arm_anchor_boxes_bs[3] # right below y
    aw = ar - al
    ah = ab - at
    ax = (al+ar)/2.0
    ay = (at+ab)/2.0
    arm_loc_preds = nd.reshape(data=arm_loc_preds,shape=(0,-1,4)) #(batch,h*w*num_anchors[:layers],4)
    arm_loc_preds_bs = nd.split(data=arm_loc_preds,axis=2,num_outputs=4)
    ox_preds = arm_anchor_boxes_bs[0]
    oy_preds = arm_anchor_boxes_bs[1]
    ow_preds = arm_anchor_boxes_bs[2]
    oh_preds = arm_anchor_boxes_bs[3]
    ## TODO: RCNN Paper object   
    ox = ox_preds * aw * 0.1 + ax
    oy = oy_preds * ah * 0.1 + ay
    ow = nd.exp(ow_preds * 0.2) * aw 
    oh = nd.exp(oh_preds * 0.2) * ah 

    out0 = ox - ow / 2.0
    out1 = oy - oh / 2.0
    out2 = ox + ow / 2.0
    out3 = oy + oh / 2.0

    refine_anchor = nd.concat(out0,out1,out2,out3,dim=2)
    # refine_anchor = nd.split(data=refine_anchor,axis=0,num_outputs=batch_size)
    return refine_anchor # (batch,h*w*num_anchors[:layers],4)



