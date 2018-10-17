# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:31:48 2018

@author: Guo Shuangshuang
"""
import mxnet as mx
from mxnet import ndarray as nd
import numpy as np

## in_data : arm_cls_preds=arm_cls_preds, odm_cls_target=odm_cls_target, odm_loc_target_mask=odm_loc_target_mask
# 挑选出最精细的 正类 和 负类用于ODM，返回的是分类和回归的mask 
class negative_filtering(mx.operator.CustomOp):
    '''
    function：根据ARM分类结果中对应负类大于0.99的锚框ID,
        将ODM中对应的cls置为-1，mask置-0，后续不参与loss计算
    input: 
        arm_cls_preds: shape (batch, 2, h*w*num_anchors[:layers])
        odm_cls_target:对应的分类结果 shape (batch , h*w*num_anchors[:layers] ) [0,1,2...]
        odm_loc_target_mask:对应的正类和负类锚框掩码 shape (batch , h*w*num_anchors[:layers]*4 )[0 0 0 0 1 1 1 1]
    
    '''
    def __init__(self, ):
        super(negative_filtering, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        arm_cls_preds = in_data[0]
        odm_cls_target = in_data[1]
        odm_loc_target_mask = in_data[2]

        arm_cls_preds = nd.softmax(data=arm_cls_preds)
        arm_cls_preds_classes = nd.split(data=arm_cls_preds,axis=1,num_outputs=2)
        # arm_cls_preds_bg shape : (batch , h*w*num_anchors[:layers]) 负类【0】
        arm_cls_preds_bg = nd.reshape(data=arm_cls_preds_classes[0],shape=(0,-1))
        prob_temp = nd.ones_like(arm_cls_preds_bg)*0.99
        cond1 = arm_cls_preds_bg >= prob_temp # > 0.99 idx is 1
        # print('negative cond1 ------- :',heapq.nlargest(2,arm_cls_preds_bg[0]))
        temp1 = nd.ones_like(odm_cls_target)*(-1) ### TODO： 0 还是-1表示背景??
        # 如果ARM分类出的负类的置信度大于0.99，将其在ODM的anchor标号中去掉（-1替代），负类转换为背景
        odm_cls_target_mask = nd.where(condition=cond1,x=temp1,y=odm_cls_target)

        # apply filtering to odm_loc_target_mask
        # odm_loc_target_mask_shape: (batch, num_anchors, 4)

        arm_cls_preds_bg = nd.reshape(data=arm_cls_preds_bg,shape=(0,-1,1))#(batch , h*w*num_anchors[:layers],1)
        # (batch , h*w*num_anchors[:layers] , 4 )
        odm_loc_target_mask = nd.reshape(data=odm_loc_target_mask,shape=(0,-1,4))
        odm_loc_target_mask = odm_loc_target_mask[:,:,0] #(batch , h*w*num_anchors[:layers])
        #(batch , h*w*num_anchors[:layers], 1)
        ## 取整个batch中 所有行的 第一列，相当于对原来的4个相同label[0 0 0 0 ],[1 1 1 1]变成[0],[1]
        odm_loc_target_mask = nd.reshape(data=odm_loc_target_mask,shape=(0,-1,1))
        loc_temp = nd.ones_like(odm_loc_target_mask)*0.99
        cond2 = arm_cls_preds_bg >= loc_temp
        temp2 = nd.zeros_like(odm_loc_target_mask) # 取0
        # 如果ARM分类出的负类的置信度大于0.99，将其在ODM的掩码置0
        ## 实际上不管IOU计算的大小，用AMR的分类结果，如果是大于0.99的负类，不管通过IOU判断的正负类结果如何，都设置为背景
        odm_loc_target_bg_mask = nd.where(cond2,temp2,odm_loc_target_mask)
        odm_loc_target_bg_mask = nd.concat(*[odm_loc_target_bg_mask]*4,dim=2)
        # 还原维度
        odm_loc_target_bg_mask = nd.reshape(odm_loc_target_bg_mask,shape=(0,-1))

        for ind, val in enumerate([odm_cls_target_mask, odm_loc_target_bg_mask]):
            self.assign(out_data[ind], req[ind], val)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register("negative_filtering")
class negative_filteringProp(mx.operator.CustomOpProp):
    def __init__(self,):
        # super(negative_filteringProp, self).__init__(need_top_grad=False)
        super(negative_filteringProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['arm_cls_preds', 'odm_cls_target', 'odm_loc_target_mask']

    def list_outputs(self):
        return ['odm_cls_target_mask', 'odm_loc_target_bg_mask']

    def infer_shape(self, in_shape):
        arm_cls_preds_shape = in_shape[0]
        odm_cls_target_shape = in_shape[1]
        odm_loc_target_mask_shape = in_shape[2]
        odm_cls_target_mask_shape = [odm_cls_target_shape[0], odm_cls_target_shape[1]]
        odm_loc_target_bg_mask_shape = [odm_loc_target_mask_shape[0], odm_loc_target_mask_shape[1]]

        return [arm_cls_preds_shape, odm_cls_target_shape, odm_loc_target_mask_shape], [odm_cls_target_mask_shape, odm_loc_target_bg_mask_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return negative_filtering()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

