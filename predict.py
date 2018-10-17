# -- coding: utf-8 --
'''
    define Predict
'''
import cv2
import matplotlib.pyplot as plt
from mxnet import gluon
from mxnet import image
from mxnet.gluon import nn, model_zoo
from mxnet import ndarray as nd
import mxnet as mx
import numpy as np
from data_loader import get_iterators
from utils import *
import time
from refine_anchor import refine_anchor_generator
from mxnet.ndarray.contrib import MultiBoxPrior,MultiBoxTarget,MultiBoxDetection
from model import sizes,ratios,normalizations,RefineDet
from commom import multibox_layer

class_names = ['meter']
num_class = len(class_names)
data_shape = (3,512,512)
resize = (512,512)
batch_size = 1
std = np.array([51.58252012, 50.01343078, 57.31053303])
rgb_mean = np.array([114.06836982, 130.57876876, 143.64666367])
ctx=mx.gpu(0)

def load_weight():
    net = RefineDet(num_classes=num_class,sizes=sizes,ratios=ratios,normalizations=normalizations,verbose=False,prefix='refineDet_')

    net.load_parameters('./Model/RefineDet_MeterDetect.param',ctx=ctx)
    return net

def predict(img_nd,net,num_classes):
    #predict
    tic = time.time()
    ssd_layers = net(img_nd)
    arm_loc_preds, arm_cls_preds, arm_anchor_boxes, odm_loc_preds, odm_cls_preds = multibox_layer(ssd_layers,\
                                                                            num_classes,sizes,ratios,normalizations)
    #process result
    odm_anchor_boxes = refine_anchor_generator(arm_anchor_boxes,arm_loc_preds)
    odm_cls_prob = nd.SoftmaxActivation(odm_cls_preds, mode='channel')
    out = MultiBoxDetection(odm_cls_prob,odm_loc_preds,odm_anchor_boxes,\
                                force_suppress=True,clip=False,nms_threshold=.5)
    out = out.asnumpy()
    print(out.shape)
    print('detect time:',time.time()-tic)
    return out    


def detector(net, img_paths, num_classes,threshold=0.3):
    img_nds = None
    print(img_paths)
    tic = time.time()
    sizes = []
    for img_path in img_paths:
        # read img
        img = plt.imread(img_path)
        sizes.append(img.shape[:2])
        # test gray img
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # grb <-> bgr
        img = cv2.resize(img, resize)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = (img - rgb_mean) / std
        # img = (cv2.resize(img, myDect_config.resize) - myDect_config.rgb_mean) / myDect_config.std
        img_nd = nd.array(img,ctx=ctx)
        img_nd = img_nd.expand_dims(0).transpose((0,3,1,2))
        if img_nds is None:
            img_nds = img_nd
        else:
            img_nds = nd.concat(img_nds,img_nd,dim=0)
        print('complete once calc')
    print('IO time:',time.time()-tic)
    outs = predict(img_nds,net,num_classes)

    all_results = []
    for i, out in enumerate(outs):
        img_w = sizes[i][1]
        img_h = sizes[i][0]
        results = []
        colom_mask = (out[:,1] > threshold) * (out[:,0] != -1)
        out = out[colom_mask, :]
        for item in out:
            class_name = class_names[int(item[0])]
            prob = float(item[1])
            cx = float((item[2]+item[4])/2)*img_w
            cy = float((item[3]+item[5])/2)*img_h
            w = float((item[4]-item[2]))*img_w
            h = float((item[5]-item[3]))*img_h
            result = [class_name,prob,[cx,cy,w,h]]
            results.append(result)
        all_results.append(results)
    return  all_results


if __name__ == '__main__':
    # print(cv2.__version__)
    colors = ['red', 'blue', 'yellow', 'green']
    # CPU上处理时间随batch线性增长（1s/图），gpu（TITAN X）上可同时算8张(约2.5s)。
    img_paths = ['test1.jpg']
    #
    # img_paths =[]
    # img_path = os.walk('./detectimage/')
    # for root,dir,files in img_path:
    #     print(root)
    #     for file in files:
    #         print(root+file)
    #         img_paths.append(root+file)
    #
    net = load_weight()
    outs = detector(net, img_paths,num_classes=1,threshold=0.5)
    print(outs)
    for i, out in enumerate(outs):
        _, figs = plt.subplots()
        img = plt.imread(img_paths[i])

        # # 灰度测试
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        figs.imshow(img)
        # plt.gca()
        # tmp = [img.shape[1], img.shape[0]] * 2
        for j,item in enumerate(out):
            box = np.array(item[2])
            rect = plt.Rectangle((box[0] - box[2] / 2, box[1] - box[3] / 2), box[2], box[3], fill=False, color=colors[j % 4] )
            figs.add_patch(rect)
            figs.text(box[0] - box[2] / 2, box[1] - box[3] / 2, item[0] + ' ' + '%4f' % (item[1]), color = colors[j % 4])
        # plt.imshow(img)
        plt.savefig('results_%d.png'%(i))
        plt.show()
    # plt.savefig('results.png')
    # print(outs)
