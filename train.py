# -- coding: utf-8 --
'''
    define Train
'''

from mxnet import gluon
from mxnet.gluon import nn, model_zoo
from mxnet import ndarray as nd
import mxnet as mx
import numpy as np
from data_loader import get_iterators
from utils import *
import time

from mxnet.ndarray.contrib import MultiBoxPrior,MultiBoxTarget,MultiBoxDetection
from model import sizes,ratios,normalizations,RefineDet


data_shape = (3,320,320)
# data_shape = (3,512,512)
std = np.array([51.58252012, 50.01343078, 57.31053303])
rgb_mean = np.array([114.06836982, 130.57876876, 143.64666367])
ctx = mx.gpu(0)
resize = data_shape[1:]
rec_prefix = './dataset/data_320/rec/img_'+str(resize[0])+'_'+str(resize[1])
# num_class = 1
'''
loss define
'''
# 1. multiply classes (tow classes use SoftmaxEntroryCross)
class FocalLoss(gluon.loss.Loss):
    def __init__(self,axis=-1,alpha=0.25,gamma=2,batch_axis=0,**kwargs):
        super(FocalLoss,self).__init__(None,batch_axis,**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.axis = axis
        self.batch_axis = batch_axis

    def hybrid_forward(self, F, y, label):
        y = F.softmax(y)
        py = y.pick(label, axis=self.axis, keepdims=True)
        loss = - (self.alpha * ((1 - py) ** self.gamma)) * py.log()
        return loss.mean(axis=self.batch_axis, exclude=True)
# 2. regession loss
class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self,batch_axis=0,**kwargs):
        super(SmoothL1Loss,self).__init__(None,batch_axis,**kwargs)
        self.batch_axis = batch_axis

    def hybrid_forward(self, F, y,label,mask):
        loss = F.smooth_l1((y-label)*mask,scalar=1.0)
        return nd.mean(loss,axis=self.batch_axis,exclude=True)

lossdoc='''
使用AP分数作为分类评价的标准。
由于在模型检测问题中，反例占据了绝大多数，即使把所有的边框全部预测为反例已然会具有不错的精度。
因此不能直接使用分类精度作为评价标准。
 AP曲线考虑在预测为正例的标签中真正为正例的概率（查准率， precise）
 以及在全部正例中预测为正例的概率（召回率， recall），更能反映模型的正确性。
 
 使用MAE（平均绝对值误差）作为回归评价的标准。
'''
from mxnet import metric
from mxnet import autograd
from mxnet.ndarray.contrib import MultiBoxDetection
import numpy as np
'''
trian net
'''
def evaluate_acc(net,data_iter,ctx):
    data_iter.reset()
    box_metric = metric.MAE()
    outs,labels = None,None
    for i, batch in enumerate(data_iter):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        # print('acc',label.shape)
        ssd_layers = net(data)
        arm_loc_preds, arm_cls_preds, arm_anchor_boxes, odm_loc_preds, odm_cls_preds = multibox_layer(ssd_layers,\
                                                                            num_classes,sizes,ratios,normalizations)
        # arm_loc_preds, arm_cls_preds, arm_anchor_boxes, odm_loc_preds, odm_cls_preds = net(data)

        label_arm = nd.Custom(label, op_type='modify_label')
        arm_tmp = MultiBoxTarget(arm_anchor_boxes,label_arm,arm_cls_preds,overlap_threshold=.5,\
                                    negative_mining_ratio=3,negative_mining_thresh=.5)
        arm_loc_target = arm_tmp[0] # box offset
        arm_loc_target_mask = arm_tmp[1] # box mask (only 0,1)
        arm_cls_target = arm_tmp[2] #  every anchor' idx

        odm_anchor_boxes = refine_anchor_generator(arm_anchor_boxes,arm_loc_preds)#(batch,h*w*num_anchors[:layers],4)
        odm_anchor_boxes_bs = nd.split(data=odm_anchor_boxes,axis=0,num_outputs=label.shape[0])# list

        odm_loc_target = []
        odm_loc_target_mask = []
        odm_cls_target = []
        label_bs = nd.split(data=label,axis=0,num_outputs=label.shape[0])
        odm_cls_preds_bs = nd.split(data=odm_cls_preds,axis=0,num_outputs=label.shape[0])
        for j in range(label.shape[0]):
            if label.shape[0] == 1:
                odm_tmp = MultiBoxTarget(odm_anchor_boxes_bs[j].expand_dims(axis=0),label_bs[j].expand_dims(axis=0),\
                                    odm_cls_preds_bs[j].expand_dims(axis=0),overlap_threshold=.5,negative_mining_ratio=2,negative_mining_thresh=.5)
                    ## 多个batch
            else:
                odm_tmp = MultiBoxTarget(odm_anchor_boxes_bs[j],label_bs[j],\
                                    odm_cls_preds_bs[j],overlap_threshold=.5,negative_mining_ratio=3,negative_mining_thresh=.5)
            odm_loc_target.append(odm_tmp[0])
            odm_loc_target_mask.append(odm_tmp[1])
            odm_cls_target.append(odm_tmp[2])

        odm_loc_target = nd.concat(*odm_loc_target,dim=0)
        odm_loc_target_mask = nd.concat(*odm_loc_target_mask,dim=0)
        odm_cls_target = nd.concat(*odm_cls_target,dim=0)

        # negitave filter
        group = nd.Custom(arm_cls_preds,odm_cls_target,odm_loc_target_mask,op_type='negative_filtering')
        odm_cls_target = group[0] #用ARM中的cls过滤后的odm_cls
        odm_loc_target_mask = group[1] #过滤掉的mask为0

        # arm_cls_prob = nd.SoftmaxActivation(arm_cls_preds, mode='channel')
        odm_cls_prob = nd.SoftmaxActivation(odm_cls_preds, mode='channel')      

        out = MultiBoxDetection(odm_cls_prob,odm_loc_preds,odm_anchor_boxes,\
                                    force_suppress=True,clip=False,nms_threshold=.5,nms_topk=400)
        # print(out.shape)
        if outs is None:
            outs = out
            labels = label
        else:
            outs = nd.concat(outs, out, dim=0)
            labels = nd.concat(labels, label, dim=0)
        box_metric.update([odm_loc_target], [odm_loc_preds * odm_loc_target_mask])

    AP = evaluate_MAP(outs,labels)
    return AP,box_metric

info = {"train_ap": [], "valid_ap": [], "loss": []}
def plot(key):
    plt.plot(range(len(info[key])), info[key], label=key)

from refine_anchor import refine_anchor_generator
from negative_filtering import negative_filtering
from modify_label import modify_label
from commom import multibox_layer
from mxboard import SummaryWriter

# sizes = [[.07, .1025], [.15,.2121], [.3, .3674], [.45, .5196], [.6, .6708], \
#         [.75, .8216], [.9, .9721]]
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],[0.88, 0.961]] # vgg11 + 3 ssd layer
ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
        [1,2,.5,3,1./3]]
# normalizations = [10, -1, -1, -1, -1, -1, -1]
normalizations = [10, -1, -1, -1, -1]

arm_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
cls_loss = FocalLoss()
box_loss = SmoothL1Loss()
def mytrain(net,num_classes,train_data,valid_data,ctx,start_epoch, end_epoch, \
            arm_cls_loss=arm_cls_loss,cls_loss=cls_loss,box_loss=box_loss,trainer=None):
    if trainer is None:
        # trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01,'momentum':0.9, 'wd':50.0})
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001,'clip_gradient':2.0})
        # trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.003})
    box_metric = metric.MAE()

    ## add visible
    # collect parameter names for logging the gradients of parameters in each epoch
    params = net.collect_params()
    # param_names = params.keys()
    # define a summary writer that logs data and flushes to the file every 5 seconds
    sw = SummaryWriter(logdir='./logs', flush_secs=5)
    global_step  = 0

    for e in range(start_epoch, end_epoch):
        # print(e)
        train_data.reset()
        valid_data.reset()
        box_metric.reset()
        tic = time.time()
        _loss = [0, 0]
        arm_loss = [0,0]
        # if e == 6 or e == 100:
        #     trainer.set_learning_rate(trainer.learning_rate * 0.2)

        outs, labels = None, None
        for i, batch in enumerate(train_data):
            # print('----- batch {} start ----'.format(i))
            data = batch.data[0].as_in_context(ctx)
            label = batch.label[0].as_in_context(ctx)
            # print('label shape: ',label.shape)
            with autograd.record():
                # 1. generate results according to extract network
                ssd_layers = net(data)
                arm_loc_preds, arm_cls_preds, arm_anchor_boxes, odm_loc_preds, odm_cls_preds = multibox_layer(ssd_layers,\
                                                                            num_classes,sizes,ratios,normalizations)
                # arm_loc_preds, arm_cls_preds, arm_anchor_boxes, odm_loc_preds, odm_cls_preds = net(data)
                # print('---------1111-----------')
                # 2. ARM predict
                ## 2.1  modify label as [-1,0,..]
                label_arm = nd.Custom(label, op_type='modify_label')
                arm_tmp = MultiBoxTarget(arm_anchor_boxes,label_arm,arm_cls_preds,overlap_threshold=.5,\
                                         negative_mining_ratio=3,negative_mining_thresh=.5)
                arm_loc_target = arm_tmp[0] # box offset
                arm_loc_target_mask = arm_tmp[1] # box mask (only 0,1)
                arm_cls_target = arm_tmp[2] #  every anchor' idx
                # print(sum(arm_cls_target[0]))
                # print('---------2222-----------')  

                # 3. ODM predict
                ## 3.1 refine anchor generator originate in ARM
                odm_anchor_boxes = refine_anchor_generator(arm_anchor_boxes,arm_loc_preds)#(batch,h*w*num_anchors[:layers],4)
                # ### debug backward err
                # odm_anchor_boxes = arm_anchor_boxes
                odm_anchor_boxes_bs = nd.split(data=odm_anchor_boxes,axis=0,num_outputs=label.shape[0])# list
                # print('---3 : odm_anchor_boxes_bs shape : {}'.format(odm_anchor_boxes_bs[0].shape))
                # print('---------3333-----------')
                ## 3.2 对当前所有batch的data计算 Target (多个gpu使用)

                odm_loc_target = []
                odm_loc_target_mask = []
                odm_cls_target = []
                label_bs = nd.split(data=label,axis=0,num_outputs=label.shape[0])
                odm_cls_preds_bs = nd.split(data=odm_cls_preds,axis=0,num_outputs=label.shape[0])
                # print('---4 : odm_cls_preds_bs shape: {}'.format(odm_cls_preds_bs[0].shape))
                # print('---4 : label_bs shape: {}'.format(label_bs[0].shape))
                
                for j in range(label.shape[0]):
                    if label.shape[0] == 1:
                        odm_tmp = MultiBoxTarget(odm_anchor_boxes_bs[j].expand_dims(axis=0),label_bs[j].expand_dims(axis=0),\
                                            odm_cls_preds_bs[j].expand_dims(axis=0),overlap_threshold=.5,negative_mining_ratio=2,negative_mining_thresh=.5)
                    ## 多个batch
                    else:
                        odm_tmp = MultiBoxTarget(odm_anchor_boxes_bs[j],label_bs[j],\
                                            odm_cls_preds_bs[j],overlap_threshold=.5,negative_mining_ratio=3,negative_mining_thresh=.5)
                    odm_loc_target.append(odm_tmp[0])
                    odm_loc_target_mask.append(odm_tmp[1])
                    odm_cls_target.append(odm_tmp[2])
                ### concat ,上面为什么会单独计算每张图,odm包含了batch，so需要拆
                odm_loc_target = nd.concat(*odm_loc_target,dim=0)
                odm_loc_target_mask = nd.concat(*odm_loc_target_mask,dim=0)
                odm_cls_target = nd.concat(*odm_cls_target,dim=0)
                
                # 4. negitave filter
                group = nd.Custom(arm_cls_preds,odm_cls_target,odm_loc_target_mask,op_type='negative_filtering')
                odm_cls_target = group[0] #用ARM中的cls过滤后的odm_cls
                odm_loc_target_mask = group[1] #过滤掉的mask为0
                # print('---------4444-----------')
                # 5. calc loss 
                # TODO：add 1/N_arm, 1/N_odm (num of positive anchors)              
                # arm_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
                arm_loss_cls = arm_cls_loss(arm_cls_preds.transpose((0,2,1)),arm_cls_target)
                arm_loss_loc = box_loss(arm_loc_preds,arm_loc_target,arm_loc_target_mask)
                # print('55555 loss->  arm_loss_cls : {} arm_loss_loc {}'.format(arm_loss_cls.shape,arm_loss_loc.shape))
                # print('arm_loss_cls loss : {}'.format(arm_loss_cls))
                # odm_cls_prob = nd.softmax(odm_cls_preds,axis=2)
                tmp = odm_cls_preds.transpose((0,2,1))
                odm_loss_cls = cls_loss(odm_cls_preds.transpose((0,2,1)),odm_cls_target)
                odm_loss_loc = box_loss(odm_loc_preds,odm_loc_target,odm_loc_target_mask)
                # print('66666 loss->  odm_loss_cls : {} odm_loss_loc {}'.format(odm_loss_cls.shape,odm_loss_loc.shape))
                # print('odm_loss_cls loss :{} '.format(odm_loss_cls))
                # print('odm_loss_loc loss :{} '.format(odm_loss_loc))
                # print('N_arm: {} ; N_odm: {} '.format(nd.sum(arm_loc_target_mask,axis=1)/4.0,nd.sum(odm_loc_target_mask,axis=1)/4.0))
                # loss = arm_loss_cls+arm_loss_loc+odm_loss_cls+odm_loss_loc
                loss = 1/(nd.sum(arm_loc_target_mask,axis=1)/4.0) *(arm_loss_cls+arm_loss_loc) + \
                        1/(nd.sum(odm_loc_target_mask,axis=1)/4.0)*(odm_loss_cls+odm_loss_loc)

            sw.add_scalar(tag='loss', value=loss.mean().asscalar(), global_step=global_step)
            global_step += 1
            loss.backward(retain_graph=False)
            # autograd.backward(loss)
            # print(net.collect_params().get('conv4_3_weight').data())
            # print(net.collect_params().get('vgg0_conv9_weight').grad())
            ### 单独测试梯度
            # arm_loss_cls.backward(retain_graph=False)
            # arm_loss_loc.backward(retain_graph=False)
            # odm_loss_cls.backward(retain_graph=False)
            # odm_loss_loc.backward(retain_graph=False)
            
            trainer.step(data.shape[0])
            _loss[0] += nd.mean(odm_loss_cls).asscalar()        
            _loss[1] += nd.mean(odm_loss_loc).asscalar()
            arm_loss[0] += nd.mean(arm_loss_cls).asscalar()            
            arm_loss[1] += nd.mean(arm_loss_loc).asscalar()
            # print(arm_loss)
            arm_cls_prob = nd.SoftmaxActivation(arm_cls_preds, mode='channel')
            odm_cls_prob = nd.SoftmaxActivation(odm_cls_preds, mode='channel')
            out = MultiBoxDetection(odm_cls_prob,odm_loc_preds,odm_anchor_boxes,\
                                        force_suppress=True,clip=False,nms_threshold=.5,nms_topk=400)
            # print('out shape: {}'.format(out.shape))
            if outs is None:
                outs = out
                labels = label
            else:
                outs = nd.concat(outs, out, dim=0)
                labels = nd.concat(labels, label, dim=0)
            box_metric.update([odm_loc_target], [odm_loc_preds * odm_loc_target_mask]) 
        print('-------{} epoch end ------'.format(e))
        train_AP = evaluate_MAP(outs, labels)
        valid_AP, val_box_metric = evaluate_acc(net,valid_data, ctx)
        info["train_ap"].append(train_AP)
        info["valid_ap"].append(valid_AP)
        info["loss"].append( _loss )
        print('odm loss: ',_loss)
        print('arm loss: ',arm_loss)
        if e == 0:
            sw.add_graph(net)
        # grads = [i.grad() for i in net.collect_params().values()]
        # grads_4_3 = net.collect_params().get('vgg0_conv9_weight').grad()
        # sw.add_histogram(tag ='vgg0_conv9_weight',values=grads_4_3,global_step=e, bins=1000 )
        grads_4_2 = net.collect_params().get('vgg0_conv5_weight').grad()
        sw.add_histogram(tag ='vgg0_conv5_weight',values=grads_4_2,global_step=e, bins=1000 )
        # assert len(grads) == len(param_names)
        # logging the gradients of parameters for checking convergence
        # for i, name in enumerate(param_names):
        #     sw.add_histogram(tag=name, values=grads[i], global_step=e, bins=1000)


        # net.export('./Model/RefineDet_MeterDetect') # net
        if (e + 1) % 5 == 0:
            print("epoch: %d time: %.2f cls loss: %.4f,reg loss: %.4f lr: %.5f" % (
            e, time.time() - tic, _loss[0], _loss[1], trainer.learning_rate))
            print("train mae: %.4f AP: %.4f" % (box_metric.get()[1], train_AP))
            print("valid mae: %.4f AP: %.4f" % (val_box_metric.get()[1], valid_AP))
        sw.add_scalar(tag='train_AP',value=train_AP,global_step=e)
        sw.add_scalar(tag='valid_AP',value=valid_AP,global_step=e)
    sw.close()
    if True:
        info["loss"] = np.array(info["loss"])
        info["cls_loss"] = info["loss"][:, 0]
        info["box_loss"] = info["loss"][:, 1]

        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plot("train_ap")
        plot("valid_ap")
        plt.legend(loc="upper right")
        plt.subplot(122)
        plot("cls_loss")
        plot("box_loss")
        plt.legend(loc="upper right")
        plt.savefig('loss_curve.png')

if __name__ == '__main__':
    batch_size = 1
    #1. get dataset and show
    train_data,valid_data,class_names,num_classes = get_iterators(rec_prefix,data_shape,batch_size)

    # train_data.reset()
    ##label数量需要大于等于3
    if train_data.next().label[0][0].shape[0] < 3:
        train_data.reshape(label_shape=(3, 5))
        valid_data.reshape(label_shape=(3, 5))
    # valid_data.sync_label_shape(train_data)

    if False:
        batch = train_data.next()
        images = batch.data[0][:].as_in_context(mx.gpu(0))
        labels = batch.label[0][:].as_in_context(mx.gpu(0))
        show_images(images.asnumpy(),labels.asnumpy(),rgb_mean,std,show_text=True,fontsize=6,MN=(2,4))
        print(labels.shape)

    #2. net initialize
    net = RefineDet(num_classes=1,sizes=sizes,ratios=ratios,normalizations=normalizations,verbose=False,prefix='refineDet_')
    net.hybridize() # MultiBoxPrior cannot support symbol
    
    #3. train
    mytrain(net,1,train_data,valid_data,ctx, 0, 120)
    mkdir_if_not_exist("./Model")
    net.save_parameters("./Model/RefineDet_MeterDetect.param") # weight
    net.export('./Model/RefineDet_MeterDetect.json') # net
