'''
dataset transform: xml to rec
img path : img_root
img xml : data_root/labels
'''

import os
from PIL import Image
from utils import parse_voc_xml,mkdir_if_not_exist

class_names = ['meter']
# class_names = ['sock','wire','slipper']

print(os.getcwd())
#resize image
os.environ['data_root'] = data_root = './dataset/data_320'
os.environ['im2rec']= "python /home/wk/anaconda3/lib/python3.6/site-packages/mxnet/tools/im2rec.py"
os.environ['img_root'] = img_root = '/home/wk/wk_ws/myTest/EData'

# resize = (512,512)
resize = (320,320)
os.environ['resize'] = resize_str = str(resize[0]) + '_' + str(resize[1])
# for imgName in os.listdir(img_root):
#     imgPath = img_root + '/' + imgName
#     img = Image.open(imgPath)
#     img = img.resize(resize,Image.BILINEAR)
#     print(data_root + '/' + 'img%d_%d'%(resize[0],resize[1]))
#     mkdir_if_not_exist(data_root + '/' + 'img%d_%d'%(resize[0],resize[1]))
#     img.save(data_root + '/'+'img%d_%d'%(resize[0],resize[1]) + '/' + imgName)

#generate lst
mkdir_if_not_exist(data_root+'/rec')
os.system('$im2rec --list --train-ratio 0.9 ${data_root}/rec/img_$resize ${data_root}/img%d_%d'%(resize[0],resize[1]))

#modify lst
'''
lst格式:
    idx \t header_width \t label_width \t [labels] \t filename
    label_width为每个label的宽度。
    header_width为label之前idx之后的数据宽度，一般为2,指label_width 和 label_data两类
    label：class_idx \t xmin \t ymin \t xmax \t ymax (anchor数量)
'''

new_lst_content = ''
with open(data_root + '/rec/img_%s_train.lst'%(resize_str)) as f:
    contents = f.read().split('\n')
    for content in contents:
        if content == '':
            break
        content = content.split('\t')
        idx = content[0]
        file = content[-1][:-3] + 'xml'
        # analyse .xml files
        bndboxs, names, filename = parse_voc_xml(data_root + '/labels/' + file)
        data = idx + '\t2\t5\t'
        for bndbox, name in zip(bndboxs,names):
            data += '%d\t%f\t%f\t%f\t%f\t'%(class_names.index(name),bndbox[0],bndbox[1],bndbox[2],bndbox[3])
            # data += filename + '\t'
            print(data)
        new_lst_content += data + filename + '\n'

# save analysis data
with open(data_root+'/rec/img_%s_train.lst'%(resize_str),'w') as f:
    f.write(new_lst_content)

new_lst_content1 = ''
with open(data_root+'/rec/img_%s_val.lst'%(resize_str)) as f:
    contents = f.read().split('\n')
    for content in contents:
        if content == '':
            break
        content = content.split('\t')
        idx = content[0]
        file = content[-1][:-3] + 'xml'

        bndboxs,names,filename = parse_voc_xml(data_root+'/labels/'+file)
        data = idx + '\t2\t5\t'
        for bndbox,name in zip(bndboxs,names):
            data += '%d\t%f\t%f\t%f\t%f\t'%(class_names.index(name),bndbox[0],bndbox[1],bndbox[2],bndbox[3])
            # data += filename + '\t'
            print(data)
        new_lst_content1 += data + filename +'\n'

with open(data_root+'/rec/img_%s_val.lst'%(resize_str),'w') as f:
    f.write(new_lst_content1)

# generate rec file
os.system('$im2rec --num-thread 6 --pass-through --pack-label $data_root/rec/img_$resize $data_root/img$resize --encoding=.jpg --quality 100')
