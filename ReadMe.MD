# My Detection
参考 https://blog.csdn.net/qq_42325732/article/details/82688585

本工程使用Python3.6+mxnet gluon (1.3.0)。



## 1. install
```bash
pip install mxboard
# if cuda 8.0 is installed
pip install mxnet-cu80 --pre --upgrade
```
## 2. rec
  打标好自己的数据后，用[generate_rec](https://github.com/wk738126046/RefineDet/blob/master/generate_rec.py)生成数据集

## 3. demo
```bash
python train.py
```
