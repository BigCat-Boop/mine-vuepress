---
title: Yolov5
---
# Yolov5解析与总结
## 1.YOLO v5中的改进点:
:::tip 优缺点
* 输入端: 数据增强(<span style="color:red">Mosaic Augmentation</span>、MixUp、CutMix、Bluring、Label Smoothing等)、<span style="color:red">自适应锚框计算、自适应的图像缩放</span>

* BackBone骨干网络: Focus结构、CSP+DarkNet等

* Neck: FPN+PAN、SPP等(FPN+PAN是特征金字塔网络)

* Prediction:更换边框回归损失函数(<span style="color:red">GloU_LOSS</span>); // yolov4中是CLoU_LOSS

* 源码网址: https://github.com/ultralytics/yolov5

:::

这块内容中得标红得部分 就是和yolov4对比后 不一样的部分
<img src="/yolov4.png"/>

## 2.YOLOv5的结构图
<img src="/yolov5简易结构.png"/>

这是根据yolov5代码中的yolov5s.yaml文件中显示 画出的yolov5结构图

但是这里面就是C3的模块的个数 需要看depth参数的值

## 3.部分层解析
### 3.1 csp层
是跨阶级连接(有点类似与resnet)

1.csp1 有残差连接

2.csp2 没有残差连接 其他都和csp1差不多

### 3.2 SPP层的工作原理(空间金字塔池化)

SPP模块通过在特征图上应用多个不同大小的池化窗口来实现。具体步骤如下：

* 1. 池化操作：在输入特征图上应用多个不同大小的池化窗口（如1x1、3x3、5x5、7x7），对每个窗口大小进行最大池化操作。

* 2. 特征拼接：将不同池化窗口得到的特征图进行拼接，形成一个包含多尺度信息的特征图。

* 3. 输出特征：拼接后的特征图保留了输入特征图的空间维度，同时包含了不同尺度的上下文信息

### 3.3 SPP在YOLOv5中的实现
在YOLOv5的配置文件中，可以看到SPP模块的定义和应用。例如：

```python
# SPP module
- [5, 1024, 1, SPP, [5, 9, 13]]

```

### 3.4 为什么每一层要去判断输出通道是不是8的倍数呢?

这是因为对于模型来说 每一层输出为8的倍数的话 这样对于GPU来说 会更加容易计算

## 4.yolov5输入端
* 1.Mosaic数据增强

* 2.自适应锚框计算

  YOLOv5中将计算先验框高宽比的程序代码集成到整个训练过程中(聚类代码)。

* 3.自适应图片缩放

  一般情况下，我们会将图像缩放到常用尺寸，然后在进行后续网络的执行;但是如果填充的比较多的话，则会存在较多的冗余信息，影响推理速度。在YOLO v5中做了自适应的处理，以希望添加最少像素的黑边。

  YOLO v5里面填充的边是灰色，并且仅在推理预测的时候添加。

## 5.YOLOv5代码说明

<span style="font-weight: 900">1.在models文件夹下得yolov5x.yaml中</span>

```python
depth_multiple: 0.33
width_mutiple: 0.25
```

其实这几个文件得差别就在于 depth_multiple和width_mutiple而已 
这两个参数能够决定网络的参数数量以及卷积核的个数 通过这两个来决定的

<span style="font-weight: 900">2.models文件夹下的common.py放的就是 我们每一层网络结构的代码</span>

其实这个文件中 做的事情 其实就是对我们的数据去做一些拼接和合并

<span style="font-weight: 900">3.hubconf.py里面定义了很多的模型</span>

在models文件夹里面也有hub文件夹 里面的网络模型结构 和hubconf.py中的是逐个对应的

<span style="font-weight: 900">4.flask_rest_api文件夹下 就是flask接口应用 可以直接运行</span>
<img src="/flaskapi1.png"/>
<br />
<img src="/flaskapi2.png"/>


:::tip 文件修改步骤

* 1.需要修改restapi.py中的这一段 把它改成本地加载模型.pt文件

* 2.把图片路径替换一下 

* 3.运行restapi.py文件和example_request.py

* 4.就会打印出来我们的测试请求

(这里会报一个问题 就是当我们的torch版本和torchvision版本的兼容度不匹配时)

会报错 解决方法: 我们pytorch官网查看版本兼容 然后重新卸载 再重新安装

:::

<span style="font-weight: 900">5.train.py 里面参数详解</span>

* 1.--weights

给定模型文件, 如果本地路径不存在, 直接从网络上下载, 如果给定为空字符串,表示不适用训练好的模型=> 这就是参数迁移, 从模型文件的参数来进行初始化

* 2.EMA
```python
这句代码的意义在于 指数移动平均
考虑历史值对于参数的影响 给训练过程带来帮助

ema = ModelEMA(model) if RANK in [-1, 0] else None

```

## 6.YOLOv5网络模型结构解析
```python
# Ultralytics YOLOv5 🚀, AGPL-3.0 license

# Parameters
nc: 80 # number of classes
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.50 # layer channel multiple
anchors: # 给定先验框大小
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  # from 就是给定当前输出是那一层的输出，-1就表示上一层
  # number 给定当前层重复的次数，实际重复次数是和参数depth_multiple有关的
  # module 给定当前层具体模块，具体查着common.py文件
  # args 模块参数列表
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, C3, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 6, C3, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, C3, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 3, C3, [1024]],
    [-1, 1, SPPF, [1024, 5]], # 9
  ]

# YOLOv5 v6.0 head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, C3, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, C3, [256, False]], # 17 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]

Model summary: 214 layers,7235389 parameters, 7235389 gradients, 16.6 GFLOPs
```


## 7.yolo.py文件 是存放生成模型的类的代码的

关键点: 通过eval函数把字符串变成真正的数据结构