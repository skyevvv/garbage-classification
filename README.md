garbage-classification
======
一个智能垃圾分类系统  
------
#### 一、主要功能<br> 
  对上传的垃圾图片进行识别，并返回分类结果（干垃圾、湿垃圾、有害垃圾、可回收垃圾）<br> 
#### 二、实现原理<br> 
![](https://raw.githubusercontent.com/skyevvv/garbage-classification/master/function.png)<br>
  该垃圾分类项目采用深度学习--图像处理模块，采用的是Sequential 序贯模型，序贯模型是函数式模型的简略版，为最简单的线性、从头到尾的结构顺序，不分叉。Sequential模型的基本组件包括① model.add，添加层；② model.compile,模型训练的BP模式设置；③ model.fit，模型训练参数设置 + 训练；④ 模型预测。<br>

#### 三、运行环境<br>  
  win10+cuda9.1+cudnn7+tensorflow-gpu-1.12.0、pytorch1.4.0+keras-2.2.4 <br>  
  
#### 四、关于库文件：<br>  
  在运行项目的过程中，我们遇到了很多报错，很大一部分是缺少各种库文件，在实现这个项目前，需要配置好环境，并且文件的版本需要对应，否则也会出现各种各样的报错问题（比如tensorflow、pytorch、keras、cudnn之间的版本对应），需要安装tensorflow、pytorch、keras、numpy、torchvision、scipy等文件包<br>
   
#### 五、数据集:<br> 
  我们在查找过程中发现网上大部分用的都是kaggle的垃圾数据集，一共六类，包括纸板（cardboard）、玻璃（glass）、金属（metal）、纸（paper）、塑料（plastic）、其他垃圾（trash） [数据集下载地址](https://www.kaggle.com/asdasdasasdas/garbage-classification)
 
#### 六、模型训练基本步骤：<br> 
  ①对数据集进行训练，生成模型，记录最终准确率（训练模型因为机子配置原因耗时较长）<br> 
  ②运用生成好的模型对图像进行测试，查看图像识别结果及垃圾分类情况，记录结果及验证情况<br> 
  ③调整参数，对模型进行优化<br> 
  
#### 七、代码运行：<br> 
  运行 garbage_ui.py 文件在前端界面进行图片上传、识别即可。
 
