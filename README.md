garbage-classification
======
一个智能垃圾分类系统  
------
#### 一、运行环境<br>  
  win10+cuda9.1+cudnn7+tensorflow-gpu-1.12.0、pytorch1.4.0+keras-2.2.4 <br>  
   
#### 二、关于库文件：<br>  
  在运行项目的过程中，我们遇到了很多报错，很大一部分是缺少各种库文件，在实现这个项目前，需要配置好环境，并且文件的版本需要对应，否则也会出现各种各样的报错问题（比如tensorflow、pytorch、keras、cudnn之间的版本对应），需要安装tensorflow、pytorch、keras、numpy、torchvision、scipy等文件包<br>
   
#### 三、数据集:<br> 
  我们在查找过程中发现网上大部分用的都是kaggle的垃圾数据集，一共六类，包括纸板（cardboard）、玻璃（glass）、金属（metal）、纸（paper）、塑料（plastic）、其他垃圾（trash），我们合并了找到的其他数据集，对数据集进行了扩充，扩充内容包括胡萝卜（carrot）、肉类（meat）<br> 
 
#### 四、实验步骤：<br> 
  ①对数据集进行训练，生成模型，记录最终准确率（训练模型因为机子配合原因耗时较长）<br> 
  ②运用生成好的模型对图像进行测试，记录结果及验证情况<br> 
  ③调整参数，对模型进行优化<br> 
 
