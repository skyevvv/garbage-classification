import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.applications.imagenet_utils import preprocess_input
import glob, os, random
from keras.models import model_from_json
from keras.preprocessing import image
from keras.optimizers import SGD
# labels = []
root_path = '.'
labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}


class Model:
    def __init__(self):
        self.model = None

    def build_model(self):
        self.model = Sequential()
        #Sequential模型，顾名思义，就是多个网络层的线性堆叠,建立模型有两种方式：一是向layer添加list的方式，二是通过.add()方式一层层添加（一个add为一层）

        #二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。例如input_shape = (128,128,3)代表128*128的彩色RGB图像
        #filters：卷积核的数目（即输出的维度）
        #kernel_size：单个整数或由两个整数构成的list / tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度
        #padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同，因为卷积核移动时在边缘会出现大小不够的情况
        #activation：激活函数，为预定义的激活函数名（参考激活函数），或逐元素（element-wise）的Theano函数

        self.model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)))

        #空间池化（也叫亚采样或下采样）降低了每个特征映射的维度，但是保留了最重要的信息。空间池化可以有很多种形式：最大(Max)，平均(Average)，求和(Sum)等等。
        # 这里使用最大池化，我们定义了空间上的邻域（2x2的窗）并且从纠正特征映射中取出窗里最大的元素。池化可以是输入表征更小更易操作，减少网络中的参数与计算数量，防止过拟合，帮助我们获得不因尺寸二改变的等效图片表征

        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))

        #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小
        self.model.add(Flatten())  # 扁平化参数

        #Dense层就是所谓的全连接神经网络层
        #units:该层有几个神经元 activation:该层使用的激活函数
        #定义了一个有64个节点，使用relu激活函数的神经层
        #ReLU是线性修正，是purelin的折线版。它的作用是如果计算出的值小于0，就让它等于0，否则保持原来的值不变。
        # 这是一种简单粗暴地强制某些数据为0的方法，使得网络可以自行引入稀疏性，同时大大地提高了训练速度
        self.model.add(Dense(64, activation='relu'))
        #Softmax 在机器学习和深度学习中有着非常广泛的应用，尤其在处理多分类。
        # 线性分类器模型最后输出层包含了6个输出值，经过Softmax处理后，数值转化为不同类别之间的相对概率，相对概率越高，预测为哪一类的可能性越大
        self.model.add(Dense(6, activation='softmax'))
        #输出模型各层的参数状况
        self.model.summary()

    def train_model(self):
        # 优化器, 主要有Adam、sgd、rmsprop等方式。
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #在训练模型之前，我们需要通过compile来对学习过程进行配置。compile接收三个参数：
        #优化器optimizer：已预定义的优化器名
        #损失函数loss：最小化的目标函数，它可为预定义的损失函数
        #指标列表metrics：对分类问题，我们一般将该列表设置为metrics=['accuracy']。
        # 指标可以是一个预定义指标的名字，也可以是一个用户定制的函数。指标函数应该返回单个张量，或一个完成metric_name - > metric_value映射的字典。

        #categorical_crossentropy(交叉熵损失函数) 交叉熵是用来评估当前训练得到的概率分布与真实分布的差异情况
        #adam一种可以替代传统随机梯度下降(SGD)过程的一阶优化算法,它能基于训练数据迭代地更新神经网络权重
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',  # sgd 'adam'
                           metrics=['accuracy'])
        # 自动扩充训练样本
        #ImageDataGenerator位于keras.preprocessing.image模块当中,可用于做数据增强,或者仅仅用于一个批次一个批次的读进图片数据
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # 数据缩放，把像素点的值除以255，使之在0到1之间
            shear_range=0.1,  # 错切变换角度
            zoom_range=0.1,  # 随机缩放范围
            width_shift_range=0.1,#浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
            height_shift_range=0.1,#浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度。
            horizontal_flip=True,#布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
            vertical_flip=True,#布尔值，进行随机竖直翻转。
            validation_split=0.1#多少数据用于验证集
        )
        # 生成验证集
        val_datagen = ImageDataGenerator(
            rescale=1. / 255, validation_split=0.1)

        # 读训练集图片
        train_generator = train_datagen.flow_from_directory(
            root_path + '/dataset-resized',#子文件路径
            # 整数元组 (height, width)，默认：(300, 300)。 所有的图像将被调整到的尺寸。
            target_size=(300, 300),#输出的图片的尺寸
            # 一批数据的大小
            batch_size=16,
            # "categorical", "binary", "sparse", "input" 或 None 之一。
            # 默认："categorical",返回one-hot 编码标签。
            class_mode='categorical',
            subset='training',
            seed=0)

        #读验证集图片
        val_generator = val_datagen.flow_from_directory(
            root_path + '/dataset-resized',
            target_size=(300, 300),
            batch_size=16,
            class_mode='categorical',
            subset='validation',
            seed=0)

        # 编译模型
        try:
            history_fit = self.model.fit_generator(train_generator,
                                                   epochs=100,  # 迭代总轮数
                                                   steps_per_epoch=2276 // 32,  # generator 产生的总步数/（批次样本）
                                                   validation_data=val_generator,  #验证数据的生成器
                                                   validation_steps=251 // 32)# 当validation_data为生成器时，本参数指定验证集的生成器返回次数

            with open(root_path + "/model/history_fit.json", "w") as json_file:
                json_file.write(str(history_fit))

            acc = history_fit.history['acc']
            val_acc = history_fit.history['val_acc']
            loss = history_fit.history['loss']
            val_loss = history_fit.history['val_loss']

            epochs = range(1, len(acc) + 1)
            plt.figure("acc")
            plt.plot(epochs, acc, 'r-', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='validation acc')
            plt.title('The comparision of train_acc and val_acc')
            plt.legend()
            plt.show()

            plt.figure("loss")
            plt.plot(epochs, loss, 'r-', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='validation loss')
            plt.title('The comparision of train_loss and val_loss')
            plt.legend()
            plt.show()
        except StopIteration:
            pass

    def save_model(self):
        model_json = self.model.to_json()
        with open(root_path + '/model/model_json.json', "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(root_path + '/model/model_weight.h5')
        self.model.save(root_path + '/model/model.h5')
        print('model saved')

    def load_model(self):
        json_file = open(root_path + '/model/model_json.json')  # 加载模型结构文件
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)  # 结构文件转化为模型
        # 加载权重
        model.load_weights(root_path + '/model/model_weight.h5')  # h5文件保存模型的权重数据
        return model


def generate_result(result):
    for i in range(6):
        if (result[0][i] == 1):
            return labels[i]


if __name__ == '__main__':
    model = Model()
    model.build_model()
    print('model built')
    model.train_model()
    print('model trained')
    model.save_model()
    print('model saved')
    # 已经有模型，直接加载，注释掉上面三个函数
    model1 = model.load_model()
    print('model loaded')
    img_path = "./dataset-resized/glass/glass2.jpg"
    # 把图片转换成为numpy数组
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model1.predict(img)
    print(generate_result(result))