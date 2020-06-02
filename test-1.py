import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.applications.imagenet_utils import preprocess_input
import glob, os, random
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from keras.optimizers import SGD
root_path = 'D:/AI/cnn-garbage'
labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

class Model:
    def __init__(self):
        self.model = None

    def build_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)))
        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2))

        self.model.add(Flatten())  # 扁平化参数
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(6, activation='softmax'))
        self.model.summary()

    def train_model(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',#sgd 'adam'
                           metrics=['accuracy'])
        # 自动扩充训练样本

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # 数据缩放，把像素点的值除以255，使之在0到1之间
            shear_range=0.1,  # 错切变换角度
            zoom_range=0.1,  # 随机缩放范围
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            horizontal_flip=True,
            vertical_flip = True,
            validation_split = 0.1
        )
        # 归一化验证集

        val_datagen = ImageDataGenerator(
            rescale=1. / 255,validation_split=0.1)

        train_generator = train_datagen.flow_from_directory(
            root_path+'/dataset-resized',
            target_size=(300, 300),
            batch_size=16,
            class_mode='categorical',
            subset='training',
            seed=0)
        val_generator = val_datagen.flow_from_directory(
            root_path+'/dataset-resized',
            target_size=(300, 300),
            batch_size=16,
            class_mode='categorical',
            subset='validation',
            seed=0)
       # global labels
       # labels = (train_generator.class_indices)
       # labels = dict((v, k) for k, v in labels.items())
        try:
            history_fit = self.model.fit_generator(train_generator,
                                              epochs=100,  # 迭代总轮数
                                              steps_per_epoch=2276//32,  # generator 产生的总步数（批次样本）
                                              validation_data=val_generator,  # 验证数据的生成器
                                              validation_steps=251//32)
            with open(root_path + "/history_fit.json", "w") as json_file:
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
        model_json=self.model.to_json()
        with open(root_path+'/model_json.json', "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(root_path+'/model_weight.h5')
        self.model.save(root_path+'/model.h5')
        print('model saved')

    def load_model(self):
        json_file = open(root_path + '/model_json.json')  # 加载模型结构文件
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)  # 结构文件转化为模型
        # 加载权重
        model.load_weights(root_path + '/model_weight.h5')  # h5文件保存模型的权重数据
        return model

def generate_result(result):
    for i in range(6):
        if(result[0][i] == 1):
            return labels[i]

if __name__=='__main__':
    model=Model()
    #model.build_model()
    print('model built')
    #model.train_model()
    print('model trained')
   # model.save_model()
    print('model saved')
    #已经有模型，直接加载，注释掉上面三个函数
    model1 = model.load_model()
    print('model loaded')
    img_path = "D:/AI/cnn-garbage/dataset-resized/cardboard/cardboard2.jpg"
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model1.predict(img)
    print(generate_result(result))
