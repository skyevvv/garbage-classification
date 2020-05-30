import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential

import glob, os, random
base_path = 'D:/AI/cnn-garbage/dataset-resized'
#glob.glob获取指定目录下的所有图片
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
print(len(img_list))#数据集一共2527个数据
#随机展示六张图片
for i, img_path in enumerate(random.sample(img_list, 6)):
    img = load_img(img_path)
    img = img_to_array(img, dtype=np.uint8)

 #   plt.subplot(2, 3, i + 1)
 #   plt.imshow(img.squeeze())

#对数据进行分组
train_datagen = ImageDataGenerator(
    rescale=1. / 225, shear_range=0.1, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
    vertical_flip=True, validation_split=0.1)

test_datagen = ImageDataGenerator(
    rescale=1. / 255, validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=16,
    class_mode='categorical', subset='training', seed=0)

validation_generator = test_datagen.flow_from_directory(
    base_path, target_size=(300, 300), batch_size=16,
    class_mode='categorical', subset='validation', seed=0)

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

print(labels)
# 0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'

#模型的建立和训练
#MaxPooling2D,epoch=50
model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=
           , kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),
    #Flatten层用来将输入“压平”，即把多维的输入一维化，
    # 常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    Dense(64, activation='relu'),
    #units=64是正整数，输出空间维度。
    #Dense 实现以下操作：output = activation(dot(input, kernel) + bias)
    # 其中 activation 是按逐个元素计算的激活函数，kernel 是由网络层创建的权值矩阵，
    # 以及 bias 是其创建的偏置向量 (只在 use_bias 为 True 时才有用)。
    #如果该层的输入的秩大于2，那么它首先被展平然后再计算与 kernel 的点乘。
    Dense(6, activation='softmax')
    #units=6，,是正整数，输出空间维度。
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#categorical_crossentropy输出张量与目标张量之间的分类交叉熵-∑p(x)logq(x)。
#p代表正确答案，q代表的是预测值。交叉熵值越小，两个概率分布越接近。
history_fit = model.fit_generator(train_generator,
                                 epochs=100, #迭代总轮数
                                 steps_per_epoch=2276//32,#generator 产生的总步数（批次样本）
                                  validation_data=validation_generator,# 验证数据的生成器
                                  validation_steps=251//32)

with open(base_path + "/history_fit.json", "w") as json_file:
    json_file.write(str(history_fit))
acc = history_fit.history['acc']
val_acc = history_fit.history['val_acc']
loss = history_fit.history['loss']

epochs = range(1, len(acc) + 1)
plt.figure("acc")
plt.plot(epochs, acc, 'r-', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('The comparision of train_acc and val_acc')
plt.legend()
plt.show()

plt.figure("loss")
plt.plot(epochs, loss, 'r-', label='loss')
plt.title('The comparision of loss')
plt.legend()
plt.show()

#结果展示
#下面我们随机抽取validation中的16张图片，展示图片以及其标签，并且给予我们的预测。 我们发现预测的准确度还是蛮高的，对于大部分图片，都能识别出其类别。
test_x, test_y = validation_generator.__getitem__(1)

preds = model.predict(test_x)

plt.figure(figsize=(16, 16))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.title('pred:%s / truth:%s' % (labels[np.argmax(preds[i])], labels[np.argmax(test_y[i])]))
    plt.imshow(test_x[i])
    plt.show()
    print(labels[np.argmax(preds[i])])