| 类型                | 数据类型                       | 结构                |               |
| ------------------- | ------------------------------ | ------------------- | ------------- |
| classify/regression | database/image/language/series | FNN, CNN, embedding |               |
| **损失函数**        | **评价指标**                   | **优化器**          | **回调**      |
| mse/crossentropy    | accuracy, mae, mse             | adam, RMSprop,      | EarlyStopping |
| **训练集规模**      | **验证集规模**                 | **测试集规模**      |               |
|                     |                                |                     |               |



```python
# template

# import data
# preprocess data
# check data
# visualize
# normalize data
# divide dataset to train and test
# separate label from data

# build model
# configure model
# train model
# visualize: history
# test model

# visualize
```





# 基础回归：预测燃油效率

```python
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import data
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()

# preprocess data
print(dataset.isna().sum())  # 查看各列数据分别有几个N/A值
dataset = dataset.dropna()   # 删除包括N/A值的行

origin = dataset.pop('Origin')         # 将类型值转换为独热码
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0

# divide dataset to train and test
train_dataset = dataset.sample(frac=0.8,random_state=0) # 取样80%
test_dataset = dataset.drop(train_dataset.index)        # 去掉train_dataset中的所有索引的项

# check data
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")  # 绘制多个变量的相关图
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# separate label from data
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# normalize data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# build model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',              # 损失函数: 均方误差
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
print(model.summary())
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense (Dense)                (None, 64)                640       
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                4160      
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 65        
# =================================================================
# Total params: 4,865
# Trainable params: 4,865
# Non-trainable params: 0

# train model
EPOCHS = 1000
# 早期停止防止过度拟合
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0, 
  callbacks=[early_stop])

# visualize: history
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

plot_history(history)

# test model
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
# 损失函数, 平均绝对误差, 均方误差

# visualize: show prediction result
test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# visualize: show error distribution
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
```





# 基本分类：将影评的态度分类

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# import data
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# num_words=10000 保留了训练数据中最常出现的 10000 个单词

# check data
print(len(train_data))     # 25000
print(train_data[0])       # [1, 14, 22, ..., 178, 32], 单词被转换为整数
print(len(train_data[0]))  # 218

# preprocess data, 使所有输入的长度相等 
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,        # 填充0
                                                        padding='post', # 后方填充
                                                        maxlen=256)     # 至256长度

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=256)

vocab_size = 10000
# build model
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))       # 嵌入层
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))    # 全连接层,ReLU激活函数
model.add(keras.layers.Dense(1, activation='sigmoid'))  # 全连接层,Logistic激活函数
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 16)          160000
# _________________________________________________________________
# global_average_pooling1d (Gl (None, 16)                0         
# _________________________________________________________________
# dense (Dense)                (None, 16)                272       
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 17        
# =================================================================
# Total params: 160,289
# Trainable params: 160,289
# Non-trainable params: 0
# _________________________________________________________________

# configure model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# divide dataset to train and test
x_val = train_data[:10000]               # 验证集
partial_x_train = train_data[10000:]     # 训练集
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train model
history = model.fit(partial_x_train,     # 训练集输入
                    partial_y_train,     # 训练集输出
                    epochs=40,           # 迭代次数(训练集的循环使用次数)
                    batch_size=512,      # batch大小(batch计算平均梯度并更新一次参数)
                    validation_data=(x_val, y_val),  # 验证集
                    verbose=1)

# test model
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)

# visualize: history
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

![png](https://tensorflow.google.cn/tutorials/keras/text_classification_files/output_nGoYf2Js-lle_0.png?hl=zh-cn)

出现过拟合。





# 基础分类：识别图片中的服装类型

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# import data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255
# labels are integers, with 0-9 corresponding to the following

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# check data
print(train_images.shape)   # (60000,28,28)
print(train_labels.shape)   # (60000)
print(train_labels)         # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
print(test_images.shape)    # (10000,28,28)
print(test_labels.shape)    # 10000

# visualize
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0

# visualize
plt.figure(figsize=(10,10))          # size of figure displayed
for i in range(25):
    plt.subplot(5,5,i+1)             # draw (i+1)th subplot of 5 rows * 5 columns
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)  # gray image
    plt.xlabel(class_names[train_labels[i]])         # x-axis(bottom) label
plt.show()

# build model
model = keras.Sequential([                        # layer sequence
    keras.layers.Flatten(input_shape=(28, 28)),   # transform (28,28) tensor to (784) tensor
    keras.layers.Dense(128, activation='relu'),   # fully connected to previous layer
    keras.layers.Dense(10)                        # last layer
])

# configure model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # loss function: cross entropy
              metrics=['accuracy'])

# train model
model.fit(train_images, train_labels, epochs=10)

# test model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# build model
probability_model = keras.Sequential([model,  # add another layer
                                         keras.layers.Softmax()])

# make prediction
predictions = probability_model.predict(test_images)

# visualize
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 12  # prediction of 13th test sample
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

num_rows = 5  # prediction of first 5*3 test samples
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
120
plt.show()
.tight_layout()
plt.show()
```





# 基础分类：MNIST



```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# import data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# check data
print(train_images.shape) # (60000, 28, 28)

# visualize: data
plt.figure(figsize=(10,10))          # size of figure displayed
for i in range(25):
    plt.subplot(5,5,i+1)             # draw (i+1)th subplot of 5 rows * 5 columns
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)  # gray image
    plt.xlabel(train_labels[i])         # x-axis(bottom) label
plt.show()    

# process data
train_images = train_images.reshape((60000, 28, 28, 1)) # standard image size: HxWxD
test_images = test_images.reshape((10000, 28, 28, 1))

# normalize data
train_images, test_images = train_images / 255.0, test_images / 255.0

# build model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 26, 26, 32)        320       
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
# _________________________________________________________________
# flatten (Flatten)            (None, 576)               0         
# _________________________________________________________________
# dense (Dense)                (None, 64)                36928     
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                650       
# =================================================================
# Total params: 93,322
# Trainable params: 93,322
# Non-trainable params: 0
# _________________________________________________________________

# configure model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train model
history = model.fit(train_images, train_labels,
                    epochs=10, validation_split = 0.2)

# visualize: history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# test model
results = model.evaluate(test_images, test_labels, verbose=2)
print(results)
```





# 基础分类：CIFAR-10



> 如果在线下载速度慢，可以选择手动下载，步骤如下：
>
> 1. download it from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
> 2. rename it as cifar-10-batches-py.tar.gz
> 3. copy it to ～./keras/datasets/

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# import data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# check data
print(train_images.shape)     # [50000, 32, 32, 3]
print(train_images[0].shape)  # [32, 32, 3]
print(train_labels[0])        # array([6], dtype=uint8)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# normalize data
train_images, test_images = train_images / 255.0, test_images / 255.0

# build model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 输入32x32RGB图片,输出32个特征映射,使用3x3卷积核,每个输出特征映射使用1个偏置
# 参数数量为3x32x(3x3)+32=896
model.add(layers.MaxPooling2D((2, 2)))
# 对每个2x2区块执行最大汇聚
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# 13%2=1,因此丢失了一行一列
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
# 将4x4x64的输出展开为1x1024向量
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary() 
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 30, 30, 32)        896       
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928     
# _________________________________________________________________
# flatten (Flatten)            (None, 1024)              0         
# _________________________________________________________________
# dense (Dense)                (None, 64)                65600     
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                650       
# =================================================================
# Total params: 122,570
# Trainable params: 122,570
# Non-trainable params: 0
# _________________________________________________________________

# configure model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels)) # 有验证集无测试集

# visualize: history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
```



