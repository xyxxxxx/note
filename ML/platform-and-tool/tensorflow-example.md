| 类型                                                 | 数据类型                              | 结构                     |               |
| ---------------------------------------------------- | ------------------------------------- | ------------------------ | ------------- |
| CV/NLP/TS: classify/regression/generation            | database/image/language/series        | FNN, CNN, RNN, embedding |               |
| **损失函数**                                         | **评价指标**                          | **优化器**               | **回调**      |
| mse/binary crossentropy/categorical crossentropy/ctc | accuracy, precision, recall, mae, mse | adam, RMSprop,           | EarlyStopping |
| **训练集规模**                                       | **验证集规模**                        | **测试集规模**           |               |
|                                                      |                                       |                          |               |

```python
# template

# import data
# preprocess data: vectorize data, normalize data, handle abnormal data
# check data
# visualize
# feature engineering
# divide dataset to train and test
# separate label from data

# prepare pretrained embedding module

# build model
# configure model
# train model
# visualize: history
# test model

# visualize
```



| Problem type                            | Last-layer activation | Loss function                  |
| --------------------------------------- | --------------------- | ------------------------------ |
| Binary classification                   | `sigmoid`             | `binary_crossentropy`          |
| Multiclass, single-label classification | `softmax`             | `categorical_crossentropy`     |
| Multiclass, multilabel classification   | `sigmoid`             | `binary_crossentropy`          |
| Regression to arbitrary values          | None                  | `mse`                          |
| Regression to values between 0 and 1    | `sigmoid`             | `mse` or `binary_crossentropy` |



# 预测燃油效率

| 类型         | 数据类型     | 结构           |               |
| ------------ | ------------ | -------------- | ------------- |
| regression   | database     | FNN            |               |
| **损失函数** | **评价指标** | **优化器**     | **回调**      |
| mse          | mae, mse     | RMSprop(0.001) | EarlyStopping |

```python
test_prediction = model.predict(test_x).flatten()import pathlib
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

# feature engineering
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")  # 绘制多个变量的相关图
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

# separate label from data
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# preprocess data: normalize data
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
  optimizer = tf.keras.optimizers.RMSprop(0.001)  # 小批量随机梯度下降法,每个batch的规模
    											  # 为所有样本的0.001
  model.compile(loss='mse',                       # 损失函数: 均方误差
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
## 早期停止防止过度拟合
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
## 损失函数, 平均绝对误差, 均方误差

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

事实上，由于样本规模太小，不应该采用本例的划分验证集的方法，而应该交叉验证，示例：

```python
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                        train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], 
                                           train_targets[(i + 1) * num_val_samples:]], 
                                           axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
    epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
```

使用验证集有几种方法：

1. 使用训练集训练模型的过程中，监视验证集在每个 `epoch` 的损失函数，若此函数在连续几个 `epoch` 中没有改善，则停止训练模型
2. 使用训练集训练模型的过程中，监视验证集在每个 `epoch` 的损失函数，若此函数在连续几个 `epoch` 中没有改善，则记录最优的超参数 `epochs` 的值，将验证集并入训练集，使用该超参数重新训练









# 识别图片中的服装类型

| 类型         | 数据类型     | 结构       |          |
| ------------ | ------------ | ---------- | -------- |
| CV: classify | image        | FNN        |          |
| **损失函数** | **评价指标** | **优化器** | **回调** |
| crossentropy | accuracy     | adam       |          |

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
                            # the first axis is called samples axis
print(train_images.dtype)   # uint8
print(train_labels)         # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
print(test_images.shape)    # (10000,28,28)
print(test_labels.shape)    # 10000

# preprocess data: normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0

# visualize
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

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
    keras.layers.Dense(10, activation='softmax')  # linear classifier
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

# make prediction
predictions = model.predict(test_images)

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
plt.show()
```





# MNIST

| 类型         | 数据类型     | 结构       |          |
| ------------ | ------------ | ---------- | -------- |
| CV: classify | image        | CNN        |          |
| **损失函数** | **评价指标** | **优化器** | **回调** |
| crossentropy | accuracy     | adam       |          |

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# import data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# check data
print(train_images.shape) # (60000, 28, 28)
print(train_labels)       # array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)

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

# preprocess data: normalize data
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
              loss='sparse_categorical_crossentropy', # 将 train_labels 中的值视作标签
              # 等价于 train_labels = to_categorical(train_labels) 
              #       loss='categorical_crossentropy'
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

# visualize: prediction
test_prediction = model.predict(test_images)
```





# CIFAR-10

| 类型         | 数据类型     | 结构       |          |
| ------------ | ------------ | ---------- | -------- |
| CV: classify | image        | CNN        |          |
| **损失函数** | **评价指标** | **优化器** | **回调** |
| crossentropy | accuracy     | adam       |          |

> 如果在线下载速度慢，可以选择手动下载，步骤如下：
>
> 1. download it from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
> 2. rename it as cifar-10-batches-py.tar.gz
> 3. copy it to ～./keras/datasets/

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matpimport tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# import data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# check data
print(train_images.shape) # (60000, 28, 28)
print(train_labels)       # array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)

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

# preprocess data: reshape data
train_images = train_images.reshape((60000, 28, 28, 1)) # standard image size: NumxHxWxD
test_images = test_images.reshape((10000, 28, 28, 1))

# preprocess data: normalize data
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
              loss='sparse_categorical_crossentropy', # 将 train_labels 中的值视作标签
              # 等价于 train_labels = to_categorical(train_labels) 
              #       loss='categorical_crossentropy'
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

# visualize: prediction
test_prediction = model.predict(test_images)
lotlib.pyplot as plt

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

# preprocess data: normalize data
train_images, test_images = train_images / 255.0, test_images / 255.0

# build model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
## 输入32x32RGB图片,输出32个特征映射,使用3x3卷积核,每个输出特征映射使用1个偏置
## 参数数量为3x32x(3x3)+32=896
model.add(layers.MaxPooling2D((2, 2)))
## 对每个2x2区块执行最大汇聚
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
## 13%2=1,因此丢失了一行一列
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
## 将4x4x64的输出展开为1x1024向量
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





# 将影评的态度分类

| 类型                | 数据类型     | 结构           |          |
| ------------------- | ------------ | -------------- | -------- |
| NLP: classify       | language     | RNN, embedding |          |
| **损失函数**        | **评价指标** | **优化器**     | **回调** |
| binary crossentropy | accuracy     | adam           |          |

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# import data
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=10000)
## num_words=10000 保留了训练数据中最常出现的 10000 个单词

# check data
print(len(train_data))     # 25000
print(train_data[0])       # [1, 14, 22, ..., 178, 32], 单词被转换为整数
print(len(train_data[0]))  # 218
print(train_labels[0])     # 1  

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
## model.add(keras.layers.SimpleRNN(32))             # SRN层
model.add(keras.layers.LSTM(64))                     # LSTM层
model.add(keras.layers.Dense(16, activation='relu'))    # 全连接层,ReLU激活函数,分类器
model.add(keras.layers.Dense(1, activation='sigmoid'))  # 全连接层,Logistic激活函数
model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 16)          160000    
# _________________________________________________________________
# lstm (LSTM)                  (None, 64)                20736     
# _________________________________________________________________
# dense (Dense)                (None, 16)                1040      
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 17        
# =================================================================
# Total params: 181,793
# Trainable params: 181,793
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
                    epochs=40,           # 迭代次数(训练集的循环迭代次数)
                    # 每个epochs结束时计算训练集和验证集的 loss & metrics
                    batch_size=512,      # batch大小, 1st batch=train_data[:512]
                    					 # 2nd batch=train_data[512:1024], ...
                                         # 每个batch计算平均梯度并更新一次参数
                    validation_data=(x_val, y_val),  # 验证集
                    verbose=1)

# test model
results = model.evaluate(test_data,  test_labels)
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





# 将影评的态度分类：原始数据处理与嵌入模块

| 类型                | 数据类型     | 结构           |          |
| ------------------- | ------------ | -------------- | -------- |
| NLP: classify       | language     | FNN, embedding |          |
| **损失函数**        | **评价指标** | **优化器**     | **回调** |
| binary crossentropy | accuracy     | adam           |          |

```python
import os

import tensorflow
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

import numpy as np
import matplotlib.pyplot as plt

# preprocess data
imdb_dir = '/Users/xyx/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

## process raw IMDB data
for label_type in ['neg', 'pos']:
	dir_name = os.path.join(train_dir, label_type)
	for fname in os.listdir(dir_name):
		if fname[-4:] == '.txt':
			f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
            	labels.append(0)
            else:
            	labels.append(1)

## vectorize text                
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# prepare pretrained embedding module
## parse GloVe word-embeddings file
glove_dir = '/Users/xyx/Downloads/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

## prepare GloVe word-embeddings matrix
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
	if i < max_words:
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

# build model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# configure model
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False  # 禁用更新

model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['acc'])

# train model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

# visualize
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```





# 生成莎士比亚风格的剧本



```python
import tensorflow as tf
import numpy as np
import os
import time

# import data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# check data
print ('Length of text: {} characters'.format(len(text)))
print(text[:250])

# preprocess data
## 创建字典
vocab = sorted(set(text))
print(vocab)
# ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

## 将字典的各字符映射到整数
char2idx = {u:i for i, u in enumerate(vocab)}
print(char2idx) # {'\n': 0, ' ': 1, ..., 'y': 63, 'z': 64}

## 从文本取样:每32个字符预测下个字符
maxlen = 32
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])
    
x = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
    	x[i, t, char2idx[char]] = 1
    y[i, char2idx[next_chars[i]]] = 1

```





# 天气预报

> https://www.tensorflow.org/tutorials/structured_data/time_series

| 类型           | 数据类型         | 结构           |          |
| -------------- | ---------------- | -------------- | -------- |
| TS: generation | database, series | FNN, CNN, RNN, |          |
| **损失函数**   | **评价指标**     | **优化器**     | **回调** |
|                |                  |                |          |

```python
import os
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# import data
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

# preprocess data
df = pd.read_csv(csv_path)
df = df[5::6]
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

# check data
df.head()
df.describe().transpose()
#                    count         mean        std      min      25%      50%       75%      max
# p (mbar)         70091.0   989.212842   8.358886   913.60   984.20   989.57   994.720  1015.29
# T (degC)         70091.0     9.450482   8.423384   -22.76     3.35     9.41    15.480    37.28
# Tpot (K)         70091.0   283.493086   8.504424   250.85   277.44   283.46   289.530   311.21
# Tdew (degC)      70091.0     4.956471   6.730081   -24.80     0.24     5.21    10.080    23.06
# rh (%)           70091.0    76.009788  16.474920    13.88    65.21    79.30    89.400   100.00
# VPmax (mbar)     70091.0    13.576576   7.739883     0.97     7.77    11.82    17.610    63.77
# VPact (mbar)     70091.0     9.533968   4.183658     0.81     6.22     8.86    12.360    28.25
# VPdef (mbar)     70091.0     4.042536   4.898549     0.00     0.87     2.19     5.300    46.01
# sh (g/kg)        70091.0     6.022560   2.655812     0.51     3.92     5.59     7.800    18.07
# H2OC (mmol/mol)  70091.0     9.640437   4.234862     0.81     6.29     8.96    12.490    28.74
# rho (g/m**3)     70091.0  1216.061232  39.974263  1059.45  1187.47  1213.80  1242.765  1393.54
# wv (m/s)         70091.0     1.702567  65.447512 -9999.00     0.99     1.76     2.860    14.01
# max. wv (m/s)    70091.0     2.963041  75.597657 -9999.00     1.76     2.98     4.740    23.50
# wd (deg)         70091.0   174.789095  86.619431     0.00   125.30   198.10   234.000   360.00

# preprocess data
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# feature engineering
# wind: change wv, wd to wx, wy
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')
wd_rad = df.pop('wd (deg)')*np.pi / 180
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
ax.axis('tight')
plt.show()

# time
timestamp_s = date_time.map(datetime.datetime.timestamp)
day = 24*60*60
year = (365.2425)*day
df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
# 周期函数的傅里叶展开的前两项
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# 如果未知气象参数与时间的周期关系,可以做频域分析
# fft = tf.signal.rfft(df['T (degC)'])
# f_per_dataset = np.arange(0, len(fft))
# n_samples_h = len(df['T (degC)'])
# hours_per_year = 24*365.2524
# years_per_dataset = n_samples_h/(hours_per_year)
# f_per_year = f_per_dataset/years_per_dataset
# plt.step(f_per_year, np.abs(fft))
# plt.xscale('log')
# plt.ylim(0, 400000)
# plt.xlim([0.1, max(plt.xlim())])
# plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
# _ = plt.xlabel('Frequency (log scale)')

# divide dataset to train, validation and test
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

# preprocess data: normalize data
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

## implement generator of window
def generator(data, lookback, delay, shuffle=False, batch_size=128):
    max_index = len(data) - delay - 1
    i = lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                lookback, max_index, size=batch_size)
        else:
            	if i + batch_size >= max_index:
            		i = lookback
            	rows = np.arange(i, min(i + batch_size, max_index))
            	i += len(rows)
        samples = np.zeros((len(rows),
         					lookback,
           					data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
           	indices = range(rows[j] - lookback, rows[j])
           	samples[j] = data.iloc[indices]
           	targets[j] = data.iloc[rows[j] + delay][1]
        yield samples, targets

train_gen = generator(train_df,
                      lookback=120,
                      delay=24,
                      shuffle=True,
                      batch_size=128)
val_gen = generator(val_df,
                    lookback=120,
                    delay=24,
                    batch_size=128)
test_gen = generator(test_df,
                      lookback=120,
                      delay=24,
                      batch_size=128)
val_steps = (len(val_df)-120-127-1)//128+1
test_steps = (len(test_df)-120-127-1)//128+1

## introduce naive method as baseline
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    return np.mean(batch_maes)

norm_celsius_mae = evaluate_naive_method()
print(norm_celsius_mae)                 # 0.319
print(norm_celsius_mae * train_std[1])  # 2.758 degree

# build model
model = keras.Sequential()
model.add(keras.layers.GRU(64, 
                            dropout=0.1,
                            recurrent_dropout=0.5,
                            return_sequences=True,
                            input_shape=(None, df.shape[-1])))
                            ## 尽管这里序列的长度是确定的(120),但也不必传入
model.add(keras.layers.GRU(64,
                            activation='relu',
                            dropout=0.1,
                            recurrent_dropout=0.5))
# 可以使用 LSTM, GRU
model.add(keras.layers.Dense(1))

# configure model
model.compile(optimizer='rmsprop', loss='mae')

# train model
history = model.fit(train_gen,
                    steps_per_epoch=500, 
                    ## 每个epoch抽选500个batch
                    epochs=10,
                    validation_data=val_gen,
                    validation_steps=val_steps)

# visualize
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# test model
results = model.evaluate(test_gen, steps=test_steps)

```

过拟合问题十分严重。