# 线性回归

```python
import numpy as np
import pandas as pd
import tensorflow as tf


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

# wash data
print(dataset.isna().sum())  # 查看各列数据分别有几个N/A值
dataset = dataset.dropna()   # 删除包括N/A值的行

# preprocess data
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

# draw history
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

# show prediction result
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

# show error distribution
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

# check data size
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
# model structure
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

# model setting
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# set validation set
x_val = train_data[:10000]               # 验证集
partial_x_train = train_data[10000:]     # 训练集
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train
history = model.fit(partial_x_train,     # 训练集输入
                    partial_y_train,     # 训练集输出
                    epochs=40,           # 迭代次数(训练集的循环使用次数)
                    batch_size=512,      # batch大小(batch计算平均梯度并更新一次参数)
                    validation_data=(x_val, y_val),  # 验证集
                    verbose=1)

# test
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)

# draw training and validation loss 
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





# 基础分类：识别图片的手写体数字

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# import data
handwrite_mnist = keras.datasets
```







# 基础分类：识别图片的服装类型

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

# check data size
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

# model structure 
model = keras.Sequential([                        # layer sequence
    keras.layers.Flatten(input_shape=(28, 28)),   # transform (28,28) tensor to (784) tensor
    keras.layers.Dense(128, activation='relu'),   # fully connected to previous layer
    keras.layers.Dense(10)                        # last layer
])

# model setting
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # loss function: cross entropy
              metrics=['accuracy'])

# train
model.fit(train_images, train_labels, epochs=10)

# test
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# classifier
probability_model = keras.Sequential([model,  # add another layer
                                         keras.layers.Softmax()])

# prediction
predictions = probability_model.predict(test_images)

# draw function
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



