# random——生成伪随机数

`random` 模块实现了各种分布的伪随机数生成器。

## 重现

### seed()

初始化随机数生成器。

```python
random.seed(a=None, version=2)
# a        如果被省略或为`None`,则使用当前系统时间;如果操作系统提供随机源,则使用它们而不是系统时间
#          如果为`int`类型,则直接使用
```

```python
>>> random.seed(0)
>>> random.random()
0.8444218515250481
```

### getstate()

捕获生成器的当前内部状态的对象并返回，这个对象用于传递给 `setstate()` 以恢复状态。

### setstate()

将生成器的内部状态恢复到 `getstate()` 被调用时的状态。

```python
>>> random.seed(0)
>>> state = random.getstate()
>>> random.setstate(state)
>>> random.random()
0.8444218515250481
>>> random.setstate(state)
>>> random.random()
0.8444218515250481
```

## 随机整数

### randrange()

```python
random.randrange(stop)
random.randrange(start, stop[, step])
```

从 `range(0,stop)` 或 `range(start,stop,step)` 返回一个随机选择的元素。

### randint()

```python
random.randint(a, b)
```

返回随机整数 *N* 满足 `a <= N <= b`。相当于 `randrange(a, b+1)`。

## 随机序列

### choice()

从非空序列中随机选择一个元素并返回。如果序列为空，则引发 `IndexError`。

```python
>>> a = list(range(10))
>>> random.choice(a)
9
>>> random.choice(a)
5
```

### choices()

从非空序列中（有放回地）随机选择多个元素并返回。如果序列为空，则引发 `IndexError`。如果指定了权重，则根据权重进行选择。

```python
>>> a = list(range(5))
>>> random.choices(a, k=3)
[2, 0, 2]
>>> random.choices(a, k=3)
[2, 1, 4]
>>> random.choices(a, [70.0, 22.0, 6.0, 1.5, 0.5], k=3)                    # 相对权重
[0, 0, 0]
>>> random.choices(a, cum_weights=[70.0, 92.0, 98.0, 99.5, 100.0], k=3)    # 累积权重
[1, 0, 0]
```

### shuffle()

随机打乱序列。

```python
random.shuffle(x[, random])
# x           序列
# random      一个不带参数的函数,返回[0.0,1.0)区间内的随机浮点数.默认为函数`random()`.
```

```python
>>> a = list(range(10))
>>> random.shuffle(a)              # 原位操作
>>> a
[8, 9, 1, 2, 5, 3, 7, 4, 0, 6]
```

### sample()

从非空序列中（无放回地）随机选择多个元素并返回。如果序列长度小于样本数量，则引发 `IndexError`。

```python
>>> random.sample(range(10), k=5)
[4, 7, 1, 9, 3]
```

要从一系列整数中选择样本，请使用 `range()` 对象作为参数，这种方法特别快速且节省空间：

```python
>>> random.sample(range(10000000), k=60)
[9787526, 3664860, 8467240, 2336625, 4728454, 2344545, 1590996, 4202798, 8934935, 2465603, 5203412, 1656973, 1237192, 5539790, 7921240, 9392115, 1689485, 5935633, 7284194, 5304900, 3430567, 9269809, 8002896, 7427162, 8746862, 4370335, 1044878, 9205646, 235580, 1564842, 6691148, 19173, 8280862, 5589080, 4092145, 5456023, 1056700, 3205573, 9521250, 3719574, 4003310, 2390659, 9109859, 7515682, 1530349, 1349656, 5369625, 8521829, 8208870, 1829687, 5057437, 9248729, 4883691, 2093976, 9184534, 5582627, 9064454, 3409161, 9180997, 9858578]
```

## 实值分布

### random()

返回服从 $[0.0,1.0)$ 区间内均匀分布的随机浮点数。

```python
>>> random.random()
0.13931343809011631
```

### uniform()

```python
random.uniform(a, b)
```

返回服从 [*a*, *b*] 区间（*a* <= *b*）或 [*b*, *a*] 区间（*b* <= *a*）内均匀分布的随机浮点数。

```python
>>> random.uniform(60, 80)
79.59813867742345
```

### gauss()

```python
random.gauss(mu, sigma)
```

返回服从平均值为 *mu*、标准差为 *sigma* 的正态分布的随机浮点数。

多线程注意事项：当两个线程同时调用此方法时，它们有可能将获得相同的返回值。这可以通过三种办法来避免。1）让每个线程使用不同的随机数生成器实例；2）在所有调用外面加锁；3）改用速度较慢但是线程安全的 `normalvariate()` 函数。

### normalviriate()

```python
random.normalvariate(mu, sigma)
```

返回服从平均值为 *mu*、标准差为 *sigma* 的正态分布的随机浮点数。

### expovariate()

```python
random.expovariate(lambd)
```

返回服从参数为 *lambd* 的指数分布的随机浮点数。

### gammavariate()

```python
random.gammavariate(alpha, beta)
```

返回服从参数为 *alpha* 和 *beta* 的 Gamma 分布的随机浮点数。

### betavariate()

```python
random.betavariate(alpha, beta)
```

返回服从参数为 *alpha* 和 *beta* 的 Beta 分布的随机浮点数。
