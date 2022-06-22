# math——数学函数

`math` 模块提供了对 C 标准定义的数学函数的访问。

## atan2()

计算向量 `(x,y)` 与 x 轴正方向的夹角，结果在 `-pi` 和 `pi` 之间。

```python
>>> math.atan2(10, 1)  # y=10,x=1
1.4711276743037347
>>> math.atan2(-1, -1) # y=-1,x=-1
-2.356194490192345
```

## ceil()

向上取整。

```python
>>> math.ceil(-0.5)
0
>>> math.ceil(0)
0
>>> math.ceil(0.5)
1
```

## comb()

组合数。

```python
>>> math.comb(6, 2)
15
```

## degrees(), radians()

角度和弧度互相转换。

```python
>>> math.degrees(math.pi)
180.0
>>> math.radians(180.0)
3.141592653589793
```

## dist()

欧几里得距离。

```python
>>> math.dist([1, 1, 1], [0, 0, 0])
1.7320508075688772
```

3.8 版本新功能。

## e

自然对数底数，精确到可用精度。

```python
>>> math.e
2.718281828459045
```

## exp()

（底数为 e 的）指数函数。

```python
>>> math.exp(1)
2.718281828459045
```

## fabs()

绝对值。

```python
>>> math.fabs(-1)
1.0                   # 返回浮点数
```

## factorial()

阶乘。

```python
>>> math.factorial(5)
120
```

## floor()

向下取整。

```python
>>> math.floor(-0.5)
-1
>>> math.floor(0)
0
>>> math.floor(0.5)
0
```

## fmod()

余数。整数计算时推荐使用 `x % y`，浮点数计算时推荐使用`fmod()`。

```python
>>> math.fmod(5.0, 1.5)
0.5
```

## fsum()

计算可迭代对象的所有元素（整数或浮点数）的和。通过跟踪多个中间部分和来避免精度损失。

```python
>>> sum([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])
0.9999999999999999
>>> math.fsum([.1, .1, .1, .1, .1, .1, .1, .1, .1, .1])
1.0
```

## gcd()

最大公约数。

```python
>>> math.gcd(20, 48)
4
>>> math.gcd(20, -48)
4
>>> math.gcd(20, 0)
20
>>> math.gcd(20, 1)
1
```

在 3.9 之后的版本可以传入任意个整数参数，之前的版本只能传入两个整数参数。

## hypot()

欧几里得范数，即点到原点的欧几里得距离。

```python
>>> math.hypot(1., 1, 1)
1.7320508075688772
```

在 3.8 之后的版本可以传入任意个实数参数，之前的版本只能传入两个实数参数。

## inf

浮点正无穷大，相当于 `float('inf')` 的返回值。浮点负无穷大用 `-math.inf` 表示。

```python
>>> math.inf
inf
>>> float('inf')
inf
```

## isclose()

若两个浮点数的值非常接近则返回 `True`，否则返回 `False`。

```python
# 默认的相对容差为1e-9,绝对容差为0.0
# 相对容差或绝对容差小于给定值时认为非常接近
>>> math.isclose(1e10, 1e10+1, rel_tol=1e-9, abs_tol=0.0)  # 相对容差为1e-10,小于1e-9
True
>>> math.isclose(1e8, 1e8+1)                               # 相对容差为1e-8,大于1e-9
False
>>> math.isclose(1e8, 1e8+1, abs_tol=2)                    # 绝对容差为1,小于2
True
```

## isfinite()

若参数值既不是无穷大又不是 `NaN`，则返回 `True`，否则返回 `False`。

```python
>>> math.isfinite(0.0)
True
>>> math.isfinite(math.inf)
False
```

## isnan()

若参数值是非数字（NaN）值，则返回 `True`，否则返回 `False`。

```python
>>> math.isnan(0.0)
False
>>> math.isnan(math.nan)
True
```

## isqrt()

平方根向下取整。

```python
>>> math.isqrt(9)
3
>>> math.isqrt(10)
3
```

平方根向上取整可以使用 `1+ isqrt(n - 1)`。

## lcm()

最大公倍数。

3.9 版本新功能。

## log(), log2(), log10()

对数函数。

```python
>>> math.log(10)      # 自然对数
2.302585092994046
>>> math.log(10, 2)   # 以2为底
3.3219280948873626
>>> math.log2(10)
3.321928094887362
>>> math.log(10, 10)  # 以10为底
1.0
>>> math.log10(10)
1.0
```

## modf()

返回浮点数参数的小数和整数部分，两个结果都是浮点数并且与参数同号。

```python
>>> math.modf(0.0)
(0.0, 0.0)
>>> math.modf(1.0)
(0.0, 1.0)
>>> math.modf(1.1)
(0.10000000000000009, 1.0)
```

## nan

浮点非数字（NaN）值，相当于 `float('nan')` 的返回值。

```python
>>> math.nan
nan
>>> float('nan')
nan
```

## perm()

排列数。

```python
>>> math.perm(5)
120
>>> math.perm(5, 2)
20
```

3.8 版本新功能。

## pi

圆周率，精确到可用精度。

```python
>>> math.pi
3.141592653589793
```

## pow()

幂运算。

```python
>>> math.pow(2, 3)
8.0
>>> math.pow(1.0, 1e10)  # 总是返回1.0
1.0
>>> math.pow(1e10, 0.0)  # 总是返回1.0
1.0
```

## prod()

计算可迭代对象的所有元素（整数或浮点数）的积。积的默认初始值为 1。

```python
>>> math.prod(range(1, 6))
120
```

3.8 版本新功能。

## remainder()

IEEE 754 风格的余数：对于有限 *x* 和有限非零 *y*，返回 `x - n*y`，其中`n`是与商`x/y`的精确值最接近的整数；如果`x/y`恰好位于两个连续整数之间，则`n`取最近的偶整数。因此余数`r = remainder(x,y)`总是满足`abs(r)<=0.5*abs(y)`。

```python
>>> math.remainder(10, 3)
1.0
>>> math.remainder(11, 3)
-1.0
```

## sin(), cos(), tan(),  asin(), acos(), atan(), sinh(), cosh(), tanh(), asinh(), acosh(), atanh()

三角函数和双曲函数。

```python
>>> math.sin(math.pi/4)
0.7071067811865475
```

## sqrt()

平方根。

```python
>>> math.sqrt(9)
3.0
>>> math.sqrt(10)
3.1622776601683795
```

## trunc()

将浮点数截断为整数。

```python
>>> math.trunc(1.1)
1
>>> math.trunc(-1.1)
-1
```
