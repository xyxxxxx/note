# secrets——生成安全随机数字用于管理密码

`secrets` 模块可用于生成高加密强度的随机数，适应管理密码、账户验证、安全凭据和相关机密数据管理的需要。

特别地，应当优先使用 `secrets` 来替代 `random` 模块中的默认伪随机数生成器，后者被设计用于建模和仿真，而不适用于安全和加密。

## 随机数

通过 `secrets` 模块可以访问你的操作系统所能提供的最安全的随机性来源。

### SystemRandom

使用操作系统所提供的最高质量源来生成随机数的类。请参阅 `random.SystemRandom` 了解更多细节。

### choice()

返回从一个非空序列中随机选取的元素。

### randbelow()

```python
secrets.randbelow(n)
```

返回一个 $[0, n)$ 区间内的随机整数。

### randbits()

```python
secrets.randbits(k)
```

返回一个具有 *k* 个随机比特位的整数。

## 生成凭据

`secrets` 模块提供了一些生成安全凭据的函数，适用于诸如密码重置、难以猜测的 URL 之类的应用场景。

### token_bytes()

```python
secrets.token_bytes([nbytes=None])
```

返回一个包含 *nbytes* 个字节的随机字节串。如果 *nbytes* 为 `None` 或未提供，则会使用一个合理的默认值。

```python
>>> token_bytes(16)  
b'\xebr\x17D*t\xae\xd4\xe3S\xb6\xe2\xebP1\x8b'
```

### token_hex()

```python
secrets.token_hex([nbytes=None])
```

返回一个十六进制数码形式的随机字符串。字符串具有 *nbytes* 个随机字节，每个字节转换为两个十六进制数码。如果 *nbytes* 为 `None` 或未提供，则会使用一个合理的默认值。


```python
>>> token_hex(16)  
'f9bf78b9a18ce6d46a0cd2b0b86df9da'
```

### token_urlsafe()

```python
secrets.token_urlsafe([nbytes=None])
```

返回一个 URL 安全的随机字符串，包含 *nbytes* 个随机字节。文本将使用 Base64 编码，因此平均来说每个字节将对应 1.3 个结果字符。如果 *nbytes* 为 `None` 或未提供，则会使用一个合理的默认值。

```python
>>> token_urlsafe(16)  
'Drmhze6EPcv0fN_81Bj-nA'
```

### 凭据应当使用多少个字节？

为了保证在面对暴力攻击时的安全，凭据必须具有足够的随机性。不幸的是，对随机性是否足够的标准会随着计算机越来越强大并能够在更短时间内进行更多猜测而不断提高。在 2015 年时，人们认为 32 字节（256 位）的随机性对于 `secrets` 模块所适合的典型用例来说是足够的。

作为想要自行管理凭据长度的用户，你可以通过为各种 `token_*` 函数指定一个 `int` 参数来显式地指定凭据要使用多大的随机性。该参数以字节数来表示要使用的随机性大小。

在其他情况下，如果未提供参数，或者如果参数为 `None`，则 `token_*` 函数将改用一个合理的默认值。

!!! note "注意"
    该默认值可能在任何时候被改变，包括在维护版本更新的时候。

## 应用技巧与最佳实践

本节展示了一些使用 `secrets` 来管理基本安全级别的应用技巧和最佳实践。

生成长度为八个字符的字母数字密码:

```python
import string
import secrets
alphabet = string.ascii_letters + string.digits
password = ''.join(secrets.choice(alphabet) for i in range(8))
```

!!! note "注意"
    应用程序不能以可恢复的格式存储密码，无论是用纯文本还是加密。它们应当使用高加密强度的单向（不可恢复）哈希函数来加盐并生成哈希值。

生成长度为十个字符的字母数字密码，其中包含至少一个小写字母，至少一个大写字母以及至少三个数字:

```python
import string
import secrets
alphabet = string.ascii_letters + string.digits
while True:
    password = ''.join(secrets.choice(alphabet) for i in range(10))
    if (any(c.islower() for c in password)
            and any(c.isupper() for c in password)
            and sum(c.isdigit() for c in password) >= 3):
        break
```

生成 [XKCD 风格的密码串](https://xkcd.com/936/):

```python
import secrets
# On standard Linux systems, use a convenient dictionary file.
# Other platforms may need to provide their own word-list.
with open('/usr/share/dict/words') as f:
    words = [word.strip() for word in f]
    password = ' '.join(secrets.choice(words) for i in range(4))
```

生成难以猜测的临时 URL，其中包含适合密码恢复应用的安全凭据:

```python
import secrets
url = 'https://mydomain.com/reset=' + secrets.token_urlsafe()
```
