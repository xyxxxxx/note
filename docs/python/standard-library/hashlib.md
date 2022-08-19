# hashlib——安全哈希与消息摘要

`hashlib` 模块针对不同的安全哈希和消息摘要算法实现了一个通用的接口，包括 FIPS 的 SHA1、SHA224、SHA256、SHA384 和 SHA512（定义于 FIPS 180-2）算法，以及 RSA 的 MD5 算法（定义于 Internet RFC 1321）。术语“安全哈希”和“消息摘要”是可互换的，较旧的算法被称为消息摘要，现代术语是安全哈希。

## 哈希算法

每种类型的哈希都有一个构造器方法，它们都返回一个具有相同的简单接口的哈希对象。例如，使用 `sha256()` 创建一个 SHA-256 哈希对象。你可以使用 `update()` 方法向这个对象输入字节类对象（通常是 `bytes`）。在任何时候你都可以使用 `digest()` 或 `hexdigest()` 方法获得到目前为止输入这个对象的拼接数据的摘要。

!!! note "注意"
    向 `update()` 输入字符串是不被支持的，因为哈希基于字节而非字符。

此模块中总是可用的哈希算法构造器有 `sha1()`、`sha224()`、`sha256()`、`sha384()`、`sha512()`、`blake2b()` 和 `blake2s()`。`md5()` 通常也是可用的，但如果你在使用少见的“FIPS 兼容”的 Python 编译版本则可能会找不到它。此外还可能有一些附加的算法，具体取决于你的平台上的 Python 所使用的 OpenSSL 库。在大部分平台上可用的还有 `sha3_224()`、`sha3_256()`、`sha3_384()`、`sha3_512()`、`shake_128()`、`shake_256()` 等等。

模块还提供了 `new()`，一个接受哈希算法的名称作为第一个形参的通用构造器。它允许访问上面列出的哈希算法以及你的 OpenSSL 库可能提供的任何其他算法。同名的构造器要比 `new()` 更快所以应当优先使用。

例如，如果想获取字节串 `b'Nobody inspects the spammish repetition'` 的 sha256 摘要:

```python
>>> m = hashlib.sha256()
>>> m.update(b'Nobody inspects')
>>> m.update(b' the spammish repetition')
>>> m.digest()
b'\x03\x1e\xdd}Ae\x15\x93\xc5\xfe\\\x00o\xa5u+7\xfd\xdf\xf7\xbcN\x84:\xa6\xaf\x0c\x95\x0fK\x94\x06'
>>> m.hexdigest()
'031edd7d41651593c5fe5c006fa5752b37fddff7bc4e843aa6af0c950f4b9406'
>>> m.digest_size
32
>>> m.block_size
64
```

更简要的写法是：

```python
>>> hashlib.sha256(b'Nobody inspects the spammish repetition').hexdigest()
'031edd7d41651593c5fe5c006fa5752b37fddff7bc4e843aa6af0c950f4b9406'
```

使用 `new()` 构造器：

```python
>>> m = hashlib.new('sha256')
>>> m.update(b'Nobody inspects the spammish repetition')
>>> m.hexdigest()
'031edd7d41651593c5fe5c006fa5752b37fddff7bc4e843aa6af0c950f4b9406'
```

哈希对象具有下列属性和方法：

### block_size

以字节为单位的哈希算法的内部块大小。

### copy()

返回哈希对象的副本。这可以被用来高效地计算共享相同初始子串的数据的摘要。

### digest()

返回到目前为止已传给 `update()` 方法的数据的摘要。这是一个大小为 `digest_size` 的字节对象，其中每个字节有 0 到 255 的完整取值范围。

### digest_size

以字节为单位的摘要的大小。

### hexdigest()

类似于 `digest()` 但摘要会以两倍长度的字符串对象的形式返回，其中仅包含十六进制数码。这可以被用于在电子邮件或其他非二进制环境中安全地交换数据值。

### name

哈希对象的规范名称，总是为小写形式并且总是可以作为 `new()` 的形参用来创建另一个此类型的哈希对象。

### update()

用字节类对象来更新哈希对象。重复调用相当于单次调用并传入所有参数的拼接结果：`m.update(a); m.update(b)` 等价于 `m.update(a+b)`。
