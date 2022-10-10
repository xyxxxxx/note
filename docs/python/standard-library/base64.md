# base64——Base16, Base32, Base64, Base85 数据编码

`base64` 模块提供了将二进制数据编码为可打印的 ASCII 字符以及将这些编码解码回二进制数据的函数，即 RFC 3548 指定的 Base16、Base32 和 Base64 编码以及已被广泛接受的 Ascii85 和 Base85 编码的编码和解码函数。

## b64encode()

对类似字节序列的对象进行 Base64 编码，返回编码后的字节序列。

```python
>>> import base64
>>> encoded = base64.b64encode(b'data to be encoded')
>>> encoded
b'ZGF0YSB0byBiZSBlbmNvZGVk'
>>> data = base64.b64decode(encoded)
>>> data
b'data to be encoded'
```

## b64decode()

对 Base64 编码过的类似字节序列的对象进行解码，返回解码后的字节序列。
