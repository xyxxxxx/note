# uuid——RFC 4122 定义的 UUID 对象

`uuid` 模块提供了不可变的 `UUID` 类和 `uuid1()`、`uuid3()`、`uuid4()`、`uuid5()` 等函数用于生成 RFC 4122 所定义的第 1, 3, 4 和 5 版 UUID。

## UUID

```python
class uuid.UUID(hex=None, bytes=None, bytes_le=None, fields=None, int=None, version=None, *, is_safe=SafeUUID.unknown)
```

用一串 32 位十六进制数字、一串大端序 16 个字节作为 *bytes* 参数、一串小端序 16 个字节作为 *bytes_le* 参数、一个由六个整数组成的元组（32 位 *time_low*，16 位 *time_mid*，16 位 *time_hi_version*，8 位 *clock_seq_hi_variant*，8 位 *clock_seq_low*，48 位 *node*）作为 *fields* 参数，或者一个 128 位整数作为 *int* 参数创建一个 UUID。当给出一串十六进制数字时，大括号、连字符和 URN 前缀都是可选的。例如，下列表达式都产生相同的 UUID:

```python
UUID('{12345678-1234-5678-1234-567812345678}')
UUID('12345678123456781234567812345678')
UUID('urn:uuid:12345678-1234-5678-1234-567812345678')
UUID(bytes=b'\x12\x34\x56\x78'*4)
UUID(bytes_le=b'\x78\x56\x34\x12\x34\x12\x78\x56' +
              b'\x12\x34\x56\x78\x12\x34\x56\x78')
UUID(fields=(0x12345678, 0x1234, 0x5678, 0x12, 0x34, 0x567812345678))
UUID(int=0x12345678123456781234567812345678)
```

必须给出 *hex*、*bytes*、*bytes_le*、*fields* 或 *int* 中的唯一一个。*version* 参数是可选的；如果给定，产生的 UUID 将根据 RFC 4122 设置其变体和版本号，覆盖给定的 *hex*、*bytes*、*bytes_le*、*fields* 或 *int* 中的位。

UUID 对象的比较是通过比较它们的 `UUID.int` 属性进行的。 与非 UUID 对象的比较会引发 `TypeError`。

`str(uuid)` 返回一个 `12345678-1234-5678-1234-567812345678` 形式的字符串，其中 32 位十六进制数字代表 UUID。

### bytes

UUID 作为一个 16 字节的字符串（包含 6 个整数字段，大端字节顺序）。

### bytes_le

UUID 作为一个 16 字节的字符串（其中 *time_low*、*time_mid* 和 *time_hi_version* 为小端字节顺序）。

### fields

UUID 的 6 个整数域构成的元组。

### hex

UUID 作为一个 32 个字符的十六进制字符串。

### int

UUID 作为一个 128 位的整数。

### version

UUID 版本号（1 到 5，只有当变体为 RFC_4122 时才有意义）。

## uuid1()

```python
uuid.uuid1(node=None, clock_seq=None)
```

根据主机 ID、序列号和当前时间生成一个 UUID。 如果没有给出 *node*，则使用 `getnode()` 来获取硬件地址。 如果给出了 *clock_seq*，它将被用作序列号；否则将选择一个随机的 14 位序列号。

## uuid4()

生成一个随机的 UUID。
