# Pillow

PIL（Python Imaging Library）是 Python 的图像处理包，Pillow 是 PIL 的一个分叉，提供了扩展的文件格式的支持、高效的内部表示和强大的图像处理功能。

## Image

### Image

图像类。此类的实例通过工厂函数 `Image.open()`、`Image.new()` 和 `Image.frombytes()` 创建得到。

#### close()

关闭文件指针。

#### convert()

返回图像转换后的副本。

```python
im_8bit = im_rgb.convert('L')   # L = R * 299/1000 + G * 587/1000 + B * 114/1000
im_1bit = im_8bit.convert('1')  # 127 -> 0, 128 -> 255 (1)
```

#### copy()

返回图像的副本。

#### create()

以给定的模式和大小创建一个新的图像。

#### crop()

返回图像的一个矩形区域。

```python
with Image.open('0.png') as im:
    im_crop = im.crop((20, 20, 100, 100))   # 元组(左,上,右,下)定义了裁剪的像素坐标
```

#### entropy()

计算并返回图像的熵。

#### filename

源文件的文件名或路径。只有由工厂函数 `open()` 创建的图像有此属性。

#### format

源文件的格式。只有由工厂函数 `open()` 创建的图像有此属性。

#### getbands()

返回包含图像各通道名称的元组。

```python
>>> im_rgb.getbands()
('R', 'G', 'B')
>>> im_8bit.getbands()
('L',)
```

#### getchannel()

返回包含图像单个通道的图像。

```python
>>> im_r = im_rgb.getchannel('R')
>>> im_r.getbands()
('L',)
```

#### getcolors()

返回图像中使用的颜色列表。

```python
>>> im = Image.effect_noise((5, 5), 32)
>>> im.getcolors()  # (num, L)
[(1, 75), (2, 101), (1, 107), (1, 108), (1, 110), (1, 111), (1, 112), (1, 114), (2, 115), (2, 117), (2, 120), (1, 127), (2, 128), (1, 149), (1, 151), (1, 153), (1, 162), (1, 163), (1, 166), (1, 182)]
```

#### getdata()

返回图像内容为包含像素值的展开的序列对象。

```python
>>> list(im_8bit.getdata())
[124, 141, 168, ..., 138]
>>> list(im_rgb.getdata())
[(255, 255, 255), (255, 255, 255), (255, 255, 255), ..., (255, 255, 255)]
```

#### getpixel()

返回图像中指定位置的像素值。

```python
>>> im_8bit.getpixel((0, 0))
124
>>> im_rgb.getpixel((0, 0))
(255, 255, 255)
```

#### height

图像的高。

#### is_animated

如果图像有超过一帧，返回 `True`，否则返回 `False`。

#### mode

图像模式。

#### n_frames

图像的帧数。

#### open()

打开并识别给定的图像文件。

```python
im = Image.open('0.png')
```

#### paste()

将另一个图像粘贴（覆盖）到图像中。

#### putpixel()

修改图像中指定位置的像素值。

#### quantize()

将图像转换为 P 模式，包含指定数量的颜色。

#### reduce()

返回图像缩小指定倍数后的副本。

```python
>>> im.size
(300, 300)
>>> im.reduce(2).size
(150, 150)
```

#### resize()

返回图像的改变大小后的副本。

```python
with Image.open('0.png') as im:
    im_resized = im.resize(((im.width // 2, im.height // 2)))
    # 宽和高各减小为原来的1/2
```

#### rotate()

返回图像的旋转后的副本。

```python
with Image.open('0.png') as im:
    im_rotated = im.rotate(angle=60, expand=True, fillcolor='white')
    # 顺时针旋转60度,扩展输出图像以容纳旋转后的整个图像,空白部分用白色填充
```

#### save()

以给定的文件名保存图像。如果没有指定格式，则格式从文件名的扩展名推断而来。

```python
with Image.open('0.png') as im:
    im.save('0-1.png')
```

#### show()

展示图像。

```python
with Image.open('0.png') as im:
    im.show()
```

#### size

图像大小，以二元组 `(width, height)` 给出。

#### split()

分割图像为单个的通道。

```python
>>> im_r, im_g, im_b = im_rgb.split()
>>> im_r.getbands()
('L',)
```

#### tobytes()

返回图像为字节对象。

#### transform()

变换图像。

#### transpose()

转置图像。

```python
im_flipped = im.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
# FLIP_LEFT_RIGHT     左右翻转
# FLIP_TOP_BOTTOM     上下翻转
# ROTATE_90           旋转90度
# ROTATE_180          旋转180度
# ROTATE_270          旋转270度
# TRANSPOSE           转置
# TRANSVERSE          转置后旋转180度
```

#### verify()

验证文件内容。

#### width

图像的宽。

### blend()

通过在两个输入图像之间插值以创建一个新的图像。

```python
im0 = Image.open('0.png')
im1 = Image.open('1.png')
im = Image.blend(im0, im1, alpha=0.2)  # 0.8 im0 + 0.2 im1
```

### effect_noise()

产生以 128 为期望的高斯噪声。

```python
im = Image.effect_noise((20, 20), 32)
                                  # 噪声的标准差
```

### eval()

将函数（应接收一个参数）应用到给定图像中的每一个像素。如果图像有多于一个通道，则相同的函数被应用到每个通道。

### fromarray()

从具有数组接口的对象创建图像。

```python
import numpy as np

im = Image.open('0.png')
a = np.asarray(im)
# array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0],
#        ...
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14, 179, 245, 236,
#         242, 254, 254, 254, 254, 245, 235,  84,   0,   0,   0,   0,   0,
#           0,   0],
#        ...
#        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
#           0,   0]], dtype=uint8)
im = Image.fromarray(a)

```

### merge()

将一组单通道图像合并为一个多通道图像。

### new()

以给定的模式和大小创建一个新的图像。

```python
im_8bit = Image.new('L', (200, 200), 128)
im_rgb = Image.new('RGB', (200, 200), (0, 206, 209))
```
