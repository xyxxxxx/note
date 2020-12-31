[Beautiful Soup](http://www.crummy.com/software/BeautifulSoup/) 是一个可以从HTML或XML文件中提取数据的Python库.它能够通过你喜欢的转换器实现惯用的文档导航,查找,修改文档的方式.Beautiful Soup会帮你节省数小时甚至数天的工作时间.



参考[Beautiful Soup 4.4.0 文档](https://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/)

下面一段HTML代码将作为例子被多次用到.这是 *爱丽丝梦游仙境的* 的一段内容:

```html
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
```

在浏览器中显示如下:

![](https://i.loli.net/2020/12/29/WZtx64Cdhf7QEAN.png)

使用BeautifulSoup解析这段代码,能够得到一个 `BeautifulSoup` 对象:

```shell
>>> from bs4 import BeautifulSoup
>>> html_doc = """
... <html><head><title>The Dormouse's story</title></head>
... <body>
... <p class="title"><b>The Dormouse's story</b></p>
... 
... <p class="story">Once upon a time there were three little sisters; and their names were
... <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
... <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
... <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
... and they lived at the bottom of a well.</p>... 
... <p class="story">...</p>
... """
>>> soup = BeautifulSoup(html_doc, 'html.parser')
```

将其按照标准的缩进格式的结构输出:

```html
>>> print(soup.prettify())
<html>
 <head>
  <title>
   The Dormouse's story
  </title>
 </head>
 <body>
  <p class="title">
   <b>
    The Dormouse's story
   </b>
  </p>
  <p class="story">
   Once upon a time there were three little sisters; and their names were
   <a class="sister" href="http://example.com/elsie" id="link1">
    Elsie
   </a>
   ,
   <a class="sister" href="http://example.com/lacie" id="link2">
    Lacie
   </a>
   and
   <a class="sister" href="http://example.com/tillie" id="link3">
    Tillie
   </a>
   ;
and they lived at the bottom of a well.
  </p>
  <p class="story">
   ...
  </p>
 </body>
</html>
```

几个简单的浏览结构化数据的方法:

```shell
>>> soup.title             # 查看<title></title>
<title>The Dormouse's story</title>

>>> soup.title.string
"The Dormouse's story"

>>> soup.title.parent.name # 查看上级名称
'head'

>>> soup.a                 # 查看(第一个)<a></a>
<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

>>> soup.find_all('a')     # 查找所有<a></a>
[<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>, <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

>>> soup.find(id='link2')  # 根据id查找
<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
```

返回文档的全部文字内容:

```shell
>>> soup.text              # 与soup.get_text()相同
"\nThe Dormouse's story\n\nThe Dormouse's story\nOnce upon a time there were three little sisters; and their names were\nElsie,\nLacie and\nTillie;\nand they lived at the bottom of a well.\n...\n"

>>> print(soup.text)

The Dormouse's story

The Dormouse's story
Once upon a time there were three little sisters; and their names were
Elsie,
Lacie and
Tillie;
and they lived at the bottom of a well.
...

```





