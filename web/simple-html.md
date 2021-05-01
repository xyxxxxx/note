> 参考https://www.w3school.com.cn/html/index.asp

[toc]



HTML(**H**yper **T**ext **M**arkup **L**anguage，超文本标记语言)



# 元素

> 参考[HTML 标签参考手册](https://www.w3school.com.cn/tags/index.asp)

## \<a>

锚（链接）。

示例：（markdown 支持此语法）

<a href="http://www.w3school.com.cn">This is a link</a>



## \<addr>

缩写。

示例：（markdown 支持此语法）

The <abbr title="People's Republic of China">PRC</abbr> was founded in 1949.



## \<area>

图像映射内部的区域。

示例参见[\<map>](#\<map>)。



## \<artical>

文章。

示例：

```html
<article>
  <h1>Internet Explorer 9</h1>
  <p>Windows Internet Explorer 9（简称 IE9）于 2011 年 3 月 14 日发布.....</p>
</article>
```



## \<audio>

音频。

示例：（markdown 支持此语法）

<audio src="/i/horse.ogg" controls="controls">
Your browser does not support the audio element.
</audio>



## \<body>

文档主体。

示例：

```html
<body>
<p>This is my first paragraph.</p>
</body>
```



## \<br />

换行。

示例：（markdown 支持此语法）

123<br />456



## \<button>

按钮。

示例：（markdown 支持此语法）

<button type="button">Click Me!</button>



## \<caption>

表格标题。

示例参见[\<table>](#\<table>)。



## \<del>

被删除文本。

示例：（markdown 支持此语法）

<del>删除文本</del>



## \<details>

细节。

示例：（markdown 支持此语法）

<details>
<summary>Copyright 2011.</summary>
<p>All pages and graphics on this web site are the property of W3School.</p>
</details>



## \<div>

文档的节。

示例：（markdown 支持此语法）

<h3>This is a header</h3>
<p>This is a paragraph.</p>
<div style="color:#0000FF">
  <h3>This is a header</h3>
  <p>This is a paragraph.</p>
</div>



## \<footer>

页脚。

示例：

<footer>
  <p>Posted by: W3School</p>
  <p>Contact information: <a href="mailto:someone@example.com">someone@example.com</a>.</p>
</footer>



## \<form>

用户表单。

示例：（markdown 支持此语法）

<form action="/demo/demo_form.asp">
First name:<br>
<input type="text" name="firstname" value="Mickey">
<br>
Last name:<br>
<input type="text" name="lastname" value="Mouse">
<br><br>
<input type="submit" value="Submit">
</form> 
<p>如果您点击提交，表单数据会被发送到名为 demo_form.asp 的页面。</p>

<p>请点击文本标记之一，就可以触发相关控件：</p>
<form>
<label for="male">Male</label>
<input type="radio" name="sex" id="male" />
<br />
<label for="female">Female</label>
<input type="radio" name="sex" id="female" />
</form>



## \<h1> - \<h6>

标题，`<h1>` 最大，`<h6>` 最小。

示例：（markdown 支持此语法）

<h1>This is a heading</h1>
<h2>This is a heading</h2>
<h3>This is a heading</h3>



## \<head>

文档信息。

示例：

```html
<head>
<title>我的第一个 HTML 页面</title>
</head>
```



## \<header>

页眉。

示例：

```html
<header>
<h1>Welcome to my homepage</h1>
<p>My name is Donald Duck</p>
</header>
```



## \<hr>

水平线。

示例：（markdown 支持此语法）

<hr />



## \<html>

整个 HTML 文档。

示例：

```html
<html>

<body>
<p>This is my first paragraph.</p>
</body>

</html>
```



## \<img>

图像。

示例：（markdown 支持此语法）

<img src="https://dummyimage.com/480x320/eee/aaa&text=-image-" width="240" height="160" />



## \<input>

输入控件。

示例参见[\<form>](#\<form>)。



## \<ins>

被插入文本（下划线文本）。

示例：（markdown 支持此语法）

<ins>下划线文本</ins>



## \<label>

input 元素的标注。

示例参见[\<form>](#\<form>)。



## \<li>

列表的项目。

示例参见[\<ol>](#\<ol>)，[\<ul>](#\<ul>)。



## \<link>

外部资源的链接，常用于链接外部样式表。

示例：

```html
<head>
<link rel="stylesheet" type="text/css" href="/html/csstest1.css" >
</head>
```



## \<main>

文档的主要内容。

示例：

```html
<main>
  <h1>Web Browsers</h1>
  <p>Google Chrome、Firefox 以及 Internet Explorer 是目前最流行的浏览器。</p>

  <article>
    <h1>Google Chrome</h1>
    <p>Google Chrome 是由 Google 开发的一款免费的开源 web 浏览器，于 2008 年发布。</p>
  </article>

  <article>
    <h1>Internet Explorer</h1>
    <p>Internet Explorer 由微软开发的一款免费的 web 浏览器，发布于 1995 年。</p>
  </article>

  <article>
    <h1>Mozilla Firefox</h1>
    <p>Firefox 是一款来自 Mozilla 的免费开源 web 浏览器，发布于 2004 年。</p>
  </article>
</main> 
```



## \<map>

图像映射。

示例：

```html
<p>请点击图像上的星球，把它们放大。</p>

<img
src="/i/eg_planets.jpg"
border="0" usemap="#planetmap"
alt="Planets" />

<map name="planetmap" id="planetmap">

<area
shape="circle"
coords="180,139,14"
href ="/example/html/venus.html"
target ="_blank"
alt="Venus" />

<area
shape="circle"
coords="129,161,10"
href ="/example/html/mercur.html"
target ="_blank"
alt="Mercury" />

<area
shape="rect"
coords="0,0,110,260"
href ="/example/html/sun.html"
target ="_blank"
alt="Sun" />

</map>
```



## \<mark>

高亮显示文本。

示例：（markdown 支持此语法）

<mark>高亮文本</mark>



## \<meta>

文档的元信息，用于向浏览器、搜索引擎、爬虫等提供文档信息。

示例：

```html
<!--某些搜索引擎在遇到此元信息时,会用这些关键字对文档进行分类-->
<meta name="keywords" content="HTML,ASP,PHP,SQL">
```

```html
<!--开放内容协议,参见https://ogp.me-->
<html prefix="og: https://ogp.me/ns#">
<head>
<title>The Rock (1996)</title>
<meta property="og:title" content="The Rock" />
<meta property="og:type" content="video.movie" />
<meta property="og:url" content="https://www.imdb.com/title/tt0117500/" />
<meta property="og:image" content="https://ia.media-imdb.com/images/rock.jpg" />
...
</head>
...
</html>
```



## \<nav>

导航链接集合。

示例：

```html
<nav>
<a href="/html/">HTML</a> |
<a href="/css/">CSS</a> |
<a href="/js/">JavaScript</a> |
<a href="/jquery/">jQuery</a>
</nav>
```



## \<object>

内嵌对象，例如图像、音频、视频、ActiveX、pdf、Flash等。其中定义了对象的数据和参数，以及可用来显示和操作数据的代码。



## \<ol>

有序列表。

示例：（markdown 支持此语法）

<p>有序列表：</p>
<ol>
  <li>打开冰箱门</li>
  <li>把大象放进去</li>
  <li>关上冰箱门</li>
</ol>



## \<option>

选择列表的选项。

示例参见[\<select>](#\<select>)。



## \<output>

输出结果。

示例：

```html
<form oninput="x.value=parseInt(a.value)+parseInt(b.value)">0
   <input type="range" id="a" value="50">100
   +<input type="number" id="b" value="50">
   =<output name="x" for="a b"></output>
</form> 
```



## \<p>

段落。

示例：（markdown 支持此语法）

<p>This is a paragraph.</p>
<p>This is another paragraph.</p>



## \<param>

对象的参数。

示例参见[\<object>](#\<object>)。



## \<progress>

进度条。

示例：（markdown 支持此语法）

下载进度：
<progress value="22" max="100">
</progress>



## \<script>

客户端脚本，通常是 JavaScript 脚本，常用于图像操作、表单验证、动态内容更新等。其既可以包含脚本语句，也可以通过 src 属性指向外部脚本文件。

示例：

```html
<script type="text/javascript">
document.write("Hello World!")
</script>
```



## \<section>

小节。

示例：

```html
<section>
  <h1>PRC</h1>
  <p>The People's Republic of China was born in 1949...</p>
</section>
```



## \<select>

选择列表（下拉列表）。

示例：（markdown 支持此语法）

<select>
  <option value ="volvo">Volvo</option>
  <option value ="saab">Saab</option>
  <option value="opel">Opel</option>
  <option value="audi">Audi</option>
</select>



## \<source>

媒介源，用于为媒介元素（video 和 audio）定义资源，并且可以提供可替换的视频/音频文件供浏览器根据它对媒体类型或者编解码器的支持进行选择。

示例：（markdown 支持此语法）

<audio controls>
<source src="/i/horse.ogg" type="audio/ogg">
<source src="/i/horse.mp3" type="audio/mpeg">
Your browser does not support the audio element.
</audio>



## \<span>

片断，用于组合行内元素，以便通过样式来格式化它们。

示例：

```html
<p class="tip"><span>提示：</span>事实上，... ...</p>
```

```css
p.tip span {
	font-weight:bold;
	color:#ff9955;
	}
```



## \<style>

文档的样式。

示例：

```html
<html>
<head>
<style type="text/css">
h1 {color:red}
p {color:blue}
</style>
</head>

<body>
<h1>Header 1</h1>
<p>A paragraph.</p>
</body>
</html>
```



## \<summary>

为 details 元素定义可见的标题。

示例参见[\<summary>](#\<summary>)。



## \<svg>

SVG 图形的容器。

示例：（markdown 支持此语法）

<svg width="100" height="100">
  <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
  抱歉，您的浏览器不支持嵌入式 SVG。
</svg>

<svg width="400" height="180">
  <rect x="50" y="20" rx="20" ry="20" width="150" height="150" style="fill:red;stroke:black;stroke-width:5;opacity:0.5" />
</svg>

<svg height="130" width="500">
<defs>
<linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
  <stop offset="0%" style="stop-color:rgb(255,255,0);stop-opacity:1" />
  <stop offset="100%" style="stop-color:rgb(255,0,0);stop-opacity:1" />
</linearGradient>
</defs>
<ellipse cx="100" cy="70" rx="85" ry="55" fill="url(#grad1)" />
<text fill="#ffffff" font-size="45" font-family="Verdana" x="50" y="86">SVG</text>
</svg>



## \<table>

表格。

示例：（markdown 支持此语法）

<table border="6">
<caption>我的标题</caption>
<tr>
  <td>100</td>
  <td>200</td>
  <td>300</td>
</tr>
<tr>
  <td>400</td>
  <td>500</td>
  <td>600</td>
</tr>
</table>

<table border="1">
  <thead>
    <tr>
      <th>Month</th>
      <th>Savings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>January</td>
      <td>$100</td>
    </tr>
    <tr>
      <td>February</td>
      <td>$80</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td>Sum</td>
      <td>$180</td>
    </tr>
  </tfoot>
</table>



## \<tbody>

表格的主体。

示例参见[\<table>](#\<table>)。




## \<td>

表格的单元。

示例参见[\<table>](#\<table>)。



## \<template>

页面加载时的隐藏内容的容器，通常与 js 配合使用。

示例：

```html
<!DOCTYPE html>
<html>
<body>

<h1>template 元素</h1>

<p>单击下面的按钮，显示 template 元素中的隐藏内容。</p>

<button onclick="showContent()">显示隐藏的内容</button>

<template>
  <h2>Flower</h2>
  <img src="/i/photo/flower.gif" width="180" height="180">
</template>

<script>
function showContent() {
  var temp = document.getElementsByTagName("template")[0];
  var clon = temp.content.cloneNode(true);
  document.body.appendChild(clon);
}
</script>

</body>
</html>
```



## \<textarea>

多行文本输入控件。

示例：（markdown 支持此语法）

<textarea rows="4" cols="30">
在w3school，你可以找到你所需要的所有的网站建设教程。
</textarea>



## \<tfoot>

表格的脚注。

示例参见[\<table>](#\<table>)。



## \<th>

表格的表头单元。

示例参见[\<table>](#\<table>)。



## \<thead>

表格的表头。

示例参见[\<table>](#\<table>)。



## \<time>

日期/时间，用于对日期和时间进行编码，以便于，例如，用户代理将排定的事件添加到用户日程表中，为搜索引擎提供更详细的日期/时间信息，等等。

示例：

<p>我们在每天早上 <time>9:00</time> 开始营业。</p>
<p>我在 <time datetime="2008-02-14">情人节</time> 有个约会。</p>



## \<title>

文档的标题，通常被浏览器放置在标签页的标题栏，并作为收藏页面时的默认名称。

示例：

```html
<html>

<head>
<title>我的第一个 HTML 页面</title>
</head>

<body>
<p>body 元素的内容会显示在浏览器中。</p>
<p>title 元素的内容会显示在浏览器的标题栏中。</p>
</body>

</html>
```



## \<tr>

表格的行。

示例参见[\<table>](#\<table>)。



## \<track>

媒体播放器的文本轨道，例如视频的字幕。

```html
<video width="320" height="240" controls="controls">
  <source src="forrest_gump.mp4" type="video/mp4" />
  <source src="forrest_gump.ogg" type="video/ogg" />
  <track kind="subtitles" src="subs_chi.srt" srclang="zh" label="Chinese">
  <track kind="subtitles" src="subs_eng.srt" srclang="en" label="English">
</video>
```



## \<ul>

无序列表。

示例：（markdown 支持此语法）

<p>无序列表：</p>
<ul>
  <li>雪碧</li>
  <li>可乐</li>
  <li>凉茶</li>
</ul>


## \<video>

视频。

示例：（markdown 支持此语法）

<video src="/i/movie.ogg" controls="controls">
your browser does not support the video tag
</video>




# 属性

下面仅介绍一些常用属性，各元素可使用的合法属性的完整列表请参考[HTML 参考手册](https://www.w3school.com.cn/tags/tag_a.asp)。

## align

对齐方式。可以取下列值：

| left     | center | right    | justify |
| -------- | ------ | -------- | ------- |
| 向左对齐 | 居中   | 向右对齐 | 自适应  |

示例：（markdown 支持此语法）

<h1 align="left">This is a heading</h1>
<h2 align="center">This is a heading</h2>
<h3 align="right">This is a heading</h3>



## bgcolor

背景颜色。

示例：

```html
<body bgcolor="lime">
<p>This is my first paragraph.</p>
</body>
```

```html
<body bgcolor=#7FFFD4>
<p>This is my first paragraph.</p>
</body>
```



## border

边框线宽。



## href

链接的 url。



## style





# 颜色

## 颜色值

使用 6 位十六进制符号定义，例如：

| black       | red      | green     | blue    | yellow  | aqua    |
| ----------- | -------- | --------- | ------- | ------- | ------- |
| #000000     | #FF0000  | #00FF00   | #0000FF | #FFFF00 | #00FFFF |
| **fuchsia** | **gray** | **white** |         |         |         |
| #FF00FF     | #C0C0C0  | #FFFFFF   |         |         |         |



## 颜色名

参见[HTML 颜色名](https://www.w3school.com.cn/html/html_colornames.asp)。

