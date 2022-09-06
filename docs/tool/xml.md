# XML

XML（eXtensible Markup Language，可扩展标记语言）是一种数据表示格式，可以描述复杂的数据结构，常用于传输和存储数据。

!!! tip "提示"
    XML 尽管功能全面，但标签繁琐，格式复杂，在 Web 上使用越来越少，而逐渐被更小、更快、更容易解析的 JSON 取而代之。

例如一个描述书的信息的 XML 文档：

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE note SYSTEM "book.dtd">
<book id="1">
    <name>Java核心技术</name>
    <author>Cay S. Horstmann</author>
    <isbn lang="CN">1234567</isbn>
    <tags>
        <tag>Java</tag>
        <tag>Network</tag>
    </tags>
    <pubDate/>
</book>
```

XML 的特点：

* 纯文本，默认使用 UTF-8 编码
* 可嵌套，适合表示结构化数据
* XML 内容保存为 `.xml` 文件，常通过网络传输

## XML 结构

```xml
<?xml version="1.0" encoding="UTF-8" ?>	 <!--首行必定是?xml version="1.0"以及可选编码-->
<!DOCTYPE note SYSTEM "book.dtd">        <!--声明文档类型DTD-->
<book id="1">							 <!--根元素-->
    <name>Java核心技术</name>			  <!--子元素-->
    <author>Cay S. Horstmann</author>
    <isbn lang="CN">1234567</isbn>		 <!--包含属性-->
    <tags>
        <tag>Java</tag>
        <tag>Network</tag>
    </tags>
    <pubDate/>
</book>									 <!-- -->
```

合法的 XML 指 XML 不但格式正确，而且其数据结构可以被 DTD 或 XSD 验证。DTD 文档可以指定一系列规则，例如 book.dtd 可以规定：

* 根元素必须是 `book`。
* `book` 元素必须包含 `name`，`author` 等指定元素。
* `isbn` 元素必须包含属性 `lang`。

XML 文件格式的正确性可以通过拖拽至浏览器验证。

**转义**

| 字符 | 表示       |
| :--- | :--------- |
| <    | ``&lt;``   |
| >    | ``&gt;``   |
| &    | ``&amp;``  |
| "    | ``&quot;`` |
| '    | ``&apos;`` |

```xml
<name>Java&lt;tm&gt;</name>
```

## 常用解析工具

### Java

#### DOM

DOM 一次性读取 XML，并在内存中表示为树形结构。以之前的 Java 核心技术.xml 为例，解析为 DOM 结构：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/adipfojh4ovejnthkfqe.PNG)

顶端的 document 代表 XML 文档，book 是其根元素

Java 提供 DOM API 以解析 XML，其使用以下对象：

* Document：代表整个 XML 文档；
* Element：代表一个 XML 元素；
* Attribute：代表一个元素的某个属性

```java
//DOM API解析XML
InputStream input = Main.class.getResourceAsStream("/book.xml");
DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
DocumentBuilder db = dbf.newDocumentBuilder();
Document doc = db.parse(input);

//遍历读取
void printNode(Node n, int indent) {
    for (int i = 0; i < indent; i++) {
        System.out.print(' ');
    }
    switch (n.getNodeType()) {
    case Node.DOCUMENT_NODE: // Document节点
        System.out.println("Document: " + n.getNodeName());
        break;
    case Node.ELEMENT_NODE: // 元素节点
        System.out.println("Element: " + n.getNodeName());
        break;
    case Node.TEXT_NODE: // 文本
        System.out.println("Text: " + n.getNodeName() + " = " + n.getNodeValue());
        break;
    case Node.ATTRIBUTE_NODE: // 属性
        System.out.println("Attr: " + n.getNodeName() + " = " + n.getNodeValue());
        break;
    default: // 其他
        System.out.println("NodeType: " + n.getNodeType() + ", NodeName: " + n.getNodeName());
    }
    for (Node child = n.getFirstChild(); child != null; child = child.getNextSibling()) {
        printNode(child, indent + 1);
    }
}
```

#### SAX

SAX（Simple API for XML）是一种基于流的解析方式，边读取 XML 边解析，并以事件回调的方式让调用者获取数据。

SAX 解析会触发一系列事件：

* startDocument：开始读取 XML 文档；
* startElement：读取到了一个元素，例如 `<book>`；
* characters：读取到了字符；
* endElement：读取到了一个结束的元素，例如 `</book>`；
* endDocument：读取 XML 文档结束。

```java
//SAX解析XML
InputStream input = Main.class.getResourceAsStream("/book.xml");
SAXParserFactory spf = SAXParserFactory.newInstance();
SAXParser saxParser = spf.newSAXParser();
saxParser.parse(input, new MyHandler());

//MyHandler()实现回溯
class MyHandler extends DefaultHandler {
    public void startDocument() throws SAXException {
        print("start document");
    }

    public void endDocument() throws SAXException {
        print("end document");
    }

    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        print("start element:", localName, qName);
    }

    public void endElement(String uri, String localName, String qName) throws SAXException {
        print("end element:", localName, qName);
    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        print("characters:", new String(ch, start, length));
    }

    public void error(SAXParseException e) throws SAXException {
        print("error:", e);
    }

    void print(Object... objs) {
        for (Object obj : objs) {
            System.out.print(obj);
            System.out.print(" ");
        }
        System.out.println();
    }
}
```

#### Jackson

观察 Java 核心技术.xml，发现其完全对应类：

```java
public class Book {
    public long id;
    public String name;
    public String author;
    public String isbn;
    public List<String> tags;
    public String pubDate;
}
```

开源的第三方库 Jackson 可以轻松做到 XML 到 JavaBean 的转换，先添加 Maven 的依赖：

```xml
<dependency>
    <groupId>com.fasterxml.jackson.dataformat</groupId>
    <artifactId>jackson-dataformat-xml</artifactId>
    <version>2.10.1</version>
</dependency>
<dependency>
    <groupId>org.codehaus.woodstox</groupId>
    <artifactId>woodstox-core-asl</artifactId>
    <version>4.4.1</version>
</dependency>
```

再进行解析

```java
InputStream input = Main.class.getResourceAsStream("/book.xml");
JacksonXmlModule module = new JacksonXmlModule();
XmlMapper mapper = new XmlMapper(module);
Book book = mapper.readValue(input, Book.class);
System.out.println(book.id);
System.out.println(book.name);
System.out.println(book.author);
System.out.println(book.isbn);
System.out.println(book.tags);
System.out.println(book.pubDate);
```

| DOM                           | SAX      |     |
| ----------------------------- | -------- | --- |
| 操作方便，可以对文档进行 CRUD | 不占内存 |     |
| 占内存                        | 只读     |     |
| 服务端                        | 客户端   |     |
