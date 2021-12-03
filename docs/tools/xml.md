## XML

XML是可扩展标记语言（eXtensible Markup Language）的缩写，是一种数据表示格式，可以描述复杂的数据结构，常用于传输和存储数据。

例如一个描述书籍的XML文档：

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

XML的特点：

+ 纯文本，默认使用UTF-8编码
+ 可嵌套，适合表示结构化数据
+ XML内容保存为.xml文件，常通过网络传输



## XML结构

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

合法的XML指XML不但格式正确，而且其数据结构可以被DTD或XSD验证。DTD文档可以指定一系列规则，例如book.dtd可以规定：

+ 根元素必须是`book`
+ `book`元素必须包含`name`，`author`等指定元素
+ `isbn`元素必须包含属性`lang`

XML文件格式的正确性可以通过拖拽至浏览器验证



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



## XML解析

### Java

#### DOM

DOM一次性读取XML，并在内存中表示为树形结构。以之前的Java核心技术.xml为例，解析为DOM结构：

![](https://raw.githubusercontent.com/xyxxxxx/image/master/adipfojh4ovejnthkfqe.PNG)



顶端的document代表XML文档，book是其根元素

Java提供DOM API以解析XML，其使用以下对象：

- Document：代表整个XML文档；
- Element：代表一个XML元素；
- Attribute：代表一个元素的某个属性

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

SAX（Simple API for XML）是一种基于流的解析方式，边读取XML边解析，并以事件回调的方式让调用者获取数据

SAX解析会触发一系列事件：

- startDocument：开始读取XML文档；
- startElement：读取到了一个元素，例如`<book>`；
- characters：读取到了字符；
- endElement：读取到了一个结束的元素，例如`</book>`；
- endDocument：读取XML文档结束。

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

观察Java核心技术.xml，发现其完全对应类：

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

开源的第三方库Jackson可以轻松做到XML到JavaBean的转换，先添加Maven的依赖：

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

| DOM                          | SAX      |      |
| ---------------------------- | -------- | ---- |
| 操作方便，可以对文档进行CRUD | 不占内存 |      |
| 占内存                       | 只读     |      |
| 服务端                       | 客户端   |      |

