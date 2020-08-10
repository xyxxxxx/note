> XML尽管功能全面，但标签繁琐，格式复杂，在Web上使用越来越少，而被JSON取而代之
> JSON更小，更快，更易解析

## JSON

JSON（JavaScript Object Notation）去除了JavaScript的执行代码，而仅保留其对象格式，如：

```json
{
    "id": 1,
    "name": "Java核心技术",
    "author": {
        "firstName": "Abc",
        "lastName": "Xyz"
    },
    "isbn": "1234567",
    "tags": ["Java", "Network"]
}
```

JSON存在以下优点：

+ 只允许UTF-8编码，不存在编码问题
+ 只允许使用双引号作为key，转义使用`\`
+ 浏览器内置JSON支持

JSON格式简单，仅支持以下数据类型：

- 键值对：`{"key": value}`
- 数组：`[1, 2, 3]`
- 字符串：`"abc"`
- 数值（整数和浮点数）：`12.34`
- 布尔值：`true`或`false`
- 空值：`null`



## JSON解析

### Java

常用的解析JSON的第三方库包含：Jackson, Gson, Fastjson, ...

例如使用Jackson，先引入以下Maven依赖：

```xml
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-databind</artifactId>
    <version>2.10.0</version>
</dependency>
```

再进行解析

```java
InputStream input = Main.class.getResourceAsStream("/book.json");
ObjectMapper mapper = new ObjectMapper();
// 反序列化时忽略不存在的JavaBean属性:
mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
Book book = mapper.readValue(input, Book.class);
```

把JSON解析为JavaBean的过程称为<u>反序列化</u>。如果把JavaBean变为JSON，那就是<u>序列化</u>。要实现JavaBean到JSON的序列化，只需要一行代码：

```java
String json = mapper.writeValueAsString(book);
```



### Go