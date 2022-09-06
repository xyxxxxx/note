# JSON

JSON（JavaScript Object Notation）去除了 JavaScript 的执行代码，而仅保留其对象格式，如：

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

JSON 存在以下优点：

* 只允许 UTF-8 编码，不存在编码问题
* 只允许使用双引号作为 key，转义使用 `\`
* 浏览器内置 JSON 支持

JSON 格式简单，仅支持以下数据类型：

* 键值对：`{"key": value}`
* 数组：`[1, 2, 3]`
* 字符串：`"abc"`
* 数值（整数和浮点数）：`12.34`
* 布尔值：`true` 或 `false`
* 空值：`null`

## 常用解析工具

### Python

### Go

### Java

常用的解析 JSON 的第三方库包含：Jackson，Gson，Fastjson，...

例如使用 Jackson，先引入以下 Maven 依赖：

```xml
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-databind</artifactId>
    <version>2.10.0</version>
</dependency>
```

再进行解析：

```java
InputStream input = Main.class.getResourceAsStream("/book.json");
ObjectMapper mapper = new ObjectMapper();
// 反序列化时忽略不存在的JavaBean属性:
mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
Book book = mapper.readValue(input, Book.class);
```

把 JSON 解析为 JavaBean 的过程称为 <u> 反序列化 </u>。如果把 JavaBean 变为 JSON，那就是 <u> 序列化 </u>。要实现 JavaBean 到 JSON 的序列化，只需要一行代码：

```java
String json = mapper.writeValueAsString(book);
```
