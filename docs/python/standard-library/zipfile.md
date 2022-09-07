# zipfile——使用 ZIP 存档

ZIP 文件格式是一个常用的归档与压缩标准。 这个模块提供了创建、读取、写入、添加及列出 ZIP 文件的工具。任何对此模块的进阶使用都将需要理解此格式，其定义参见 [PKZIP 应用程序笔记](https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT)。

此模块目前不能处理分卷 ZIP 文件。它可以处理使用 ZIP64 扩展的 ZIP 文件（超过 4 GB 的 ZIP 文件）。它支持解密 ZIP 归档中的加密文件，但是目前不能创建一个加密的文件。解密非常慢，因为它是使用原生 Python 而不是 C 实现的。
