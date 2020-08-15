**数据库（DataBase）**是一个用来存储和管理数据的仓库。它的存储空间很大，并且有一定的数据存放规则。通过由行和列组成的二维表（类似 Excel 工作表）来管理数据。数据库中可以同时存储多个表。

数据库按照数据结构来组织、存储和管理数据. 数据库一共有三种模型：

+ 层次模型
+ 网状模型
+ 关系模型

主流的关系数据库主要分为以下几类：

1. 商用数据库，例如：[Oracle](https://www.oracle.com/)，[SQL Server](https://www.microsoft.com/sql-server/)，[DB2](https://www.ibm.com/db2/)等；
2. 开源数据库，例如：[MySQL](https://www.mysql.com/)，[PostgreSQL](https://www.postgresql.org/)等；
3. 桌面数据库，以微软[Access](https://products.office.com/access)为代表，适合桌面应用程序使用；
4. 嵌入式数据库，以[Sqlite](https://sqlite.org/)为代表，适合手机应用和桌面程序。

**管理系统（Management System）**是一个软件，我们可以通过它来插入（insert）、查询（query）、修改（modify）或删除（delete）表中的数据。

**SQL(Structured Query Language)**即结构化查询语言，用来访问和操作数据库系统. 不同的数据库都支持SQL. 现实情况是，如果我们只使用标准SQL的核心功能，那么所有数据库通常都可以执行。不常用的SQL功能，不同的数据库支持的程度都不一样。而各个数据库支持的各自扩展的功能，通常我们把它们称之为“方言”.

SQL语言定义了如下几种操作数据库的能力：

SQL 包含以下 4 部分：

**1）数据定义语言（Data Definition Language，DDL）**

用来创建或删除数据库以及表等对象，主要包含以下几种命令：

- DROP：删除数据库和表等对象
- CREATE：创建数据库和表等对象
- ALTER：修改数据库和表等对象的结构

**2）数据操作语言（Data Manipulation Language，DML）**

用来变更表中的记录，主要包含以下几种命令：

- SELECT：查询表中的数据
- INSERT：向表中插入新数据
- UPDATE：更新表中的数据
- DELETE：删除表中的数据

**3）数据查询语言（Data Query Language，DQL）**

用来查询表中的记录，主要包含 SELECT 命令，来查询表中的数据。

**4）数据控制语言（Data Control Language，DCL）**

用来确认或者取消对数据库中的数据进行的变更。除此之外，还可以对数据库中的用户设定权限。主要包含以下几种命令：

- GRANT：赋予用户操作权限
- REVOKE：取消用户的操作权限
- COMMIT：确认对数据库中的数据进行的变更
- ROLLBACK：取消对数据库中的数据进行的变更