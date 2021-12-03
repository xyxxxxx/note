# 数据类型

## 整数

| 类型名称      | 说明           | 存储需求 |
| ------------- | -------------- | -------- |
| TINYINT       | 很小的整数     | 1个字节  |
| SMALLINT      | 小的整数       | 2个宇节  |
| MEDIUMINT     | 中等大小的整数 | 3个字节  |
| INT (INTEGHR) | 普通大小的整数 | 4个字节  |
| BIGINT        | 大整数         | 8个字节  |

| 类型名称      |                                           |                         |
| ------------- | ----------------------------------------- | ----------------------- |
| TINYINT       | -128〜127                                 | 0 〜255                 |
| SMALLINT      | -32768〜32767                             | 0〜65535                |
| MEDIUMINT     | -8388608〜8388607                         | 0〜16777215             |
| INT (INTEGER) | -2147483648〜2147483647                   | 0〜4294967295           |
| BIGINT        | -9223372036854775808〜9223372036854775807 | 0〜18446744073709551615 |

## 浮点数和定点数

| 类型名称            | 说明               | 存储需求   |
| ------------------- | ------------------ | ---------- |
| FLOAT               | 单精度浮点数       | 4 个字节   |
| DOUBLE              | 双精度浮点数       | 8 个字节   |
| DECIMAL (M, D)，DEC | 压缩的“严格”定点数 | M+2 个字节 |

FLOAT 类型的取值范围如下：

- 有符号的取值范围：-3.402823466E+38～-1.175494351E-38。
- 无符号的取值范围：0 和 -1.175494351E-38～-3.402823466E+38。

DOUBLE 类型的取值范围如下：

- 有符号的取值范围：-1.7976931348623157E+308～-2.2250738585072014E-308。
- 无符号的取值范围：0 和 -2.2250738585072014E-308～-1.7976931348623157E+308。

> 对精度要求比较高的时候使用DECIMAL的类型较好

## 时间

| 类型名称  | 日期格式            | 日期范围                                          | 存储需求 |
| --------- | ------------------- | ------------------------------------------------- | -------- |
| YEAR      | YYYY                | 1901 ~ 2155                                       | 1 个字节 |
| TIME      | HH:MM:SS            | -838:59:59 ~ 838:59:59                            | 3 个字节 |
| DATE      | YYYY-MM-DD          | 1000-01-01 ~ 9999-12-3                            | 3 个字节 |
| DATETIME  | YYYY-MM-DD HH:MM:SS | 1000-01-01 00:00:00 ~ 9999-12-31 23:59:59         | 8 个字节 |
| TIMESTAMP | YYYY-MM-DD HH:MM:SS | 1980-01-01 00:00:01 UTC ~ 2040-01-19 03:14:07 UTC | 4 个字节 |

## 字符串

| 类型名称   | 说明                                         | 存储需求                                                   |
| ---------- | -------------------------------------------- | ---------------------------------------------------------- |
| CHAR(M)    | 固定长度非二进制字符串                       | M 字节，1<=M<=255                                          |
| VARCHAR(M) | 变长非二进制字符串                           | L+1字节，在此，L< = M和 1<=M<=255                          |
| TINYTEXT   | 非常小的非二进制字符串                       | L+1字节，在此，L<2^8                                       |
| TEXT       | 小的非二进制字符串                           | L+2字节，在此，L<2^16                                      |
| MEDIUMTEXT | 中等大小的非二进制字符串                     | L+3字节，在此，L<2^24                                      |
| LONGTEXT   | 大的非二进制字符串                           | L+4字节，在此，L<2^32                                      |
| ENUM       | 枚举类型，只能有一个枚举字符串值             | 1或2个字节，取决于枚举值的数目 (最大值为65535)             |
| SET        | 一个设置，字符串对象可以有零个或 多个SET成员 | 1、2、3、4或8个字节，取决于集合 成员的数量（最多64个成员） |

| 插入值   | CHAR(4) | 存储需求 | VARCHAR(4) | 存储需求 |
| -------- | ------- | -------- | ---------- | -------- |
| ' '      | '  '    | 4字节    | ''         | 1字节    |
| 'ab'     | 'ab '   | 4字节    | 'ab'       | 3字节    |
| 'abc'    | 'abc '  | 4字节    | 'abc'      | 4字节    |
| 'abcd'   | 'abcd'  | 4字节    | 'abcd'     | 5字节    |
| 'abcdef' | 'abcd'  | 4字节    | 'abcd'     | 5字节    |

```mysql
<字段名> ENUM( '值1', '值1', …, '值n' )
```

| 值     | 索引 |
| ------ | ---- |
| NULL   | NULL |
| ''     | 0    |
| first  | 1    |
| second | 2    |
| third  | 3    |

```mysql
SET( '值1', '值1', …, '值n' )
```

## 二进制类型

| 类型名称       | 说明                 | 存储需求               |
| -------------- | -------------------- | ---------------------- |
| BIT(M)         | 位字段类型           | 大约 (M+7)/8 字节      |
| BINARY(M)      | 固定长度二进制字符串 | M 字节                 |
| VARBINARY (M)  | 可变长度二进制字符串 | M+1 字节               |
| TINYBLOB (M)   | 非常小的BLOB         | L+1 字节，在此，L<2^8  |
| BLOB (M)       | 小 BLOB              | L+2 字节，在此，L<2^16 |
| MEDIUMBLOB (M) | 中等大小的BLOB       | L+3 字节，在此，L<2^24 |
| LONGBLOB (M)   | 非常大的BLOB         | L+4 字节，在此，L<2^32 |

| 数据类型   | 存储范围                              |
| ---------- | ------------------------------------- |
| TINYBLOB   | 最大长度为255 (28-1)字节              |
| BLOB       | 最大长度为65535 (216-1)字节           |
| MEDIUMBLOB | 最大长度为16777215 (224-1)字节        |
| LONGBLOB   | 最大长度为4294967295 (231-1)字节或4GB |

数据类型的选择http://c.biancheng.net/view/7175.html

# 存储引擎

```mysql
SHOW ENGINES;						--查看所有引擎
SET default_storage_engine=<引擎名>  --设置默认引擎

ALTER TABLE <表名> ENGINE=<存储引擎名>;	--修改表的存储引擎

```

| 特性         | MyISAM | InnoDB | MEMORY |
| ------------ | ------ | ------ | ------ |
| 存储限制     | 有     | 支持   | 有     |
| 事务安全     | 不支持 | 支持   | 不支持 |
| 锁机制       | 表锁   | 行锁   | 表锁   |
| B树索引      | 支持   | 支持   | 支持   |
| 哈希索引     | 不支持 | 不支持 | 支持   |
| 全文索引     | 支持   | 不支持 | 不支持 |
| 集群索引     | 不支持 | 支持   | 不支持 |
| 数据缓存     |        | 支持   | 支持   |
| 索引缓存     | 支持   | 支持   | 支持   |
| 数据可压缩   | 支持   | 不支持 | 不支持 |
| 空间使用     | 低     | 高     | N/A    |
| 内存使用     | 低     | 高     | 中等   |
| 批量插入速度 | 高     | 低     | 高     |
| 支持外键     | 不支持 | 支持   | 不支持 |

MyISAM 存储引擎不支持事务和外键，所以访问速度比较快。如果应用主要以读取和写入为主，只有少量的更新和删除操作，并且对事务的完整性、并发性要求不是很高，那么选择 MyISAM 存储引擎是非常适合的。

InnoDB 存储引擎在事务上具有优势，即支持具有提交、回滚和崩溃恢复能力的事务安装，所以比 MyISAM 存储引擎占用更多的磁盘空间。如果应用对事务的完整性有比较高的要求，在并发条件下要求数据的一致性，数据操作除了插入和查询以外，还包括很多的更新、删除操作，那么 InnoDB 存储引擎是比较合适的选择。

MEMORY 存储引擎将所有数据保存在 RAM 中，所以该存储引擎的数据访问速度快，但是安全上没有保障。

# 约束

 MySQL 中支持以下 6 种约束：

**1）主键约束**

主键是表的一个特殊字段，该字段能唯一标识该表中的每条信息。例如，学生信息表中的学号是唯一的

**2）外键约束**

**3）唯一约束**

确保列中每个值的唯一性

**4）检查约束**

检查是否有效

**5）非空约束**

**6）默认值约束**

## 主键

- 每个表只能定义一个主键。
- 主键值必须<u>唯一</u>标识表中的每一行，且不能为 NULL，即表中不可能存在有相同主键值的两行数据
- 一个字段名只能在联合主键字段表中出现一次。
- 联合主键不能包含不必要的多余字段。当把联合主键的某一字段删除后，如果剩下的字段构成的主键仍然满足唯一性原则，那么这个联合主键是不正确的

## 外键

在两个表的数据之间建立链接

- 父表必须已经存在于数据库中，或者是当前正在创建的表。如果是后一种情况，则父表与子表是同一个表，这样的表称为自参照表，这种结构称为自参照完整性。
- 必须为父表定义主键。
- 主键不能包含空值，但允许在外键中出现空值。也就是说，只要外键的每个非空值出现在指定的主键中，这个外键的内容就是正确的。
- 在父表的表名后面指定列名或列名的组合。这个列或列的组合必须是父表的主键或候选键。
- 外键中列的数目必须和父表的主键中列的数目相同。
- 外键中列的数据类型必须和父表主键中对应列的数据类型相同。

# 视图view

# 索引index

使用索引可以大大提高数据库的工作效率. 索引根据数据结构分为：

+ B-树索引

  目前大部分的索引采用 B-树索引存储. B-树索引主要包含：

  - 叶子节点：包含的条目直接指向表里的数据行。叶子节点之间彼此相连，一个叶子节点有一个指向下一个叶子节点的指针。
  - 分支节点：包含的条目指向索引里其他的分支节点或者叶子节点。
  - 根节点：一个 B-树索引只有一个根节点，实际上就是位于树的最顶端的分支节点。

  B树索引可以进行全键值、键值范围和键值前缀查询，也可以对查询结果进行排序. 但 B-树索引必须遵循左边前缀原则，要考虑以下几点约束：

  - 查询必须从索引的最左边的列开始。
  - 查询不能跳过某一索引列，必须按照从左到右的顺序进行匹配。
  - 存储引擎不能使用索引中范围条件右边的列。

+ 哈希索引

  目前仅有MEMORY存储引擎和HEAP存储引擎支持该类索引. 哈希索引的最大特点是访问速度快，但也存在下面的一些缺点：

  - MySQL 需要读取表中索引列的值来参与散列计算，散列计算是一个比较耗时的操作。也就是说，相对于 B- 树索引来说，建立哈希索引会耗费更多的时间。
  - 不能使用 HASH 索引排序。
  - HASH 索引只支持等值比较，如“=”“IN()”或“<=>”。
  - HASH 索引不支持键的部分匹配，因为在计算 HASH 值的时候是通过整个索引值来计算的。

建立索引的时候应该遵循以下原则：

+ 选择唯一性索引
+ 为经常需要排序、分组和联合操作的字段建立索引
+ 为常作为查询条件的字段建立索引
+ 限制索引的数目
+ 尽量使用数据量少的索引
+ 数据量小的表不要使用索引
+ 尽量使用前缀来索引

# 存储过程

存储过程一个可编程的函数，它在数据库中创建并保存，一般由 SQL 语句和一些特殊的控制结构组成

# 触发器

触发器和存储过程一样，都是嵌入到 MySQL 中的一段程序，但只有执行 INSERT、UPDATE 和 DELETE 操作时才能激活触发器

触发器与数据表关系密切，主要用于保护表中的数据。特别是当有多个表具有一定的相互联系的时候，触发器能够让不同的表保持数据的一致性。

# 事务

事务可以将一系列的数据操作捆绑成一个整体进行统一管理，如果某一事务执行成功，则在该事务中进行的所有数据更改均会提交，成为数据库中的永久组成部分。如果事务执行时遇到错误，则就必须取消或回滚。取消或回滚后，数据将全部恢复到操作前的状态，所有数据的更改均被清除。

在数据库系统上执行并发操作时，事务是作为最小的控制单元来使用的，特别适用于多用户同时操作的数据库系统。例如，航空公司的订票系统、银行、保险公司以及证券交易系统等

事务具有 4 个特性，即**原子性（Atomicity）**、**一致性（Consistency）**、**隔离性（Isolation）**和**持久性（Durability）**，这 4 个特性通常简称为 ACID：

+ 原子性 不可分割的最小操作单位，或同时成功，或同时失败
+ 持久性 事务提交或回滚后，数据库会持久地保存数据
+ 隔离性 多个事务之间相互影响
+ 一致性 事务操作前后总量不变

```mysql
BEGIN;		-- 开始
COMMIT;		-- 提交
ROLLBACK;	-- 回滚

mysql> USE mybank;
mysql> BEGIN;
mysql> UPDATE bank SET currentMoney = currentMoney-500
    -> WHERE customerName='张三';
mysql> UPDATE bank SET currentMoney = currentMoney+500
    -> WHERE customerName='李四';
mysql> COMMIT;    
```

事务是一项非常消耗资源的功能，使用过程中应注意：

1) **事务尽可能简短**

2) **事务中访问的数据量尽量最少**

3) **查询数据时尽量不要使用事务**

4) **在事务处理过程中尽量不要出现等待用户输入的操作**

## 设置自动提交

```mysql
-- SHOW VARIABLES LIKE 'autocommit';	
SELECT @@autocommit;				-- 查看自动提交是否开启
SET @@autocommit = 0|1|ON|OFF;		-- 设置自动提交
```

mysql默认自动提交；若关闭自动提交，则所有改变不会提交，直到出现COMMIT语句；

## 事务隔离级别

问题

**脏读** 一个事务正在访问数据，并且对数据进行了修改，但是这种修改还没有提交到数据库中，这时，另外一个事务也访问这个数据，然后使用了这个数据

**不可重复读** 一个事务内，多次读取的同一个数据不同

**幻读** 一个事务操作表中的所有记录，另一个事务添加了一条数据

MySQL 包括的事务隔离级别如下：

- **读未提交（READ UNCOMITTED）**

  一个事务可以读取另一个未提交事务修改过的数据

  （脏读，不可重复读，幻读）

- **读提交（READ COMMITTED）**

  一个事务只能读取到另一个已提交事务修改过的数据

  （不可重复读，幻读）

- **可重复读（REPEATABLE READ）**

  一个事务只能读取到另一个已提交事务修改过的数据，但是第一次读过某条记录后，即使其它事务修改了该记录的值并且提交，之后该事务再读该条记录时，读到的仍是第一次读到的值

  （幻读）

- **串行化（SERIALIZABLE）**

隔离级别越高，安全性越高，效率越低

```mysql
-- 查看TIL
SELECT @@transaction_isolation;

-- 设置TIL
SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET GLOBAL TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
```

# 用户管理

user 表是 MySQL 中最重要的一个权限表，用来记录允许连接到服务器的账号信息

| 字段名                 | 字段类型                          | 是否为空 | 默认值                | 说明                                                         |
| ---------------------- | --------------------------------- | -------- | --------------------- | ------------------------------------------------------------ |
| Host                   | char(60)                          | NO       | 无                    | 主机名                                                       |
| User                   | char(32)                          | NO       | 无                    | 用户名                                                       |
| authentication_string  | text                              | YES      | 无                    | 密码                                                         |
| Select_priv            | enum('N','Y')                     | NO       | N                     | 是否可以通过SELECT  命令查询数据                             |
| Insert_priv            | enum('N','Y')                     | NO       | N                     | 是否可以通过  INSERT 命令插入数据                            |
| Update_priv            | enum('N','Y')                     | NO       | N                     | 是否可以通过UPDATE  命令修改现有数据                         |
| Delete_priv            | enum('N','Y')                     | NO       | N                     | 是否可以通过DELETE  命令删除现有数据                         |
| Create_priv            | enum('N','Y')                     | NO       | N                     | 是否可以创建新的数据库和表                                   |
| Drop_priv              | enum('N','Y')                     | NO       | N                     | 是否可以删除现有数据库和表                                   |
| Reload_priv            | enum('N','Y')                     | NO       | N                     | 是否可以执行刷新和重新加载MySQL所用的各种内部缓存的特定命令，包括日志、权限、主机、查询和表 |
| Shutdown_priv          | enum('N','Y')                     | NO       | N                     | 是否可以关闭MySQL服务器。将此权限提供给root账户之外的任何用户时，都应当非常谨慎 |
| Process_priv           | enum('N','Y')                     | NO       | N                     | 是否可以通过SHOW  PROCESSLIST命令查看其他用户的进程          |
| File_priv              | enum('N','Y')                     | NO       | N                     | 是否可以执行SELECT  INTO OUTFILE和LOAD DATA INFILE命令       |
| Grant_priv             | enum('N','Y')                     | NO       | N                     | 是否可以将自己的权限再授予其他用户                           |
| References_priv        | enum('N','Y')                     | NO       | N                     | 是否可以创建外键约束                                         |
| Index_priv             | enum('N','Y')                     | NO       | N                     | 是否可以对索引进行增删查                                     |
| Alter_priv             | enum('N','Y')                     | NO       | N                     | 是否可以重命名和修改表结构                                   |
| Show_db_priv           | enum('N','Y')                     | NO       | N                     | 是否可以查看服务器上所有数据库的名字，包括用户拥有足够访问权限的数据库 |
| Super_priv             | enum('N','Y')                     | NO       | N                     | 是否可以执行某些强大的管理功能，例如通过KILL命令删除用户进程；使用SET  GLOBAL命令修改全局MySQL变量，执行关于复制和日志的各种命令。（超级权限） |
| Create_tmp_table_priv  | enum('N','Y')                     | NO       | N                     | 是否可以创建临时表                                           |
| Lock_tables_priv       | enum('N','Y')                     | NO       | N                     | 是否可以使用LOCK  TABLES命令阻止对表的访问/修改              |
| Execute_priv           | enum('N','Y')                     | NO       | N                     | 是否可以执行存储过程                                         |
| Repl_slave_priv        | enum('N','Y')                     | NO       | N                     | 是否可以读取用于维护复制数据库环境的二进制日志文件           |
| Repl_client_priv       | enum('N','Y')                     | NO       | N                     | 是否可以确定复制从服务器和主服务器的位置                     |
| Create_view_priv       | enum('N','Y')                     | NO       | N                     | 是否可以创建视图                                             |
| Show_view_priv         | enum('N','Y')                     | NO       | N                     | 是否可以查看视图                                             |
| Create_routine_priv    | enum('N','Y')                     | NO       | N                     | 是否可以更改或放弃存储过程和函数                             |
| Alter_routine_priv     | enum('N','Y')                     | NO       | N                     | 是否可以修改或删除存储函数及函数                             |
| Create_user_priv       | enum('N','Y')                     | NO       | N                     | 是否可以执行CREATE  USER命令，这个命令用于创建新的MySQL账户  |
| Event_priv             | enum('N','Y')                     | NO       | N                     | 是否可以创建、修改和删除事件                                 |
| Trigger_priv           | enum('N','Y')                     | NO       | N                     | 是否可以创建和删除触发器                                     |
| Create_tablespace_priv | enum('N','Y')                     | NO       | N                     | 是否可以创建表空间                                           |
| ssl_type               | enum('','ANY','X509','SPECIFIED') | NO       |                       | 支持ssl标准加密安全字段                                      |
| ssl_cipher             | blob                              | NO       |                       | 支持ssl标准加密安全字段                                      |
| x509_issuer            | blob                              | NO       |                       | 支持x509标准字段                                             |
| x509_subject           | blob                              | NO       |                       | 支持x509标准字段                                             |
| plugin                 | char(64)                          | NO       | mysql_native_password | 引入plugins以进行用户连接时的密码验证，plugin创建外部/代理用户 |
| password_expired       | enum('N','Y')                     | NO       | N                     | 密码是否过期 (N  未过期，y 已过期)                           |
| password_last_changed  | timestamp                         | YES      |                       | 记录密码最近修改的时间                                       |
| password_lifetime      | smallint(5)  unsigned             | YES      |                       | 设置密码的有效时间，单位为天数                               |
| account_locked         | enum('N','Y')                     | NO       | N                     | 用户是否被锁定（Y  锁定，N 未锁定）                          |
| max_questions          | int(11)  unsigned                 | NO       | 0                     | 规定每小时允许执行查询的操作次数                             |
| max_updates            | int(11)  unsigned                 | NO       | 0                     | 规定每小时允许执行更新的操作次数                             |
| max_connections        | int(11)  unsigned                 | NO       | 0                     | 规定每小时允许执行的连接操作次数                             |
| max_user_connections   | int(11)  unsigned                 | NO       | 0                     | 规定允许同时建立的连接次数                                   |

## 用户

**创建用户**

```mysql
-- CREATE语句
CREATE USER 'username'@'host' IDENTIFIED BY 'password';

-- 不设定主机名,即对所有主机开放该用户的使用
CREATE USER 'test1' IDENTIFIED BY 'test1';

-- 避免明文存储,可以存储哈希值
CREATE USER 'test1'@'localhost' IDENTIFIED BY PASSWORD '*06C0BF5B64ECE2F648B5F048A71903906BA08E5C';

-- INSERT语句
INSERT INTO mysql.user(Host, User, authentication_string, ssl_cipher, x509_issuer, x509_subject) 
VALUES ('localhost', 'test2', PASSWORD('test2'), '', '', '');

FLUSH PRIVILEGES;		-- 刷新系统权限相关表

-- GRANT语句
GRANT SELECT ON *.* TO 'test3'@localhost IDENTIFIED BY 'test3';

```

**修改和删除用户**

```mysql
-- 重命名
RENAME USER <旧用户> TO <新用户>;

DROP USER 'username'@'host';
```

## 用户权限

**查看权限**

```mysql
SELECT * FROM mysql.user;

SHOW GRANTS FOR 'username'@'hostname';
```

**用户授权**

```mysql
mysql> GRANT SELECT,INSERT ON *.*
    -> TO 'testUser'@'localhost'
    -> IDENTIFIED BY 'testPwd'
    -> WITH GRANT OPTION;			-- 可以授权自己的权力
```

 GRANT 语句中可用于指定权限级别的值有以下几类格式：

- *：表示当前数据库中的所有表。
- \*.*：表示所有数据库中的所有表。
- db_name.*：表示某个数据库中的所有表，db_name 指定数据库名。
- db_name.tbl_name：表示某个数据库中的某个表或视图，db_name 指定数据库名，tbl_name 指定表名或视图名。
- db_name.routine_name：表示某个数据库中的某个存储过程或函数，routine_name 指定存储过程名或函数名。
- TO 子句：如果权限被授予给一个不存在的用户，MySQL 会自动执行一条 CREATE USER 语句来创建这个用户，但同时必须为该用户设置密码。

**收回权限**

```mysql
mysql> REVOKE INSERT ON *.*
    -> FROM 'testUser'@'localhost';
```

**登录和注销**

```mysql
mysql -h localhost -u root -p;

QUIT;
```

**修改普通用户密码**

```mysql
-- root操作
SET PASSWORD FOR 'testuser'@'localhost' = PASSWORD("newpwd");

-- 用户操作
SET PASSWORD = PASSWORD('newpwd1');

-- UPDATE语句
UPDATE MySQL.user SET authentication_string = PASSWORD("newpwd") WHERE User = "username" AND Host = "hostname";

-- GRANT语句
GRANT USAGE ON *.* TO 'testuser'@'localhost' IDENTIFIED BY 'newpwd3';
```

**修改root密码**

```mysql
SET PASSWORD = password ("rootpwd3");

-- 命令行
mysqladmin -u username -h hostname -p password "newpwd"

-- 修改user表
UPDATE mysql.user set authentication_string = password ("rootpwd2")
WHERE User = "root" and Host = "localhost";
```

# 数据库备份与恢复

## 备份与恢复

**mysqldump**

```mysql 
-- 备份一张表
mysqldump -u username -p dbname [tbname]> filename.sql

-- 备份多个数据库
mysqldump -u username -p --databases dbname1 dbname2 > filename.sql

-- 备份所有数据库
mysqldump -u username -p --all-databases >filename.sql

```

**mysql**

```mysql
mysql -u username -p [dbname] < filename.sql
```

## 导出与导入

**INTO OUTFILE**

```mysql
SELECT * FROM test.person INTO OUTFILE 'C:/person.txt';

SELECT * FROM test.person INTO OUTFILE 'C:/person.txt'
FIELDS TERMINATED BY '\、' 		-- 字段之间用、分隔
OPTIONALLY ENCLOSED BY '\"' 	-- str字段用"引用
LINES STARTING BY '\-'			-- 每行以-开头
TERMINATED BY '\r\n';			-- 每行以回车换行符结尾
```

```
-1、"Java"、12
-2、"MySQL"、13
-3、"C"、15
-4、"C++"、22
-5、"Python"、18
```

**LOAD DATA**

```mysql
mysql> LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 5.7/
Uploads/file.txt'
    -> INTO TABLE test_db.tb_students_copy
    -> FIELDS TERMINATED BY '\、'
    -> OPTIONALLY ENCLOSED BY '\"'
    -> LINES TERMINATED BY '\-';
```

# 字符集

**字符（Character）**是计算机中字母、数字、符号的统称，一个字符可以是一个中文汉字、一个英文字母、一个阿拉伯数字、一个标点符号等。

**字符集（Character set）**定义了字符和二进制的对应关系，为字符分配了唯一的编号。常见的字符集有 ASCII、GBK、IOS-8859-1 等。

**字符编码（Character encoding）**也可以称为字集码，规定了如何将字符的编号存储到计算机中。

> 大部分字符集都只对应一种字符编码，例如：ASCII、IOS-8859-1、GB2312、GBK，都是既表示了字符集又表示了对应的字符编码。所以一般情况下，可以将两者视为同义词。Unicode 字符集除外，Unicode 有三种编码方案，即 UTF-8、UTF-16 和 UTF-32。最为常用的是 UTF-8 编码。

```mysql
-- 查看字符集
SHOW VARIABLES LIKE 'character%';

-- 查看校对规则
SHOW VARIABLES LIKE 'collation%';
```

| 名称                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| character_set_client     | MySQL 客户端使用的字符集                                     |
| character_set_connection | 连接数据库时使用的字符集                                     |
| character_set_database   | 创建数据库使用的字符集                                       |
| character_set_filesystem | MySQL 服务器文件系统使用的字符集，默认值为 binary，不做任何转换 |
| character_set_results    | 数据库给客户端返回数据时使用的字符集                         |
| character_set_server     | MySQL 服务器使用的字符集，建议由系统自己管理，不要人为定义   |
| character_set_system     | 数据库系统使用的字符集，默认值为 utf8，不需要设置            |
| character_sets_dir       | 字符集的安装目录                                             |

```mysql
-- 修改字符集和校对规则
-- 数据库
ALTER DATABASE <数据库名>
CHARACTER SET utf8;

-- 表

-- 列
ALTER TABLE tb_students_info MODIFY name VARCHAR(10) CHARACTER SET gbk;

```

# 数据库范式

+ 第一范式1NF：每一列都是不可分割的原子数据项
+ 第二范式2NF：非码属性必须完全依赖于码属性（消除部分依赖）
+ 第三范式3NF：任何非主属性不依赖其他非主属性（消除传递依赖）

