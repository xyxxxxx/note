# 数据库

**查看数据库**

```mysql
SHOW DATABASES;						--显示所有db
SHOW DATABASES LIKE 'db_name';		--匹配db名称

SHOW CREATE TABLE <db_name>;		--查看创建db
```

```mysql
SHOW DATABASES LIKE 'test_db';		--完全匹配
SHOW DATABASES LIKE '%test%';		--包含匹配
SHOW DATABASES LIKE 'db%';			--开头匹配
SHOW DATABASES LIKE '%db';			--结尾匹配
```

**创建数据库**

```mysql
CREATE DATABASE [IF NOT EXISTS] <数据库名>
[DEFAULT CHARACTER SET <字符集名>] 
[DEFAULT COLLATE <校对规则名>];

CREATE DATABASE test_db;			--创建db

CREATE DATABASE IF NOT EXISTS test_db_char
DEFAULT CHARACTER SET utf8			--默认字符集utf8
DEFAULT COLLATE utf8_chinese_ci;	--默认校对规则utf8_chinese_ci
```

**修改数据库**

```mysql
ALTER DATABASE <数据库名> { 
[DEFAULT CHARACTER SET <字符集名>] 
[DEFAULT COLLATE <校对规则名>];
```

**删除数据库**

```mysql
DROP DATABASE [IF EXISTS] <数据库名>
```

**选择数据库**

```mysql
USE <数据库名>
```





# 表和字段

**创建表**

```mysql
CREATE TABLE <表名> ([表定义选项])[表选项][分区选项];

CREATE TABLE tb_emp1
(
id INT(11),
name VARCHAR(25),
deptId INT(11),
salary FLOAT
);
```

**查看表**

```mysql
SHOW TABLES;	--显示所有表
DESC <表名>;	   --查看表的结构
SHOW CREATE TABLE <表名>;		--查看创建表
```

**修改表**

```mysql
ALTER TABLE <表名>
{ ADD COLUMN <列名> <类型>
| CHANGE COLUMN <旧列名> <新列名> <新列类型>
| ALTER COLUMN <列名> { SET DEFAULT <默认值> | DROP DEFAULT }
| MODIFY COLUMN <列名> <类型>
| DROP COLUMN <列名>
| RENAME TO <新表名>
| CHARACTER SET <字符集名>
| COLLATE <校对规则名> }
ALTER TABLE <旧表名> RENAME TO <新表名>;

ALTER TABLE 表名 [DEFAULT] CHARACTER SET <字符集名> [DEFAULT] COLLATE <校对规则名>;		-- 修改表字符集

```

**克隆表**

create table like方式会完整地克隆表结构，但不会插入数据，需要单独使用insert into或load data方式加载数据
create table as  方式会部分克隆表结构，完整保留数据



**删除表**

```mysql
DROP TABLE [IF EXISTS] 表名1 [ ,表名2, 表名3 ...]
-- 必须先解除外键约束

```

**添加字段**

```mysql
-- 开头添加字段
ALTER TABLE <表名> ADD <新字段名> <数据类型> [约束条件] FIRST;

-- 末尾添加字段
ALTER TABLE <表名> ADD <新字段名><数据类型> [约束条件];

-- 中间添加字段
ALTER TABLE <表名> ADD <新字段名> <数据类型> [约束条件] AFTER <已经存在的字段名>;
```

**修改字段**

```mysql
ALTER TABLE <表名> CHANGE <旧字段名> <新字段名> <新数据类型>；

ALTER TABLE <表名> MODIFY <字段名> <数据类型>

ALTER TABLE <表名> DROP <字段名>；
```





# 约束

**主键约束**

```mysql
-- 创建主键约束
mysql> CREATE TABLE tb_emp3
    -> (
    -> id INT(11) PRIMARY KEY,
    -> name VARCHAR(25),
    -> deptId INT(11),
    -> salary FLOAT
    -> );

mysql> CREATE TABLE tb_emp4
    -> (
    -> id INT(11),
    -> name VARCHAR(25),
    -> deptId INT(11),
    -> salary FLOAT,
    -> PRIMARY KEY(id)
    -> );
    
-- 创建联合主键约束
mysql> CREATE TABLE tb_emp5
    -> (
    -> name VARCHAR(25),
    -> deptId INT(11),
    -> salary FLOAT,
    -> PRIMARY KEY(name,deptId)
    -> );
    
-- 修改主键约束
ALTER TABLE <数据表名> ADD PRIMARY KEY(<字段名>);

-- 删除主键约束
ALTER TABLE <数据表名> DROP PRIMARY KEY;
```

**外键约束**

```mysql
-- 创建外键约束
mysql> CREATE TABLE tb_emp6
    -> (
    -> id INT(11) PRIMARY KEY,
    -> name VARCHAR(25),
    -> deptId INT(11),
    -> salary FLOAT,
    -> CONSTRAINT fk_emp_dept1
    -> FOREIGN KEY(deptId) REFERENCES tb_dept1(id)
    -> );

-- 增加外键约束
mysql> ALTER TABLE tb_emp2
    -> ADD CONSTRAINT fk_tb_dept1 
    -> FOREIGN KEY(deptId) REFERENCES tb_dept1(id);

-- 删除外键约束
ALTER TABLE <表名> DROP FOREIGN KEY <外键约束名>;
```

**唯一约束**

```mysql
-- 创建
mysql> CREATE TABLE tb_dept2
    -> (
    -> id INT(11) PRIMARY KEY,
    -> name VARCHAR(22) UNIQUE,
    -> location VARCHAR(50)
    -> );

-- 增加
mysql> ALTER TABLE tb_dept1
    -> ADD CONSTRAINT unique_name UNIQUE(name);
    
-- 删除
ALTER TABLE <表名> DROP INDEX <唯一约束名>;
```

**检查约束**

```mysql
-- 创建
mysql> CREATE TABLE tb_emp7
    -> (
    -> id INT(11) PRIMARY KEY,
    -> name VARCHAR(25),
    -> deptId INT(11),
    -> salary FLOAT,
    -> CHECK(salary>0 AND salary<100),
    -> FOREIGN KEY(deptId) REFERENCES tb_dept1(id)
    -> );

-- 增加
mysql> ALTER TABLE tb_emp7
    -> ADD CONSTRAINT check_id
    -> CHECK(id>0);

-- 删除
ALTER TABLE <数据表名> DROP CONSTRAINT <检查约束名>;
```

**默认值约束**

```mysql
-- 创建
mysql> CREATE TABLE tb_dept3
    -> (
    -> id INT(11) PRIMARY KEY,
    -> name VARCHAR(22),
    -> location VARCHAR(50) DEFAULT 'Beijing'
    -> );

-- 增加
mysql> ALTER TABLE tb_dept3
    -> CHANGE COLUMN location
    -> location VARCHAR(50) DEFAULT 'Shanghai';

-- 删除
mysql> ALTER TABLE tb_dept3
    -> CHANGE COLUMN location
    -> location VARCHAR(50) DEFAULT NULL;
```

**非空约束**

```mysql
-- 创建
mysql> CREATE TABLE tb_dept4
    -> (
    -> id INT(11) PRIMARY KEY,
    -> name VARCHAR(22) NOT NULL,
    -> location VARCHAR(50)
    -> );

-- 增加
mysql> ALTER TABLE tb_dept4
    -> CHANGE COLUMN location
    -> location VARCHAR(50) NOT NULL;

-- 删除
mysql> ALTER TABLE tb_dept4
    -> CHANGE COLUMN location
    -> location VARCHAR(50) NULL;
```





# 计算

**基本运算**

```mysql
--算术运算
+ - * / %
--逻辑运算
! && || XOR
--比较,真返回1,假返回0
= <=> !=或<> <= >= < > ISNULL BETWEEN AND
--位运算
& | ^ ~ << >>
```

**IN & NOT IN**

```mysql
expr IN ( value1, value2, value3 ... valueN )
expr NOT IN ( value1, value2, value3 ... valueN )

+---------------------+---------------------------+
| 2 IN (1,3,5,'thks') | 'thks' IN (1,3,5, 'thks') |
+---------------------+---------------------------+
|                   0 |                         1 |
+---------------------+---------------------------+

--NULL的处理
+------------------------+-------------------------+--------------------------+
| NULL IN (1,3,5,'thks') | 10 IN (1,3,NULL,'thks') | 10 IN (1,10,NULL,'thks') |
+------------------------+-------------------------+--------------------------+
|                   NULL |                    NULL |                        1 |
+------------------------+-------------------------+--------------------------+
```

**函数**

http://c.biancheng.net/mysql/function/





# 查询

**查询**

```mysql
-- 查询全表
SELECT * FROM <表名>;	

-- 查询指定字段
SELECT <字段名1>,<字段名2> FROM <表名>;

-- 去重查询,即给出所有组合值
SELECT DISTINCT <字段名1>,<字段名2> FROM <表名>;
SELECT COUNT(DISTINCT <字段名1>,<字段名2>) FROM <表名>;	-- 计数
```

**别名**

```mysql
SELECT stu.name,stu.height FROM tb_students_info AS stu;
SELECT name AS student_name, age AS student_age FROM tb_students_info;
```

**限制查询结果条数**

```mysql
SELECT * FROM tb_students_info LIMIT 5;		-- LIMIT 记录数
SELECT * FROM tb_students_info LIMIT 3,5;	-- LIMIT 初始位置,记录数
SELECT * FROM tb_students_info LIMIT 5 OFFSET 3;
```

**排序**

```mysql
SELECT * FROM tb_students_info ORDER BY height;			-- 排升序
SELECT * FROM tb_students_info ORDER BY height DESC;	-- 排降序
SELECT * FROM tb_students_info ORDER BY height,name;	-- 依次排升序
```

**条件查询**

```mysql
SELECT name,height FROM tb_students_info
WHERE height=170;

SELECT name,age,height FROM tb_students_info
WHERE age>21 && height>=175;

```

**模糊查询**

```mysql
SELECT name FROM tb_students_info
WHERE name [NOT] LIKE 'T%';			-- %占任意位,'%'匹配任意字符串除NULL

SELECT name FROM tb_students_info
WHERE name LIKE '____y';			-- _占1位

SELECT name FROM tb_students_info
WHERE name [NOT] LIKE BINARY 't%';	-- 区分大小写


```

**范围查询**

```mysql
SELECT name,age FROM tb_students_info 
WHERE age BETWEEN 20 AND 23;
```

**空值查询**

```mysql
SELECT name,login_date FROM tb_students_info 
WHERE login_date IS [NOT] NULL;
```

**分组查询**

```mysql
SELECT name,sex FROM tb_students_info 
GROUP BY sex;			-- 查询每组的第一条记录

SELECT GROUP_CONCAT(name),sex,age FROM tb_students_info 
GROUP BY age,sex;		-- 查询每组的所有记录

SELECT sex,COUNT(sex) FROM tb_students_info 
GROUP BY sex;			-- 统计每组数量

SELECT GROUP_CONCAT(name),sex FROM tb_students_info 
GROUP BY sex WITH ROLLUP;	-- 查询每组的所有记录后求和
```

**过滤分组**

```mysql
SELECT name,sex FROM tb_students_info 
WHERE height>150;		 

SELECT name,sex FROM tb_students_info 
HAVING height>150;		-- error,查询字段中没有height

SELECT GROUP_CONCAT(name),sex,height FROM tb_students_info 
GROUP BY height
HAVING AVG(height)>170; -- ???过滤条件
```

**交叉连接**

不建议使用

**内连接**

```mysql
SELECT s.name,c.course_name FROM tb_students_info s INNER JOIN tb_course c 
ON s.course_id = c.id;		-- 连接条件

SELECT s.name,c.course_name FROM tb_students_info s, tb_course c 
WHERE s.course_id = c.id;	-- 隐式内连接


```

**外连接**

```mysql
SELECT s.name,c.course_name FROM tb_students_info s LEFT OUTER JOIN tb_course c 				-- 左连接,以左表为基表
ON s.`course_id`=c.`id`;

SELECT s.name,c.course_name FROM tb_students_info s RIGHT OUTER JOIN tb_course c 				 -- 右连接,以右表为基表
ON s.`course_id`=c.`id`;
```

**子查询**

```mysql
SELECT name FROM tb_students_info 
WHERE course_id IN (SELECT id FROM tb_course WHERE course_name = 'Java');		-- IN连接的嵌套查询,可以用=替换
-- 相反的连接使用NOT IN或<>

SELECT * FROM tb_students_info
WHERE age>24 AND EXISTS(SELECT course_name FROM tb_course WHERE id=1);
-- 若EXISTS语句正确,则返回TRUE

-- SELECT语句中的子查询
SELECT (子查询) FROM 表名;
SELECT * FROM (子查询) AS 表的别名;
```

**正则表达式匹配**

```mysql
SELECT * FROM tb_students_info 
WHERE name REGEXP '^J';
```





# 数据操作

**插入**

```mysql
-- INSERT VALUES语句
mysql> INSERT INTO tb_courses
    -> (course_id,course_name,course_grade,course_info)
    -> VALUES(1,'Network',3,'Computer Network');

mysql> INSERT INTO tb_courses	-- 自定义字段顺序
    -> (course_name,course_info,course_id,course_grade)
    -> VALUES('Database','MySQL',2,3);

mysql> INSERT INTO tb_courses	-- 默认字段顺序
    -> VALUES(3,'Java',4,'Java EE');
    
mysql> INSERT INTO tb_courses	-- 部分字段
    -> (course_name,course_grade,course_info)
    -> VALUES('System',3,'Operation System');    

mysql> INSERT INTO tb_courses_new	-- 复制表
    -> (course_id,course_name,course_grade,course_info)
    -> SELECT course_id,course_name,course_grade,course_info
    -> FROM tb_courses;



-- INSERT SET语句

```

**修改**

```mysql
mysql> UPDATE tb_courses_new
    -> SET course_grade=4;
    
mysql> UPDATE tb_courses_new
    -> SET course_name='DB',course_grade=3.5
    -> WHERE course_id=2;    
    
```

**删除**

```mysql
DELETE FROM tb_courses_new;		-- 删除全部

mysql> DELETE FROM tb_courses
    -> WHERE course_id=4;
```





# 视图

**创建**

```mysql
mysql> CREATE VIEW v_students_info
    -> (s_id,s_name,d_id,s_age,s_sex,s_height,s_date)
    -> AS SELECT id,name,dept_id,age,sex,height,login_date
    -> FROM tb_students_info;
```

**查看**

```mysql
DESC <视图名>;

SHOW CREATE VIEW <视图名> \G;	-- 查看创建
```

**修改**

```mysql
mysql> ALTER VIEW view_students_info	-- 相当于重新创建
    -> AS SELECT id,name,age
    -> FROM tb_students_info;
    
mysql> UPDATE view_students_info
    -> SET age=25 WHERE id=1;    
```

**删除**

```mysql
DROP VIEW IF EXISTS <视图名>;
```





# 索引

**创建**

```mysql
CREATE <索引名> ON <表名>

mysql> CREATE TABLE tb_stu_info2
    -> (
    -> id INT NOT NULL,
    -> name CHAR(45) DEFAULT NULL,
    -> dept_id INT DEFAULT NULL,
    -> age INT DEFAULT NULL,
    -> height INT DEFAULT NULL,
    -> UNIQUE INDEX(height)			-- 唯一索引
    -> );

ALTER TABLE <表名>
ADD UNIQUE INDEX(<列名>);
```

**查看**

```mysql
SHOW INDEX FROM <表名> FROM <数据库名>;
```

| 参数         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| Table        | 表示创建索引的数据表名。                                     |
| Non_unique   | 表示该索引是否是唯一索引。若不是唯一索引，则该列的值为 1；若是唯一索引，则该列的值为 0。 |
| Key_name     | 表示索引的名称。                                             |
| Seq_in_index | 表示该列在索引中的位置，如果索引是单列的，则该列的值为 1；如果索引是组合索引，则该列的值为每列在索引定义中的顺序。 |
| Column_name  | 表示定义索引的列字段。                                       |
| Collation    | 表示列以何种顺序存储在索引中。在 MySQL 中，升序显示值“A”（升序），若显示为 NULL，则表示无分类。 |
| Cardinality  | 索引中唯一值数目的估计值。基数根据被存储为整数的统计数据计数，所以即使对于小型表，该值也没有必要是精确的。基数越大，当进行联合时，MySQL 使用该索引的机会就越大。 |
| Sub_part     | 表示列中被编入索引的字符的数量。若列只是部分被编入索引，则该列的值为被编入索引的字符的数目；若整列被编入索引，则该列的值为 NULL。 |
| Packed       | 指示关键字如何被压缩。若没有被压缩，值为 NULL。              |
| Null         | 用于显示索引列中是否包含 NULL。若列含有 NULL，该列的值为 YES。若没有，则该列的值为 NO。 |
| Index_type   | 显示索引使用的类型和方法（BTREE、FULLTEXT、HASH、RTREE）。   |
| Comment      | 显示评注。                                                   |

**修改**



**删除**

```mysql
DROP INDEX <索引名> ON <表名>

mysql> ALTER TABLE tb_stu_info2
    -> DROP INDEX height;
```





# 存储过程

**创建**

```mysql
mysql> DELIMITER //		-- 更改结束符
mysql> CREATE PROCEDURE ShowStuScore()
    -> BEGIN
    -> SELECT * FROM tb_students_score;
    -> END //			-- 结束符
mysql> DELIMITER ;
mysql> CALL ShowStuScore();

mysql> DELIMITER //
mysql> CREATE PROCEDURE GetScoreByStu
    -> (IN name VARCHAR(30))	-- IN表示传入参数
    -> BEGIN
    -> SELECT student_score FROM tb_students_score
    -> WHERE student_name=name;
    -> END //
mysql> DELIMITER ;
mysql> CALL GetScoreByStu('Green');    
```

**查看**

```mysql
mysql> SHOW PROCEDURE STATUS LIKE 'showstuscore' \G

mysql> SHOW CREATE PROCEDURE showstuscore \G		-- 查看创建
```

**修改**

```mysql
ALTER PROCEDURE 存储过程名 [特征...]
```

`特征`指定了存储过程的特性，可能的取值有：

- CONTAINS SQL 表示子程序包含 SQL 语句，但不包含读或写数据的语句。
- NO SQL 表示子程序中不包含 SQL 语句。
- READS SQL DATA 表示子程序中包含读数据的语句。
- MODIFIES SQL DATA 表示子程序中包含写数据的语句。
- SQL SECURITY { DEFINER |INVOKER } 指明谁有权限来执行。
- DEFINER 表示只有定义者自己才能够执行。
- INVOKER 表示调用者可以执行。
- COMMENT 'string' 表示注释信息。

**删除**

```mysql
DROP PROCEDURE IF EXISTS <过程名>;
```





# 触发器

**创建**

```mysql
CREATE TRIGGER <触发器名> 
<BEFORE | AFTER> <INSERT | UPDATE | DELETE> ON <表名>
<FOR EACH Row>
<触发器主体>

-- BEFORE类型
mysql> CREATE TRIGGER SumOfSalary
    -> BEFORE INSERT ON tb_emp8
    -> FOR EACH ROW
    -> SET @sum=@sum+NEW.salary;
    
mysql> SET @sum=0;		-- 定义用户变量
mysql> INSERT INTO tb_emp8
    -> VALUES(1,'A',1,1000),(2,'B',1,500);
mysql> SELECT @sum;     -- 查看用户变量

-- AFTER类型
mysql> CREATE TRIGGER double_salary
    -> AFTER INSERT ON tb_emp6
    -> FOR EACH ROW
    -> INSERT INTO tb_emp7
    -> VALUES (NEW.id,NEW.name,deptId,2*NEW.salary);
```

**查看**

```mysql
SHOW TRIGGERS \G;

-- 查询information_schema 数据库的 triggers 表
SELECT * FROM information_schema.triggers WHERE trigger_name= '触发器名' \G;


```

**删除**

```mysql
DROP TRIGGER IF EXISTS <触发器名>;
```





# 备份和还原

```mysql
-- 备份
mysqldump -uroot -p <dbname> > d://codes//mysql//back.sql

-- 还原
CREATE DATABASE <dbname>;
USE <dbname>;
SOURCE d://codes//mysql//back.sql; 

```

