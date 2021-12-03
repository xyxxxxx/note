使用数据库：[中国5级行政区域mysql库](https://github.com/xyxxxxx/china_area_mysql)

## 单表查询

1）显示名称包含“南京”但不属于南京市的所有行政区.

```mysql
SELECT name, merger_name
FROM cnarea_2018
WHERE name LIKE '%南京%' AND city_code NOT IN 
(
		SELECT city_code
		FROM cnarea_2018
		WHERE `name` = '南京市'
			
)
;
```

2）显示香港特别行政区的所有辖区.

```mysql
SELECT name
FROM `cnarea_2018`
WHERE city_code LIKE '00852';

SELECT name
FROM cnarea_2018
WHERE city_code IN 
(
		SELECT city_code
		FROM cnarea_2018
		WHERE `name` = '香港特别行政区'
);
```

3）显示名称首字符为”海“的所有三级行政区.

4）显示各级行政区中最靠南者.

```mysql
SELECT name, merger_name
FROM cnarea_2018
ORDER BY lat LIMIT 1;
```

5）显示上海市各级行政区的所有邮政编码.

```mysql
SELECT DISTINCT zip_code
FROM `cnarea_2018`
WHERE city_code IN
(
		SELECT city_code
		FROM cnarea_2018
		WHERE `short_name` = '上海' AND `level`=1
)
ORDER BY zip_code
;
```

6）显示所有3位区号的行政区.

```MYSQL
SELECT merger_name, city_code
FROM `cnarea_2018`
WHERE `level` = 1 AND city_code LIKE '___';
```

7）显示1级行政区中名称最长者.

```mysql
SELECT `name`,merger_name
FROM `cnarea_2018`
WHERE `name` LIKE '______' AND `level`=1;
```

8）哪个大陆省（市，自治区）有最多的村（社区）？

```MYSQL

```

