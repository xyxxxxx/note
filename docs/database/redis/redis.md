## intro

非关系型数据库将数据存储在内存中

redis是用C语言开发的一款高性能的键值对数据库，50个并发执行100 000个请求，读的速度110 000次/s，写的速度81 000次/s

应用场景

+ 缓存
+ 在线好友
+ 任务队列
+ 应用排行榜
+ 网站统计
+ 数据国企处理
+ 分布式集群



## 快速入门

### 数据结构

键值对<key,value>，其中key为字符串，value有5种数据结构：

1. 字符串 string
2. 哈希 hash
3. 列表 list
4. 集合 set
5. 有序集合 sortedset

### 数据操作

```redis
//string
SET <key> <value>
GET <key>
DEL <key>

//hash
HSET <map> <field> <value>
HGET <map> <field>
HGETALL <map>
HDEL <map> <field>
HDEL <map>

//list
LPUSH <list> <value>
RPUSH <list> <value>
LRANGE <list> <start> <end>
	LRANGE <list> 0 -1
LPOP <list>
RPOP <list>

//set
SADD <set> <value>
SMEMBERS <set>
SREM <set> <value>

//sortedset
ZADD <set> <score> <value>
ZRANGE <set> <start> <end>
ZRANGE <set> <start> <end> WITHSCORES
ZREM <set> <value>
```

通用命令

```redis
KEYS *
	KEYS <RegularExpression>
//返回所有数据项

TYPE <key>

DEL <KEY>

```



## 持久化



appendonly



