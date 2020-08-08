

# 结构

## 结构体

```c
struct date    //定义日期结构体由3个int字段组成
{
    int year, month, day;
}today;        //变量today包括3个int字段，占用连续的3*2字节

struct student //结构体的嵌套
{
    int stuno;
    char name[20];
    struct date birthday;
};
struct student stu1={2001001,"Zhang San",1990,1,1};//结构体变量的初始化

stu1.num=2001001;   //数据访问
stu1.birthday.year=1990;
struct student stu[]={{},{},...,{}}; //结构体数组的初始化
stu[1].num=2001001;                  //数据访问

struct date *p, date[]={{2001,1,1},{2002,2,2},{2003,3,3}}; //结构体指针
p=date;                              //即p=date[0]
//p->year==(*p).year==2001 表示指针p所在结构变量的year字段，或者*p取所在结构变量
//(++p)->month==2 由于p已经被定义为date结构的指针,结构变量将被视作一个整体,++p会从第一个结构变量的起始地址跳到第二个的起始地址,
//++p->day 将第二个结构变量的day字段+1,因为->的运算优先级高于++

sizeof(stu1);        //返回结构变量的字节数
#define NKEYS (sizeof(stu)/sizeof(stu[0])) //结构数组的规模
```

## 联合

```c
union u
{
    char u1;
    int u2;
    long u3;
};             //同一内存段中可存放几种类型的变量，变量会覆盖存放
u.u1           //引用方式
&u=&u.u1=&u.u2 //联合仅有一地址
//联合不能初始化、赋值，不能作为函数参数，但可以使用指针访问
```

## 链表

```c
//创建单向链表
struct node         //1.定义结点结构
{ char name[20],addr[20],tel[15];
  struct node *link //包含同结构指针
};
typedef struct node NODE;//结点结构重命名
NODE *head;//2.表头指针head
head=(NODE *)malloc(sizeof(NODE));////3.表头结点  开辟新存储区,强制转换malloc的返回值类型为NODE指针
head->link=NULL;
NODE *p;//4.添加数据节点
p=(NODE *)malloc(sizeof(NODE));
gets(p->name);gets(p->addr);gets(p->tel);
p->link=NULL;
head->link=p;
NODE *p;//5.插入数据节点
p=(NODE *)malloc(sizeof(NODE));
gets(p->name);gets(p->addr);gets(p->tel);
p->link=head->link;
head.link->=p;
-----
void output(NODE *head) //创建函数访问链表数据
{
    NODE *p;
    p=head->link;
    while (p!=NULL)
    {
        puts(p->name);
        p=p->link;
    }
}    

void insert(NODE *head, NODE *p, int i)//第i个位置插入节点p
{
    NODE *q;
    int n=0;
    for (q=head;n<i && q->link!=NULL;++n)
        q=q->link;
    p->link=q->link;
    q->link=p;
}
   
void delete(NODE *head, int i) //删除第i个节点
{
    NODE *q, *p;//p指针指向被删结点
    int n;
    for (n=0,q=head;n<i-1 && q->link!=NULL;++n)
        q=q->link;
    p=q->link;
    q->link=p->link;
    free(p);//释放被删除结点内存
}
```

