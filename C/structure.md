# 结构

## 结构体

结构体`struct`定义了包含多个字段的数据类型。

```c
struct date //定义日期结构体由3个int字段组成
{
   int year, month, day;
} today; 　　//结构体变量today包括3个int字段，占用连续的3*2字节

struct student //结构体嵌套
{
   int stuno;
   char name[20];
   struct date birthday;
};


struct student stu1 = {2001001, "Zhang San", 1990, 1, 1}; //初始化结构体变量
stu1.num = 2001001; //数据访问和赋值
stu1.birthday.year = 1990;

struct student stu[] = {  //初始化结构体数组
    {2001001, "Zhang San", 1990, 1, 1},
    {2001002, "Li Si", 1990, 2, 2},
    {2001003, "Wang Wu", 1990, 3, 3}
    }; 
stu[1].num = 2001001;     //数据访问和赋值


struct date *p, dates[] = {{2001, 1, 1}, {2002, 2, 2}, {2003, 3, 3}}; //结构体指针
p = dates;                                                            //即p=dates[0]
//p->year==(*p).year==2001 表示指针p所在结构体变量的year字段，或者使用*p取所在结构变量
//(++p)->month==2 由于p已经被定义为date结构的指针,结构体变量将被视作一个整体,++p会从第一个结构体变量的起始地址跳到第二个的起始地址,
//++p->day 将第二个结构变量的day字段+1,因为->的运算优先级高于++


sizeof(stu1);                                //结构体变量的大小为所有字段的大小之和
#define NKEYS (sizeof(stu) / sizeof(stu[0])) //结构体数组的规模
```



## 联合

联合`union`可在同一段内存中存放几种类型的变量，变量会覆盖存放。

```c
union u_tag    //定义联合可以存放char,int,double类型
{
    char chu;
    int iu;
    double fu;
}u;


u.iu = 65;     // 0x0000000000000041
printf("%d\n",u.iu);  // 65
printf("%c\n",u.chu); // A
printf("%f\n",u.fu);  // 0.000000


int *p = &u.u1; //&u==&u.u1==&u.u2 联合仅有一地址
*p = 65;


sizeof(u);     //联合变量的大小为所有类型的大小的最大值
```

联合不能初始化、赋值，不能作为函数参数，但可以使用指针访问



## 位域

预定义宽度的变量称为位域，在结构中定义如下：

```c
struct{
    unsigned int r: 1;
    unsigned int w: 1;
    unsigned int x: 1;
}access;　　　　　//该变量占用４个字节,但只有３位用于存储值
```

+ 部分机器从左到右使用位，部分机器从右到左，因此造成程序无法移植
+ 指针不适用

