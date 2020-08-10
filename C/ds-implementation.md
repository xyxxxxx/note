# 链表

## 单向链表

```c
struct node        　　　　　　　　　 //1.定义结点结构
{ char name[20],addr[20],tel[15];
  struct node *link; //包含同结构体指针
};
typedef struct node NODE;//结点结构重命名


NODE *head;　　　　　　　　　　　　　　//2.表头指针head
head=(NODE *)malloc(sizeof(NODE));//3.为表头结点分配新存储区,强制转换malloc的返回值类型为NODE指针
head->link=NULL;
NODE *p;                          //4.添加数据节点
p=(NODE *)malloc(sizeof(NODE));
gets(p->name);gets(p->addr);gets(p->tel);
```

```c
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





# 栈

```c

```





