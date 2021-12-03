# 素数

素数计数函数：小于或等于x的素数的个数，用 $\pi(x)$ 表示

## 素数判定

```c++
bool isPrime(a) {
  if (a < 2) return 0;
  for (int i = 2; i * i <= a; ++i)
    if (a % i) return 0;
  return 1;
}
```

# 最大公约数和最小公倍数gcd&lcm

**最大公约数gcd,最小公倍数lcm**

**定理** 若 $a|m,b|m$，则 ${\rm lcm}(a,b)|m$ ；若 $d|a,d|b$，则 $d|{\rm gcd}(a,b)$ 

**定理** 设 $a=qb+r$，其中 $a,b,q,r$ 是整数，则 ${\rm gcd}(a,b)={\rm gcd}(b,r)$ 

**定理：欧几里得算法** 设a,b不全为0，则存在整数x和y使得 ${\rm gcd}(a,b)=xa+yb$ 

**定理** 整数a和b互素当且仅当存在整数x和y使得 $xa+yb=1$ 

## 欧几里得算法

## 算数基本定理

每个大于1的自然数，要么本身就是质数，要么可以写为2个或以上的质数的积，而且这些质因子按大小排列之后，写法仅有一种方式

设 $a=p_1^{k_{a_1}}p_2^{k_{a_2}}\cdots p_s^{k_{a_s}},b=p_1^{k_{b_1}}p_2^{k_{b_2}}\cdots p_s^{k_{b_s}}$，则 $gcd(a,b)=p_1^{k_{\min(a_1,b_1)}}p_2^{k_{\min(a_2,b_2)}}\cdots p_s^{k_{\min(a_s,b_s)}}$ , $lsm(a,b)=p_1^{k_{\max(a_1,b_1)}}p_2^{k_{\max(a_2,b_2)}}\cdots p_s^{k_{\max(a_s,b_s)}}$ , 故 $ab=gcd(a,b)lcm(a,b)$ 

# 欧拉定理

# 筛法

### 双平方定理

任一素数p可表示为一对整数的平方和，当且仅当 $p\%4=1$ 