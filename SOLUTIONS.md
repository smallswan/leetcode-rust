# 经典算法
## 计算机大师
- Brian Kernighan (布莱恩·克尼汉)
- Donald Knuth （唐纳德·克努特 ，中文名：高德纳） 经典巨著《计算机程序设计的艺术》的年轻作者。
- Doung Mcllroy （著名计算机科学家，美国工程院院士）

## Floyd判圈算法（又称快慢指针法、龟兔赛跑法）
### 数学证明
假设环长为 L，从起点到环的入口的步数是 aa，从环的入口继续走 b 步到达相遇位置，从相遇位置继续走 c 步回到环的入口，则有 b+c=L，其中 L、a、b、c 都是正整数。根据上述定义，慢指针走了 a+b 步，快指针走了 2(a+b) 步。
从另一个角度考虑，在相遇位置，快指针比慢指针多走了若干圈，因此快指针走的步数还可以表示成 a+b+kL其中 k 表示快指针在环上走的圈数。联立等式，可以得到

2(a+b)=a+b+kL

解得 a=kL−b，整理可得

a=(k−1)L+(L−b)=(k−1)L+c

从上述等式可知，如果慢指针从起点出发，快指针从相遇位置出发，每次两个指针都移动一步，则慢指针走了 aaa 步之后到达环的入口，快指针在环里走了 k−1k-1k−1 圈之后又走了 ccc 步，由于从相遇位置继续走 ccc 步即可回到环的入口，因此快指针也到达环的入口。两个指针在环的入口相遇，相遇点就是答案。

作者：LeetCode-Solution  
链接：https://leetcode-cn.com/problems/find-the-duplicate-number/solution/xun-zhao-zhong-fu-shu-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

### 应用场景
- 判断图形结构是否存在环
- 判断重复数字

### 相关问题
- [26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/ )
- [202. 快乐数](https://leetcode-cn.com/problems/happy-number/ ) 
- [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/ )

### 实际应用
[C++ 标准库的 unique 方法](http://www.cplusplus.com/reference/algorithm/unique/ )

## 数字二进制异或运算

### 应用场景
- 只出现一次的数字

### 相关问题
- [136. 只出现一次的数字]( https://leetcode-cn.com/problems/single-number/)
- [137. 只出现一次的数字 II]( https://leetcode-cn.com/problems/single-number-ii/)
- [260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/) 和 [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/) 是同一题


## 数字二进制按位与运算
Brian Kernighan 算法
```
n & (n - 1) 使得二进制位最右边一位（不一定是最后一位，比如12（1100 变为1000））为0
```

### 应用场景
- 统计二进制中1的个数

### 相关问题
- [力扣（191. 位1的个数)](https://leetcode-cn.com/problems/number-of-1-bits/)
- [力扣（338. 比特位计数）](https://leetcode-cn.com/problems/counting-bits/)
- [力扣（231. 2的幂）](https://leetcode-cn.com/problems/power-of-two/)
- [力扣（201. 数字范围按位与） ](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/)
- [力扣（461. 汉明距离）](https://leetcode-cn.com/problems/hamming-distance/ )

## 双指针法

### 相关问题

## 旋转操作
Doung Mcllroy 给出了将十元数组向上旋转5个位置的翻手例子。初始时掌心对着我们的脸，左手在右手上面。
通过“翻转左手”、“翻转右手”、“翻转双手”三次翻转，达到模拟向左旋转5位的效果。

rotate(旋转) 可以通过三次reverse实现；
reverse(反转，颠倒，翻转) 可以通过交换（swap）实现。

```
rotate可以通过三次reverse实现，具体来说：
（1）rotate_left(mid) 可以通过以下三次reverse实现：
reverse(0,mid);
reverse(mid,len);
reverse(0,len);

（2）rotate_right(mid) 可以通过以下三次reverse实现：
reverse(0,len);
reverse(0,mid);
reverse(mid,len);
```
### 相关问题
[189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/) 