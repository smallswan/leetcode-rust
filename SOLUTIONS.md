# 经典算法

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
- 查找只出现一次的数字

### 相关问题
- [136. 只出现一次的数字]( https://leetcode-cn.com/problems/single-number/)
- [137. 只出现一次的数字 II]( https://leetcode-cn.com/problems/single-number-ii/)
- [260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/) 和 [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/) 是同一题

## 双指针法

### 相关问题

