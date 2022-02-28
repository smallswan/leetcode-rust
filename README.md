# leet-code-rust
A collection of [LeetCode](https://leetcode-cn.com/) problems that I've solved in [Rust](Rust).

## leetcode实战总结 
1. 条条大路通罗马
   
条条大路通罗马，但有的道路崎岖难走，有的路途遥远困难重重，如何在有限的时间内付出有限的代价到达罗马呢？
因此合理规划好一条路线，选择合适的交通工具，采用巧妙的方法解决路途中的各种问题就显得十分重要。
解决力扣题也是如此，好的算法就好比选择了一条比较快捷的路线，采用合理的数据结构就好比选择了恰当的交通工具，
而数学方面的公式、定理、方法往往能为解决遇到的各种问题提供有力地支持，往往达到事半功倍的效果。

```
问题  
  |--分类：排序、查找、
  |--特点：前置条件、
  
解决方案  
   |--计算机方面  
          |--编程语言  
          |--设计模式：状态模式
          |--数据结构：字符串、数组、向量、双端队列、哈希集合、哈希表、二叉堆、链表、  
          |--算法：KMP(Knuth-Morris-Pratt)算法、Rabin-Karp算法、排序算法（归并排序）、查询算法（二分查找、深度优先搜索、广度优先搜索）、动态规划、分治算法、递归算法、回溯法、位运算（Brian Kernighan 算法、SWAR算法）、摩尔投票法、滑动窗口、旋转（rotate）、快慢指针、DFA算法  		  
   |--数学方面
          |--概念：质数、合数、快乐数、不快乐数、丑数、完全平方数、完美数、梅森数
          |--公式：阶乘、幂运算和对数运算、算术平方根、一元二次方程、 
          |--定理：欧几里得-欧拉定理 
	   |--方法：数学归纳法、牛顿迭代法、厄拉多塞筛法（埃氏筛）、快速乘（俄罗斯农夫乘法）、快速幂、
```
2. 八仙过海各显神通

针对同一个问题，前人已经在计算机、数学方面积累了大量的解决方案，同时每个解决方案可能还适用于解决特定或其他的问题，
只要我们善于归纳总结，熟练掌握各种解决方案、抓住问题的关键，就能够做到举一反三，触类旁通。
此外，每个开发者的知识储备各不相同，各有所长。只有找到适合自己的解决方案就是好的方案。
在解决具体问题的过程中，可能要综合运用多种解决方案。


3. 注意事项
- 定义辅助函数需要在主函数及“impl Solution {” 前定义
- 递归函数的写法
- 数字是否溢出（overflow）
（1）注意各种数字类型（uszie,i32,i64等）的范围，特别是rust用作下标的usize类型。
（2）相乘、幂运算等运算容易溢出。

4. 刷题攻略
## 刷题攻略	
[大一计算机学生如何高效刷力扣？](https://www.zhihu.com/question/392882083/answer/1860538172 )	
对于刷题，我们都是想用最短的时间把经典题目都做一篇，这样效率才是最高的！
[leetcode-master](https://github.com/youngyangyang04/leetcode-master )


## 代码阅读
- 代码按照困难程度（简单 simple,中等 medium，困难 hard）将代码写在对应的.rs文件中；
- 相同级别的题目又按照题目编号从小到大，自上而下排序；
- 每题的方法都写有相应的备注、链接方便快速查找；
- 题目如果有多种解法，则在原方法名后加“_v2”之类的后缀进行区分。

## 问题及解决方案

### 问题：计算两个数的最大公约数
- [BigInteger gcd](https://docs.oracle.com/javase/7/docs/api/java/math/BigInteger.html )
- [最大公约数 GCD](https://baike.baidu.com/item/%E6%9C%80%E5%A4%A7%E5%85%AC%E7%BA%A6%E6%95%B0/869308?fr=aladdin )
- [中国剩余定理](https://baike.baidu.com/item/%E5%AD%99%E5%AD%90%E5%AE%9A%E7%90%86?fromtitle=%E4%B8%AD%E5%9B%BD%E5%89%A9%E4%BD%99%E5%AE%9A%E7%90%86&fromid=11200132 )
- [欧几里德算法(辗转相除法)](https://baike.baidu.com/item/%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E7%AE%97%E6%B3%95/1647675?fromtitle=%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%B7%E7%AE%97%E6%B3%95&fromid=9002848 )

- [Stein算法](https://baike.baidu.com/item/Stein%E7%AE%97%E6%B3%95/7874057 )
### 问题：汉明距离
[汉明距离](https://baike.baidu.com/item/%E6%B1%89%E6%98%8E%E8%B7%9D%E7%A6%BB/475174?fr=aladdin#4 )  
[布赖恩·克尼根算法](https://www.e-learn.cn/topic/3779838 )

### 问题：寻找众数
- [Boyer-Moore算法](https://baike.baidu.com/item/Boyer-%20Moore%E7%AE%97%E6%B3%95/16548374?fr=aladdin )
- [Boyer-Moore 投票算法](https://zhuanlan.zhihu.com/p/85474828 )
- [229. 求众数 II](https://leetcode-cn.com/problems/majority-element-ii/)

### 问题：如何检测一个链表是否有环（循环节）
[弗洛伊德的兔子与乌龟(Floyd's Tortoise and Hare algorithm)](https://zhuanlan.zhihu.com/p/105269431 )，快慢指针法、龟兔赛跑法

### 问题：电话号码的字母组合
[17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)
[回溯法](https://baike.baidu.com/item/%E5%9B%9E%E6%BA%AF%E6%B3%95/86074?fr=aladdin)

### 问题：旋转操作
[189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/) 

### 问题：字符串模式匹配(strStr)
[28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)

### 链表
- [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/ )
- [19. 删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/ )
- [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/ )
- [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/ )
- [24. 两两交换链表中的节点 ](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)
- [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/ )
- [92. 反转链表 II  ](https://leetcode-cn.com/problems/reverse-linked-list-ii/)
- [203. 移除链表元素](https://leetcode-cn.com/problems/remove-linked-list-elements/ )
- [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/ )
- [707. 设计链表](https://leetcode-cn.com/problems/design-linked-list/ )
- [剑指 Offer 06. 从尾到头打印链表 ](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)
- [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/submissions/ )
- 

### 二叉树
二叉树的遍历：前序遍历、中序遍历、后序遍历、层序遍历

- [94. 二叉树的中序遍历 ](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/ )
- [98. 验证二叉搜索树 ](https://leetcode-cn.com/problems/validate-binary-search-tree/)
- [100. 相同的树](https://leetcode-cn.com/problems/same-tree/ )
- [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/ )
- [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/ )
- [103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/ )
- [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/ )
- [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/ )
- [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/ )
- [107. 二叉树的层序遍历 II](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/ )
- [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/ )
- [109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/)
- [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/ )
- [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/ )
- [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/ )
- [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/ )
- [222. 完全二叉树的节点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/ )
- [226. 翻转二叉树 ](https://leetcode-cn.com/problems/invert-binary-tree/ )
- [655. 输出二叉树](https://leetcode-cn.com/problems/print-binary-tree/)
