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
          |--数据结构：数组、向量、双端队列、哈希集合、哈希表、二叉堆、、  
          |--算法：排序算法（归并排序）、查询算法（二分查找）、动态规划、分治算法、递归算法、回溯法  		  
   |--数学方面
          |--概念：质数、合数、快乐数、不快乐数、丑数
          |--公式： 
          |--定理： 
	   |--方法：数学归纳法、牛顿迭代法、厄拉多塞筛法（埃氏筛）、快速乘（俄罗斯农夫乘法）
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

### 问题：如何检测一个链表是否有环（循环节）
[弗洛伊德的兔子与乌龟(Floyd's Tortoise and Hare algorithm)](https://zhuanlan.zhihu.com/p/105269431 )，快慢指针法、龟兔赛跑法

### 问题：电话号码的字母组合
[17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)
[回溯法](https://baike.baidu.com/item/%E5%9B%9E%E6%BA%AF%E6%B3%95/86074?fr=aladdin)

### 问题：旋转操作
[189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/) 