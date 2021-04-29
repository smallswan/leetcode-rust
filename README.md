# leet-code-rust
A collection of [LeetCode](https://leetcode-cn.com/) problems that I've solved in [Rust](Rust).

## leetcode实战总结 
1. 同一个问题有多种解决方案（计算机方面、数学方面），要善于归纳总结问题以及其解决方案；

```
问题  
  |--分类：
  |--特点：前置条件、
  
解决方案  
   |--计算机方面  
          |--编程语言  
          |--设计模式：状态模式  
          |--算法：排序算法、查询算法、动态规划、分治算法、递归算法  		  
   |--数学方面 
          |--公式： 
          |--定理： 
	   |--方法：数学归纳法、牛顿迭代法、
```
2. 每个解决方案能够解决特定的问题，在解决具体问题的过程中，可能要综合运用多种解决方案。

3. 注意边界值
- 数字是否溢出（overflow）

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
## 问题：汉明距离
[汉明距离](https://baike.baidu.com/item/%E6%B1%89%E6%98%8E%E8%B7%9D%E7%A6%BB/475174?fr=aladdin#4 )  
[布赖恩·克尼根算法](https://www.e-learn.cn/topic/3779838 )

### 问题：寻找众数
- [Boyer-Moore算法](https://baike.baidu.com/item/Boyer-%20Moore%E7%AE%97%E6%B3%95/16548374?fr=aladdin )
- [Boyer-Moore 投票算法](https://zhuanlan.zhihu.com/p/85474828 )
