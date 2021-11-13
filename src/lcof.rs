//! 剑指Offer
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_lcoff() {
        let mut c = 80;
        c >>= 1;
        println!("{}", c);
        println!("{}", c & 1);
        println!("{}", 1 & 1);
        println!("{}", 2 & 1);
        println!("{}", 3 & 1);
        println!("{}", 4 & 1);

        let sum = sum_nums(10000);
        println!("{}", sum);

        let sum = sum_nums_v2(10000);
        println!("{}", sum);

        println!("{}", sum >> 1);
        println!("{}", sum);

        let nums = vec![2, 3, 1, 0, 2, 5, 3];
        let find_repeat_number_result = find_repeat_number(nums);
        dbg!(find_repeat_number_result);

        let nums = vec![2, 3, 1, 0, 2, 5, 3];
        let find_repeat_number_v3_result = find_repeat_number_v3(nums);
        dbg!(find_repeat_number_v3_result);
    }
}

/// 剑指 Offer 03. 数组中重复的数字 https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/
///  方法1：哈希集合
pub fn find_repeat_number(nums: Vec<i32>) -> i32 {
    use std::collections::HashSet;
    let mut nums_set = HashSet::<i32>::new();
    for num in nums {
        if nums_set.contains(&num) {
            return num;
        } else {
            nums_set.insert(num);
        }
    }

    // 题目没有指明如果没有重复的数字就返回-1，是测试出来的
    -1
}

/// 剑指 Offer 03. 数组中重复的数字
pub fn find_repeat_number_v2(nums: Vec<i32>) -> i32 {
    use std::collections::HashSet;
    let mut nums_set = HashSet::<i32>::new();
    for num in nums {
        if !nums_set.insert(num) {
            return num;
        }
    }

    // 题目没有指明如果没有重复的数字就返回-1，是测试出来的
    -1
}

/// 剑指 Offer 03. 数组中重复的数字
/// 方法2：原地交换
pub fn find_repeat_number_v3(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let mut new_nums = nums;
    let mut i = 0;
    while i < len {
        let num = new_nums[i] as usize;
        if num == i {
            i += 1;
            continue;
        }
        if new_nums[num] as usize == num {
            return num as i32;
        }
        new_nums.swap(i, num);
    }

    -1
}

/// 剑指 Offer 06. 从尾到头打印链表 https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/
use crate::simple::ListNode;
pub fn reverse_print(head: Option<Box<ListNode>>) -> Vec<i32> {
    let mut res = Vec::new();
    let mut next = &head;
    while next.is_some() {
        res.push(next.as_ref().unwrap().val);
        next = &(next.as_ref().unwrap().next);
    }
    res.reverse();
    res
}

/// 剑指 Offer 10- I. 斐波那契数列   https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/
pub fn fib(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }

    let mut f0 = 0i64;
    let mut f1 = 1i64;
    let mut current = 0i64;
    let mut i = 1;
    while i < n {
        // 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
        current = (f0 + f1) % 1000000007;
        f0 = f1;
        f1 = current;
        i += 1;
    }

    current as i32
}

/// 剑指 Offer 10- I. 斐波那契数列
pub fn fib_v2(n: i32) -> i32 {
    let mut f0 = 0;
    let mut f1 = 1;
    let mut i = 0;
    let mut sum = 0;
    while i < n {
        sum = (f0 + f1) % 1000000007;
        f0 = f1;
        f1 = sum;
        i += 1;
    }
    f0
}

/// 剑指 Offer 64. 求1+2+…+n  https://leetcode-cn.com/problems/qiu-12n-lcof/
/// 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
/// 使用match语句是否违规呢？
/// 方法1：递归法
pub fn sum_nums(n: i32) -> i32 {
    let mut sum = n;
    let flage = match n > 0 {
        true => {
            sum += self::sum_nums(n - 1);
        }
        false => {}
    };
    sum
}

/// 剑指 Offer 64. 求1+2+…+n
/// 1+2+…+n = n*(n+1)/2 = n*(n+1) >> 1
/// 方法2：快速乘法（俄罗斯乘法）
pub fn sum_nums_v2(n: i32) -> i32 {
    let mut sum = 0;
    let mut a = n;
    let mut b = n + 1;

    // 因为题目数据范围 n 为 [1,10000]，所以 n 二进制展开最多不会超过 14 位，我们手动展开 14 层代替循环即可
    // (b & 1) = 1 表示b为奇数，如果(b & 1) = 0 则b为偶数
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;

    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;

    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;

    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;

    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;
    match (b & 1) > 0 {
        true => {
            sum += a;
        }
        false => {}
    };
    a <<= 1;
    b >>= 1;

    sum >> 1
}
