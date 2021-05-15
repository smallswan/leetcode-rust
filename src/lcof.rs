//! 剑指Offer

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
}

/// 剑指 Offer 10- I. 斐波那契数列   https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/
pub fn fib(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }

    let mut a = 0i64;
    let mut b = 1i64;
    let mut c = 0i64;
    let mut i = 1;
    while i < n {
        // 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
        c = (a + b) % 1000000007;
        a = b;
        b = c;
        i += 1;
    }

    c as i32
}

/// 剑指 Offer 10- I. 斐波那契数列
pub fn fib_v2(n: i32) -> i32 {
    let mut a = 0;
    let mut b = 1;
    let mut i = 0;
    let mut sum = 0;
    while i < n {
        sum = (a + b) % 1000000007;
        a = b;
        b = sum;
        i += 1;
    }
    a
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
