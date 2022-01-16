/// 力扣（190. 颠倒二进制位） https://leetcode-cn.com/problems/reverse-bits/
//  方法一：逐位颠倒
pub fn reverse_bits(x: u32) -> u32 {
    let mut n = x;
    let mut rev = 0;
    let mut i = 0;
    while i < 32 && n > 0 {
        rev |= (n & 1) << (31 - i);
        n >>= 1;
        i += 1;
    }
    rev
}

/// 力扣（190. 颠倒二进制位）
//  方法二：位运算分治
pub fn reverse_bits_v2(x: u32) -> u32 {
    const M1: u32 = 0x55555555; // 01010101010101010101010101010101
    const M2: u32 = 0x33333333; // 00110011001100110011001100110011
    const M4: u32 = 0x0f0f0f0f; // 00001111000011110000111100001111
    const M8: u32 = 0x00ff00ff; // 00000000111111110000000011111111

    let mut n = x;
    n = n >> 1 & M1 | (n & M1) << 1;
    n = n >> 2 & M2 | (n & M2) << 2;
    n = n >> 4 & M4 | (n & M4) << 4;
    n = n >> 8 & M8 | (n & M8) << 8;
    n >> 16 | n << 16
}

/// 力扣（190. 颠倒二进制位）
pub fn reverse_bits_v3(x: u32) -> u32 {
    let mut n = x;
    n.reverse_bits()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn left_right() {
        dbg!(reverse_bits(43261596));
    }
}
