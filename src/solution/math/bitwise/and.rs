//！ 与运算

/// 力扣（191. 位1的个数)
pub fn hamming_weight_v2(n: u32) -> i32 {
    format!("{:b}", n).chars().filter(|c| *c == '1').count() as i32
}

/// 力扣（191. 位1的个数)
pub fn hamming_weight_v3(n: u32) -> i32 {
    count_ones(n)
}

/// Brian Kernighan 算法
fn count_ones(n: u32) -> i32 {
    let mut ones = 0;
    let mut n = n;
    while n > 0 {
        n &= (n - 1);
        ones += 1;
    }
    ones
}

/// 力扣（201. 数字范围按位与） https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/
/// 我们可以将问题重新表述为：给定两个整数，我们要找到它们对应的二进制字符串的公共前缀。
/// 方法1：位移
pub fn range_bitwise_and(left: i32, right: i32) -> i32 {
    let mut left = left;
    let mut right = right;
    let mut shift = 0;
    while left < right {
        left >>= 1;
        right >>= 1;
        shift += 1;
    }
    left << shift
}

/// 力扣（201. 数字范围按位与）
/// 方法2：Brian Kernighan 算法
pub fn range_bitwise_and_v2(left: i32, right: i32) -> i32 {
    let mut right = right;
    while left < right {
        right &= (right - 1);
    }
    right
}

/// 力扣（338. 比特位计数） https://leetcode-cn.com/problems/counting-bits/
/// 与 力扣（191. 位1的个数) 类似
pub fn count_bits(n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut result = vec![0; n + 1];

    for (num, item) in result.iter_mut().enumerate().take(n + 1) {
        *item = num.count_ones() as i32;
    }

    result
}

/// 力扣（338. 比特位计数）
pub fn count_bits_v2(n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut result = vec![0; n + 1];
    for (num, item) in result.iter_mut().enumerate().take(n + 1).skip(1) {
        *item = count_ones(num as u32);
    }

    result
}

/// 力扣（476. 数字的补数） https://leetcode-cn.com/problems/number-complement/
/// 方法1：左移求出最高位所需移动的次数，然后构建与之匹配的掩码
pub fn find_complement(num: i32) -> i32 {
    let mut num = num;
    //最高位为1的位
    let mut high_bit = 0;
    for i in 1..=30 {
        if num >= (1 << i) {
            high_bit = i;
        } else {
            break;
        }
    }

    let mark = match (high_bit == 30) {
        true => 0x7fffffff,
        false => (1 << (high_bit + 1)) - 1,
    };

    num ^ mark
}

/// 力扣（476. 数字的补数）
/// 方法2：右移统计数字的有效位数，同时构建与之匹配的掩码
pub fn find_complement_v2(num: i32) -> i32 {
    let mut temp = num;
    let mut mask = 0;
    while temp > 0 {
        temp >>= 1;
        mask = (mask << 1) + 1;
    }

    num ^ mask
}

/// 1009. 十进制整数的反码 https://leetcode-cn.com/problems/complement-of-base-10-integer/
pub fn bitwise_complement(n: i32) -> i32 {
    let mut num = n;
    let mut high_bit = 0;
    for i in 1..=30 {
        if num >= (1 << i) {
            high_bit = i;
        } else {
            break;
        }
    }

    let mark = match (high_bit == 30) {
        true => 0x7fffffff,
        false => (1 << (high_bit + 1)) - 1,
    };

    num ^ mark
}

/// 1009. 十进制整数的反码
pub fn bitwise_complement_v2(n: i32) -> i32 {
    if n == 0 {
        return 1;
    }
    let mut num = n;
    let mut mark = 1;
    let mut high_bit = 0;
    while num > 0 {
        num >>= 1;
        high_bit += 1;
    }

    //dbg!(high_bit);
    let mark = match (high_bit == 31) {
        true => i32::MAX,
        false => (1 << high_bit) - 1,
    };

    n ^ mark
}

/// 393. UTF-8 编码验证 https://leetcode-cn.com/problems/utf-8-validation/
/// TODO
pub fn valid_utf8(data: Vec<i32>) -> bool {
    let len = data.len();
    match len {
        1 => {
            //0xxxxxxx
            data[0] <= 0x7f
        }
        2 => {
            // 110xxxxx 10xxxxxx
            (data[0] >= 0xb0 && data[0] <= 0xdf) && (data[1] >= 0x80 && data[1] <= 0xbf)
        }
        3 => {
            // 1110xxxx 10xxxxxx 10xxxxxx
            (data[0] >= 0xd0 && data[0] <= 0xef)
                && (data[1] >= 0x80 && data[1] <= 0xbf)
                && (data[2] >= 0x80 && data[2] <= 0xbf)
        }
        4 => {
            // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            (data[0] >= 0xf0 && data[0] <= 0xf7)
                && (data[1] >= 0x80 && data[1] <= 0xbf)
                && (data[2] >= 0x80 && data[2] <= 0xbf)
                && (data[3] >= 0x80 && data[3] <= 0xbf)
        }
        _ => false,
    }
}

pub fn valid_utf8_v2(data: Vec<i32>) -> bool {
    let bytes: Vec<u8> = data.into_iter().map(|b| b as u8).collect();
    if let Ok(utf8_str) = String::from_utf8(bytes) {
        dbg!(utf8_str);
        return true;
    } else {
        return false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwise_operators() {
        dbg!(bitwise_complement_v2(0));
        dbg!(bitwise_complement_v2(5));
        dbg!(bitwise_complement_v2(7));
        dbg!(bitwise_complement_v2(1022));
        //100000000
        //214748364
        dbg!(bitwise_complement_v2(100000000));
        dbg!(bitwise_complement_v2(i32::MAX));
        dbg!(i32::MAX);
        println!("{:b}", i32::MAX);

        let data = vec![
            194, 155, 231, 184, 185, 246, 176, 131, 161, 222, 174, 227, 162, 134, 241, 154, 168,
            185, 218, 178, 229, 187, 139, 246, 178, 187, 139, 204, 146, 225, 148, 179, 245, 139,
            172, 134, 193, 156, 233, 131, 154, 240, 166, 188, 190, 216, 150, 230, 145, 144, 240,
            167, 140, 163, 221, 190, 238, 168, 139, 241, 154, 159, 164, 199, 170, 224, 173, 140,
            244, 182, 143, 134, 206, 181, 227, 172, 141, 241, 146, 159, 170, 202, 134, 230, 142,
            163, 244, 172, 140, 191,
        ];
        dbg!(valid_utf8_v2(data));
    }
}
