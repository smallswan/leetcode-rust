//! 异或运算
//! 异或运算应用：查找出现一次的数字；

/// 力扣（136. 只出现一次的数字） https://leetcode-cn.com/problems/single-number/
/// 使用异或运算的规律
pub fn single_number(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let mut single_number = nums[0];
    for &num in nums.iter().take(len).skip(1) {
        single_number ^= num;
    }

    single_number
}

/// 力扣（137. 只出现一次的数字 II） https://leetcode-cn.com/problems/single-number-ii/
/// 方法1：哈希表
pub fn single_number_ii(nums: Vec<i32>) -> i32 {
    use std::collections::HashMap;
    let mut counts_map = HashMap::<i32, i32>::new();
    for num in nums {
        match counts_map.get_mut(&num) {
            Some(count) => {
                *count += 1;
            }
            None => {
                counts_map.insert(num, 1);
            }
        }
    }

    for (key, value) in counts_map {
        if value == 1 {
            return key;
        }
    }
    -1
}

/// 力扣（137. 只出现一次的数字 II）
/// 方法2 ：依次确定每一个二进制位
pub fn single_number_ii_v2(nums: Vec<i32>) -> i32 {
    let mut answer = 0;
    for i in 0..32 {
        let mut total = 0;
        for &num in &nums {
            total += ((num >> i) & 1);
        }
        if total % 3 != 0 {
            answer |= (1 << i);
        }
    }

    answer
}

/// 力扣（260. 只出现一次的数字 III） https://leetcode-cn.com/problems/single-number-iii/
/// 方法1：分组异或
pub fn single_number_260(nums: Vec<i32>) -> Vec<i32> {
    // ret 为 a,b两个数异或的结果
    let mut ret = 0;
    for &num in &nums {
        ret ^= num;
    }
    // div 为 a,b 二进制位上不相同时，最低的位
    let mut div = 1;
    while div & ret == 0 {
        div <<= 1;
    }
    let (mut a, mut b) = (0, 0);
    for &num in &nums {
        if div & num != 0 {
            a ^= num;
        } else {
            b ^= num;
        }
    }
    vec![a, b]
}

/// 力扣（268. 丢失的数字） https://leetcode-cn.com/problems/missing-number/
pub fn missing_number(nums: Vec<i32>) -> i32 {
    use std::collections::HashSet;
    let len = nums.len();
    let mut nums_set = HashSet::with_capacity(len);
    nums.iter().for_each(|num| {
        nums_set.insert(*num as usize);
    });

    for num in 0..=len {
        if !nums_set.contains(&num) {
            return num as i32;
        }
    }

    -1
}

/// 力扣（268. 丢失的数字）
/// 方法：数学法
pub fn missing_number_v2(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let expect_sum = (len * (len + 1) / 2) as i32;
    // let sum = nums.iter().fold(0, |sum, num| sum + *num);
    let sum: i32 = nums.iter().sum();
    expect_sum - sum
}

/// 力扣（268. 丢失的数字）
/// 方法：位运算（异或）
pub fn missing_number_v3(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let mut missing = len;
    for (idx, num) in nums.iter().enumerate().take(len) {
        missing ^= idx ^ (*num as usize);
    }
    missing as i32
}

/// 力扣（389. 找不同） https://leetcode-cn.com/problems/find-the-difference/
pub fn find_the_difference(s: String, t: String) -> char {
    let mut chars_vec = vec![0; 26];
    let a = b'a' as usize;
    t.as_bytes().iter().for_each(|x| {
        chars_vec[(*x as usize) - a] += 1;
    });
    s.as_bytes().iter().for_each(|x| {
        chars_vec[(*x as usize) - a] -= 1;
    });

    for (index, count) in chars_vec.iter().enumerate() {
        if *count == 1 {
            return (index as u8 + b'a') as char;
        }
    }
    ' '
}

/// 力扣（389. 找不同）
pub fn find_the_difference_v2(s: String, t: String) -> char {
    let mut result = 0u8;
    s.as_bytes().iter().fold(result, |acc, b| {
        result ^= b;
        result
    });

    t.as_bytes().iter().fold(result, |acc, b| {
        result ^= b;
        result
    });

    result as char
}

/// 力扣（461. 汉明距离） https://leetcode-cn.com/problems/hamming-distance/
pub fn hamming_distance(x: i32, y: i32) -> i32 {
    let z = x ^ y;
    z.count_ones() as i32
}

/// 力扣（461. 汉明距离）
/// 这里采用右移位，每个位置都会被移动到最右边。移位后检查最右位的位是否为 1 即可。检查最右位是否为 1，可以使用取模运算（i % 2）或者 AND 操作（i & 1），这两个操作都会屏蔽最右位以外的其他位。
pub fn hamming_distance_v2(x: i32, y: i32) -> i32 {
    let mut xor = x ^ y;
    let mut distance = 0;
    while xor != 0 {
        if xor % 2 == 1 {
            distance += 1;
        }
        xor >>= 1;
    }
    distance
}

/// 力扣（461. 汉明距离）
/// 布赖恩·克尼根算法
pub fn hamming_distance_v3(x: i32, y: i32) -> i32 {
    let mut xor = x ^ y;
    let mut distance = 0;
    while xor != 0 {
        distance += 1;
        xor = xor & (xor - 1);
    }
    distance
}

/// 693. 交替位二进制数 https://leetcode-cn.com/problems/binary-number-with-alternating-bits/
pub fn has_alternating_bits(n: i32) -> bool {
    let n = n as u32;
    ((n ^ (n >> 1)) + 1).is_power_of_two()
}

/// 力扣（1486. 数组异或操作） https://leetcode-cn.com/problems/xor-operation-in-an-array/
/// 方法一：模拟
pub fn xor_operation(n: i32, start: i32) -> i32 {
    (1..n).fold(start, |acc, i| acc ^ (start + 2 * i))
}

/// 力扣（1486. 数组异或操作）
/// 方法二：数学
pub fn xor_operation_v2(n: i32, start: i32) -> i32 {
    let (s, e) = (start >> 1, n & start & 1);
    let result = sum_xor(s - 1) ^ sum_xor(s + n - 1);

    result << 1 | e
}

fn sum_xor(x: i32) -> i32 {
    match x % 4 {
        0 => x,
        1 => 1,
        2 => x + 1,
        _ => 0,
    }
}

/// 力扣（1863. 找出所有子集的异或总和再求和）https://leetcode-cn.com/problems/sum-of-all-subset-xor-totals/
pub fn subset_xor_sum(nums: Vec<i32>) -> i32 {
    let mut xor_sum = 0;
    let n = nums.len();
    for num in nums {
        xor_sum |= num;
    }
    xor_sum << (n - 1)
}

/// 力扣（1863. 找出所有子集的异或总和再求和）
pub fn subset_xor_sum_v2(nums: Vec<i32>) -> i32 {
    let mut xor_sum = 0;
    let n = nums.len();
    let two_pow_n = 1 << n;
    for i in 0..two_pow_n {
        let mut temp = 0;
        for j in 0..n {
            if i & (1 << j) != 0 {
                temp ^= nums[j];
            }
        }
        xor_sum += temp;
    }
    xor_sum
}

/// 剑指 Offer 56 - I. 数组中数字出现的次数 https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/
/// 方法1：分组异或
pub fn single_numbers(nums: Vec<i32>) -> Vec<i32> {
    // ret 为 a,b两个数异或的结果
    let mut ret = 0;
    for &num in &nums {
        ret ^= num;
    }
    // div 为 a,b 二进制位上不相同时，最低的位
    let mut div = 1;
    while div & ret == 0 {
        div <<= 1;
    }
    let (mut a, mut b) = (0, 0);
    for &num in &nums {
        if div & num != 0 {
            a ^= num;
        } else {
            b ^= num;
        }
    }
    vec![a, b]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor() {
        let ones = hamming_distance(1, 4);
        dbg!(ones);

        let distance = hamming_distance_v2(4, 255);
        dbg!(distance);

        let distance = hamming_distance_v3(4, 65535);
        dbg!(distance);

        let nums = vec![1, 3];
        dbg!(subset_xor_sum(nums));

        let nums = vec![3, 4, 5, 6, 7, 8];
        dbg!(subset_xor_sum(nums));
        let (n, start) = (4, 3);
        dbg!(xor_operation(n, start));

        let nums = vec![6, i32::MIN, 6, 6, 7, 8, 7, 8, 8, 7];
        dbg!(single_number_ii_v2(nums));

        let nums = vec![1, 2, 1, 3, 2, 5];
        dbg!(single_number_260(nums));
    }
}
