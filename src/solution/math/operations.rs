/// 力扣（50. Pow(x, n)） https://leetcode-cn.com/problems/powx-n/
pub fn my_pow(x: f64, n: i32) -> f64 {
    x.powi(n)
}

/// 力扣（50. Pow(x, n)）
/// 方法一：快速幂 + 递归
pub fn my_pow_v2(x: f64, n: i32) -> f64 {
    if n >= 0 {
        quick_mul(x, n)
    } else {
        1.0 / quick_mul(x, -n)
    }
}

/// 快速幂
fn quick_mul(x: f64, n: i32) -> f64 {
    if n == 0 {
        return 1.0;
    }

    let y = quick_mul(x, n / 2);
    if n % 2 == 0 {
        y * y
    } else {
        y * y * x
    }
}

fn quick_mul_v2(x: f64, n: i32) -> f64 {
    let mut ans = 1.0f64;
    let mut x_contribute = x;
    let mut n_mut = n;
    while n_mut > 0 {
        if n_mut % 2 == 1 {
            ans *= x_contribute;
        }
        x_contribute *= x_contribute;
        n_mut /= 2;
    }
    ans
}

/// 力扣（50. Pow(x, n)）
/// 方法二：快速幂 + 迭代 (该方法容易造成溢出不可取)
pub fn my_pow_v3(x: f64, n: i32) -> f64 {
    if n >= 0 {
        quick_mul_v2(x, n)
    } else {
        1.0f64 / quick_mul_v2(x, -n)
    }
}

/// 力扣（50. Pow(x, n)）
pub fn my_pow_v4(x: f64, n: i32) -> f64 {
    let mut x = if n > 0 { x } else { 1f64 / x };
    let mut n = n;
    let mut r = 1f64;
    while n != 0 {
        if n & 1 == 1 {
            r *= x;
        }
        x *= x;
        n /= 2;
    }
    r
}

/// 力扣（66. 加一） https://leetcode-cn.com/problems/plus-one/
pub fn plus_one(digits: Vec<i32>) -> Vec<i32> {
    // 以下算法参考了：https://leetcode-cn.com/problems/plus-one/solution/java-shu-xue-jie-ti-by-yhhzw/
    let len = digits.len();
    let mut new_digits = digits.clone();

    let mut i = len - 1;
    loop {
        let b = (digits[i] + 1) % 10;
        new_digits[i] = b;
        if b != 0 {
            return new_digits;
        }
        if i > 0 {
            i -= 1;
        } else {
            break;
        }
    }

    // digits为全部是9的情况
    let mut new_digits = vec![0; len + 1];
    new_digits[0] = 1;
    new_digits
}

/// 力扣（67. 二进制求和） https://leetcode-cn.com/problems/add-binary/
pub fn add_binary(a: String, b: String) -> String {
    let mut result = String::new();
    let mut ca = 0;
    let mut s = true;
    let mut t = true;
    let mut a_rev = a.chars().rev();
    let mut b_rev = b.chars().rev();

    while s || t {
        let mut sum = ca;

        if let Some(x) = a_rev.next() {
            let temp = x as i32 - 48;

            sum += temp;
        } else {
            s = false;
        }

        if let Some(x) = b_rev.next() {
            let temp = x as i32 - 48;
            sum += temp;
        } else {
            t = false;
        }

        if !s && !t {
            break;
        }

        if sum % 2 == 0 {
            result.push('0');
        } else {
            result.push('1');
        }
        ca = sum / 2;
    }

    if ca == 1 {
        result.push('1');
    }
    //字符串翻转
    result.chars().rev().collect()
}

/// 力扣（69. x 的平方根） https://leetcode-cn.com/problems/sqrtx/
/// 方法1：二分查找，注意相乘溢出
pub fn my_sqrt(x: i32) -> i32 {
    let mut left = 0;
    let mut right = x;
    let mut ans = -1;
    while left <= right {
        let mid = left + (right - left) / 2;
        if (mid as i64) * (mid as i64) <= x as i64 {
            ans = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    ans
}

/// 力扣（69. x 的平方根）
/// 方法2：换底公式（自然数e为底）
pub fn my_sqrt_v2(x: i32) -> i32 {
    if x == 0 {
        return 0;
    }

    // x1/2=(elnx)1/2=e21​lnx
    //java int ans = (int) Math.exp(0.5 * Math.log(x));
    let x = x as f64;

    //exp(self) = e^(self)
    //ln(self) = self's  natural logarithm
    let ans = (0.5 * (x).ln()).exp();
    let squar = (ans + 1f64) * (ans + 1f64);

    if squar <= x {
        (ans as i32) + 1
    } else {
        ans as i32
    }
}

/// 力扣（69. x 的平方根）
/// 方法3：系统内置方法
pub fn my_sqrt_v3(x: i32) -> i32 {
    let y = x as f64;
    y.sqrt().floor() as i32
}

/// 力扣（69. x 的平方根）
/// 牛顿迭代法
/// 我们用 C表示待求出平方根的那个整数。显然，C 的平方根就是函数: y = f(x) = x^2 - C
pub fn my_sqrt_v4(x: i32) -> i32 {
    if x == 0 {
        return 0i32;
    }

    let c = x as f64;
    let mut x0 = c;
    loop {
        // 迭代方程
        let xi = 0.5 * (x0 + c / x0);
        // 1e-7即0.0000001
        if (x0 - xi).abs() < 1e-7 {
            break;
        }
        x0 = xi;
    }

    x0 as i32
}

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

/// 力扣（172. 阶乘后的零） https://leetcode-cn.com/problems/factorial-trailing-zeroes/
pub fn trailing_zeroes(n: i32) -> i32 {
    let mut count_fives = 0;
    let mut steps: Vec<i32> = (5..=n).into_iter().filter(|x| *x % 5 == 0).collect();
    // println!("{:?}",steps);
    for step in steps {
        let mut remaining = step;
        while remaining % 5 == 0 {
            count_fives += 1;
            remaining /= 5;
        }
    }

    count_fives
}

/// 力扣（172. 阶乘后的零）
/// f(n) = n/5^1 + n/5^2 + n/5^3 + n/5^m (n < 5^m)
pub fn trailing_zeroes_v2(n: i32) -> i32 {
    let mut count_fives = 0;
    let mut remaining = n;
    while remaining > 0 {
        remaining /= 5;
        count_fives += remaining;
    }
    count_fives
}

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

/// 方法2：Brian Kernighan 算法
pub fn range_bitwise_and_v2(left: i32, right: i32) -> i32 {
    let mut right = right;
    while left < right {
        right &= (right - 1);
    }
    right
}

/// 力扣（231. 2的幂） https://leetcode-cn.com/problems/power-of-two/
pub fn is_power_of_two(n: i32) -> bool {
    if n <= 0 {
        return false;
    }
    if n == 1 {
        return true;
    }

    let mut m = n;
    loop {
        if m % 2 == 0 {
            m /= 2;
            if m <= 1 {
                break;
            }
        } else {
            return false;
        }
    }

    true
}

/// 力扣（231. 2的幂）
pub fn is_power_of_two_v2(n: i32) -> bool {
    if n == 0 {
        return false;
    }

    n & (n - 1) == 0
}

/// 力扣（326. 3的幂) https://leetcode-cn.com/problems/power-of-three/
pub fn is_power_of_three(n: i32) -> bool {
    n > 0 && 1162261467 % n == 0
}

/// 力扣（326. 3的幂)
pub fn is_power_of_three_v2(n: i32) -> bool {
    if n < 1 {
        return false;
    }
    let mut n = n;
    while n % 3 == 0 {
        n /= 3;
    }
    n == 1
}

/// 力扣（326. 3的幂)
pub fn is_power_of_three_v3(n: i32) -> bool {
    if n <= 0 {
        return false;
    }
    let mut n = n;
    while n > 1 {
        if n % 3 != 0 {
            return false;
        }
        n /= 3;
    }
    true
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

/// 力扣（338. 比特位计数）
/// 动态规划
pub fn count_bits_v3(n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut result = vec![0; n + 1];
    let mut high_bit = 0;
    for num in 1..=n {
        if num & (num - 1) == 0 {
            high_bit = num;
        }
        result[num] = result[num - high_bit] + 1;
    }

    result
}

/// 力扣（338. 比特位计数）
/// 动态规划——最低有效位
pub fn count_bits_v4(n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut result = vec![0i32; n + 1];
    for num in 1..=n {
        result[num] = result[num >> 1] + ((num as i32) & 1);
    }

    result
}

/// 力扣（342. 4的幂） https://leetcode-cn.com/problems/power-of-four/
pub fn is_power_of_four(n: i32) -> bool {
    n > 0 && (n & (n - 1)) == 0 && (n & 0x2aaaaaaa == 0)
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
    fn add_sub_mul_div() {
        dbg!(plus_one(vec![9, 1, 9]));
        let a = String::from("0");
        let b = String::from("0");
        dbg!(add_binary(a, b));
    }

    #[test]
    fn pow_sqrt() {
        dbg!(my_pow(2.10000, 3));

        dbg!(my_pow_v2(2.00000, -4));
        dbg!(my_pow_v2(2.00000, 10));

        dbg!("i32 max :{},min :{}", std::i32::MAX, std::i32::MIN);
        // dbg!("{}",my_pow_v3(2.00000,-2147483648));
        // dbg!("{}",my_pow_v3(2.00000,-2147483647));

        dbg!(my_pow_v4(2.00000, -2147483648));
        dbg!(my_pow_v4(2.00000, -2147483647));
        dbg!(my_pow_v4(2.00000, 2147483647));

        assert_eq!(my_sqrt(4), 2);
        assert_eq!(my_sqrt(8), 2);

        dbg!(my_sqrt(2147395599));
        dbg!(my_sqrt_v2(2147395599));
        dbg!(my_sqrt_v2(256));
        let num = 2147395599f64;
        dbg!(num.sqrt().floor());
        dbg!(my_sqrt_v3(2147395599));
        dbg!(my_sqrt_v4(2147395599));

        let nums = vec![1, 16, 218];
        for num in nums {
            assert_eq!(is_power_of_two(num), is_power_of_two_v2(num));
        }

        dbg!(is_power_of_three(81 * 3));
    }

    #[test]
    fn xor() {
        let nums = vec![1, 3];
        dbg!(subset_xor_sum(nums));

        let nums = vec![3, 4, 5, 6, 7, 8];
        dbg!(subset_xor_sum(nums));
        let (n, start) = (4, 3);
        dbg!(xor_operation(n, start));

        let nums = vec![6, i32::MIN, 6, 6, 7, 8, 7, 8, 8, 7];
        dbg!(single_number_ii_v2(nums));

        dbg!(reverse_bits(43261596));

        let nums = vec![1, 2, 1, 3, 2, 5];
        dbg!(single_number_260(nums));
    }
}
