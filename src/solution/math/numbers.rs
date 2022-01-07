use std::{
    borrow::Borrow,
    cmp::max,
    cmp::Ordering,
    collections::HashMap,
    ops::{BitAndAssign, BitOr, DerefMut},
    str::Chars,
};

/// 力扣（9. 回文数） https://leetcode-cn.com/problems/palindrome-number/
/// 数字转字符串
pub fn is_palindrome(x: i32) -> bool {
    // 负数不是回文数
    if x < 0 {
        return false;
    }
    let s = x.to_string();
    let arr = s.as_bytes();
    let len = arr.len();
    for (c1, c2) in (0..len / 2)
        .into_iter()
        .zip((len / 2..len).rev().into_iter())
    {
        if arr[c1] != arr[c2] {
            return false;
        }
    }
    true
}

/// 力扣（9. 回文数） https://leetcode-cn.com/problems/palindrome-number/
/// 反转一半的数字
pub fn is_palindrome_v2(x: i32) -> bool {
    // 负数以及以0结尾但不是0的数字都不是回文数
    if x < 0 || (x % 10 == 0 && x != 0) {
        return false;
    }
    let mut y = x;
    let mut reverted_num = 0;
    while y > reverted_num {
        reverted_num = reverted_num * 10 + y % 10;
        y /= 10;
    }
    y == reverted_num || y == reverted_num / 10
}

/// rust slice pattern
fn is_palindrome_str(chars: &[char]) -> bool {
    match chars {
        [first, middle @ .., last] => first == last && is_palindrome_str(middle),
        [] | [_] => true,
    }
}

/// 力扣（9. 回文数）
/// rust slice pattern
pub fn is_palindrome_v3(x: i32) -> bool {
    // 负数以及以0结尾但不是0的数字都不是回文数
    if x < 0 || (x % 10 == 0 && x != 0) {
        return false;
    }
    let s = x.to_string().chars().collect::<Vec<_>>();
    is_palindrome_str(s.as_slice())
}

/// 力扣（9. 回文数）
/// 双指针
pub fn is_palindrome_v4(x: i32) -> bool {
    // 负数以及以0结尾但不是0的数字都不是回文数
    if x < 0 || (x % 10 == 0 && x != 0) {
        return false;
    }
    let s = x.to_string().chars().collect::<Vec<_>>();
    let len = s.len();
    let (mut first, mut last) = (0, len - 1);
    while first < last {
        if s[first] == s[last] {
            first += 1;
            last -= 1;
        } else {
            return false;
        }
    }
    true
}

/// 力扣（13. 罗马数字转整数） https://leetcode-cn.com/problems/roman-to-integer/
/// 以下解法不正确
pub fn roman_to_int(s: String) -> i32 {
    let len = s.len();
    let mut sum = 0;

    let mut map = HashMap::<&str, i32>::new();
    map.insert("I", 1);
    map.insert("V", 5);
    map.insert("X", 10);
    map.insert("L", 50);
    map.insert("C", 100);
    map.insert("D", 500);
    map.insert("M", 1000);
    map.insert("IV", 4);
    map.insert("IX", 9);
    map.insert("XL", 40);
    map.insert("XC", 90);
    map.insert("CD", 400);
    map.insert("CM", 900);

    let mut i = 1;

    while i < len {
        if let Some(x) = s.get(i - 1..=i) {
            println!("x:{}", x);
            if let Some(value) = map.get(x) {
                println!("value:{}", value);
                sum += value;
            }
        }
        i += 1;
    }

    while let Some(x) = s.get(i - 1..=i) {
        println!("x:{}", x);

        if let Some(v) = map.get(x) {
            sum += v;
        }
        if i < len {
            i += 1;
        } else {
            break;
        }
    }

    let chars_vec: Vec<char> = s.chars().collect();

    let chars = s.char_indices();

    let mut split_idx = 0;
    for (idx, ch) in chars {
        if idx != 0 && idx == split_idx {
            continue;
        }
        let num = match ch {
            'I' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'V' {
                        split_idx = idx + 1;
                        4
                    } else if next_char == 'X' {
                        split_idx = idx + 1;
                        9
                    } else {
                        split_idx = idx;
                        1
                    }
                } else {
                    split_idx = idx;
                    1
                }
            }
            'V' => 5,
            'X' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'L' {
                        split_idx = idx + 1;
                        40
                    } else if next_char == 'C' {
                        split_idx = idx + 1;
                        90
                    } else {
                        split_idx = idx;
                        10
                    }
                } else {
                    split_idx = idx;
                    10
                }
            }

            'L' => 50,
            'C' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'D' {
                        split_idx = idx + 1;
                        400
                    } else if next_char == 'M' {
                        split_idx = idx + 1;
                        900
                    } else {
                        split_idx = idx;
                        100
                    }
                } else {
                    split_idx = idx;
                    100
                }
            }
            'D' => 500,
            'M' => 1000,
            _ => panic!("No valid character"),
        };
        sum += num;
    }

    sum
}

static ROMAN_NUMBERS: [(char, i32); 7] = [
    ('I', 1),
    ('V', 5),
    ('X', 10),
    ('L', 50),
    ('C', 100),
    ('D', 500),
    ('M', 1000),
];
/// 力扣（13. 罗马数字转整数）
pub fn roman_to_int_v2(s: String) -> i32 {
    use std::collections::HashMap;
    // [(a,b)] convert to HashMap<a,b>
    let map: HashMap<char, i32> = ROMAN_NUMBERS.iter().cloned().collect();
    let mut ret = 0;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        let v = map.get(&c).unwrap();
        match it.peek() {
            Some(n) if v < map.get(n).unwrap() => ret -= v,
            _ => ret += v,
        }
    }
    ret
}

/// 力扣（13. 罗马数字转整数）
pub fn roman_to_int_v3(s: String) -> i32 {
    let mut sum = 0;
    let chars_vec: Vec<char> = s.chars().collect();

    let chars = s.char_indices();
    let len = s.len();

    let mut split_idx = 0;
    for (idx, ch) in chars {
        if idx != 0 && idx == split_idx {
            continue;
        }
        let num = match ch {
            'I' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'V' {
                        split_idx = idx + 1;
                        4
                    } else if next_char == 'X' {
                        split_idx = idx + 1;
                        9
                    } else {
                        split_idx = idx;
                        1
                    }
                } else {
                    split_idx = idx;
                    1
                }
            }
            'V' => 5,
            'X' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'L' {
                        split_idx = idx + 1;
                        40
                    } else if next_char == 'C' {
                        split_idx = idx + 1;
                        90
                    } else {
                        split_idx = idx;
                        10
                    }
                } else {
                    split_idx = idx;
                    10
                }
            }

            'L' => 50,
            'C' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'D' {
                        split_idx = idx + 1;
                        400
                    } else if next_char == 'M' {
                        split_idx = idx + 1;
                        900
                    } else {
                        split_idx = idx;
                        100
                    }
                } else {
                    split_idx = idx;
                    100
                }
            }
            'D' => 500,
            'M' => 1000,
            _ => panic!("No valid character"),
        };
        sum += num;
        // println!("num:{},sum:{}",num,sum);
    }

    sum
}

/// 计算数字number各个位上的数字的平方和
fn get_next(number: i32) -> i32 {
    let mut total_sum = 0;
    let mut num = number;
    while num > 0 {
        let d = num % 10;
        num /= 10;
        total_sum += d * d;
    }
    total_sum
}

// 计算数字number各个位上的数字的平方和
fn bit_square_sum(number: i32) -> i32 {
    let total_sum = number.to_string().chars().fold(0, |acc, x| {
        let y = (x as i32) - 48;
        acc + y * y
    });
    total_sum
}

use std::collections::HashSet;
// const cycle_number:HashSet<i32> = [4, 16, 37,58,89,145,42,20].iter().cloned().collect();
/// 力扣（202. 快乐数) https://leetcode-cn.com/problems/happy-number/
/// 不是快乐数的数称为不快乐数（unhappy number），所有不快乐数的数位平方和计算，最後都会进入 4 → 16 → 37 → 58 → 89 → 145 → 42 → 20 → 4 的循环中。
pub fn is_happy(n: i32) -> bool {
    let mut cycle_number = HashSet::new();
    cycle_number.insert(4);
    cycle_number.insert(16);
    cycle_number.insert(37);
    cycle_number.insert(58);
    cycle_number.insert(89);
    cycle_number.insert(145);
    cycle_number.insert(42);
    cycle_number.insert(20);

    let mut num = n;
    while num != 1 && !cycle_number.contains(&num) {
        num = get_next(num);
        // num = bit_square_sum(num);
    }

    num == 1
}

/// 力扣（202. 快乐数)
/// 快慢指针法、龟兔赛跑法
pub fn is_happy_v2(n: i32) -> bool {
    let mut slow_runner = n;
    let mut fast_runner = get_next(n);
    while fast_runner != 1 && slow_runner != fast_runner {
        slow_runner = get_next(slow_runner);
        fast_runner = get_next(fast_runner);
        fast_runner = get_next(fast_runner);
    }

    fast_runner == 1
}

fn is_prime(x: i32) -> bool {
    let mut i = 2;
    while i * i <= x {
        if x % i == 0 {
            return false;
        }
        i += 1;
    }
    true
}

/// 力扣（204. 计数质数) https://leetcode-cn.com/problems/count-primes/
pub fn count_primes(n: i32) -> i32 {
    let mut ans = 0;
    let mut i = 2;
    while i < n {
        if is_prime(i) {
            ans += 1;
        }
        i += 1;
    }
    ans
}

/// 力扣（204. 计数质数)
/// 方法二：厄拉多塞筛法（埃氏筛）
pub fn count_primes_v2(n: i32) -> i32 {
    let n = n as usize;
    let mut primes = vec![1; n];
    let mut ans = 0;
    let mut i = 2_usize;

    while i < n {
        if primes[i] == 1 {
            ans += 1;
        }
        if let Some(squar) = i.checked_mul(i) {
            if squar < n {
                let mut j = squar;
                while j < n {
                    primes[j] = 0;
                    j += i;
                }
            }
        }

        i += 1;
    }
    ans
}

/// 力扣（263. 丑数）   https://leetcode-cn.com/problems/ugly-number/
/// 丑数 就是只包含质因数 2、3 和/或 5 的正整数。
/// 1 通常被视为丑数。
pub fn is_ugly(num: i32) -> bool {
    if num <= 0 {
        return false;
    }
    let mut num = num;
    for x in [2, 3, 5].iter() {
        while num > 1 && num % x == 0 {
            num /= x;
        }
    }
    num == 1
}

use std::cmp::Reverse;
use std::collections::BinaryHeap;
static UGLY_NUMBER_FACTORS: [i64; 3] = [2, 3, 5];
/// 力扣（264. 丑数 II） https://leetcode-cn.com/problems/ugly-number-ii/
/// 1 通常被视为丑数。
/// 方法一：最小堆
pub fn nth_ugly_number(n: i32) -> i32 {
    //let factors = vec![2, 3, 5];
    let mut seen = HashSet::new();
    let mut heap = BinaryHeap::new();
    seen.insert(1i64);
    heap.push(Reverse(1i64));
    let mut ugly = 0;
    for _ in 0..n {
        if let Some(Reverse(curr)) = heap.pop() {
            ugly = curr;
            for factor in &UGLY_NUMBER_FACTORS {
                let next: i64 = curr * (*factor);
                if seen.insert(next) {
                    heap.push(Reverse(next));
                }
            }
        };
    }
    ugly as i32
}

use std::cmp::min;
/// 力扣（264. 丑数 II）
/// 方法二：动态规划
pub fn nth_ugly_number_v2(n: i32) -> i32 {
    let n = n as usize;
    let mut dp = vec![0; n + 1];
    dp[1] = 1;

    let (mut p2, mut p3, mut p5) = (1, 1, 1);

    for i in 2..=n {
        let (num2, num3, num5) = (dp[p2] * 2, dp[p3] * 3, dp[p5] * 5);
        dp[i] = min(min(num2, num3), num5);
        if dp[i] == num2 {
            p2 += 1;
        }
        if dp[i] == num3 {
            p3 += 1;
        }
        if dp[i] == num5 {
            p5 += 1;
        }
    }

    dp[n]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_numbers() {
        dbg!(is_palindrome(121));
        dbg!(is_palindrome(-121));
        dbg!(is_palindrome(10));
        dbg!(is_palindrome(1));
        dbg!(is_palindrome_v3(8848));

        let roman_numbers = String::from("MCMXCIV");
        dbg!(roman_to_int(roman_numbers));
        let roman_numbers_v2 = String::from("MCMXCIV");
        dbg!(roman_to_int_v2(roman_numbers_v2));

        let roman_numbers_v3 = String::from("MCMXCIV");
        dbg!(roman_to_int_v3(roman_numbers_v3));

        let rand_num = rand::random::<u16>() as i32;
        dbg!(is_happy(rand_num));
        dbg!(is_happy_v2(rand_num));

        let nums3 = vec![-12, 6, 10, 11, 12, 13, 15];
        for num in nums3 {
            dbg!(is_ugly(num));
        }

        dbg!(bit_square_sum(123456));

        let ans = count_primes(10000);
        dbg!(ans);

        let ans = count_primes_v2(10000);
        dbg!(ans);

        dbg!("nth_ugly_number    {}", nth_ugly_number(1690));
        dbg!("nth_ugly_number_v2 {}", nth_ugly_number_v2(1690));
    }
}
