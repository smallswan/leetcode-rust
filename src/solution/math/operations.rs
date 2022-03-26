use std::i32::{MAX, MIN};
/// 力扣（7. 整数反转） ， https://leetcode-cn.com/problems/reverse-integer/
/// 关键是防止转换后的数据溢出（overflow）
pub fn reverse(x: i32) -> i32 {
    //MIN:-2147483648,MAX:2147483647
    println!("MIN:{},MAX:{}", MIN, MAX);
    let mut ret = 0;
    let mut y = x;
    while y != 0 {
        let pop = y % 10;
        y /= 10;
        if ret > MAX / 10 || (ret == MAX / 10 && pop > 7) {
            return 0;
        }

        if ret < MIN / 10 || (ret == MIN / 10 && pop < -8) {
            return 0;
        }
        ret = ret * 10 + pop;
    }
    ret
}

/// 力扣（7. 整数反转）
/// 参考： 吴翱翔 https://zhuanlan.zhihu.com/p/340649000
pub fn reverse2(x: i32) -> i32 {
    || -> Option<i32> {
        let mut ret = 0i32;
        let mut y = x;
        while y.abs() != 0 {
            ret = ret.checked_mul(10)?.checked_add(y % 10)?;
            y /= 10;
        }
        Some(ret)
    }()
    .unwrap_or(0)
}

/// 29. 两数相除 https://leetcode-cn.com/problems/divide-two-integers/
pub fn divide(dividend: i32, divisor: i32) -> i32 {
    if dividend == i32::MIN {
        if divisor == 1 {
            return i32::MIN;
        }

        if divisor == -1 {
            //溢出返回
            return i32::MAX;
        }
    }

    if divisor == i32::MIN {
        if dividend == i32::MIN {
            return 1;
        } else {
            return 0;
        }
    }

    let mut dividend = dividend;
    let mut rev = false;
    if dividend > 0 {
        dividend = -dividend;
        rev = !rev;
    }
    let mut divisor = divisor;
    if divisor > 0 {
        divisor = -divisor;
        rev = !rev;
    }
    let (mut left, mut right, mut ans) = (1, i32::MAX, 0);
    while left <= right {
        let mid = left + ((right - left) >> 1);
        let check = quick_add(divisor, mid, dividend);
        if check {
            ans = mid;
            if mid == i32::MAX {
                break;
            }
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    if rev {
        -ans
    } else {
        ans
    }
}

fn quick_add(y: i32, z: i32, x: i32) -> bool {
    let mut result = 0;
    let mut add = y;
    let mut z = z;
    while z != 0 {
        if (z & 1) != 0 {
            if result < x - add {
                return false;
            }
            result += add;
        }
        if z != 1 {
            if add < x - add {
                return false;
            }
            add += add;
        }
        z >>= 1;
    }
    true
}

/// 41. 缺失的第一个正数 https://leetcode-cn.com/problems/first-missing-positive/
pub fn first_missing_positive(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    let len = nums.len();
    let mut i = 0;
    let mut c = 0;
    while i < len {
        let num = nums[i];
        if num > 0 && num - 1 < (len as i32) {
            c += 1;
            nums.swap((num - 1) as usize, i);
            if (num - 1) > (i as i32) && (num != nums[i]) {
                continue;
            }
        }
        i += 1;
    }

    for (i, &num) in nums.iter().enumerate() {
        if num != ((i + 1) as i32) {
            return (i + 1) as i32;
        }
    }
    return (len + 1) as i32;
}

/// 41. 缺失的第一个正数
pub fn first_missing_positive_v2(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    for i in &mut nums {
        if *i <= 0 {
            *i = i32::MAX;
        }
    }
    let len = nums.len();
    for i in 0..len {
        if nums[i].abs() <= len as i32 {
            let idx = nums[i].abs() as usize - 1;
            if nums[idx] > 0 {
                nums[idx] = -nums[idx];
            }
        }
    }

    for (i, x) in nums.iter().enumerate() {
        if *x >= 0 {
            return i as i32 + 1;
        }
    }
    len as i32 + 1
}

/// 43. 字符串相乘  https://leetcode-cn.com/problems/multiply-strings/
use std::collections::VecDeque;
pub fn multiply(num1: String, num2: String) -> String {
    if &num1 == "0" || &num2 == "0" {
        return "0".to_string();
    }
    //计算num1和num2有多少个后缀0，并把后缀0去掉
    let (num1, num2, zero_cnt) = zero_tail(num1, num2);

    let n1: Vec<u8> = num1.into_bytes().into_iter().map(|c| c - 48).collect();
    let mut n1 = VecDeque::from(n1);
    n1.insert(0, 0);

    let n2: Vec<u8> = num2.into_bytes().into_iter().map(|c| c - 48).collect();
    //克隆第一个字符串，分别保存在9个双端列表中
    let mut vn: Vec<VecDeque<u8>> = vec![n1.clone(); 9];

    //把1到9和第一个字符串相乘的结果保存到每个双端列表中
    for i in 1..9 {
        multi_1_9(&mut vn, i);
    }
    let len2 = n2.len();
    let mut ans = vn[(n2[len2 - 1] - 1) as usize].clone();

    let mut cnt = 1;
    //遍历第二个字符串，每个字符对应前面1到9的双端列表，移位相加
    for i in (0..len2 - 1).rev() {
        ans.insert(0, 0);
        ans.insert(0, 0);
        if n2[i] > 0 {
            let tmp = &vn[(n2[i] - 1) as usize];
            let mut index_b = tmp.len();
            let mut flag = 0;
            for j in (0..ans.len() - cnt).rev() {
                let a = ans[j] + flag;
                let b = if index_b > 0 {
                    index_b -= 1;
                    tmp[index_b]
                } else {
                    0
                };
                ans[j] = (a + b) % 10;
                flag = if a + b > 9 { 1 } else { 0 };
            }
        }
        cnt += 1;
    }
    //追加后缀0
    for _ in 0..zero_cnt {
        ans.push_back(0);
    }
    //去掉前缀0
    while ans[0] == 0 {
        ans.pop_front();
    }
    let ans: Vec<u8> = ans.into_iter().map(|a| a + 48).collect();
    String::from_utf8(ans).unwrap()
}
//计算后缀0个数，并去掉后缀0
fn zero_tail(s1: String, s2: String) -> (String, String, usize) {
    let mut zero_cnt = 0;
    let mut v = vec![s1, s2];
    for s in v.iter_mut() {
        let mut tmp = 0;
        for i in (0..s.len()).rev() {
            if &s[i..i + 1] == "0" {
                tmp += 1;
            } else {
                break;
            }
        }
        if tmp > 0 {
            zero_cnt += tmp;
            *s = format!("{}", &s[..s.len() - tmp]);
        }
    }
    (v[0].clone(), v[1].clone(), zero_cnt)
}

//1到9和第一个字符串相乘的结果保存到双端列表中
fn multi_1_9(vn: &mut Vec<VecDeque<u8>>, index: usize) {
    let mut flag = 0;
    let len = vn[0].len();
    for i in (0..len).rev() {
        let sum = vn[index - 1][i] + vn[index][i] + flag;
        vn[index][i] = sum % 10;
        flag = if sum > 9 { 1 } else { 0 };
    }
}

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

/// 力扣（70. 爬楼梯）
/// 方法2：通用公式
pub fn climb_stairs_v2(n: i32) -> i32 {
    let sqrt5 = 5.0_f64.sqrt();
    let fibn = ((1.0 + sqrt5) / 2.0).powi(n as i32 + 1) + ((1.0 - sqrt5) / 2.0).powi(n as i32 + 1);

    (fibn / sqrt5).round() as i32
}

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::convert::TryInto;

fn gcd(mut x: i32, mut y: i32) -> i32 {
    while y != 0 {
        let next_y = x % y;

        x = y;
        y = next_y;
    }

    x
}

fn get_line_key([x_1, y_1]: [i32; 2], [x_2, y_2]: [i32; 2]) -> (i32, i32) {
    let d_x = x_2 - x_1;
    let d_y = y_2 - y_1;
    let g = gcd(d_x, d_y);
    let r_x = d_x / g;
    let r_y = d_y / g;

    if r_x < 0 {
        (-r_x, -r_y)
    } else {
        (r_x, r_y)
    }
}

/// 149. 直线上最多的点数 https://leetcode-cn.com/problems/max-points-on-a-line/
pub fn max_points(points: Vec<Vec<i32>>) -> i32 {
    let mut result = 0;
    let mut counts = HashMap::new();

    for (i, point_1) in points
        .iter()
        .map(|p| p.as_slice().try_into().unwrap())
        .enumerate()
    {
        for point_2 in points[i + 1..]
            .iter()
            .map(|p| p.as_slice().try_into().unwrap())
        {
            match counts.entry(get_line_key(point_1, point_2)) {
                Entry::Occupied(entry) => *entry.into_mut() += 1,
                Entry::Vacant(entry) => {
                    entry.insert(1);
                }
            }
        }

        result = result.max(1 + counts.drain().map(|(_, v)| v).max().unwrap_or(0));
    }

    result
}

/// 166. 分数到小数 https://leetcode-cn.com/problems/fraction-to-recurring-decimal/
pub fn fraction_to_decimal(numerator: i32, denominator: i32) -> String {
    let numerator = i64::from(numerator);
    let denominator = i64::from(denominator);
    let integer_part = numerator / denominator;
    let mut remainder = numerator % denominator;

    let mut result = if integer_part == 0
        && if denominator < 0 {
            numerator > 0
        } else {
            numerator < 0
        } {
        String::from("-0")
    } else {
        integer_part.to_string()
    };

    if remainder != 0 {
        let denominator = denominator.abs();
        let mut remainder_to_index = HashMap::new();

        result.push('.');
        remainder = remainder.abs();
        loop {
            match remainder_to_index.entry(remainder) {
                Entry::Occupied(entry) => {
                    result.insert(*entry.into_mut(), '(');
                    result.push(')');

                    break;
                }
                Entry::Vacant(entry) => {
                    let temp = remainder * 10;

                    entry.insert(result.len());
                    result.push(char::from(b'0' + (temp / denominator) as u8));
                    remainder = temp % denominator;
                }
            }

            if remainder == 0 {
                break;
            }
        }
    }

    result
}

/// 力扣（168. Excel表列名称） https://leetcode-cn.com/problems/excel-sheet-column-title/
pub fn convert_to_title(column_number: i32) -> String {
    let mut ret = String::new();
    let mut column_number = column_number;
    while column_number > 0 {
        let a0 = (column_number - 1) % 26 + 1;
        let ch = b'A' + (a0 - 1) as u8;
        ret.push(ch as char);
        column_number = (column_number - a0) / 26;
    }

    ret.chars().rev().collect()
}

/// 力扣（171. Excel 表列序号） https://leetcode-cn.com/problems/excel-sheet-column-number/submissions/
pub fn title_to_number(column_title: String) -> i32 {
    let mut sum = 0;
    let mut hex_base = 1;
    for ch in column_title.chars().rev() {
        sum += (hex_base * (ch as u8 - b'A' + 1) as i32);
        hex_base *= 26;
    }

    sum
}

/// 力扣（172. 阶乘后的零） https://leetcode-cn.com/problems/factorial-trailing-zeroes/
pub fn trailing_zeroes(n: i32) -> i32 {
    let mut count_fives = 0;
    // let mut steps: Vec<i32> = (5..=n).into_iter().filter(|x| *x % 5 == 0).collect();
    let mut steps: Vec<i32> = (5..=n).into_iter().step_by(5).collect();
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

/// 223. 矩形面积 https://leetcode-cn.com/problems/rectangle-area/
pub fn compute_area(
    ax1: i32,
    ay1: i32,
    ax2: i32,
    ay2: i32,
    bx1: i32,
    by1: i32,
    bx2: i32,
    by2: i32,
) -> i32 {
    //
    let (area1, area2) = ((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1));
    let overlap_width = min(ax2, bx2) - max(ax1, bx1);
    let overlap_height = min(ay2, by2) - max(ay1, by1);
    let overlap_area = max(overlap_width, 0) * max(overlap_height, 0);

    area1 + area2 - overlap_area
}

/// 224. 基本计算器 https://leetcode-cn.com/problems/basic-calculator/
pub fn calculate(s: String) -> i32 {
    let mut stack = Vec::new();
    let mut lhs = 0;
    let mut rhs = 0;
    let mut sign = 1;

    for c in s.bytes() {
        match c {
            b'+' => {
                lhs += rhs * sign;
                rhs = 0;
                sign = 1;
            }
            b'-' => {
                lhs += rhs * sign;
                rhs = 0;
                sign = -1;
            }
            b'(' => {
                stack.push((lhs, sign));
                lhs = 0;
                sign = 1;
            }
            b')' => {
                rhs = lhs + rhs * sign;

                let (saved_lhs, saved_sign) = stack.pop().unwrap();

                lhs = saved_lhs;
                sign = saved_sign;
            }
            b'0'..=b'9' => rhs = rhs * 10 + i32::from(c - b'0'),
            _ => {}
        }
    }

    lhs + rhs * sign
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

/// 233. 数字 1 的个数 https://leetcode-cn.com/problems/number-of-digit-one/
pub fn count_digit_one(n: i32) -> i32 {
    let mut n = i64::from(n);
    let mut result = 0;
    let mut ten_to_the_power = 1;
    let mut base = 0;
    let mut processed = 0;

    while n > 0 {
        let digit = n % 10;

        match digit {
            0 => {}
            1 => result += base + processed + 1,
            _ => result += base * digit + ten_to_the_power,
        }

        processed += ten_to_the_power * digit;
        base = ten_to_the_power + base * 10;
        ten_to_the_power *= 10;
        n /= 10;
    }

    result as _
}

/// 力扣（258. 各位相加） https://leetcode-cn.com/problems/add-digits/
pub fn add_digits(num: i32) -> i32 {
    (num - 1) % 9 + 1
}

/// 力扣（292. Nim 游戏） https://leetcode-cn.com/problems/nim-game/
pub fn can_win_nim(n: i32) -> bool {
    n % 4 != 0
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

/// 力扣（342. 4的幂） https://leetcode-cn.com/problems/power-of-four/
pub fn is_power_of_four(n: i32) -> bool {
    n > 0 && (n & (n - 1)) == 0 && (n & 0x2aaaaaaa == 0)
}

fn less_than_thousand(num: i32, result: &mut String) {
    const SINGLES: [&str; 19] = [
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
        "Ten",
        "Eleven",
        "Twelve",
        "Thirteen",
        "Fourteen",
        "Fifteen",
        "Sixteen",
        "Seventeen",
        "Eighteen",
        "Nineteen",
    ];

    const TENS: [&str; 8] = [
        "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety",
    ];
    match num {
        1..=19 => result.push_str(SINGLES[(num - 1) as usize]),
        20..=99 => {
            result.push_str(TENS[(num / 10 - 2) as usize]);

            let remainder = num % 10;

            if remainder != 0 {
                result.push(' ');
                less_than_thousand(remainder, result);
            }
        }
        _ => {
            less_than_thousand(num / 100, result);

            result.push_str(" Hundred");

            let remainder = num % 100;

            if remainder != 0 {
                result.push(' ');
                less_than_thousand(remainder, result);
            }
        }
    }
}

/// 273. 整数转换英文表示 https://leetcode-cn.com/problems/integer-to-english-words/
pub fn number_to_words(num: i32) -> String {
    let mut result = String::new();

    if num == 0 {
        result.push_str("Zero");
    } else {
        let mut num = num;

        for (name, base) in [
            ("Billion", 1_000_000_000),
            ("Million", 1_000_000),
            ("Thousand", 1_000),
        ] {
            if num >= base {
                less_than_thousand(num / base, &mut result);
                result.push(' ');
                result.push_str(name);

                num %= base;

                if num == 0 {
                    return result;
                }

                result.push(' ');
            }
        }

        less_than_thousand(num, &mut result);
    }

    result
}

/// 441. 排列硬币 https://leetcode-cn.com/problems/arranging-coins/
pub fn arrange_coins(n: i32) -> i32 {
    (((1.0f64 + 8.0f64 * n as f64).sqrt() - 1.0f64) / 2.0).floor() as i32
}

/// 力扣（412. Fizz Buzz） https://leetcode-cn.com/problems/fizz-buzz/
pub fn fizz_buzz(n: i32) -> Vec<String> {
    let len = n as usize;
    let mut result = Vec::<String>::with_capacity(len);
    for i in 1..=len {
        let divisible_by_3 = i % 3 == 0;
        let divisible_by_5 = i % 5 == 0;
        if divisible_by_3 && divisible_by_5 {
            result.push("FizzBuzz".to_string());
        } else if divisible_by_3 {
            result.push("Fizz".to_string());
        } else if divisible_by_5 {
            result.push("Buzz".to_string());
        } else {
            result.push(i.to_string());
        }
    }

    result
}

/// 力扣（412. Fizz Buzz）
pub fn fizz_buzz_v2(n: i32) -> Vec<String> {
    let len = n as usize;
    let mut result = Vec::<String>::with_capacity(len);
    for i in 1..=len {
        if i % 3 == 0 {
            if i % 5 == 0 {
                result.push("FizzBuzz".to_string());
            } else {
                result.push("Fizz".to_string());
            }
        } else if i % 5 == 0 {
            result.push("Buzz".to_string());
        } else {
            result.push(i.to_string());
        }
    }

    result
}

/// 力扣（415. 字符串相加） https://leetcode-cn.com/problems/add-strings/
pub fn add_strings(num1: String, num2: String) -> String {
    let mut i = (num1.len() - 1) as i32;
    let mut j = (num2.len() - 1) as i32;
    let mut add = 0;
    let mut ans = Vec::<u8>::new();
    let num1_chars = num1.into_bytes();
    let num2_chars = num2.into_bytes();

    while i >= 0 || j >= 0 || add != 0 {
        let x = if i >= 0 {
            num1_chars[i as usize] - b'0'
        } else {
            0
        };
        let y = if j >= 0 {
            num2_chars[j as usize] - b'0'
        } else {
            0
        };
        let result = x + y + add;
        ans.push((result % 10 + b'0') as u8);
        add = result / 10;
        i -= 1;
        j -= 1;
    }
    ans.reverse();
    String::from_utf8(ans).unwrap()
}

/// 463. 岛屿的周长 https://leetcode-cn.com/problems/island-perimeter/
pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
    let mut result = 0;

    for (i, row) in grid.iter().enumerate() {
        for (j, &cell) in row.iter().enumerate() {
            if cell != 0 {
                if grid.get(i.wrapping_sub(1)).map_or(true, |r| r[j] == 0) {
                    result += 1;
                }

                if row.get(j.wrapping_sub(1)).map_or(true, |&c| c == 0) {
                    result += 1;
                }

                if row.get(j + 1).map_or(true, |&c| c == 0) {
                    result += 1;
                }

                if grid.get(i + 1).map_or(true, |r| r[j] == 0) {
                    result += 1;
                }
            }
        }
    }

    result
}

/// 492. 构造矩形 https://leetcode-cn.com/problems/construct-the-rectangle/
pub fn construct_rectangle(area: i32) -> Vec<i32> {
    let mut w = (area as f32).sqrt() as i32;
    while area % w != 0 {
        w -= 1;
    }
    vec![area / w, w]
}

use std::cmp::min;
/// 598. 范围求和 II https://leetcode-cn.com/problems/range-addition-ii/
pub fn max_count(m: i32, n: i32, ops: Vec<Vec<i32>>) -> i32 {
    let (mut min_a, mut min_b) = (m, n);

    for op in ops.iter() {
        min_a = min(op[0], min_a);
        min_b = min(op[1], min_b);
    }

    min_a * min_b
}

/// 剑指 Offer 17. 打印从1到最大的n位数 https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/
pub fn print_numbers(n: i32) -> Vec<i32> {
    let max = (10i32.pow(n as u32) - 1);
    let mut result = Vec::<i32>::with_capacity(max as usize);
    for num in 1..=max {
        result.push(num);
    }
    result
}

use std::cmp::max;
/// 628. 三个数的最大乘积 https://leetcode-cn.com/problems/maximum-product-of-three-numbers/
pub fn maximum_product(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort_unstable();
    let len = nums.len();

    max(
        nums[0] * nums[1] * nums[len - 1],
        nums[len - 3] * nums[len - 2] * nums[len - 1],
    )
}

/// 762. 二进制表示中质数个计算置位 https://leetcode-cn.com/problems/prime-number-of-set-bits-in-binary-representation/
pub fn count_prime_set_bits(left: i32, right: i32) -> i32 {
    let mut count = 0;
    for num in left..=right {
        match num.count_ones() {
            2 | 3 | 5 | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 => {
                count += 1;
            }
            _ => (),
        }
    }
    count
}

/// 812. 最大三角形面积 https://leetcode-cn.com/problems/largest-triangle-area/
pub fn largest_triangle_area(points: Vec<Vec<i32>>) -> f64 {
    //鞋带公式
    fn area(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> f64 {
        0.5f64 * (x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1).abs()
    }
    let mut ans = 0f64;
    let len = points.len();
    for i in 0..len {
        for j in i + 1..len {
            for k in j + 1..len {
                let temp = area(
                    points[i][0] as f64,
                    points[i][1] as f64,
                    points[j][0] as f64,
                    points[j][1] as f64,
                    points[k][0] as f64,
                    points[k][1] as f64,
                );
                ans = if temp > ans { temp } else { ans }
            }
        }
    }

    ans
}

/// 836. 矩形重叠 https://leetcode-cn.com/problems/rectangle-overlap/
pub fn is_rectangle_overlap(rec1: Vec<i32>, rec2: Vec<i32>) -> bool {
    (min(rec1[2], rec2[2]) > max(rec1[0], rec2[0]) && min(rec1[3], rec2[3]) > max(rec1[1], rec2[1]))
}

/// 868. 二进制间距 https://leetcode-cn.com/problems/binary-gap/
pub fn binary_gap(n: i32) -> i32 {
    let (mut last, mut ans) = (-1, 0);
    for i in 0..32 {
        if ((n >> i) & 1) > 0 {
            if last >= 0 {
                ans = max(ans, i - last);
            }
            last = i;
        }
    }

    ans
}

/// 883. 三维形体投影面积 https://leetcode-cn.com/problems/projection-area-of-3d-shapes/
pub fn projection_area(grid: Vec<Vec<i32>>) -> i32 {
    let len = grid.len();
    let mut ans = 0;
    for i in 0..len {
        let (mut best_row, mut best_col) = (0, 0);
        for j in 0..len {
            if grid[i][j] > 0 {
                ans += 1;
            }
            best_row = max(best_row, grid[i][j]);
            best_col = max(best_col, grid[j][i]);
        }
        ans += best_row + best_col;
    }

    ans
}

/// 892. 三维形体的表面积 https://leetcode-cn.com/problems/surface-area-of-3d-shapes/
pub fn surface_area(grid: Vec<Vec<i32>>) -> i32 {
    let dr = vec![0, 1, 0, -1];
    let dc = vec![1, 0, -1, 0];
    let len = grid.len();
    let mut ans = 0;
    for r in 0..len {
        for c in 0..len {
            if grid[r][c] > 0 {
                ans += 2;
                for k in 0..4 {
                    let nr = (r as i32 + dr[k]) as usize;
                    let nc = (c as i32 + dc[k]) as usize;

                    let mut nv = 0;
                    if nr < len && nc < len {
                        nv = grid[nr][nc];
                    }

                    ans += max(grid[r][c] - nv, 0);
                }
            }
        }
    }

    ans
}

/// 908. 最小差值 I  https://leetcode-cn.com/problems/smallest-range-i/
pub fn smallest_range_i(nums: Vec<i32>, k: i32) -> i32 {
    let (mut min, mut max) = (nums[0], nums[0]);
    for x in nums {
        min = std::cmp::min(min, x);
        max = std::cmp::max(max, x);
    }
    std::cmp::max(0, max - min - 2 * k)
}

/// 1025. 除数博弈 https://leetcode-cn.com/problems/divisor-game/
pub fn divisor_game(n: i32) -> bool {
    n % 2 == 0
}

/// 剑指 Offer 17. 打印从1到最大的n位数
pub fn print_numbers_v2(n: i32) -> Vec<i32> {
    let max = (10i32.pow(n as u32) - 1);
    let mut result = Vec::<i32>::with_capacity(max as usize);
    const NUMBERS: [char; 10] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

    fn dfs(
        result: &mut Vec<i32>,
        num: &mut [char],
        idx: usize,
        len: usize,
        start: &mut usize,
        nine: &mut usize,
    ) {
        if idx == len {
            let number: String = num.iter().skip(*start).collect();
            if ("0" != number) {
                result.push(number.parse().unwrap());
            }

            if len - (*start) == (*nine) {
                if (*start > 0) {
                    *start -= 1;
                }
            }
            return;
        }
        for i in NUMBERS {
            if i == '9' {
                *nine += 1;
            }
            num[idx] = i;
            dfs(result, num, idx + 1, len, start, nine);
        }
        *nine -= 1;
    }

    let mut num: Vec<char> = vec!['0'; n as usize];
    let mut start = (n - 1) as usize;
    let mut nine = 0;
    dfs(&mut result, &mut num, 0, n as usize, &mut start, &mut nine);

    result
}

/// 剑指 Offer 65. 不用加减乘除做加法 https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/
pub fn add(a: i32, b: i32) -> i32 {
    let (mut a, mut b) = (a, b);

    while b != 0 {
        // c 进位
        let c = (a & b) << 1;
        // a 非进位和
        a ^= b;
        b = c;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse() {
        dbg!(reverse(132));
        dbg!(reverse(-1999999999));
    }

    #[test]
    fn add_sub_mul_div() {
        dbg!(plus_one(vec![9, 1, 9]));
        let a = String::from("0");
        let b = String::from("0");
        dbg!(add_binary(a, b));

        dbg!(add_strings("11".to_string(), "123".to_string()));
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

        dbg!(construct_rectangle(10_000_000));
    }

    #[test]
    fn lcof() {
        dbg!(print_numbers_v2(2));
    }

    #[test]
    fn test_area() {
        let (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) = (-3, 0, 3, 4, 0, -1, 9, 2);
        dbg!(compute_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2));
    }
}
