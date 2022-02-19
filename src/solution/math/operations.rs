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

/// 力扣（342. 4的幂） https://leetcode-cn.com/problems/power-of-four/
pub fn is_power_of_four(n: i32) -> bool {
    n > 0 && (n & (n - 1)) == 0 && (n & 0x2aaaaaaa == 0)
}

/// 441. 排列硬币 https://leetcode-cn.com/problems/arranging-coins/
pub fn arrange_coins(n: i32) -> i32 {
    (((1.0f64 + 8.0f64 * n as f64).sqrt() - 1.0f64) / 2.0).floor() as i32
}

/// 492. 构造矩形 https://leetcode-cn.com/problems/construct-the-rectangle/
pub fn construct_rectangle(area: i32) -> Vec<i32> {
    let mut w = (area as f32).sqrt() as i32;
    while area % w != 0 {
        w -= 1;
    }
    vec![area / w, w]
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

        dbg!(construct_rectangle(10_000_000));
    }

    #[test]
    fn lcof() {
        dbg!(print_numbers_v2(2));
    }
}
