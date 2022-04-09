#[derive(Debug)]
enum Pattern {
    Char(char), // just char, or dot
    Wild(char), // char *
    Fill,       // 只是占位
}

/// 力扣（10. 正则表达式匹配）  https://leetcode-cn.com/problems/regular-expression-matching/
pub fn is_match(s: String, p: String) -> bool {
    // 将pattern拆成一个数组，*和前面的一个字符一组，其它字符单独一组
    // 从后往前拆
    let mut patterns: Vec<Pattern> = Vec::new();
    {
        let mut p: Vec<char> = p.chars().collect();
        while let Some(c) = p.pop() {
            match c {
                '*' => {
                    patterns.insert(0, Pattern::Wild(p.pop().unwrap()));
                }
                _ => {
                    patterns.insert(0, Pattern::Char(c));
                }
            }
        }
        patterns.insert(0, Pattern::Fill);
    }

    //println!("{:?}", &patterns);

    let mut s: Vec<char> = s.chars().collect();
    s.insert(0, '0');

    let mut matrix: Vec<Vec<bool>> = vec![vec![false; s.len()]; patterns.len()];
    matrix[0][0] = true;

    for i in 1..patterns.len() {
        match patterns[i] {
            Pattern::Char(c) => {
                for (j, &item) in s.iter().enumerate().skip(1) {
                    if (item == c || c == '.') && matrix[i - 1][j - 1] {
                        matrix[i][j] = true;
                    }
                }
            }
            Pattern::Wild(c) => {
                for j in 0..s.len() {
                    if matrix[i - 1][j] {
                        matrix[i][j] = true;
                    }
                }

                for (j, &item) in s.iter().enumerate().skip(1) {
                    if matrix[i][j - 1] && (c == '.' || c == item) {
                        matrix[i][j] = true;
                    }
                }
            }
            _ => {
                println!("{}", "error".to_string());
            }
        }
    }
    //print(&matrix);

    matrix[patterns.len() - 1][s.len() - 1]
}

/// 力扣（10. 正则表达式匹配）  
/// 动态规划
pub fn is_match_v2(s: String, p: String) -> bool {
    let chars: Vec<char> = p.chars().collect();
    let s_len = s.len();
    let p_len = p.len();
    let mut dp = Vec::<Vec<bool>>::with_capacity(s_len + 1);
    for i in 0..=s_len {
        dp.push(vec![false; p_len + 1]);
    }
    dp[0][0] = true;

    for i in 0..=s_len {
        for j in 1..=p_len {
            if chars[j - 1] == '*' {
                dp[i][j] = dp[i][j - 2];
                if matches(&s, &p, i, j - 1) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                }
            } else if matches(&s, &p, i, j) {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }

    dp[s_len][p_len]
}

fn matches(s: &str, p: &str, i: usize, j: usize) -> bool {
    if i == 0 {
        return false;
    }
    let p_chars: Vec<char> = p.chars().collect();
    if p_chars[j - 1] == '.' {
        return true;
    }

    let s_chars: Vec<char> = s.chars().collect();
    s_chars[i - 1] == p_chars[j - 1]
}

/// 力扣（10. 正则表达式匹配）  
/// 动态规划
pub fn is_match_v3(s: String, p: String) -> bool {
    let chars: Vec<char> = p.chars().collect();
    let s_len = s.len();
    let p_len = p.len();
    let mut dp = Vec::<Vec<bool>>::with_capacity(s_len + 1);
    for i in 0..=s_len {
        dp.push(vec![false; p_len + 1]);
    }
    dp[0][0] = true;

    let s_chars: Vec<char> = s.chars().collect();
    for i in 0..=s_len {
        for j in 1..=p_len {
            if chars[j - 1] == '*' {
                dp[i][j] = dp[i][j - 2];
                if matches_v2(&s_chars, &chars, i, j - 1) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                }
            } else if matches_v2(&s_chars, &chars, i, j) {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }

    dp[s_len][p_len]
}

fn matches_v2(s_chars: &[char], p_chars: &[char], i: usize, j: usize) -> bool {
    if i == 0 {
        return false;
    }

    if p_chars[j - 1] == '.' {
        return true;
    }
    s_chars[i - 1] == p_chars[j - 1]
}

/// 62. 不同路径 https://leetcode-cn.com/problems/unique-paths/
pub fn unique_paths(m: i32, n: i32) -> i32 {
    let mut current = vec![1; n as usize];
    for i in 1..m as usize {
        for j in 1..n as usize {
            current[j] += current[j - 1]
        }
    }
    current[(n - 1) as usize]
}

/// 62. 不同路径
pub fn unique_paths_v2(m: i32, n: i32) -> i32 {
    let mut ans = 1_i64;
    let (mut x, mut y) = (n as i64, 1_i64);
    while y < m as i64 {
        ans = ans * x / y;
        x += 1_i64;
        y += 1_i64;
    }

    ans as i32
}

/// 63. 不同路径 II https://leetcode-cn.com/problems/unique-paths-ii/
pub fn unique_paths_with_obstacles(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let columns = obstacle_grid[0].len();
    let mut cache = vec![0; columns];

    cache[columns - 1] = 1;

    for row in obstacle_grid.into_iter().rev() {
        let mut prev = 0;

        for (cell, cache_item) in row.into_iter().zip(&mut cache).rev() {
            if cell == 0 {
                *cache_item += prev;
            } else {
                *cache_item = 0;
            }

            prev = *cache_item;
        }
    }

    cache[0]
}

use std::cmp::min;
/// 力扣（264. 丑数 II） https://leetcode-cn.com/problems/ugly-number-ii/
/// 方法二：动态规划
pub fn nth_ugly_number(n: i32) -> i32 {
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

/// 力扣（338. 比特位计数） https://leetcode-cn.com/problems/counting-bits/
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

/// 1137. 第 N 个泰波那契数 https://leetcode-cn.com/problems/n-th-tribonacci-number/
pub fn tribonacci(n: i32) -> i32 {
    match n {
        0 => 0,
        1 | 2 => 1,
        _ => {
            let (mut t, mut t0, mut t1, mut t2) = (0, 0, 1, 1);
            for i in 3..=n {
                t = t0;
                t0 = t1;
                t1 = t2;
                t2 = t + t0 + t1;
            }
            t2
        }
    }
}

/// 1646. 获取生成数组中的最大值  https://leetcode-cn.com/problems/get-maximum-in-generated-array/
pub fn get_maximum_generated(n: i32) -> i32 {
    if n == 0 {
        return 0;
    }

    let mut nums = vec![0; (n + 1) as usize];
    nums[1] = 1;
    for i in 2..=n {
        let flag = (i % 2);
        let half = (i / 2) as usize;
        if flag == 1 {
            nums[i as usize] = nums[half] + nums[half + 1];
        } else {
            nums[i as usize] = nums[half];
        }
    }

    *nums.iter().max().unwrap()
}

/// 2100. 适合打劫银行的日子 https://leetcode-cn.com/problems/find-good-days-to-rob-the-bank/
pub fn good_days_to_rob_bank(security: Vec<i32>, time: i32) -> Vec<i32> {
    let n = security.len();
    if (n as i32) < time {
        return vec![];
    }
    let mut left = vec![0i32; n];
    let mut right = vec![0i32; n];
    for i in 1..n {
        if security[i] <= security[i - 1] {
            left[i] = left[i - 1] + 1;
        }
        if security[n - i - 1] <= security[n - i] {
            right[n - i - 1] = right[n - i] + 1;
        }
    }
    let mut result = Vec::new();
    //let time = time as usize;
    for i in (time as usize)..n - (time as usize) {
        if left[i] >= time && right[i] >= time {
            result.push(i as i32);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp() {
        dbg!(
            "{}",
            is_match("mississippi".to_string(), "mis*is*p*.".to_string())
        );
        dbg!(is_match("aab".to_string(), "c*a*b".to_string()));
        dbg!(is_match("ab".to_string(), ".*".to_string()));
        dbg!(is_match("a".to_string(), "ab*a".to_string()));

        dbg!(
            "{}",
            is_match_v2("mississippi".to_string(), "mis*is*p*.".to_string())
        );
        dbg!(is_match_v2("aab".to_string(), "c*a*b".to_string()));
        dbg!(is_match_v2("ab".to_string(), ".*".to_string()));
        dbg!(is_match_v2("a".to_string(), "ab*a".to_string()));

        dbg!(
            "{}",
            is_match_v3("mississippi".to_string(), "mis*is*p*.".to_string())
        );
        dbg!(is_match_v3("aab".to_string(), "c*a*b".to_string()));
        dbg!(is_match_v3("ab".to_string(), ".*".to_string()));
        dbg!(is_match_v3("a".to_string(), "ab*a".to_string()));

        dbg!("nth_ugly_number {}", nth_ugly_number(1690));

        dbg!(unique_paths(3, 3));
        dbg!(unique_paths_v2(3, 3));
    }
}
