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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp() {
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
    }
}
