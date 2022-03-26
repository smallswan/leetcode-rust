use std::mem;
/// 72. 编辑距离 https://leetcode-cn.com/problems/edit-distance/
pub fn min_distance(word1: String, word2: String) -> i32 {
    let (word1, word2) = if word2.len() < word1.len() {
        (word2, word1)
    } else {
        (word1, word2)
    };

    let mut cache = (0..=word1.len() as _).rev().collect::<Box<_>>();

    for (prev_base, c2) in word2.as_bytes().iter().rev().enumerate() {
        let mut prev = prev_base as _;

        cache[word1.len()] = prev + 1;

        for (i, c1) in word1.as_bytes().iter().enumerate().rev() {
            let distance = if c1 == c2 {
                prev
            } else {
                cache[i].min(cache[i + 1]).min(prev) + 1
            };

            prev = mem::replace(&mut cache[i], distance);
        }
    }

    cache[0]
}

/// 72. 编辑距离
/// 方法2：动态规划
use std::cmp::min;
pub fn min_distance_v2(word1: String, word2: String) -> i32 {
    let bytes1 = word1.as_bytes();
    let bytes2 = word2.as_bytes();
    let len1 = word1.len();
    let len2 = word2.len();
    let mut dp = vec![vec![0; len2 + 1]; len1 + 1];
    for i in 1..=len2 {
        dp[0][i] = dp[0][i - 1] + 1;
    }

    for i in 1..=len1 {
        dp[i][0] = dp[i - 1][0] + 1;
    }

    for i in 1..=len1 {
        for j in 1..=len2 {
            if bytes1[i - 1] == bytes2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = min(min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
            }
        }
    }

    dp[len1][len2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_distance() {
        dbg!(min_distance(
            "intention".to_string(),
            "execution".to_string()
        ));
        dbg!(min_distance_v2(
            "intention".to_string(),
            "execution".to_string()
        ));
    }
}
