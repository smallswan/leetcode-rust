pub struct Solution;
impl Solution {
    /// 力扣（20. 有效的括号）https://leetcode-cn.com/problems/valid-parentheses/
    fn is_valid(s: String) -> bool {
        let len = s.len();
        if len == 0 {
            return false;
        }
        let chars: Vec<char> = s.chars().collect();
        //使用 Vec模拟Stack
        let mut stack = Vec::<char>::with_capacity(len);

        for char in chars {
            if char == ')' || char == '}' || char == ']' {
                let prev_ch = stack.pop();
                match prev_ch {
                    Some(ch) => {
                        let m = Solution::is_match_brackets(ch, char);
                        if !m {
                            return false;
                        }
                    }
                    None => {
                        return false;
                    }
                };
            } else {
                stack.push(char);
            }
        }

        stack.is_empty()
    }

    /// 判断括号是否匹配
    fn is_match_brackets(left: char, right: char) -> bool {
        match left {
            '(' => right == ')',
            '{' => right == '}',
            '[' => right == ']',
            _ => false,
        }
    }

    /// 22. 括号生成 https://leetcode-cn.com/problems/generate-parentheses/
    pub fn generate_parenthesis(n: i32) -> Vec<String> {
        if n < 1 {
            return vec![];
        }
        fn dfs(n: i32, left: i32, right: i32, result: &mut Vec<String>, mut path: String) {
            if left == n && right == n {
                result.push(path);
                return;
            }
            if left < n {
                let mut new_path = path.clone();
                new_path.push('(');
                dfs(n, left + 1, right, result, new_path);
            }
            if right < left {
                // reuse path to avoid clone overhead
                path.push(')');
                dfs(n, left, right + 1, result, path);
            }
        }
        let mut result = Vec::new();
        dfs(n, 0, 0, &mut result, String::new());
        result
    }

    /// 32. 最长有效括号  https://leetcode-cn.com/problems/longest-valid-parentheses/
    pub fn longest_valid_parentheses(s: String) -> i32 {
        let mut seq: Vec<char> = s.chars().collect();
        let forward_max = Solution::longest(&seq, '(');
        seq.reverse();
        let backward_max = Solution::longest(&seq, ')');
        i32::max(forward_max, backward_max)
    }

    fn longest(seq: &Vec<char>, plus_char: char) -> i32 {
        let mut stack = 0;
        let mut max_len = 0;
        let (mut i, mut j) = (0_usize, 0_usize);
        while j < seq.len() {
            if seq[j] == plus_char {
                stack += 1;
            } else {
                // stack exhausted, shift over
                if stack < 1 {
                    i = j + 1;
                } else {
                    stack -= 1;
                    if stack == 0 {
                        max_len = i32::max(max_len, (j - i + 1) as i32);
                    }
                }
            }
            j += 1;
        }
        max_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parentheses() {
        let valid_string = String::from("(){{}");
        dbg!(Solution::is_valid(valid_string));

        dbg!(Solution::generate_parenthesis(3));

        let s = String::from("))()())");
        dbg!(Solution::longest_valid_parentheses(s));
    }
}
