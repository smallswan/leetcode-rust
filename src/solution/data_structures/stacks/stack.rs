//! 栈（stack）
//! https://leetcode-cn.com/tag/stack/problemset/

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
                    let m = is_match_brackets(ch, char);
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
    let forward_max = longest(&seq, '(');
    seq.reverse();
    let backward_max = longest(&seq, ')');
    i32::max(forward_max, backward_max)
}

fn longest(seq: &[char], plus_char: char) -> i32 {
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

/// 力扣（150. 逆波兰表达式求值） https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/
/// 方法二：数组模拟栈
pub fn eval_rpn(tokens: Vec<String>) -> i32 {
    let len = (tokens.len() + 1) / 2;
    let mut stack = vec![0; len];
    let mut index = -1;
    for token in &tokens {
        match (token.as_str()) {
            "+" => {
                index -= 1;
                stack[index as usize] += stack[(index + 1) as usize];
            }
            "-" => {
                index -= 1;
                stack[index as usize] -= stack[(index + 1) as usize];
            }
            "*" => {
                index -= 1;
                stack[index as usize] *= stack[(index + 1) as usize];
            }
            "/" => {
                index -= 1;
                stack[index as usize] /= stack[(index + 1) as usize];
            }

            _ => {
                index += 1;
                stack[index as usize] = token.parse::<i32>().unwrap();
            }
        };
    }

    stack[index as usize]
}

/// 225. 用队列实现栈 https://leetcode-cn.com/problems/implement-stack-using-queues/
use std::collections::VecDeque;
pub struct MyStack {
    q: VecDeque<i32>,
}

impl MyStack {
    fn new() -> Self {
        Self { q: VecDeque::new() }
    }

    fn push(&mut self, x: i32) {
        self.q.push_back(x);

        for _ in 1..self.q.len() {
            let value = self.q.pop_front().unwrap();

            self.q.push_back(value);
        }
    }

    fn pop(&mut self) -> i32 {
        self.q.pop_front().unwrap()
    }

    fn top(&self) -> i32 {
        *self.q.front().unwrap()
    }

    fn empty(&self) -> bool {
        self.q.is_empty()
    }
}

/// 重构字符串
fn build(s: String) -> String {
    let mut chars_vec = Vec::new();
    for ch in s.chars() {
        if ch != '#' {
            chars_vec.push(ch);
        } else if !chars_vec.is_empty() {
            chars_vec.pop();
        }
    }
    chars_vec.into_iter().collect()
}

/// 682. 棒球比赛 https://leetcode-cn.com/problems/baseball-game/
pub fn cal_points(ops: Vec<String>) -> i32 {
    let mut stack: Vec<i32> = Vec::with_capacity(ops.len());

    for op in ops {
        match op.parse().map_err(|_| op.as_str()) {
            Ok(value) => stack.push(value),
            Err("C") => {
                stack.pop();
            }
            Err("D") => stack.push(stack.last().unwrap() * 2),
            Err(_) => {
                let len = stack.len();
                stack.push(stack[len - 2] + stack[len - 1]);
            }
        }
    }

    stack.iter().sum()
}

/// 力扣（844. 比较含退格的字符串)  https://leetcode-cn.com/problems/backspace-string-compare/
/// 方法一：重构字符串
pub fn backspace_compare(s: String, t: String) -> bool {
    build(s) == (build(t))
}

/// 力扣（844. 比较含退格的字符串)
/// 方法二：双指针
pub fn backspace_compare_v2(s: String, t: String) -> bool {
    let mut i = s.len() as i32 - 1;
    let mut j = t.len() as i32 - 1;
    let mut skip_s = 0;
    let mut skip_t = 0;
    let s_chars: Vec<char> = s.chars().into_iter().collect();
    let t_chars: Vec<char> = t.chars().into_iter().collect();

    while i >= 0 || j >= 0 {
        while i >= 0 {
            if s_chars[i as usize] == '#' {
                skip_s += 1;
                i -= 1;
            } else if skip_s > 0 {
                skip_s -= 1;
                i -= 1;
            } else {
                break;
            }
        }

        while j >= 0 {
            if t_chars[j as usize] == '#' {
                skip_t += 1;
                j -= 1;
            } else if skip_t > 0 {
                skip_t -= 1;
                j -= 1;
            } else {
                break;
            }
        }

        if i >= 0 && j >= 0 {
            if s_chars[i as usize] != t_chars[j as usize] {
                return false;
            }
        } else if i >= 0 || j >= 0 {
            return false;
        }

        i -= 1;
        j -= 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_stack() {
        // let tokens = vec![
        //     "2".to_string(),
        //     "1".to_string(),
        //     "+".to_string(),
        //     "3".to_string(),
        //     "*".to_string(),
        // ];
        let tokens = vec![
            "10".to_string(),
            "6".to_string(),
            "9".to_string(),
            "3".to_string(),
            "+".to_string(),
            "-11".to_string(),
            "*".to_string(),
            "/".to_string(),
            "*".to_string(),
            "17".to_string(),
            "+".to_string(),
            "5".to_string(),
            "+".to_string(),
        ];
        dbg!("rpn {}", eval_rpn(tokens));

        let s = String::from("ab#c");
        let t = String::from("ad#c");
        dbg!(backspace_compare(s, t));

        let ops = ["5", "-2", "4", "C", "D", "9", "+", "+"]
            .iter()
            .map(|str| str.to_string())
            .collect();
        dbg!("{}", cal_points(ops));
    }

    #[test]
    fn parentheses() {
        let valid_string = String::from("(){{}");
        dbg!(is_valid(valid_string));

        dbg!(generate_parenthesis(3));

        let s = String::from("))()())");
        dbg!(longest_valid_parentheses(s));
    }
}
