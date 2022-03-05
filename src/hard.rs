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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other() {
        dbg!(
            "{}",
            is_match("mississippi".to_string(), "mis*is*p*.".to_string())
        );
        dbg!(is_match("aab".to_string(), "c*a*b".to_string()));
        dbg!(is_match("ab".to_string(), ".*".to_string()));
        dbg!(is_match("a".to_string(), "ab*a".to_string()));
    }
}
