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

/// 力扣（37. 解数独） https://leetcode-cn.com/problems/sudoku-solver/
pub fn solve_sudoku(board: &mut Vec<Vec<char>>) {
    let mut line = Vec::<Vec<bool>>::with_capacity(9);
    let mut column = Vec::<Vec<bool>>::with_capacity(9);
    let mut block = Vec::<Vec<Vec<bool>>>::with_capacity(3);
    let mut spaces = Vec::<(usize, usize)>::new();
    let mut valid = false;
    for i in 0..9 {
        let row = vec![false; 9];
        line.push(row);
        let col = vec![false; 9];
        column.push(col);
    }
    for i in 0..3 {
        let row = vec![vec![false; 9]; 3];
        block.push(row);
    }

    for i in 0..9 {
        for (j, col) in column.iter_mut().enumerate().take(9) {
            if board[i][j] == '.' {
                spaces.push((i, j));
            } else {
                let digit = (board[i][j] as usize) - ('0' as usize) - 1;
                line[i][digit] = true;
                col[digit] = true;
                block[i / 3][j / 3][digit] = true;
            }
        }
    }

    dfs(
        board,
        0,
        &spaces,
        &mut valid,
        &mut line,
        &mut column,
        &mut block,
    );
}
static DIGITS: [char; 10] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
fn dfs(
    board: &mut Vec<Vec<char>>,
    pos: usize,
    spaces: &[(usize, usize)],
    valid: &mut bool,
    line: &mut Vec<Vec<bool>>,
    column: &mut Vec<Vec<bool>>,
    block: &mut Vec<Vec<Vec<bool>>>,
) {
    if pos == spaces.len() {
        *valid = true;
        return;
    }

    let space = spaces.get(pos).unwrap();
    let i = space.0;
    let j = space.1;
    let mut digit = 0;
    while digit < 9 && !(*valid) {
        if !line[i][digit] && !column[j][digit] && !block[i / 3][j / 3][digit] {
            line[i][digit] = true;
            column[j][digit] = true;
            block[i / 3][j / 3][digit] = true;
            board[i][j] = DIGITS[digit + 1];
            dfs(board, pos + 1, spaces, valid, line, column, block);
            line[i][digit] = false;
            column[j][digit] = false;
            block[i / 3][j / 3][digit] = false;
        }
        digit += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn soku() {
        let mut board = Vec::<Vec<char>>::with_capacity(9);
        board.push(vec!['5', '3', '.', '.', '7', '.', '.', '.', '.']);
        board.push(vec!['6', '.', '.', '1', '9', '5', '.', '.', '.']);
        board.push(vec!['.', '9', '8', '.', '.', '.', '.', '6', '.']);
        board.push(vec!['8', '.', '.', '.', '6', '.', '.', '.', '3']);
        board.push(vec!['4', '.', '.', '8', '.', '3', '.', '.', '1']);
        board.push(vec!['7', '.', '.', '.', '2', '.', '.', '.', '6']);
        board.push(vec!['.', '6', '.', '.', '.', '.', '2', '8', '.']);
        board.push(vec!['.', '.', '.', '4', '1', '9', '.', '.', '5']);
        board.push(vec!['.', '.', '.', '.', '8', '.', '.', '7', '9']);

        solve_sudoku(&mut board);

        for row in 0..9 {
            println!("{:?}", board[row]);
        }
    }

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
