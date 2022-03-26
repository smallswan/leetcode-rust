/// 36. 有效的数独 https://leetcode-cn.com/problems/valid-sudoku/
pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
    let mut rows = vec![vec![0; 9]; 9];
    let mut columns = vec![vec![0; 9]; 9];
    let mut sub_boxes = vec![vec![vec![0; 9]; 3]; 3];
    for i in 0..9 {
        for j in 0..9 {
            let c = board[i][j];
            if c != '.' {
                let index = (c as u8 - b'0' - 1) as usize;
                rows[i][index] += 1;
                columns[j][index] += 1;
                sub_boxes[i / 3][j / 3][index] += 1;
                if rows[i][index] > 1 || columns[j][index] > 1 || sub_boxes[i / 3][j / 3][index] > 1
                {
                    return false;
                }
            }
        }
    }
    true
}

/// 36. 有效的数独
pub fn is_valid_sudoku_v2(board: Vec<Vec<char>>) -> bool {
    // row: 第一层代表第 1-9 数字，第二层代表第 1-9 行；col、 block 类似
    let [mut row, mut col, mut block] = [[[0u8; 9]; 9]; 3];
    let exists = |arr: &mut [[u8; 9]; 9], number: usize, idx: usize| -> bool {
        arr[number][idx] += 1;
        return if arr[number][idx] > 1 { true } else { false };
    };
    for i in 0..9 {
        for j in 0..9 {
            let ch = board[i][j];
            if ch != '.' {
                let number = ch as usize - 49; // '1' 转换 u8 为 49
                if exists(&mut row, number, i)
                    || exists(&mut col, number, j)
                    || exists(&mut block, number, i / 3 * 3 + j / 3)
                {
                    return false;
                }
            }
        }
    }
    true
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

/// 48. 旋转图像 https://leetcode-cn.com/problems/rotate-image/
pub fn rotate(matrix: &mut Vec<Vec<i32>>) {
    let n = matrix.len();

    for first in 0..n / 2 {
        let last = n - 1 - first;

        for i in first..last {
            let j = n - 1 - i;

            let temp = matrix[first][i];

            matrix[first][i] = matrix[j][first];
            matrix[j][first] = matrix[last][j];
            matrix[last][j] = matrix[i][last];
            matrix[i][last] = temp;
        }
    }
}

/// 54. 螺旋矩阵 https://leetcode-cn.com/problems/spiral-matrix/
pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let m = matrix.len();
    if m == 0 {
        return vec![];
    }
    let n = matrix[0].len();

    let mut result = Vec::<i32>::with_capacity(m * n);

    let mut i = 0;
    let mut j = 0;
    let mut x = m - 1; //i的最大值
    let mut y = n - 1; //j的最大值
    let mut s = 0; //i的最小值
    let mut t = 0; //j的最小值
    let mut direct = 0;

    let mut push_times = 1;
    result.push(matrix[0][0]);

    while push_times < m * n {
        match direct % 4 {
            0 => {
                //右
                if j < y {
                    j += 1;
                    result.push(matrix[i][j]);
                    push_times += 1;
                    continue;
                } else {
                    s += 1;
                    direct += 1;
                }
            }
            1 => {
                //下
                if i < x {
                    i += 1;
                    result.push(matrix[i][j]);
                    push_times += 1;
                    continue;
                } else {
                    y -= 1;
                    direct += 1;
                }
            }
            2 => {
                //左
                if j > t {
                    j -= 1;
                    result.push(matrix[i][j]);
                    push_times += 1;
                    continue;
                } else {
                    x -= 1;
                    direct += 1;
                }
            }
            3 => {
                //上
                if i > s {
                    i -= 1;
                    result.push(matrix[i][j]);
                    push_times += 1;
                    continue;
                } else {
                    t += 1;
                    direct += 1;
                }
            }
            _ => {
                println!("不可能发生这种情况");
            }
        }
    }
    result
}

/// 剑指 Offer 29. 顺时针打印矩阵 https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/
/// 注意：本题与主站 54 题相同：https://leetcode-cn.com/problems/spiral-matrix/
pub fn spiral_order_v2(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let rows = matrix.len();
    if rows == 0 {
        return vec![];
    }
    let columns = matrix[0].len();
    let mut result = Vec::<i32>::with_capacity(rows * columns);
    let mut start = 0usize;
    while rows > start * 2 && columns > start * 2 {
        clockwise(&matrix, &mut result, rows, columns, start);
        start += 1;
    }

    fn clockwise(
        matrix: &Vec<Vec<i32>>,
        result: &mut Vec<i32>,
        rows: usize,
        columns: usize,
        start: usize,
    ) {
        let end_x = columns - 1 - start;
        let end_y = rows - 1 - start;
        // 从左往右
        for i in start..=end_x {
            result.push(matrix[start][i]);
        }
        // 从上往下
        if start < end_y {
            for i in start + 1..=end_y {
                result.push(matrix[i][end_x]);
            }
        }
        // 从右往左
        if start < end_x && start < end_y {
            let mut i = end_x - 1;
            while i >= start {
                result.push(matrix[end_y][i]);
                if i > 0 {
                    i -= 1;
                } else {
                    break;
                }
            }
        }

        // 从下往上
        if end_y < 1 {
            return;
        }
        if start < end_x && start < end_y - 1 {
            let mut i = end_y - 1;
            while i >= start + 1 {
                result.push(matrix[i][start]);
                if i > 0 {
                    i -= 1;
                } else {
                    break;
                }
            }
        }
    }

    result
}

/// 59. 螺旋矩阵 II https://leetcode-cn.com/problems/spiral-matrix-ii/
pub fn generate_matrix(n: i32) -> Vec<Vec<i32>> {
    let n = n as usize;
    let mut result = vec![vec![0; n]; n];
    let mut previous_value = 0;

    let mut next_value = move || {
        previous_value += 1;

        previous_value
    };

    for i in 0..n / 2 {
        // Right.
        result[i][i..n - i].fill_with(&mut next_value);

        // Down.
        for row in &mut result[i + 1..n - i - 1] {
            row[n - i - 1] = next_value();
        }

        // Left.
        for target in result[n - i - 1][i..n - i].iter_mut().rev() {
            *target = next_value();
        }

        // Up.
        for row in result[i + 1..n - i - 1].iter_mut().rev() {
            row[i] = next_value();
        }
    }

    if n % 2 == 1 {
        result[n / 2][n / 2] = next_value();
    }

    result
}

/// 力扣（73.矩阵置零) https://leetcode-cn.com/problems/set-matrix-zeroes/
pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
    let m = matrix.len();
    let n = matrix[0].len();
    let mut row = vec![false; m];
    let mut col = vec![false; n];

    for i in 0..m {
        for (j, item) in col.iter_mut().enumerate().take(n) {
            if matrix[i][j] == 0 {
                row[i] = true;
                *item = true;
            }
        }
    }

    for i in 0..m {
        for (j, &item) in col.iter().enumerate().take(n) {
            if row[i] || item {
                matrix[i][j] = 0;
            }
        }
    }
}

/// 498. 对角线遍历  https://leetcode-cn.com/problems/diagonal-traverse/
pub fn find_diagonal_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let m = matrix.len();
    if m == 0 {
        return vec![];
    }
    let n = matrix[0].len();

    let mut result = Vec::<i32>::with_capacity(m * n);

    let mut i = 0;
    let mut j = 0;
    for _ in 0..m * n {
        result.push(matrix[i][j]);
        if (i + j) % 2 == 0 {
            //往右上角移动，即i-,j+
            if j == n - 1 {
                i += 1;
            } else if i == 0 {
                j += 1;
            } else {
                i -= 1;
                j += 1;
            }
        } else {
            //往左下角移动，即i+,j-
            if i == m - 1 {
                j += 1;
            } else if j == 0 {
                i += 1;
            } else {
                i += 1;
                j -= 1;
            }
        }
    }

    result
}

/// 566. 重塑矩阵 https://leetcode-cn.com/problems/reshape-the-matrix/
pub fn matrix_reshape(mat: Vec<Vec<i32>>, r: i32, c: i32) -> Vec<Vec<i32>> {
    let r = r as usize;
    let c = c as usize;
    let old_rows = mat.len();
    let old_columns = mat.first().map_or(0, Vec::len);

    if old_rows * old_columns == r * c && old_rows != r {
        let mut iter = mat.into_iter().flatten();

        (0..r).map(|_| iter.by_ref().take(c).collect()).collect()
    } else {
        mat
    }
}

/// 力扣（867. 转置矩阵) https://leetcode-cn.com/problems/transpose-matrix/
/// matrixT[i][j] = matrix[j][i]
pub fn transpose(matrix: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut transposed = Vec::<Vec<i32>>::with_capacity(n);
    for i in 0..n {
        transposed.push(vec![0; m]);
    }

    for (i, item) in matrix.iter().enumerate().take(m) {
        for (j, trans) in transposed.iter_mut().enumerate().take(n) {
            trans[i] = item[j];
        }
    }
    transposed
}
/// 力扣（867. 转置矩阵)
pub fn transpose_v2(matrix: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut transposed = Vec::with_capacity(n);
    for j in 0..n {
        let mut row = Vec::with_capacity(m);
        for item in matrix.iter().take(m) {
            row.push(item[j]);
        }
        transposed.push(row);
    }
    transposed
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
    fn test_matrix() {
        let mut matrix = Vec::<Vec<i32>>::new();
        matrix.push(vec![1, 2, 3, 4]);
        matrix.push(vec![5, 6, 7, 8]);
        matrix.push(vec![9, 10, 11, 12]);

        println!("{:?}", matrix);
        //    dbg!("{:?}",find_diagonal_order(matrix));

        dbg!("spiral_order: {:?}", spiral_order(matrix));

        let mut matrix = Vec::<Vec<i32>>::new();
        matrix.push(vec![1, 2, 3]);
        matrix.push(vec![4, 5, 6]);
        matrix.push(vec![7, 8, 9]);
        dbg!(spiral_order_v2(matrix));

        let mut matrix = Vec::<Vec<i32>>::new();
        matrix.push(vec![1, 2, 3]);
        matrix.push(vec![4, 5, 6]);
        // matrix.push(vec![7, 8, 9]);

        let new_matrix = transpose(matrix);
        dbg!(new_matrix);

        let mut matrix2 = Vec::<Vec<i32>>::new();
        matrix2.push(vec![1, 2, 3]);
        matrix2.push(vec![4, 5, 6]);
        // matrix.push(vec![7, 8, 9]);

        let new_matrix2 = transpose_v2(matrix2);
        dbg!(new_matrix2);
    }
}
