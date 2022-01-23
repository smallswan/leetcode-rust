pub struct Solution;
impl Solution {
    /// 51. N 皇后 https://leetcode-cn.com/problems/n-queens/
    /// 代码来源：https://github.com/aylei/leetcode-rust/blob/master/src/solution/s0051_n_queens.rs
    pub fn solve_n_queens(n: i32) -> Vec<Vec<String>> {
        let mut board = vec![vec!['.'; n as usize]; n as usize];
        let mut solution = Vec::new();
        Solution::schedule_queens(&mut board, &mut solution, n as usize, 0);
        solution
    }

    /// 52. N皇后 II https://leetcode-cn.com/problems/n-queens-ii/
    /// 代码来源：https://github.com/aylei/leetcode-rust/blob/master/src/solution/s0052_n_queens_ii.rs
    pub fn total_n_queens(n: i32) -> i32 {
        let mut board = vec![vec!['.'; n as usize]; n as usize];
        let mut num = 0;
        Solution::schedule_queens_v2(&mut board, &mut num, n as usize, 0);
        num
    }

    fn schedule_queens_v2(board: &mut Vec<Vec<char>>, num: &mut i32, len: usize, row: usize) {
        for col in 0..len {
            if !Solution::collision(&board, len, row, col) {
                board[row][col] = 'Q';
                if row == len - 1 {
                    *num += 1;
                } else {
                    Solution::schedule_queens_v2(board, num, len, row + 1);
                }
                board[row][col] = '.';
            }
        }
    }

    fn schedule_queens(
        board: &mut Vec<Vec<char>>,
        solution: &mut Vec<Vec<String>>,
        len: usize,
        row: usize,
    ) {
        for col in 0..len {
            if !Solution::collision(&board, len, row, col) {
                board[row][col] = 'Q';
                if row == len - 1 {
                    solution.push(board.iter().map(|vec| vec.iter().collect()).collect());
                } else {
                    Solution::schedule_queens(board, solution, len, row + 1);
                }
                board[row][col] = '.';
            }
        }
    }

    /// 判断冲突
    #[inline(always)]
    fn collision(board: &Vec<Vec<char>>, len: usize, x: usize, y: usize) -> bool {
        // 同一列
        for i in 0..x {
            if board[i][y] == 'Q' {
                return true;
            }
        }
        // 从右下到左上
        let (mut i, mut j) = (x as i32 - 1, y as i32 - 1);
        while i >= 0 && j >= 0 {
            if board[i as usize][j as usize] == 'Q' {
                return true;
            }
            i -= 1;
            j -= 1;
        }
        //  从左下到右上
        let (mut i, mut j) = (x as i32 - 1, y as i32 + 1);
        while i >= 0 && j < len as i32 {
            if board[i as usize][j as usize] == 'Q' {
                return true;
            }
            i -= 1;
            j += 1;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn n_queens() {
        dbg!(Solution::solve_n_queens(8));
        dbg!(Solution::total_n_queens(8));
    }
}
