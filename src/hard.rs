/// 力扣（4. 寻找两个正序数组的中位数）https://leetcode-cn.com/problems/median-of-two-sorted-arrays/
/// 归并算法  
pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let len1 = nums1.len();
    let len2 = nums2.len();
    // let mut merge_vec = Vec::<i32>::with_capacity(len1 + len2);
    let mut merge_vec = vec![0; len1 + len2];

    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    while i < len1 && j < len2 {
        // println!("i:{},j:{},k:{}", i, j, k);
        if nums1[i] < nums2[j] {
            merge_vec[k] = nums1[i];
            k += 1;
            i += 1;
        } else {
            merge_vec[k] = nums2[j];
            k += 1;
            j += 1;
        }
    }
    while i < len1 {
        merge_vec[k] = nums1[i];
        k += 1;
        i += 1;
    }
    while j < len2 {
        merge_vec[k] = nums2[j];
        k += 1;
        j += 1;
    }

    let t1 = (len1 + len2) % 2;
    if t1 == 1 {
        merge_vec[(len1 + len2) / 2] as f64
    } else {
        let t2 = (len1 + len2) / 2;
        ((merge_vec[t2 - 1] + merge_vec[t2]) as f64) / 2.0
    }
}

/// 力扣（4. 寻找两个正序数组的中位数）
/// 二分查找
pub fn find_median_sorted_arrays_v2(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let len1 = nums1.len();
    let len2 = nums2.len();

    let total_len = len1 + len2;
    if total_len % 2 == 1 {
        let medium_idx = total_len / 2;
        kth_elem(&nums1, &nums2, medium_idx + 1)
    } else {
        let medium_idx1 = total_len / 2 - 1;
        let medium_idx2 = total_len / 2;
        (kth_elem(&nums1, &nums2, medium_idx1 + 1) + kth_elem(&nums1, &nums2, medium_idx2 + 1))
            / 2.0
    }
}

use std::cmp::min;

/// 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
/// 这里的 "/" 表示整除
/// nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
/// nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
/// 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
/// 这样 pivot 本身最大也只能是第 k-1 小的元素
/// 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
/// 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
/// 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
fn kth_elem(nums1: &[i32], nums2: &[i32], k: usize) -> f64 {
    let mut k = k;
    let len1 = nums1.len();
    let len2 = nums2.len();

    let mut idx1 = 0;
    let mut idx2 = 0;
    let mut kth = 0;
    loop {
        if idx1 == len1 {
            return nums2[idx2 + k - 1] as f64;
        }
        if idx2 == len2 {
            return nums1[idx1 + k - 1] as f64;
        }
        if k == 1 {
            return min(nums1[idx1], nums2[idx2]) as f64;
        }

        let half = k / 2;
        let new_idx1 = min(idx1 + half, len1) - 1;
        let new_idx2 = min(idx2 + half, len2) - 1;
        let pivot1 = nums1[new_idx1];
        let pivot2 = nums2[new_idx2];
        if pivot1 <= pivot2 {
            k -= new_idx1 - idx1 + 1;
            idx1 = new_idx1 + 1;
        } else {
            k -= new_idx2 - idx2 + 1;
            idx2 = new_idx2 + 1;
        }
    }
}
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

use crate::simple::ListNode;
/// 力扣（23. 合并K个升序链表） https://leetcode-cn.com/problems/merge-k-sorted-lists/
pub fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    let mut result = Vec::new();
    for node in lists.iter() {
        let mut head = node;
        while head.is_some() {
            let value = head.as_ref().unwrap().val;
            result.push(value);
            head = &(head.as_ref().unwrap().next);
        }
    }

    result.sort();

    let mut head = None;
    for i in result.iter().rev() {
        let mut node = ListNode::new(*i);
        node.next = head;
        head = Some(Box::new(node));
    }
    head
}

/// 力扣（23. 合并K个升序链表）
pub fn merge_k_lists_v2(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    use std::collections::BinaryHeap;
    let mut result = BinaryHeap::new();
    for node in lists.iter() {
        let mut head = node;
        while head.is_some() {
            let value = head.as_ref().unwrap().val;
            result.push(value);
            head = &(head.as_ref().unwrap().next);
        }
    }

    let mut head = None;
    for i in result.into_sorted_vec().iter().rev() {
        let mut node = ListNode::new(*i);
        node.next = head;
        head = Some(Box::new(node));
    }
    head
}

/// 力扣（23. 合并K个升序链表）
/// 方法3：小顶堆
pub fn merge_k_lists_v3(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut dummy_head = Box::new(ListNode::new(0));
    let mut pans = &mut dummy_head;
    let mut result = BinaryHeap::new();
    for node in lists {
        let mut head = &node;
        while let Some(node) = head {
            result.push(Reverse(node.val));
            head = &(node.next);
        }
    }

    while let Some(Reverse(num)) = result.pop() {
        pans.next = Some(Box::new(ListNode::new(num)));
        pans = pans.next.as_mut().unwrap();
    }
    dummy_head.next
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

    use crate::simple;
    #[test]
    fn test_k_merge() {
        // lists: Vec<Option<Box<ListNode>>
        let node1 = crate::simple::vec_to_list(&vec![1, 2, 3]);
        let node2 = crate::simple::vec_to_list(&vec![1, 3, 4]);
        let node3 = crate::simple::vec_to_list(&vec![2, 6]);
        let lists = vec![node1, node2, node3];
        let res = merge_k_lists(lists);
        crate::simple::display(res);

        let node4 = crate::simple::vec_to_list(&vec![1, 2, 3]);
        let node5 = crate::simple::vec_to_list(&vec![1, 3, 4]);
        let node6 = crate::simple::vec_to_list(&vec![2, 6]);
        let lists2 = vec![node4, node5, node6];

        let res2 = merge_k_lists_v2(lists2);
        crate::simple::display(res2);

        let node7 = crate::simple::vec_to_list(&vec![1, 2, 3]);
        let node8 = crate::simple::vec_to_list(&vec![1, 3, 4]);
        let node9 = crate::simple::vec_to_list(&vec![2, 6]);
        let lists3 = vec![node7, node8, node9];
        let res3 = merge_k_lists_v3(lists3);

        crate::simple::display(res3);
    }
}

#[test]
fn hard() {
    let nums1: Vec<i32> = vec![1, 3];
    let nums2: Vec<i32> = vec![2];
    let median_num = find_median_sorted_arrays(nums1, nums2);
    println!("median_num v1:{}", median_num);

    let mut nums3: Vec<i32> = vec![1, 3];
    let mut nums4: Vec<i32> = vec![2];
    let median_num = find_median_sorted_arrays_v2(nums3, nums4);
    println!("median_num v2:{}", median_num);

    println!(
        "{}",
        is_match("mississippi".to_string(), "mis*is*p*.".to_string())
    );
    println!("{}", is_match("aab".to_string(), "c*a*b".to_string()));
    println!("{}", is_match("ab".to_string(), ".*".to_string()));
    println!("{}", is_match("a".to_string(), "ab*a".to_string()));

    println!(
        "{}",
        is_match_v2("mississippi".to_string(), "mis*is*p*.".to_string())
    );
    println!("{}", is_match_v2("aab".to_string(), "c*a*b".to_string()));
    println!("{}", is_match_v2("ab".to_string(), ".*".to_string()));
    println!("{}", is_match_v2("a".to_string(), "ab*a".to_string()));

    println!(
        "{}",
        is_match_v3("mississippi".to_string(), "mis*is*p*.".to_string())
    );
    println!("{}", is_match_v3("aab".to_string(), "c*a*b".to_string()));
    println!("{}", is_match_v3("ab".to_string(), ".*".to_string()));
    println!("{}", is_match_v3("a".to_string(), "ab*a".to_string()));
}
