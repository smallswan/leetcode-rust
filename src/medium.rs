//! 中等难度

// 计算以第i个字符为中点的字符串的最长回文，从中间往两边检查，复杂度为O(N)。
// 返回值：(最长回文的起点、最长回文的终点)
fn longest_of(i: usize, s: &[u8]) -> (usize, usize) {
    // 先检查奇数长度的字符串，中点只有一个，就是第i个字符
    let mut ret1 = (i, i);
    for (f, t) in (0..=i).rev().zip(i..s.len()) {
        if s[f] != s[t] {
            break;
        }
        ret1 = (f, t);
    }

    // 再检查偶数长度的字符串，中点有两个，此处指定为第i、i+1个
    let mut ret2 = (i, i);
    for (f, t) in (0..=i).rev().zip(i + 1..s.len()) {
        if s[f] != s[t] {
            break;
        }
        ret2 = (f, t);
    }

    if ret2.1 - ret2.0 > ret1.1 - ret1.0 {
        ret2
    } else {
        ret1
    }
}

/// 力扣（5. 最长回文子串） https://leetcode-cn.com/problems/longest-palindromic-substring/solution/zui-chang-hui-wen-zi-chuan-by-leetcode-solution/
/// 中心拓展算法
pub fn longest_palindrome(s: String) -> String {
    if s.is_empty() {
        return "".to_string();
    }

    let mut range = (0, 0);
    let s = s.as_bytes();
    // 遍历每个字符，找出以当前字符为中点的最长回文字符串
    for i in 0..s.len() {
        let r = longest_of(i, &s);
        if r.1 - r.0 > range.1 - range.0 {
            range = r;
        }
    }
    return std::str::from_utf8(&s[range.0..=range.1])
        .unwrap()
        .to_string();
}

/// 力扣（5. 最长回文子串）
/// 代码不正确
pub fn longest_palindrome_v2(s: String) -> String {
    //    let s_chars = s.chars();
    let s_bytes = s.as_bytes();
    let mut res_bytes = vec![b'#'; 2 * s_bytes.len() + 1];

    let mut index = 0;
    let mut i = 0;
    while i != res_bytes.len() {
        if i & 1 != 0 {
            index += 1;
            res_bytes[i] = s_bytes[index - 1];
        }
        i += 1;
    }
    //    let new_s = String::from_utf8(res_bytes).unwrap();

    let mut rl = vec![0; 2 * s_bytes.len() + 1];

    println!("rl:{:?}", rl);
    let mut max_right = 0;
    let mut pos = 0;
    let mut max_len = 0;

    for j in 0..res_bytes.len() {
        if j < max_right {
            rl[j] = if rl[2 * pos - j] < max_right - j {
                rl[2 * pos - j]
            } else {
                max_right - j
            }
        } else {
            rl[j] = 1;
        }
        println!("j:{},rl[j]:{}", j, rl[j]);
        while (j - rl[j]) > 0
            && (j + rl[j]) < res_bytes.len()
            && res_bytes[j - rl[j]] == res_bytes[j + rl[j]]
        {
            rl[j] += 1;
        }

        if rl[j] + j - 1 > max_right {
            max_right = rl[j] + j - 1;
            pos = j;
        }
        if max_len < rl[j] {
            max_len = rl[j];
        }
    }

    println!("max_len:{}", max_len);

    String::from_utf8(res_bytes).unwrap()
}

/// 力扣（11. 盛最多水的容器） https://leetcode-cn.com/problems/container-with-most-water/
pub fn max_area(height: Vec<i32>) -> i32 {
    use std::cmp::max;
    let mut max_area = 0;
    let mut left = 0;
    let mut right = height.len() - 1;
    while left < right {
        if height[left] < height[right] {
            let area = height[left] * ((right - left) as i32);
            max_area = max(max_area, area);
            left += 1;
        } else {
            let area = height[right] * ((right - left) as i32);
            max_area = max(max_area, area);
            right -= 1;
        }
    }

    max_area
}

/// 力扣（11. 盛最多水的容器）
pub fn max_area_v2(height: Vec<i32>) -> i32 {
    use std::cmp::{max, min};
    let mut max_area = 0;
    let mut left = 0;
    let mut right = height.len() - 1;
    while left < right {
        let area = min(height[left], height[right]) * ((right - left) as i32);
        max_area = max(max_area, area);

        if height[left] <= height[right] {
            left += 1;
        } else {
            right -= 1;
        }
    }

    max_area
}

/// 力扣（11. 盛最多水的容器）
pub fn max_area_v3(height: Vec<i32>) -> i32 {
    use std::cmp::{max, min};
    let mut max_area = 0;
    let mut left = 0;
    let mut right = height.len() - 1;
    let mut min_height = 0;
    while left < right {
        let current_min_height = min(height[left], height[right]);
        if current_min_height > min_height {
            let area = current_min_height * ((right - left) as i32);
            max_area = max(max_area, area);
            min_height = current_min_height;
        }

        if height[left] <= height[right] {
            left += 1;
        } else {
            right -= 1;
        }
    }

    max_area
}

use std::collections::HashMap;
fn backtrace(
    combinations: &mut Vec<String>,
    nums_map: &HashMap<char, &[char]>,
    digits: &[char],
    index: usize,
    combination: &mut Vec<char>,
) {
    if index == digits.len() {
        let abc = combination.iter().collect();
        combinations.push(abc);
    } else {
        let digit = digits[index];

        match nums_map.get(&digit) {
            Some(&letters) => {
                //let len = letters.len();
                for &letter in letters {
                    combination.push(letter);
                    backtrace(combinations, nums_map, digits, index + 1, combination);
                    combination.remove(index);
                }
            }
            None => {
                panic!("digits is invalid.");
            }
        }
    }
}

fn backtrace_v2(
    combinations: &mut Vec<String>,
    nums_map: &HashMap<char, Vec<char>>,
    digits: &[char],
    index: usize,
    combination: &mut Vec<char>,
) {
    if index == digits.len() {
        let abc = combination.iter().collect();
        combinations.push(abc);
    } else {
        let digit = digits[index];

        match nums_map.get(&digit) {
            Some(letters) => {
                //let len = letters.len();
                for &letter in letters {
                    combination.push(letter);
                    backtrace_v2(combinations, nums_map, digits, index + 1, combination);
                    combination.remove(index);
                }
            }
            None => {
                panic!("digits is invalid.");
            }
        }
    }
}

/// 力扣（15. 三数之和） https://leetcode-cn.com/problems/3sum/
/// 方法1：排序 + 双指针
pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut result = Vec::<Vec<i32>>::new();
    let len = nums.len();
    let mut new_nums = nums;
    new_nums.sort_unstable();
    // 枚举 a
    for (first, &a) in new_nums.iter().enumerate() {
        // 需要和上一次枚举的数不相同
        if first > 0 && a == new_nums[first - 1] {
            continue;
        }
        let mut third = len - 1;
        let target = -a;
        let mut second = first + 1;
        while second < len {
            // 需要和上一次枚举的数不相同
            if second > first + 1 && new_nums[second] == new_nums[second - 1] {
                second += 1;
                continue;
            }

            // 需要保证 b 的指针在 c 的指针的左侧
            while second < third && new_nums[second] + new_nums[third] > target {
                third -= 1;
            }

            // 如果指针重合，随着 b 后续的增加
            // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
            if second == third {
                break;
            }

            if new_nums[second] + new_nums[third] == target {
                result.push(vec![a, new_nums[second], new_nums[third]]);
            }

            second += 1;
        }
    }

    result
}

/// 力扣（16. 最接近的三数之和） https://leetcode-cn.com/problems/3sum-closest/
/// 方法1：排序 + 双指针
pub fn three_sum_closest(nums: Vec<i32>, target: i32) -> i32 {
    let len = nums.len();
    let mut new_nums = nums;
    new_nums.sort_unstable();
    // -10^4 <= target <= 10^4
    let mut best = 10000;
    // 枚举 a
    for (first, &a) in new_nums.iter().enumerate() {
        // 需要和上一次枚举的数不相同
        if first > 0 && a == new_nums[first - 1] {
            continue;
        }
        let mut second = first + 1;
        let mut third = len - 1;
        while second < third {
            let sum = a + new_nums[second] + new_nums[third];
            if sum == target {
                return target;
            }

            if (sum - target).abs() < (best - target).abs() {
                best = sum;
            }

            if sum > target {
                let mut third0 = third - 1;
                while second < third0 && new_nums[third0] == new_nums[third] {
                    third0 -= 1;
                }

                third = third0;
            } else {
                let mut second0 = second + 1;
                while second0 < third && new_nums[second0] == new_nums[second] {
                    second0 += 1;
                }
                second = second0;
            }
        }
    }

    best
}

const PHONE_LETTER: [(char, &[char]); 8] = [
    ('2', &['a', 'b', 'c']),
    ('3', &['d', 'e', 'f']),
    ('4', &['g', 'h', 'i']),
    ('5', &['j', 'k', 'l']),
    ('6', &['m', 'n', 'o']),
    ('7', &['p', 'q', 'r', 's']),
    ('8', &['t', 'u', 'v']),
    ('9', &['w', 'x', 'y', 'z']),
];
/// 力扣（17. 电话号码的字母组合） https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/
/// 回溯法（递归+深度优先）
pub fn letter_combinations(digits: String) -> Vec<String> {
    let mut combinations = vec![];
    let len = digits.len();
    if len == 0 {
        return combinations;
    }
    let char_map = PHONE_LETTER
        .iter()
        .map(|(ch, letter)| (*ch, *(letter)))
        .collect();
    let digits_chars = digits.chars().collect::<Vec<char>>();
    let mut combination = vec![];
    backtrace(
        &mut combinations,
        &char_map,
        &digits_chars,
        0,
        &mut combination,
    );
    combinations
}

/// 力扣（17. 电话号码的字母组合） https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/
/// 回溯法（递归+深度优先）
pub fn letter_combinations_v2(digits: String) -> Vec<String> {
    let mut combinations = vec![];
    let len = digits.len();
    if len == 0 {
        return combinations;
    }
    let mut char_map = HashMap::<char, Vec<char>>::new();

    char_map.insert('2', vec!['a', 'b', 'c']);
    char_map.insert('3', vec!['d', 'e', 'f']);
    char_map.insert('4', vec!['g', 'h', 'i']);
    char_map.insert('5', vec!['j', 'k', 'l']);
    char_map.insert('6', vec!['m', 'n', 'o']);
    char_map.insert('7', vec!['p', 'q', 'r', 's']);
    char_map.insert('8', vec!['t', 'u', 'v']);
    char_map.insert('9', vec!['w', 'x', 'y', 'z']);

    let digits_chars = digits.chars().collect::<Vec<char>>();
    let mut combination = vec![];
    backtrace_v2(
        &mut combinations,
        &char_map,
        &digits_chars,
        0,
        &mut combination,
    );
    combinations
}

/// 力扣（18. 四数之和) https://leetcode-cn.com/problems/4sum/
/// 方法1：排序 + 双指针
pub fn four_sum(nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    use std::cmp::Ordering;

    let mut result = Vec::<Vec<i32>>::new();
    let len = nums.len();

    if len < 4 {
        return result;
    }
    let mut new_nums = nums;
    new_nums.sort_unstable();

    // 枚举 a
    for (first, &a) in new_nums.iter().take(len - 3).enumerate() {
        // 需要和上一次枚举的数不相同
        if first > 0 && a == new_nums[first - 1] {
            continue;
        }
        let min_fours = a + new_nums[first + 1] + new_nums[first + 2] + new_nums[first + 3];
        if min_fours > target {
            break;
        }
        let max_fours = a + new_nums[len - 3] + new_nums[len - 2] + new_nums[len - 1];
        if max_fours < target {
            continue;
        }

        let mut second = first + 1;

        while second < len - 2 {
            if second > first + 1 && new_nums[second] == new_nums[second - 1] {
                second += 1;
                continue;
            }

            if a + new_nums[second] + new_nums[second + 1] + new_nums[second + 2] > target {
                break;
            }

            if a + new_nums[second] + new_nums[len - 2] + new_nums[len - 1] < target {
                second += 1;
                continue;
            }
            let mut third = second + 1;
            let mut fourth = len - 1;
            while third < fourth {
                let sum = a + new_nums[second] + new_nums[third] + new_nums[fourth];

                match sum.cmp(&target) {
                    Ordering::Equal => {
                        result.push(vec![a, new_nums[second], new_nums[third], new_nums[fourth]]);
                        // 相等的情況下，不能break;還需要继续遍历
                        while third < fourth && new_nums[third + 1] == new_nums[third] {
                            third += 1;
                        }
                        third += 1;
                        while third < fourth && new_nums[fourth - 1] == new_nums[fourth] {
                            fourth -= 1
                        }
                        fourth -= 1;
                    }
                    Ordering::Greater => {
                        fourth -= 1;
                    }
                    Ordering::Less => {
                        third += 1;
                    }
                }
            }

            second += 1;
        }
    }

    result
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

/// 31. 下一个排列 https://leetcode-cn.com/problems/next-permutation/
pub fn next_permutation(nums: &mut Vec<i32>) {
    let n = nums.len();
    let mut i = n - 1;
    while i > 0 && nums[i - 1] >= nums[i] {
        i -= 1;
    }
    if i > 0 {
        let mut j = n - 1;
        while nums[i - 1] >= nums[j] {
            j -= 1;
        }
        // 较小数nums[i-i]与较大数nums[j]交换位置
        nums.swap(i - 1, j);
    }
    nums[i..].reverse();
}

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

/// 力扣（38. 外观数列） https://leetcode-cn.com/problems/count-and-say/
pub fn count_and_say(n: i32) -> String {
    let mut s = "1".to_string();
    for _ in 0..n - 1 {
        let mut ret = "".to_string();
        let mut count = 0;
        // use peekable to check next char
        let mut it = s.chars().peekable();
        while let Some(c) = it.next() {
            match it.peek() {
                Some(next) if next == &c => count += 1,
                _ => {
                    ret.push_str(&(count + 1).to_string());
                    ret.push(c);
                    count = 0;
                }
            }
        }
        s = ret;
    }
    s
}

/// 39. 组合总和  https://leetcode-cn.com/problems/combination-sum/
pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    let mut res: Vec<Vec<i32>> = Vec::with_capacity(150);
    let mut v: Vec<i32> = Vec::with_capacity(150);
    // println!("\ncandidates: {:?} target: {}", candidates, target);
    combination_sum_backtrace(&candidates, 0, target, &mut res, &mut v);
    res
}

fn combination_sum_backtrace(
    candidates: &[i32],
    i: usize,
    target: i32,
    res: &mut Vec<Vec<i32>>,
    v: &mut Vec<i32>,
) {
    if i == candidates.len() {
        return;
    }
    if target == 0 {
        res.push(v.clone());
        return;
    }
    combination_sum_backtrace(candidates, i + 1, target, res, v);
    let d = candidates[i];
    if target >= d {
        v.push(d);

        combination_sum_backtrace(candidates, i, target - d, res, v);
        v.pop();
    }
}

/// 力扣（49. 字母异位词分组） https://leetcode-cn.com/problems/group-anagrams/
pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
    let mut result: Vec<Vec<String>> = vec![];
    use std::collections::HashMap;
    let mut anagrams_map = HashMap::<String, Vec<String>>::new();
    for str in strs {
        // 字母异位词拥有相同的签名（所有字符排序后得到的字符串）
        let sign = {
            if str.is_empty() {
                "".to_string()
            } else {
                let mut chars: Vec<char> = str.chars().collect();
                chars.sort_unstable();
                chars.iter().collect()
            }
        };
        match anagrams_map.get_mut(&sign) {
            Some(anagrams) => {
                anagrams.push(str);
            }
            None => {
                anagrams_map.insert(sign, vec![str]);
            }
        }
    }

    for value in anagrams_map.values() {
        let mut anagrams = Vec::with_capacity(value.len());
        for v in value {
            anagrams.push(v.to_owned());
        }
        result.push(anagrams);
    }

    result
}

/// 力扣（46. 全排列） https://leetcode-cn.com/problems/permutations/
pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let len = nums.len();
    let mut result: Vec<Vec<i32>> = Vec::new();
    if len == 0 {
        return result;
    }
    let mut used_vec = vec![false; len];
    let mut path = Vec::<i32>::new();
    dfs(&nums, len, 0, &mut path, &mut used_vec, &mut result);
    result
}

// use std::collections::VecDeque;
fn dfs(
    nums: &Vec<i32>,
    len: usize,
    dept: usize,
    path: &mut Vec<i32>,
    used_vec: &mut Vec<bool>,
    result: &mut Vec<Vec<i32>>,
) {
    if dept == len {
        let full_path: Vec<i32> = path.into_iter().map(|&mut num| num).collect();
        result.push(full_path);
        return;
    }

    for i in 0..len {
        if used_vec[i] {
            continue;
        }
        path.push(nums[i]);
        used_vec[i] = true;
        dfs(&nums, len, dept + 1, path, used_vec, result);
        path.pop();
        used_vec[i] = false;
    }
}

/// 力扣（54. 螺旋矩阵） https://leetcode-cn.com/problems/spiral-matrix/
pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let len = matrix.len();
    if len == 0 {
        return vec![];
    }
    let row_len = matrix[0].len();

    let mut result = Vec::<i32>::with_capacity(len * row_len);

    let mut row = 0;
    let mut col = 0;
    let mut x = len - 1; //i的最大值
    let mut y = row_len - 1; //j的最大值
    let mut row_s = 0; //i的最小值
    let mut col_t = 0; //j的最小值
    let mut direct = 0;

    let mut push_times = 1;
    result.push(matrix[0][0]);

    while push_times < len * row_len {
        match direct % 4 {
            0 => {
                //右
                if col < y {
                    col += 1;
                    result.push(matrix[row][col]);
                    push_times += 1;
                    continue;
                } else {
                    row_s += 1;
                    direct += 1;
                }
            }
            1 => {
                //下
                if row < x {
                    row += 1;
                    result.push(matrix[row][col]);
                    push_times += 1;
                    continue;
                } else {
                    y -= 1;
                    direct += 1;
                }
            }
            2 => {
                //左
                if col > col_t {
                    col -= 1;
                    result.push(matrix[row][col]);
                    push_times += 1;
                    continue;
                } else {
                    x -= 1;
                    direct += 1;
                }
            }
            3 => {
                //上
                if row > row_s {
                    row -= 1;
                    result.push(matrix[row][col]);
                    push_times += 1;
                    continue;
                } else {
                    col_t += 1;
                    direct += 1;
                }
            }
            _ => {
                unreachable!();
            }
        }
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

/// 力扣（189. 旋转数组） https://leetcode-cn.com/problems/rotate-array/
pub fn rotate(nums: &mut Vec<i32>, k: i32) {
    let len = nums.len();
    if len <= 1 {
        return;
    }
    let offset = (k as usize) % len;
    if offset == 0 {
        return;
    }

    //三次翻转
    nums.reverse();

    for i in 0..offset / 2 {
        nums.swap(i, offset - i - 1);
    }

    for j in 0..(len - offset) / 2 {
        nums.swap(j + offset, len - j - 1);
    }
}

///  力扣（209. 长度最小的子数组） https://leetcode-cn.com/problems/minimum-size-subarray-sum/
pub fn min_sub_array_len(s: i32, nums: Vec<i32>) -> i32 {
    let mut k = 0;
    let mut i = 0usize;
    let mut j = 0usize;
    let len = nums.len();
    let mut sum = if len >= 1 { nums[0] } else { 0 };

    while i <= j && j < len {
        // 当 sum>=s 时，i++
        if sum >= s {
            if k == 0 {
                k = j - i + 1;
            } else {
                let temp = j - i + 1;
                if temp < k {
                    k = temp;
                }
            }
            sum -= nums[i];
            i += 1;
        } else {
            // 当 sum<s 时，j++
            j += 1;
            if j < len {
                sum += nums[j];
            } else {
                break;
            }
        }
    }

    k as i32
}

/// 力扣（209. 长度最小的子数组)
/// 滑动窗口
pub fn min_sub_array_len_v2(target: i32, nums: Vec<i32>) -> i32 {
    use std::cmp::min;
    let mut result = i32::MAX;
    let mut sum = 0;
    let mut i = 0;
    let mut sub_length = 0;
    let len = nums.len();
    for j in 0..len {
        sum += nums[j];
        while sum >= target {
            sub_length = (j - i + 1);
            result = min(result, sub_length as i32);
            sum -= nums[i];
            i += 1;
        }
    }
    if result == i32::MAX {
        0
    } else {
        result
    }
}

/// 力扣（210. 课程表II），https://leetcode-cn.com/problems/course-schedule-ii/
pub fn find_order(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::VecDeque;

    let u_num_courses = num_courses as usize;

    // 存储有向图
    let mut edges = vec![vec![]; u_num_courses];
    // 存储答案
    let mut result = vec![0; u_num_courses];

    // 答案下标
    let mut index = 0usize;

    // 存储每个节点（课程）的入度，indeg[1]即为其他课程选修前必须选修课程1的总次数
    let mut indeg = vec![0; u_num_courses];

    for info in prerequisites {
        edges[info[1] as usize].push(info[0]);
        indeg[info[0] as usize] += 1;
    }

    let mut queue = VecDeque::<i32>::new();
    // 将所有入度为 0 的节点放入队列中
    for (course, item) in indeg.iter().enumerate().take(u_num_courses) {
        if *item == 0 {
            queue.push_back(course as i32);
        }
    }

    while !queue.is_empty() {
        if let Some(u) = queue.pop_back() {
            result[index] = u;
            index += 1;

            if let Some(edge_vec) = edges.get(u as usize) {
                for v in edge_vec {
                    indeg[*v as usize] -= 1;
                    if indeg[*v as usize] == 0 {
                        queue.push_back(*v);
                    }
                }
            }
        }
    }

    if index != u_num_courses {
        vec![]
    } else {
        result
    }
}

/// 力扣（229. 求众数 II） https://leetcode-cn.com/problems/majority-element-ii/
pub fn majority_element(nums: Vec<i32>) -> Vec<i32> {
    let mut count1 = 0;
    let mut candidate1 = nums[0];
    let mut count2 = 0;
    let mut candidate2 = nums[0];

    // 摩尔投票法，分为两个阶段：配对阶段和计数阶段
    // 配对阶段（找出两个候选人）
    for num in &nums {
        if *num == candidate1 {
            count1 += 1;
        } else if *num == candidate2 {
            count2 += 1;
        } else if count1 == 0 {
            candidate1 = *num;
            count1 = 1;
        } else if count2 == 0 {
            candidate2 = *num;
            count2 = 1;
        } else {
            count1 -= 1;
            count2 -= 1;
        }
    }

    // 计数阶段（重新计票为检查候选人是否符合条件）
    let mut count1 = 0;
    let mut count2 = 0;
    for num in &nums {
        if *num == candidate1 {
            count1 += 1;
        } else if *num == candidate2 {
            count2 += 1;
        }
    }

    let mut result = vec![];
    let condition = nums.len() / 3;
    if count1 > condition {
        result.push(candidate1);
    }
    if count2 > condition {
        result.push(candidate2);
    }

    result
}

/// 力扣（287. 寻找重复数） https://leetcode-cn.com/problems/find-the-duplicate-number/
/// 方法1：快慢指针法（龟兔赛跑法）
pub fn find_duplicate(nums: Vec<i32>) -> i32 {
    let mut slow = 0;
    let mut fast = 0;
    let new_nums = nums.iter().map(|n| *n as usize).collect::<Vec<_>>();

    slow = new_nums[slow];
    fast = new_nums[new_nums[fast]];
    while slow != fast {
        slow = new_nums[slow];
        fast = new_nums[new_nums[fast]];
    }

    slow = 0usize;
    while slow != fast {
        slow = new_nums[slow];
        fast = new_nums[fast];
    }

    slow as i32
}

/// 力扣（287. 寻找重复数）
/// 方法1：快慢指针法（龟兔赛跑法）V2
pub fn find_duplicate_v2(nums: Vec<i32>) -> i32 {
    let (mut slow, mut fast) = (nums[0], nums[nums[0] as usize]);
    while slow != fast {
        slow = nums[slow as usize];
        fast = nums[nums[fast as usize] as usize];
    }
    slow = 0;
    while slow != fast {
        slow = nums[slow as usize];
        fast = nums[fast as usize];
    }
    slow
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

// TODO 500

// TODO 600
/// 1208. 尽可能使字符串相等 https://leetcode-cn.com/problems/get-equal-substrings-within-budget/
pub fn equal_substring(s: String, t: String, max_cost: i32) -> i32 {
    use std::cmp::max;
    let (mut left, mut right, mut cost, mut result) = (0, 0, 0, 0);
    let len = s.len();
    let s_bytes = s.as_bytes();
    let t_bytes = t.as_bytes();
    while right < len {
        cost += (s_bytes[right] as i32 - t_bytes[right] as i32).abs();
        right += 1;
        while cost > max_cost {
            cost -= (s_bytes[left] as i32 - t_bytes[left] as i32).abs();
            left += 1;
        }
        result = max(result, right - left);
    }
    result as i32
}

/// 1208. 尽可能使字符串相等
pub fn equal_substring_v2(s: String, t: String, max_cost: i32) -> i32 {
    use std::cmp::max;
    let (mut left, mut right, mut cost, mut result) = (0, 0, 0, 0);
    let len = s.len();

    let s_bytes = s.as_bytes();
    let t_bytes = t.as_bytes();
    let mut diff: Vec<i32> = vec![0; len];
    for idx in 0..len {
        diff[idx] = (s_bytes[idx] as i32 - t_bytes[idx] as i32).abs();
    }
    while right < len {
        cost += diff[right];
        right += 1;
        while cost > max_cost {
            cost -= diff[left];
            left += 1;
        }
        result = max(result, right - left);
    }
    result as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn medium() {
        use super::*;
        let s = count_and_say(6);
        dbg!(s);

        let mut matrix = Vec::<Vec<i32>>::new();
        matrix.push(vec![1, 2, 3, 4]);
        matrix.push(vec![5, 6, 7, 8]);
        matrix.push(vec![9, 10, 11, 12]);

        println!("{:?}", matrix);
        //    dbg!("{:?}",find_diagonal_order(matrix));

        dbg!("spiral_order: {:?}", spiral_order(matrix));

        let s = 7;
        let nums = vec![1, 2, 3, 4, 3, 7, 2, 2];
        let min_len = min_sub_array_len(s, nums);
        dbg!("min_len:{}", min_len);

        let target = 7;
        let nums = vec![1, 2, 3, 4, 3, 7, 2, 2];

        let min_len = min_sub_array_len_v2(target, nums);
        dbg!("min_len:{}", min_len);

        let mut rotate_vec = vec![1, 2];
        rotate(&mut rotate_vec, 1);

        dbg!("{:?}", rotate_vec);

        let mut prerequisites = Vec::<Vec<i32>>::new();
        prerequisites.push(vec![1, 0]);
        prerequisites.push(vec![2, 0]);
        prerequisites.push(vec![3, 1]);
        prerequisites.push(vec![3, 2]);

        let result = find_order(4, prerequisites);
        dbg!("result-> : {:?}", result);

        let result = find_order(2, vec![]);
        dbg!("{:?}", result);

        dbg!(longest_palindrome("babad".to_string()));
        // Fixed dbg!("{}",longest_palindrome_v2("babad".to_string()));

        let heights = vec![1, 8, 6, 2, 5, 4, 8, 3, 7];
        dbg!("max area : {}", max_area(heights));

        let heights = vec![4, 3, 2, 1, 4];
        dbg!("max area : {}", max_area(heights));

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

        let nums = vec![3, 2];
        let result = majority_element(nums);
        dbg!("majority_element: {:?}", result);

        let digits = String::from("234");
        let combination = letter_combinations(digits);
        dbg!("combination1: {:?}", combination);

        let digits = String::from("234");
        let combination = letter_combinations_v2(digits);
        dbg!("combination2: {:?}", combination);

        let mut nums = vec![4, 5, 2, 6, 3, 1];
        next_permutation(&mut nums);
        println!("nums: {:?}", nums);
    }

    #[test]
    fn medium2() {
        let strs = vec![
            "eat".to_string(),
            "tea".to_string(),
            "tan".to_string(),
            "ate".to_string(),
            "nat".to_string(),
            "bat".to_string(),
        ];
        let anagrams = group_anagrams(strs);
        dbg!("anagrams: {:?}", anagrams);

        let nums = vec![-1, 0, 1, 2, -1, -4];
        let three_sum_result = three_sum(nums);
        dbg!(three_sum_result);

        let nums = vec![-1, 2, 1, -4];
        let target = 1;
        let three_sum_closest_result = three_sum_closest(nums, target);
        dbg!(three_sum_closest_result);

        let nums = vec![-3, -2, -1, 0, 0, 1, 2, 3];
        let target = 0;

        let four_sum_result = four_sum(nums, target);
        dbg!("{:?}", four_sum_result);

        dbg!(permute(vec![1, 2, 3]));
    }

    fn test_equal_substring() {
        dbg!(equal_substring("abcd".to_string(), "bcdf".to_string(), 3));
    }
}
