//! 中等难度

use crate::simple::ListNode;
///  力扣（2. 两数相加） https://leetcode-cn.com/problems/add-two-numbers/
pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut l1 = l1;
    let mut l2 = l2;
    let mut head = None;

    // 进位
    let mut left = 0;

    loop {
        if l1.is_none() && l2.is_none() && left == 0 {
            break;
        }

        let mut sum = left;
        if let Some(n) = l1 {
            sum += n.val;
            l1 = n.next;
        }
        if let Some(n) = l2 {
            sum += n.val;
            l2 = n.next;
        }

        left = sum / 10;
        let mut node = ListNode::new(sum % 10);
        node.next = head;
        head = Some(Box::new(node));
    }

    // reverse the list
    // 8->0->7 => 7->0->8
    let mut tail = None;
    while let Some(mut n) = head.take() {
        head = n.next;
        n.next = tail;
        tail = Some(n);
    }
    tail
}

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

/// 力扣（6. Z 字形变换） https://leetcode-cn.com/problems/zigzag-conversion/
pub fn convert(s: String, num_rows: i32) -> String {
    if num_rows == 1 {
        return s;
    }

    let mut result_vec = vec![vec![]; num_rows as usize];
    let mut row = 0usize;
    let mut direct_down = true;
    for ch in s.chars() {
        if row == 0 {
            direct_down = true;
        } else if row == (num_rows - 1) as usize {
            direct_down = false;
        }

        if let Some(row_vec) = result_vec.get_mut(row) {
            row_vec.push(ch);
        }

        if direct_down {
            row += 1;
        } else {
            row -= 1;
        }
    }

    let mut result_str_vec = Vec::<char>::new();
    for row_vec in result_vec {
        result_str_vec.extend_from_slice(&row_vec);
    }

    result_str_vec.iter().collect()
}

enum State {
    Init,
    ExpectNumber, // 已经碰到了+或者-，下一个字符必须是数字
    Number(i32),
}

/// 力扣（8. 字符串转整数（atoi）） https://leetcode-cn.com/problems/string-to-integer-atoi/
pub fn my_atoi(s: String) -> i32 {
    let mut state = State::Init;
    let mut neg = 1;
    for c in s.chars() {
        match c {
            ' ' => match state {
                State::Init => {}
                State::ExpectNumber => return 0,
                State::Number(n) => return neg * n,
            },
            '+' | '-' => match state {
                State::Init => {
                    state = State::ExpectNumber;
                    if c == '-' {
                        neg = -1;
                    }
                }
                State::ExpectNumber => return 0,
                State::Number(n) => return neg * n,
            },
            '0'..='9' => {
                let digit = c.to_digit(10).unwrap() as i32;
                match state {
                    State::Init | State::ExpectNumber => state = State::Number(digit),
                    State::Number(n) => {
                        match n.checked_mul(10).and_then(|x| x.checked_add(digit)) {
                            Some(number) => state = State::Number(number),
                            _ => {
                                return if neg < 0 {
                                    std::i32::MIN
                                } else {
                                    std::i32::MAX
                                }
                            }
                        }
                    }
                }
            }
            _ => match state {
                State::Init | State::ExpectNumber => return 0,
                State::Number(n) => return neg * n,
            },
        }
    }

    match state {
        State::Number(n) => neg * n,
        _ => 0,
    }
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

/// 力扣（12. 整数转罗马数字）  https://leetcode-cn.com/problems/integer-to-roman/
/// 贪心算法
pub fn int_to_roman(num: i32) -> String {
    let arr = vec![
        (1, "I"),
        (4, "IV"),
        (5, "V"),
        (9, "IX"),
        (10, "X"),
        (40, "XL"),
        (50, "L"),
        (90, "XC"),
        (100, "C"),
        (400, "CD"),
        (500, "D"),
        (900, "CM"),
        (1000, "M"),
    ];

    fn find(n: i32, arr: &[(i32, &'static str)]) -> (i32, &'static str) {
        for (value, s) in arr.iter().rev() {
            if n >= *value {
                return (*value, *s);
            }
        }
        unreachable!()
    }

    // 上次用除法，这次用减法
    let mut ret = "".to_string();
    let mut num = num;
    while num > 0 {
        let (v, s) = find(num, &arr);
        ret.push_str(s);
        num -= v;
    }

    ret
}

/// 力扣（12. 整数转罗马数字）
pub fn int_to_roman_v2(num: i32) -> String {
    let arr = vec![
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];

    fn find(n: i32, arr: &[(i32, &'static str)]) -> (i32, &'static str) {
        for (value, s) in arr {
            if n >= *value {
                return (*value, *s);
            }
        }
        unreachable!()
    }

    // 上次用除法，这次用减法
    let mut ret = "".to_string();
    let mut num = num;
    while num > 0 {
        let (v, s) = find(num, &arr);
        ret.push_str(s);
        num -= v;
    }

    ret
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

/// 力扣(19. 删除链表的倒数第 N 个结点) https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let mut dummy_head = Some(Box::new(ListNode {
        val: 0i32,
        next: head,
    }));
    //链表长度（不包括虚拟节点的长度）
    let mut len = 0;
    {
        let mut current = &dummy_head.as_ref().unwrap().next;
        while current.is_some() {
            len += 1;
            current = &(current.as_ref().unwrap().next);
        }
    }
    {
        let steps = len - n;
        let mut current = dummy_head.as_mut();
        for _ in 0..steps {
            current = current.unwrap().next.as_mut();
        }
        // current 被删除元素的前一个元素， next 为 被删除元素的后一个元素
        let next = current
            .as_mut()
            .unwrap()
            .next
            .as_mut()
            .unwrap()
            .next
            .clone();
        current.as_mut().unwrap().next = next;
    }
    dummy_head.unwrap().next
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

/// 33. 搜索旋转排序数组 https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
/// 方法一：二分查找
pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    let len = nums.len();
    if len == 0 {
        return -1;
    }
    if len == 1 {
        if nums[0] == target {
            return 0;
        } else {
            return -1;
        }
    }

    let (mut left, mut right) = (0, len - 1);
    while left <= right {
        let mut middle = (left + right) / 2;
        if nums[middle] == target {
            return middle as i32;
        }
        if nums[0] <= nums[middle] {
            if nums[0] <= target && target < nums[middle] {
                right = middle - 1;
            } else {
                left = middle + 1;
            }
        } else {
            if nums[middle] < target && target <= nums[right] {
                left = middle + 1;
            } else {
                right = middle - 1;
            }
        }
    }

    -1
}

/// 力扣（34. 在排序数组中查找元素的第一个和最后一个位置) https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
/// 先用二分查找算法找到target的下标，然后向左右两边继续查找
pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
    use std::cmp::Ordering;
    let mut range = vec![-1, -1];
    let mut left = 0;
    let mut right = nums.len();
    while left < right {
        let mut middle = (left + right) / 2;
        match nums[middle].cmp(&target) {
            Ordering::Greater => {
                right = middle;
            }
            Ordering::Less => {
                left = middle + 1;
            }
            Ordering::Equal => {
                // 找到target的第一个位置后则向左右两边拓展查找
                range[0] = middle as i32;
                range[1] = middle as i32;
                let mut l = middle;
                let mut r = middle;
                while r < right - 1 {
                    if nums[r + 1] == target {
                        r += 1;
                    } else {
                        break;
                    }
                }

                while l > 0 {
                    if nums[l - 1] == target {
                        l -= 1;
                    } else {
                        break;
                    }
                }

                range[0] = l as i32;
                range[1] = r as i32;
                break;
            }
        }
    }

    range
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
/// 力扣（50. Pow(x, n)） https://leetcode-cn.com/problems/powx-n/
pub fn my_pow(x: f64, n: i32) -> f64 {
    x.powi(n)
}

/// 力扣（50. Pow(x, n)）
/// 方法一：快速幂 + 递归
pub fn my_pow_v2(x: f64, n: i32) -> f64 {
    if n >= 0 {
        quick_mul(x, n)
    } else {
        1.0 / quick_mul(x, -n)
    }
}

/// 快速幂
fn quick_mul(x: f64, n: i32) -> f64 {
    if n == 0 {
        return 1.0;
    }

    let y = quick_mul(x, n / 2);
    if n % 2 == 0 {
        y * y
    } else {
        y * y * x
    }
}

fn quick_mul_v2(x: f64, n: i32) -> f64 {
    let mut ans = 1.0f64;
    let mut x_contribute = x;
    let mut n_mut = n;
    while n_mut > 0 {
        if n_mut % 2 == 1 {
            ans *= x_contribute;
        }
        x_contribute *= x_contribute;
        n_mut /= 2;
    }
    ans
}

/// 力扣（50. Pow(x, n)）
/// 方法二：快速幂 + 迭代 (该方法容易造成溢出不可取)
pub fn my_pow_v3(x: f64, n: i32) -> f64 {
    if n >= 0 {
        quick_mul_v2(x, n)
    } else {
        1.0f64 / quick_mul_v2(x, -n)
    }
}

/// 力扣（50. Pow(x, n)）
pub fn my_pow_v4(x: f64, n: i32) -> f64 {
    let mut x = if n > 0 { x } else { 1f64 / x };
    let mut n = n;
    let mut r = 1f64;
    while n != 0 {
        if n & 1 == 1 {
            r *= x;
        }
        x *= x;
        n /= 2;
    }
    r
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

/// 力扣（137. 只出现一次的数字 II） https://leetcode-cn.com/problems/single-number-ii/
/// 方法1：哈希表
pub fn single_number(nums: Vec<i32>) -> i32 {
    use std::collections::HashMap;
    let mut counts_map = HashMap::<i32, i32>::new();
    for num in nums {
        match counts_map.get_mut(&num) {
            Some(count) => {
                *count += 1;
            }
            None => {
                counts_map.insert(num, 1);
            }
        }
    }

    for (key, value) in counts_map {
        if value == 1 {
            return key;
        }
    }
    -1
}

/// 力扣（137. 只出现一次的数字 II）
/// 方法2 ：依次确定每一个二进制位
pub fn single_number_v2(nums: Vec<i32>) -> i32 {
    let mut answer = 0;
    for i in 0..32 {
        let mut total = 0;
        for &num in &nums {
            total += ((num >> i) & 1);
        }
        if total % 3 != 0 {
            answer |= (1 << i);
        }
    }

    answer
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

/// 力扣（201. 数字范围按位与） https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/
/// 我们可以将问题重新表述为：给定两个整数，我们要找到它们对应的二进制字符串的公共前缀。
/// 方法1：位移
pub fn range_bitwise_and(left: i32, right: i32) -> i32 {
    let mut left = left;
    let mut right = right;
    let mut shift = 0;
    while left < right {
        left >>= 1;
        right >>= 1;
        shift += 1;
    }
    left << shift
}

/// 方法2：Brian Kernighan 算法
pub fn range_bitwise_and_v2(left: i32, right: i32) -> i32 {
    let mut right = right;
    while left < right {
        right &= (right - 1);
    }
    right
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

/// 力扣（215. 数组中的第K个最大元素） https://leetcode-cn.com/problems/kth-largest-element-in-an-array/
pub fn find_kth_largest(nums: Vec<i32>, k: i32) -> i32 {
    let mut heap = BinaryHeap::from(nums);
    for _ in 1..k {
        heap.pop();
    }
    heap.pop().unwrap()
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

/// 力扣（260. 只出现一次的数字 III） https://leetcode-cn.com/problems/single-number-iii/
/// 方法1：分组异或
pub fn single_number_260(nums: Vec<i32>) -> Vec<i32> {
    // ret 为 a,b两个数异或的结果
    let mut ret = 0;
    for &num in &nums {
        ret ^= num;
    }
    // div 为 a,b 二进制位上不相同时，最低的位
    let mut div = 1;
    while div & ret == 0 {
        div <<= 1;
    }
    let (mut a, mut b) = (0, 0);
    for &num in &nums {
        if div & num != 0 {
            a ^= num;
        } else {
            b ^= num;
        }
    }
    vec![a, b]
}

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
static UGLY_NUMBER_FACTORS: [i64; 3] = [2, 3, 5];
/// 力扣（264. 丑数 II） https://leetcode-cn.com/problems/ugly-number-ii/
/// 1 通常被视为丑数。
/// 方法一：最小堆
pub fn nth_ugly_number(n: i32) -> i32 {
    //let factors = vec![2, 3, 5];
    let mut seen = HashSet::new();
    let mut heap = BinaryHeap::new();
    seen.insert(1i64);
    heap.push(Reverse(1i64));
    let mut ugly = 0;
    for _ in 0..n {
        if let Some(Reverse(curr)) = heap.pop() {
            ugly = curr;
            for factor in &UGLY_NUMBER_FACTORS {
                let next: i64 = curr * (*factor);
                if seen.insert(next) {
                    heap.push(Reverse(next));
                }
            }
        };
    }
    ugly as i32
}

use std::cmp::min;
/// 力扣（264. 丑数 II）
/// 方法二：动态规划
pub fn nth_ugly_number_v2(n: i32) -> i32 {
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

/// 力扣（468. 验证IP地址）  https://leetcode-cn.com/problems/validate-ip-address/
/// 使用标准库中的方法
use std::net::IpAddr;
pub fn valid_ip_address(ip: String) -> String {
    match ip.parse::<IpAddr>() {
        Ok(IpAddr::V4(x)) => {
            let array: Vec<Vec<char>> = ip.split('.').map(|x| x.chars().collect()).collect();
            for item in &array {
                if (item[0] == '0' && item.len() > 1) {
                    return String::from("Neither");
                }
            }
            String::from("IPv4")
        }
        Ok(IpAddr::V6(_)) => {
            let array: Vec<Vec<char>> = ip.split(':').map(|x| x.chars().collect()).collect();
            for item in array {
                if item.is_empty() {
                    return String::from("Neither");
                }
            }
            String::from("IPv6")
        }
        _ => String::from("Neither"),
    }
}

/// 力扣（468. 验证IP地址）  https://leetcode-cn.com/problems/validate-ip-address/
/// 使用分治法解
pub fn valid_ip_address2(ip: String) -> String {
    if ip.chars().filter(|ch| *ch == '.').count() == 3 {
        //println!("valid_ipv4_address..");
        // return valid_ipv4_address(ip);
        valid_ipv4_address_v2(ip)
    } else if ip.chars().filter(|ch| *ch == ':').count() == 7 {
        //println!("valid_ipv6_address..");
        // return valid_ipv6_address(ip);
        valid_ipv6_address_v2(ip)
    } else {
        String::from("Neither")
    }
}

fn valid_ipv4_address(ip: String) -> String {
    let array: Vec<Vec<char>> = ip.split('.').map(|x| x.chars().collect()).collect();
    for item in array {
        //Validate integer in range (0, 255):
        //1. length of chunk is between 1 and 3
        if item.is_empty() || item.len() > 3 {
            return String::from("Neither");
        }
        //2. no extra leading zeros
        if (item[0] == '0' && item.len() > 1) {
            return String::from("Neither");
        }
        //3. only digits are allowed
        for ch in &item {
            if !ch.is_digit(10) {
                return String::from("Neither");
            }
        }
        //4. less than 255
        let num_str: String = item.iter().collect();
        let num = num_str.parse::<u16>().unwrap();
        if num > 255 {
            return String::from("Neither");
        }
    }
    "IPv4".to_string()
}

fn valid_ipv4_address_v2(ip: String) -> String {
    // let array: Vec<Vec<char>> = ip.split('.').map(|x| x.chars().collect()).collect();
    let array: Vec<&str> = ip.split('.').collect();
    for item in array {
        let len = item.len();
        //Validate integer in range (0, 255):
        //1. length of chunk is between 1 and 3
        if len == 0 || len > 3 {
            return String::from("Neither");
        }
        //2. no extra leading zeros
        let mut chars = item.chars().peekable();
        // let first_char = chars.peek();

        if let Some(first) = chars.peek() {
            if *first == '0' && len > 1 {
                return String::from("Neither");
            }

            if !(*first).is_digit(10) {
                return String::from("Neither");
            }
        }
        //3. only digits are allowed

        for ch in chars {
            if !(ch).is_digit(10) {
                return String::from("Neither");
            }
        }
        //4. less than 255
        // let num_str: String = array[i].iter().collect();
        let num = item.parse::<u16>().unwrap();
        if num > 255 {
            return String::from("Neither");
        }
    }
    "IPv4".to_string()
}

fn valid_ipv6_address(ip: String) -> String {
    let array: Vec<Vec<char>> = ip.split(':').map(|x| x.chars().collect()).collect();
    for item in &array {
        let num = item;
        if num.is_empty() || num.len() > 4 {
            return String::from("Neither");
        }
        //2.
        for ch in num {
            if !(*ch).is_digit(16) {
                return String::from("Neither");
            }
        }
    }
    String::from("IPv6")
}

fn valid_ipv6_address_v2(ip: String) -> String {
    // let array: Vec<Vec<char>> = ip.split(':').map(|x| x.chars().collect()).collect();
    let array: Vec<&str> = ip.split(':').collect();
    for item in array {
        let len = item.len();
        if len == 0 || len > 4 {
            return String::from("Neither");
        }
        //2.
        for ch in item.chars() {
            if !(ch).is_digit(16) {
                return String::from("Neither");
            }
        }
    }
    String::from("IPv6")
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

/// 力扣（707. 设计链表) https://leetcode-cn.com/problems/design-linked-list/
/**
 * Your MyLinkedList object will be instantiated and called as such:
 * let obj = MyLinkedList::new();
 * let ret_1: i32 = obj.get(index);
 * obj.add_at_head(val);
 * obj.add_at_tail(val);
 * obj.add_at_index(index, val);
 * obj.delete_at_index(index);
 */
pub struct MyLinkedList {
    len: usize,
    pub root: Option<Box<ListNode>>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl MyLinkedList {
    /** Initialize your data structure here. */
    fn new() -> Self {
        MyLinkedList {
            len: 0usize,
            root: None,
        }
    }

    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    fn get(&self, index: i32) -> i32 {
        if self.root.is_none() || index < 0 || index as usize >= self.len {
            -1
        } else {
            let mut current_index = 0;

            let mut current = &self.root;
            while current.is_some() {
                if current_index == index {
                    return current.as_ref().unwrap().val;
                }
                current_index += 1;
                current = &(current.as_ref().unwrap().next);
            }

            -1
        }
    }

    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    fn add_at_head(&mut self, val: i32) {
        if self.root.is_none() {
            self.root = Some(Box::new(ListNode::new(val)));
            self.len = 1;
        } else {
            let next = self.root.take();
            self.root = Some(Box::new(ListNode { val, next }));
            self.len += 1;
        }
    }

    /** Append a node of value val to the last element of the linked list. */
    fn add_at_tail(&mut self, val: i32) {
        if self.root.is_none() {
            self.root = Some(Box::new(ListNode::new(val)));
            self.len = 1;
        } else {
            let mut current = &mut self.root;
            while current.is_some() {
                let next = &current.as_ref().unwrap().next;
                //判断当前元素current是否为最后一个元素
                if next.is_none() {
                    if let Some(last) = current.as_deref_mut() {
                        last.next = Some(Box::new(ListNode::new(val)));
                        self.len += 1;
                        break;
                    }
                }

                current = &mut current.as_deref_mut().unwrap().next;
            }
        }
    }

    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    fn add_at_index(&mut self, index: i32, val: i32) {
        if index < 0 || index > self.len as i32 {
            eprintln!("invalid index :{}", index);
            return;
        }
        if index == 0 {
            self.add_at_head(val);
            return;
        } else if self.root.is_some() && index == self.len as i32 {
            self.add_at_tail(val);
            return;
        }
        let mut current_index = 0;
        let mut current = &mut self.root;
        while current.is_some() {
            if (current_index == index - 1) {
                // if let Some(cur) = current {
                //     if let Some(next) = cur.next.take().as_ref() {
                //         cur.next = Some(Box::new(ListNode {
                //             val,
                //             next: Some(Box::new(*next.clone())),
                //         }));
                //     }

                //     break;
                // }
                let next = current.as_mut().unwrap().next.take();

                let node = Some(Box::new(ListNode { val, next }));

                current.as_mut().unwrap().next = node;
                self.len += 1;
                break;
            }
            current = &mut current.as_deref_mut().unwrap().next;

            current_index += 1;
        }
    }

    /** Delete the index-th node in the linked list, if the index is valid. */
    fn delete_at_index(&mut self, index: i32) {
        if index < 0 || index >= self.len as i32 {
            eprintln!("invalid index :{}", index);
            return;
        }

        if index == 0 {
            let mut current = &mut self.root;
            if current.is_some() {
                *current = current.as_mut().unwrap().next.take();
                self.len -= 1;
            }
        }

        // 遍历到 index - 1的位置，则删除index位置的元素，并将index+1位置以后的元素添加到当前元素的后面
        let mut current_index = 0;
        let mut current = &mut self.root;
        while current.is_some() {
            if current_index == index - 1 {
                let next_next = current.as_mut().unwrap().next.as_mut().unwrap().next.take();
                current.as_mut().unwrap().next = next_next;
                self.len -= 1;
                break;
            }
            current_index += 1;
            current = &mut current.as_deref_mut().unwrap().next;
        }
    }
}
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
        let s = String::from("LEETCODEISHIRING");
        let zz = convert(s, 4);
        dbg!(zz);

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

        let ip = String::from("2001:0db8:85a3:0:0:8A2E:0370:7334");

        let ret = valid_ip_address2(ip);
        dbg!(ret);

        let mut prerequisites = Vec::<Vec<i32>>::new();
        prerequisites.push(vec![1, 0]);
        prerequisites.push(vec![2, 0]);
        prerequisites.push(vec![3, 1]);
        prerequisites.push(vec![3, 2]);

        let result = find_order(4, prerequisites);
        dbg!("result-> : {:?}", result);

        let result = find_order(2, vec![]);
        dbg!("{:?}", result);

        dbg!(my_atoi(" -423456".to_string()));
        dbg!(my_atoi("4193 with words".to_string()));
        dbg!(my_atoi("words and 987".to_string()));
        dbg!(my_atoi("-91283472332".to_string()));

        dbg!(longest_palindrome("babad".to_string()));
        // Fixed dbg!("{}",longest_palindrome_v2("babad".to_string()));

        let heights = vec![1, 8, 6, 2, 5, 4, 8, 3, 7];
        dbg!("max area : {}", max_area(heights));

        let heights = vec![4, 3, 2, 1, 4];
        dbg!("max area : {}", max_area(heights));

        dbg!("3999 to roman {}", int_to_roman(3999));
        dbg!("3999 to roman {}", int_to_roman_v2(3999));

        dbg!(my_pow(2.10000, 3));

        dbg!(my_pow_v2(2.00000, -4));
        dbg!(my_pow_v2(2.00000, 10));

        dbg!("i32 max :{},min :{}", std::i32::MAX, std::i32::MIN);
        // dbg!("{}",my_pow_v3(2.00000,-2147483648));
        // dbg!("{}",my_pow_v3(2.00000,-2147483647));

        dbg!(my_pow_v4(2.00000, -2147483648));
        dbg!(my_pow_v4(2.00000, -2147483647));
        dbg!(my_pow_v4(2.00000, 2147483647));

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

        dbg!("nth_ugly_number    {}", nth_ugly_number(1690));
        dbg!("nth_ugly_number_v2 {}", nth_ugly_number_v2(1690));

        let nums = vec![3, 2, 3, 1, 2, 4, 5, 5, 6];
        let kth_largest = find_kth_largest(nums, 4);
        dbg!("kth_largest    {}", kth_largest);

        let nums = vec![3, 2, 1, 5, 6, 4];
        let kth_largest = find_kth_largest(nums, 6);
        dbg!("kth_largest    {}", kth_largest);

        let nums = vec![8, 8, 8, 8, 8, 8];
        let range = search_range(nums, 7);
        dbg!("range {:?}", range);

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

        let nums = vec![6, i32::MIN, 6, 6, 7, 8, 7, 8, 8, 7];
        let single_number_v2_result = single_number_v2(nums);
        dbg!(single_number_v2_result);

        let nums = vec![1, 2, 1, 3, 2, 5];
        let single_number_260_result = single_number_260(nums);
        dbg!(single_number_260_result);

        dbg!(permute(vec![1, 2, 3]));
    }

    #[test]
    fn test_add_linked_list() {
        let l1 = ListNode {
            val: 2,
            next: Some(Box::new(ListNode {
                val: 4,
                next: Some(Box::new(ListNode::new(3))),
            })),
        };

        let l2 = ListNode {
            val: 5,
            next: Some(Box::new(ListNode {
                val: 6,
                next: Some(Box::new(ListNode::new(4))),
            })),
        };

        let result = add_two_numbers(Some(Box::new(l1)), Some(Box::new(l2)));
        dbg!("{:?}", result);
    }

    #[test]
    fn test_linked_list_() {
        //use MyLinkedList;
        let mut linked_list = MyLinkedList::new();
        linked_list.add_at_head(7);
        linked_list.add_at_head(6);
        linked_list.add_at_head(8);
        linked_list.add_at_tail(9);
        dbg!(linked_list.get(2));
        dbg!(linked_list.get(3));

        linked_list.add_at_index(2, 10);

        dbg!(linked_list.get(2));
        dbg!(linked_list.get(3));
        dbg!(linked_list.get(4));

        linked_list.delete_at_index(0);
        dbg!(linked_list.get(0));
        dbg!(linked_list.get(3));

        linked_list.delete_at_index(1);
        dbg!(linked_list.get(0));
        dbg!(linked_list.get(1));
        dbg!(linked_list.get(2));

        linked_list.delete_at_index(3);
    }

    /// ["MyLinkedList","addAtHead","addAtHead","addAtHead","addAtIndex","deleteAtIndex","addAtHead","addAtTail","get","addAtHead","addAtIndex","addAtHead"]
    /// [[],[7],[2],[1],[3,0],[2],[6],[4],[4],[4],[5,0],[6]]
    #[test]
    fn test_linked_list_fn_add_at_head() {
        //use MyLinkedList;
        let mut linked_list = MyLinkedList::new();
        linked_list.add_at_head(7);
        linked_list.add_at_head(2);
        linked_list.add_at_head(1);
        linked_list.add_at_index(3, 0);
        linked_list.delete_at_index(2);
        linked_list.add_at_head(6);
        linked_list.add_at_tail(4);
        let val = linked_list.get(4);
        dbg!(val);
        linked_list.add_at_head(4);
        linked_list.add_at_index(5, 0);
        linked_list.add_at_head(6);
    }

    //["MyLinkedList","addAtIndex","addAtIndex","addAtIndex","get"]
    //[[],[0,10],[0,20],[1,30],[0]]
    #[test]
    fn test_linked_list_fn_add_at_index() {
        //use MyLinkedList;
        let mut linked_list = MyLinkedList::new();
        linked_list.delete_at_index(0);
        linked_list.add_at_index(0, 10);
        linked_list.add_at_index(0, 20);
        linked_list.add_at_index(0, 30);
        dbg!(linked_list.get(0));

        linked_list.add_at_tail(40);
        linked_list.add_at_tail(50);
        linked_list.add_at_tail(60);
        dbg!(linked_list.get(4));
        dbg!(linked_list.get(5));
        dbg!(linked_list.get(6));

        linked_list.delete_at_index(3);
        dbg!(linked_list.get(3));
    }

    #[test]
    fn test_equal_substring() {
        dbg!(equal_substring("abcd".to_string(), "bcdf".to_string(), 3));
    }
}
