//！ 简单难度

use core::num;
use std::{
    borrow::Borrow,
    cmp::max,
    cmp::Ordering,
    collections::HashMap,
    ops::{BitAndAssign, BitOr, DerefMut},
    str::Chars,
};

lazy_static! {
    static ref VERSIONS: Vec<bool> = {
        let len: usize = 100;
        let mut versions = vec![false; len];
        let mut version = rand::random::<usize>() % len;
        while version >= len {
            version = rand::random::<usize>();
        }
        for item in versions.iter_mut().take(len).skip(version) {
            *item = true;
        }
        versions
    };
}

/// 力扣（1. 两数之和） https://leetcode-cn.com/problems/two-sum
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut nums_map = HashMap::<i32, i32>::new();
    for (idx, num) in nums.into_iter().enumerate() {
        let complement = target - num;

        let j = idx as i32;
        if let Some(idx) = nums_map.get(&complement) {
            return vec![*idx, j];
        }
        nums_map.insert(num, j);
    }
    vec![]
}

/// 力扣（3. 无重复的字符串的最长子串）https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/submissions/
pub fn length_of_longest_substring(s: String) -> i32 {
    if s.is_empty() {
        return 0;
    }

    let s: &[u8] = s.as_bytes();

    // 查找以第i个字符为起始的最长不重复的字符串，返回值：(不重复字符串长度，下一次查询的起始位置)
    fn get_len(i: usize, s: &[u8]) -> (i32, usize) {
        let mut len = 0;
        //字符 0-z（包含了数字、符号、空格） 对应的u8 范围为[48,122]，这里分配长度为128的数组绰绰有余
        // 例如：bits[48] 存储字符0出现的位置
        let mut bits = [0usize; 128]; // 用数组记录每个字符是否出现过
        let mut to = s.len() - 1;
        for (j, &item) in s.iter().enumerate().skip(i) {
            let index = item as usize;
            if bits[index] == 0 {
                bits[index] = j + 1;
                len += 1;
            } else {
                to = bits[index]; // 下一次开始搜索的位置，从与当前重复的字符的下一个字符开始
                break;
            }
        }
        (len, to)
    }

    let mut ret = 1;
    let mut i = 0;
    while i < s.len() - 1 {
        //println!("i={}", i);
        let (len, next) = get_len(i, &s);
        if len > ret {
            ret = len;
        }
        i = next;
    }

    ret
}

use std::i32::{MAX, MIN};
/// 力扣（7. 整数反转） ， https://leetcode-cn.com/problems/reverse-integer/
/// 关键是防止转换后的数据溢出（overflow）
pub fn reverse(x: i32) -> i32 {
    //MIN:-2147483648,MAX:2147483647
    println!("MIN:{},MAX:{}", MIN, MAX);
    let mut ret = 0;
    let mut y = x;
    while y != 0 {
        let pop = y % 10;
        y /= 10;
        if ret > MAX / 10 || (ret == MAX / 10 && pop > 7) {
            return 0;
        }

        if ret < MIN / 10 || (ret == MIN / 10 && pop < -8) {
            return 0;
        }
        ret = ret * 10 + pop;
    }
    ret
}

/// 力扣（7. 整数反转）
/// 参考： 吴翱翔 https://zhuanlan.zhihu.com/p/340649000
pub fn reverse2(x: i32) -> i32 {
    || -> Option<i32> {
        let mut ret = 0i32;
        let mut y = x;
        while y.abs() != 0 {
            ret = ret.checked_mul(10)?.checked_add(y % 10)?;
            y /= 10;
        }
        Some(ret)
    }()
    .unwrap_or(0)
}

/// rust slice pattern
fn is_palindrome_str(chars: &[char]) -> bool {
    match chars {
        [first, middle @ .., last] => first == last && is_palindrome_str(middle),
        [] | [_] => true,
    }
}

/// 力扣（9. 回文数） https://leetcode-cn.com/problems/palindrome-number/
/// 数字转字符串
pub fn is_palindrome(x: i32) -> bool {
    // 负数不是回文数
    if x < 0 {
        return false;
    }
    let s = x.to_string();
    let arr = s.as_bytes();
    let len = arr.len();
    for (c1, c2) in (0..len / 2)
        .into_iter()
        .zip((len / 2..len).rev().into_iter())
    {
        if arr[c1] != arr[c2] {
            return false;
        }
    }
    true
}

/// 力扣（9. 回文数） https://leetcode-cn.com/problems/palindrome-number/
/// 反转一半的数字
pub fn is_palindrome_v2(x: i32) -> bool {
    // 负数以及以0结尾但不是0的数字都不是回文数
    if x < 0 || (x % 10 == 0 && x != 0) {
        return false;
    }
    let mut y = x;
    let mut reverted_num = 0;
    while y > reverted_num {
        reverted_num = reverted_num * 10 + y % 10;
        y /= 10;
    }
    y == reverted_num || y == reverted_num / 10
}

/// 力扣（9. 回文数）
/// rust slice pattern
pub fn is_palindrome_v3(x: i32) -> bool {
    // 负数以及以0结尾但不是0的数字都不是回文数
    if x < 0 || (x % 10 == 0 && x != 0) {
        return false;
    }
    let s = x.to_string().chars().collect::<Vec<_>>();
    is_palindrome_str(s.as_slice())
}

/// 力扣（9. 回文数）
/// 双指针
pub fn is_palindrome_v4(x: i32) -> bool {
    // 负数以及以0结尾但不是0的数字都不是回文数
    if x < 0 || (x % 10 == 0 && x != 0) {
        return false;
    }
    let s = x.to_string().chars().collect::<Vec<_>>();
    let len = s.len();
    let (mut first, mut last) = (0, len - 1);
    while first < last {
        if s[first] == s[last] {
            first += 1;
            last -= 1;
        } else {
            return false;
        }
    }
    true
}

/// 力扣（13. 罗马数字转整数） https://leetcode-cn.com/problems/roman-to-integer/
/// 以下解法不正确
pub fn roman_to_int(s: String) -> i32 {
    let len = s.len();
    let mut sum = 0;

    let mut map = HashMap::<&str, i32>::new();
    map.insert("I", 1);
    map.insert("V", 5);
    map.insert("X", 10);
    map.insert("L", 50);
    map.insert("C", 100);
    map.insert("D", 500);
    map.insert("M", 1000);
    map.insert("IV", 4);
    map.insert("IX", 9);
    map.insert("XL", 40);
    map.insert("XC", 90);
    map.insert("CD", 400);
    map.insert("CM", 900);

    let mut i = 1;

    while i < len {
        if let Some(x) = s.get(i - 1..=i) {
            println!("x:{}", x);
            if let Some(value) = map.get(x) {
                println!("value:{}", value);
                sum += value;
            }
        }
        i += 1;
    }

    while let Some(x) = s.get(i - 1..=i) {
        println!("x:{}", x);

        if let Some(v) = map.get(x) {
            sum += v;
        }
        if i < len {
            i += 1;
        } else {
            break;
        }
    }

    let chars_vec: Vec<char> = s.chars().collect();

    let chars = s.char_indices();

    let mut split_idx = 0;
    for (idx, ch) in chars {
        if idx != 0 && idx == split_idx {
            continue;
        }
        let num = match ch {
            'I' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'V' {
                        split_idx = idx + 1;
                        4
                    } else if next_char == 'X' {
                        split_idx = idx + 1;
                        9
                    } else {
                        split_idx = idx;
                        1
                    }
                } else {
                    split_idx = idx;
                    1
                }
            }
            'V' => 5,
            'X' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'L' {
                        split_idx = idx + 1;
                        40
                    } else if next_char == 'C' {
                        split_idx = idx + 1;
                        90
                    } else {
                        split_idx = idx;
                        10
                    }
                } else {
                    split_idx = idx;
                    10
                }
            }

            'L' => 50,
            'C' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'D' {
                        split_idx = idx + 1;
                        400
                    } else if next_char == 'M' {
                        split_idx = idx + 1;
                        900
                    } else {
                        split_idx = idx;
                        100
                    }
                } else {
                    split_idx = idx;
                    100
                }
            }
            'D' => 500,
            'M' => 1000,
            _ => panic!("No valid character"),
        };
        sum += num;
    }

    sum
}

static ROMAN_NUMBERS: [(char, i32); 7] = [
    ('I', 1),
    ('V', 5),
    ('X', 10),
    ('L', 50),
    ('C', 100),
    ('D', 500),
    ('M', 1000),
];
/// 力扣（13. 罗马数字转整数）
pub fn roman_to_int_v2(s: String) -> i32 {
    use std::collections::HashMap;
    // [(a,b)] convert to HashMap<a,b>
    let map: HashMap<char, i32> = ROMAN_NUMBERS.iter().cloned().collect();
    let mut ret = 0;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        let v = map.get(&c).unwrap();
        match it.peek() {
            Some(n) if v < map.get(n).unwrap() => ret -= v,
            _ => ret += v,
        }
    }
    ret
}

/// 力扣（13. 罗马数字转整数）
pub fn roman_to_int_v3(s: String) -> i32 {
    let mut sum = 0;
    let chars_vec: Vec<char> = s.chars().collect();

    let chars = s.char_indices();
    let len = s.len();

    let mut split_idx = 0;
    for (idx, ch) in chars {
        if idx != 0 && idx == split_idx {
            continue;
        }
        let num = match ch {
            'I' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'V' {
                        split_idx = idx + 1;
                        4
                    } else if next_char == 'X' {
                        split_idx = idx + 1;
                        9
                    } else {
                        split_idx = idx;
                        1
                    }
                } else {
                    split_idx = idx;
                    1
                }
            }
            'V' => 5,
            'X' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'L' {
                        split_idx = idx + 1;
                        40
                    } else if next_char == 'C' {
                        split_idx = idx + 1;
                        90
                    } else {
                        split_idx = idx;
                        10
                    }
                } else {
                    split_idx = idx;
                    10
                }
            }

            'L' => 50,
            'C' => {
                if idx + 1 < len {
                    let next_char = chars_vec[idx + 1];
                    if next_char == 'D' {
                        split_idx = idx + 1;
                        400
                    } else if next_char == 'M' {
                        split_idx = idx + 1;
                        900
                    } else {
                        split_idx = idx;
                        100
                    }
                } else {
                    split_idx = idx;
                    100
                }
            }
            'D' => 500,
            'M' => 1000,
            _ => panic!("No valid character"),
        };
        sum += num;
        // println!("num:{},sum:{}",num,sum);
    }

    sum
}

/// 力扣（14. 最长公共前缀） https://leetcode-cn.com/problems/longest-common-prefix/
pub fn longest_common_prefix(strs: Vec<String>) -> String {
    let len = strs.len();
    if len == 0 {
        return "".to_owned();
    }
    let mut prefix = &strs[0][..];
    let mut i = 1;
    let mut idx = -1;
    while i < len {
        let next_str = &strs[i][..];
        match next_str.find(prefix) {
            None => idx = -1,
            Some(i) => idx = i as i32,
        }

        while idx != 0 {
            if let Some(p) = prefix.get(0..prefix.len() - 1) {
                prefix = p;
                if prefix.is_empty() {
                    return "".to_owned();
                }
            }

            match next_str.find(prefix) {
                None => idx = -1,
                Some(i) => idx = i as i32,
            }
        }
        i += 1;
    }

    prefix.to_owned()
}

/// 力扣（14. 最长公共前缀）
/// 方法二：纵向扫描
pub fn longest_common_prefix_v2(strs: Vec<String>) -> String {
    let len = strs.len();
    if len == 0 {
        return "".to_owned();
    }
    let mut prefix = &strs[0][..];
    let len_of_prefix = prefix.len();
    for i in 0..len_of_prefix {
        if let Some(ch) = prefix.bytes().nth(i) {
            for item in strs.iter().take(len).skip(1) {
                if i == item.len() || item.bytes().nth(i) != Some(ch) {
                    return String::from(prefix.get(0..i).unwrap());
                }
            }
        }
    }

    prefix.to_owned()
}

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

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        ListNode { val, next: None }
    }
}

impl Ord for ListNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.val.cmp(&self.val)
    }
}
impl PartialOrd for ListNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 力扣（21. 合并两个有序链表) https://leetcode-cn.com/problems/merge-two-sorted-lists/
pub fn merge_two_lists(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    match (l1, l2) {
        (Some(v1), None) => Some(v1),
        (None, Some(v2)) => Some(v2),
        (Some(mut v1), Some(mut v2)) => {
            if v1.val < v2.val {
                let n = v1.next.take();
                v1.next = self::merge_two_lists(n, Some(v2));
                Some(v1)
            } else {
                let n = v2.next.take();
                v2.next = self::merge_two_lists(Some(v1), n);
                Some(v2)
            }
        }
        _ => None,
    }
}

/// 将数组转为链表（从尾到头构建）
pub fn vec_to_list(v: &[i32]) -> Option<Box<ListNode>> {
    let mut head = None;
    for i in v.iter().rev() {
        let mut node = ListNode::new(*i);
        node.next = head;
        head = Some(Box::new(node));
    }
    head
}

/// 将数组转为链表（从头到尾构建）
pub fn vec_to_list_v2(v: &[i32]) -> Option<Box<ListNode>> {
    let mut dummy_head = Box::new(ListNode::new(0));
    let mut head = &mut dummy_head;
    for i in v {
        head.next = Some(Box::new(ListNode::new(*i)));
        head = head.next.as_mut().unwrap();
    }
    dummy_head.next
}

/// 打印链表的值
pub fn display(l: Option<Box<ListNode>>) {
    let mut head = &l;
    while head.is_some() {
        print!("{}, ", head.as_ref().unwrap().val);
        head = &(head.as_ref().unwrap().next);
    }
    println!();
}

/// 力扣（26. 删除有序数组中的重复项) https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/
pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    let len = nums.len();
    if len <= 1 {
        return len as i32;
    }
    let mut slow_index = 0;
    let mut fast_index = 1;
    while fast_index < len {
        if nums[slow_index] != nums[fast_index] {
            nums[slow_index + 1] = nums[fast_index];
            slow_index += 1;
        }
        fast_index += 1;
    }

    (slow_index + 1) as i32
}

/// 力扣（26. 删除有序数组中的重复项)
pub fn remove_duplicates_v2(nums: &mut Vec<i32>) -> i32 {
    let len = nums.len();
    if len == 0 {
        return 0;
    }
    let mut slow_index = 1;
    let mut fast_index = 1;
    while fast_index < len {
        if nums[fast_index] != nums[fast_index - 1] {
            nums[slow_index] = nums[fast_index];
            slow_index += 1;
        }

        fast_index += 1;
    }

    slow_index as i32
}

/// 力扣（26. 删除有序数组中的重复项)
pub fn remove_duplicates_v3(nums: &mut Vec<i32>) -> i32 {
    nums.dedup();
    nums.len() as i32
}

///  力扣（27. 移除元素）https://leetcode-cn.com/problems/remove-element/
pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
    let mut k = 0;
    for i in 0..nums.len() {
        if nums[i] != val {
            nums[k] = nums[i];
            k += 1;
        }
    }
    k as i32
}

/// 力扣（28. 实现 strStr()）  https://leetcode-cn.com/problems/implement-strstr/
/// 当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。
pub fn str_str(haystack: String, needle: String) -> i32 {
    // 参考Java String.indexOf()的代码
    let source = haystack.as_bytes();
    let target = needle.as_bytes();

    let source_offset = 0usize;
    let source_count = source.len();
    let target_offset = 0usize;
    let target_count = target.len();
    let from_index = 0usize;
    if target_count == 0usize {
        return 0;
    }

    if target_count > source_count {
        return -1;
    }

    let first = target[target_offset];
    let max = source_offset + (source_count - target_count);

    let mut i = source_offset + from_index;
    while i <= max {
        // 首先匹配首字母
        while source[i] != first {
            i += 1;
            if i <= max {
                continue;
            } else {
                break;
            }
        }

        if i <= max {
            let mut j = i + 1;
            let end = j + target_count - 1;
            let mut k = target_offset + 1;
            // 匹配剩余的字符
            while j < end && source[j] == target[k] {
                j += 1;
                k += 1;
            }

            if j == end {
                return (i - source_offset) as i32;
            }
        }

        i += 1;
    }

    -1
}

/// 力扣（28. 实现 strStr()）
/// 系统内置方法
pub fn str_str_v2(haystack: String, needle: String) -> i32 {
    match haystack.find(&needle) {
        Some(index) => index as i32,
        None => return -1,
    }
}

/// 力扣（28. 实现 strStr()）
/// KMP(Knuth-Morris-Pratt)算法
/// 前缀函数，记作 π(i)，其定义如下：
/// 对于长度为 m 的字符串 s，其前缀函数 π(i)(0≤i<m) 表示 s 的子串 s[0:i] 的最长的相等的真前缀与真后缀的长度。特别地，如果不存在符合条件的前后缀，那么 π(i)=0。
pub fn str_str_v3(haystack: String, needle: String) -> i32 {
    let (m, n) = (needle.len(), haystack.len());
    if m == 0 {
        return 0;
    }
    let haystack_chars = haystack.chars().collect::<Vec<char>>();
    let needle_chars = needle.chars().collect::<Vec<char>>();
    let mut pi = vec![0; m];
    let (mut i, mut j) = (1, 0);
    while i < m {
        while j > 0 && (&needle_chars[i] != &needle_chars[j]) {
            j = pi[j - 1];
        }
        // 如果 s[i]=s[π(i−1)]，那么 π(i)=π(i−1)+1。
        if &needle_chars[i] == &needle_chars[j] {
            j += 1;
        }
        pi[i] = j;
        i += 1;
    }

    let (mut i, mut j) = (0, 0);
    while i < n {
        while j > 0 && (&haystack_chars[i] != &needle_chars[j]) {
            j = pi[j - 1];
        }
        if &haystack_chars[i] == &needle_chars[j] {
            j += 1;
        }
        if (j == m) {
            return (i - m + 1) as i32;
        }
        i += 1;
    }

    -1
}

/// 力扣（35. 搜索插入位置） https://leetcode-cn.com/problems/search-insert-position/
/// 提示:nums 为无重复元素的升序排列数组
pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
    let len = nums.len();
    let mut idx = 0;
    while idx < len {
        if target <= nums[idx] {
            return idx as i32;
        }

        if target > nums[idx] {
            if idx != len - 1 {
                if target < nums[idx + 1] {
                    return (idx + 1) as i32;
                } else {
                    idx += 1;
                    continue;
                }
            } else {
                return len as i32;
            }
        }
        idx += 1;
    }

    idx as i32
}

/// 力扣（35. 搜索插入位置）
/// 二分查找
pub fn search_insert_v2(nums: Vec<i32>, target: i32) -> i32 {
    use std::cmp::Ordering;
    let mut left = 0;
    let mut right = (nums.len() - 1) as i32;
    while left <= right {
        let middle = (left + (right - left) / 2) as usize;
        match nums[middle].cmp(&target) {
            Ordering::Greater => {
                right = (middle as i32) - 1;
            }
            Ordering::Less => {
                left = (middle + 1) as i32;
            }
            Ordering::Equal => {
                return middle as i32;
            }
        }
    }
    (right + 1) as i32
}

/// 力扣（53. 最大子序和） https://leetcode-cn.com/problems/maximum-subarray/
/// 动态规划转移方程： f(i)=max{f(i−1)+nums[i],nums[i]}  
///  f(i) 代表以第 i 个数结尾的「连续子数组的最大和」
pub fn max_sub_array(nums: Vec<i32>) -> i32 {
    let mut prev = 0;
    let mut max_ans = nums[0];
    for x in nums {
        prev = max(prev + x, x);
        max_ans = max(max_ans, prev);
    }
    max_ans
}

/// 力扣（58. 最后一个单词的长度） https://leetcode-cn.com/problems/length-of-last-word/submissions/
pub fn length_of_last_word(s: String) -> i32 {
    let s_trim = s.trim_end();
    let words: Vec<&str> = s_trim.rsplitn(2, ' ').collect();
    words[0].len() as i32
}

/// 力扣（66. 加一） https://leetcode-cn.com/problems/plus-one/
pub fn plus_one(digits: Vec<i32>) -> Vec<i32> {
    // 以下算法参考了：https://leetcode-cn.com/problems/plus-one/solution/java-shu-xue-jie-ti-by-yhhzw/
    let len = digits.len();
    let mut new_digits = digits.clone();

    let mut i = len - 1;
    loop {
        let b = (digits[i] + 1) % 10;
        new_digits[i] = b;
        if b != 0 {
            return new_digits;
        }
        if i > 0 {
            i -= 1;
        } else {
            break;
        }
    }

    let mut new_digits = vec![0; len + 1];
    new_digits[0] = 1;
    new_digits
}

/// 力扣（67. 二进制求和） https://leetcode-cn.com/problems/add-binary/
pub fn add_binary(a: String, b: String) -> String {
    let mut result = String::new();
    let mut ca = 0;
    let mut s = true;
    let mut t = true;
    let mut a_rev = a.chars().rev();
    let mut b_rev = b.chars().rev();

    while s || t {
        let mut sum = ca;

        if let Some(x) = a_rev.next() {
            let temp = x as i32 - 48;

            sum += temp;
        } else {
            s = false;
        }

        if let Some(x) = b_rev.next() {
            let temp = x as i32 - 48;
            sum += temp;
        } else {
            t = false;
        }

        if !s && !t {
            break;
        }

        if sum % 2 == 0 {
            result.push('0');
        } else {
            result.push('1');
        }
        ca = sum / 2;
    }

    if ca == 1 {
        result.push('1');
    }
    //字符串翻转
    result.chars().rev().collect()
}

/// 力扣（69. x 的平方根） https://leetcode-cn.com/problems/sqrtx/
/// 二分查找，注意相乘溢出
pub fn my_sqrt(x: i32) -> i32 {
    let mut left = 0;
    let mut right = x;
    let mut ans = -1;
    while left <= right {
        let mid = left + (right - left) / 2;
        if (mid as i64) * (mid as i64) <= x as i64 {
            ans = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    ans
}

/// 力扣（69. x 的平方根）
/// 换底公式（自然数e为底）
pub fn my_sqrt_v2(x: i32) -> i32 {
    if x == 0 {
        return 0;
    }

    // x1/2=(elnx)1/2=e21​lnx
    //java int ans = (int) Math.exp(0.5 * Math.log(x));
    let x = x as f64;

    //exp(self) = e^(self)
    //ln(self) = self's  natural logarithm
    let ans = (0.5 * (x).ln()).exp();
    let squar = (ans + 1f64) * (ans + 1f64);

    if squar <= x {
        (ans as i32) + 1
    } else {
        ans as i32
    }
}

/// 力扣（69. x 的平方根）
///
pub fn my_sqrt_v3(x: i32) -> i32 {
    let y = x as f64;
    y.sqrt().floor() as i32
}

/// 力扣（69. x 的平方根）
/// 牛顿迭代法
/// 我们用 C表示待求出平方根的那个整数。显然，C 的平方根就是函数: y = f(x) = x^2 - C
pub fn my_sqrt_v4(x: i32) -> i32 {
    if x == 0 {
        return 0i32;
    }

    let c = x as f64;
    let mut x0 = c;
    loop {
        // 迭代方程
        let xi = 0.5 * (x0 + c / x0);
        // 1e-7即0.0000001
        if (x0 - xi).abs() < 1e-7 {
            break;
        }
        x0 = xi;
    }

    x0 as i32
}

use std::cell::RefCell;
use std::rc::Rc;
/// 力扣（70. 爬楼梯） https://leetcode-cn.com/problems/climbing-stairs/
pub fn climb_stairs(n: i32) -> i32 {
    if n == 1 {
        return 1;
    } else if n == 2 {
        return 2;
    }

    let mut dp = vec![0; (n + 1) as usize];
    dp[1] = 1;
    dp[2] = 2;
    for i in 3..=n as usize {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    dp[n as usize]
}

pub fn climb_stairs_memo(n: i32, memo: Rc<RefCell<Vec<i32>>>) -> i32 {
    if n == 1 {
        memo.borrow_mut()[n as usize] = 1;
    } else if n == 2 {
        memo.borrow_mut()[n as usize] = 2;
    } else {
        let step1 = climb_stairs_memo(n - 1, Rc::clone(&memo));
        let step2 = climb_stairs_memo(n - 2, Rc::clone(&memo));
        memo.borrow_mut()[n as usize] = step1 + step2;
    }

    memo.borrow_mut()[n as usize]
}

/// 力扣（83. 删除排序链表中的重复元素) https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/
pub fn delete_duplicates(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut dummy_head = Box::new(ListNode::new(0));
    let mut dummy_node = &mut dummy_head;
    let mut node = &head;
    while node.is_some() {
        let value = node.as_ref().unwrap().val;
        node = &(node.as_ref().unwrap().next);
        if node.is_none() || value != node.as_ref().unwrap().val {
            dummy_node.next = Some(Box::new(ListNode::new(value)));
            dummy_node = dummy_node.next.as_mut().unwrap();
        }
    }

    dummy_head.next
}

/// 力扣（88. 合并两个有序数组） https://leetcode-cn.com/problems/merge-sorted-array/
pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    let mut m = m;
    let mut index: usize = 0;
    for &item in nums2.iter().take(n as usize) {
        while (index < m as usize) && nums1[index] <= item {
            index += 1;
        }

        if index < (m as usize) {
            for j in (index + 1..nums1.len()).rev() {
                nums1[j] = nums1[j - 1];
            }
            m += 1;
        }
        nums1[index] = item;
        index += 1;
    }
}

/// 力扣（88. 合并两个有序数组）
/// 双指针/从后往前
pub fn merge_v2(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    let mut p1 = m - 1;
    let mut p2 = n - 1;
    let mut p = m + n - 1;
    while p1 >= 0 && p2 >= 0 {
        if nums1[p1 as usize] < nums2[p2 as usize] {
            nums1[p as usize] = nums2[p2 as usize];
            p2 -= 1;
        } else {
            nums1[p as usize] = nums1[p1 as usize];
            p1 -= 1;
        }
        p -= 1;
    }
    nums1[..((p2 + 1) as usize)].clone_from_slice(&nums2[..((p2 + 1) as usize)]);
}

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

/// 力扣（101. 对称二叉树)  https://leetcode-cn.com/problems/symmetric-tree/
pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    false
}

/// 力扣（118. 杨辉三角） https://leetcode-cn.com/problems/pascals-triangle/
/// 动态规划算法：由上一行的数据推导出下一行的数据
pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
    let rows = num_rows as usize;
    let mut result = Vec::<Vec<i32>>::with_capacity(rows);

    for row in 1..=rows {
        if row <= 2 {
            let r = vec![1; row];
            result.push(r);
        } else if let Some(last_row) = result.last() {
            let mut r = vec![1; row];
            for i in 1..row - 1 {
                r[i] = last_row[i - 1] + last_row[i];
            }
            result.push(r);
        }
    }
    result
}

/// 力扣（121. 买卖股票的最佳时机）  https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
/// 暴力解法，容易超时
pub fn max_profit(prices: Vec<i32>) -> i32 {
    let len = prices.len();
    if len <= 1 {
        return 0;
    }
    let mut buy_day = 0;
    let mut sale_day = 1;
    let mut max_profit = 0;
    while buy_day < len - 1 {
        while sale_day < len {
            let profit = prices[sale_day] - prices[buy_day];
            if profit > 0 {
                max_profit = max(max_profit, profit);
            }
            sale_day += 1;
        }
        buy_day += 1;
        sale_day = buy_day + 1;
    }

    max_profit
}

/// 力扣（121. 买卖股票的最佳时机）
/// 最低价格、最大利润
pub fn max_profit_v2(prices: Vec<i32>) -> i32 {
    let len = prices.len();
    let mut min_prince = i32::MAX;
    let mut max_profit = 0;
    for price in prices {
        if price < min_prince {
            min_prince = price;
        } else if price - min_prince > max_profit {
            max_profit = price - min_prince;
        }
    }

    max_profit
}

/// 力扣（119. 杨辉三角 II） https://leetcode-cn.com/problems/pascals-triangle-ii/
pub fn get_row(row_index: i32) -> Vec<i32> {
    let rows = (row_index + 1) as usize;

    if rows <= 2 {
        return vec![1; rows];
    } else {
        let mut result_vec = vec![1; rows];
        for i in 2..rows {
            let mut j = i - 1;
            while j > 0 {
                result_vec[j] += result_vec[j - 1];
                j -= 1;
            }
        }

        result_vec
    }
}

/// 力扣（125. 验证回文串)  https://leetcode-cn.com/problems/valid-palindrome/
pub fn is_palindrome_125(s: String) -> bool {
    let chars: Vec<char> = s.chars().collect();
    let mut left = 0;
    let mut right = chars.len() - 1;
    while left < right {
        if !chars[left].is_alphanumeric() {
            left += 1;
            continue;
        }
        if !chars[right].is_alphanumeric() {
            right -= 1;
            continue;
        }
        if chars[left].eq_ignore_ascii_case(&chars[right]) {
            left += 1;
            right -= 1;
            continue;
        } else {
            break;
        }
    }
    left >= right
}

/// 力扣（125. 验证回文串)
pub fn is_palindrome_125_v2(s: String) -> bool {
    let chars: Vec<char> = s.chars().filter(|c| c.is_alphanumeric()).collect();
    let len = chars.len();
    if len == 0 {
        return true;
    }

    let mut left = 0;
    let mut right = len - 1;
    while left < right {
        if chars[left].eq_ignore_ascii_case(&chars[right]) {
            left += 1;
            right -= 1;
            continue;
        } else {
            break;
        }
    }
    left >= right
}

/// 力扣（136. 只出现一次的数字） https://leetcode-cn.com/problems/single-number/
/// 使用异或运算的规律
pub fn single_number(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let mut single_number = nums[0];
    for &num in nums.iter().take(len).skip(1) {
        single_number ^= num;
    }

    single_number
}

/// 力扣（167. 两数之和 II - 输入有序数组）https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/
pub fn two_sum2(numbers: Vec<i32>, target: i32) -> Vec<i32> {
    let mut result = Vec::<i32>::with_capacity(2);

    let mut index1 = 0;
    let mut index2 = numbers.len() - 1;
    while index2 >= 1 {
        let sum = numbers[index1] + numbers[index2];
        match sum.cmp(&target) {
            Ordering::Less => {
                index1 += 1;
                continue;
            }
            Ordering::Greater => {
                index2 -= 1;
                continue;
            }
            Ordering::Equal => {
                result.push((index1 + 1) as i32);
                result.push((index2 + 1) as i32);
                break;
            }
        }
    }

    result
}

/// 力扣（169. 多数元素） https://leetcode-cn.com/problems/majority-element/
/// Boyer-Moore 投票算法
pub fn majority_element(nums: Vec<i32>) -> i32 {
    let mut count = 0;
    let mut candidate = 0;
    for num in nums {
        if count == 0 {
            candidate = num;
        }
        if num == candidate {
            count += 1;
        } else {
            count -= 1;
        }
    }

    candidate
}

/// 力扣（168. Excel表列名称） https://leetcode-cn.com/problems/excel-sheet-column-title/
pub fn convert_to_title(column_number: i32) -> String {
    let mut ret = String::new();
    let mut column_number = column_number;
    while column_number > 0 {
        let a0 = (column_number - 1) % 26 + 1;
        let ch = b'A' + (a0 - 1) as u8;
        ret.push(ch as char);
        column_number = (column_number - a0) / 26;
    }

    ret.chars().rev().collect()
}

/// 力扣（169. 多数元素）
/// 哈希表
pub fn majority_element_v2(nums: Vec<i32>) -> i32 {
    let mut counts_map: HashMap<i32, usize> = HashMap::with_capacity(nums.len() / 2 + 1);
    for num in nums {
        // if !counts_map.contains_key(&num) {
        //     counts_map.insert(num, 1);
        // } else {
        //     if let Some(value) = counts_map.get_mut(&num) {
        //         *value += 1;
        //     };
        // }
        match counts_map.get_mut(&num) {
            Some(value) => {
                *value += 1;
            }
            None => {
                counts_map.insert(num, 1);
            }
        };
    }

    let mut major_entry: (i32, usize) = (0, 0);
    for (key, val) in counts_map.iter() {
        if *val > major_entry.1 {
            major_entry.0 = *key;
            major_entry.1 = *val;
            // major_entry = (*key,*val);
        }
    }
    major_entry.0
}

/// 力扣（171. Excel 表列序号） https://leetcode-cn.com/problems/excel-sheet-column-number/submissions/
pub fn title_to_number(column_title: String) -> i32 {
    let mut sum = 0;
    let mut hex_base = 1;
    for ch in column_title.chars().rev() {
        sum += (hex_base * (ch as u8 - b'A' + 1) as i32);
        hex_base *= 26;
    }

    sum
}

/// 力扣（172. 阶乘后的零） https://leetcode-cn.com/problems/factorial-trailing-zeroes/
pub fn trailing_zeroes(n: i32) -> i32 {
    let mut count_fives = 0;
    let mut steps: Vec<i32> = (5..=n).into_iter().filter(|x| *x % 5 == 0).collect();
    // println!("{:?}",steps);
    for step in steps {
        let mut remaining = step;
        while remaining % 5 == 0 {
            count_fives += 1;
            remaining /= 5;
        }
    }

    count_fives
}

/// 力扣（172. 阶乘后的零）
/// f(n) = n/5^1 + n/5^2 + n/5^3 + n/5^m (n < 5^m)
pub fn trailing_zeroes_v2(n: i32) -> i32 {
    let mut count_fives = 0;
    let mut remaining = n;
    while remaining > 0 {
        remaining /= 5;
        count_fives += remaining;
    }
    count_fives
}
/// 力扣（190. 颠倒二进制位） https://leetcode-cn.com/problems/reverse-bits/
//  方法一：逐位颠倒
pub fn reverse_bits(x: u32) -> u32 {
    let mut n = x;
    let mut rev = 0;
    let mut i = 0;
    while i < 32 && n > 0 {
        rev |= (n & 1) << (31 - i);
        n >>= 1;
        i += 1;
    }
    rev
}

/// 力扣（190. 颠倒二进制位）
//  方法二：位运算分治
pub fn reverse_bits_v2(x: u32) -> u32 {
    const M1: u32 = 0x55555555; // 01010101010101010101010101010101
    const M2: u32 = 0x33333333; // 00110011001100110011001100110011
    const M4: u32 = 0x0f0f0f0f; // 00001111000011110000111100001111
    const M8: u32 = 0x00ff00ff; // 00000000111111110000000011111111

    let mut n = x;
    n = n >> 1 & M1 | (n & M1) << 1;
    n = n >> 2 & M2 | (n & M2) << 2;
    n = n >> 4 & M4 | (n & M4) << 4;
    n = n >> 8 & M8 | (n & M8) << 8;
    n >> 16 | n << 16
}

/// 力扣（190. 颠倒二进制位）
pub fn reverse_bits_v3(x: u32) -> u32 {
    let mut n = x;
    n.reverse_bits()
}

/// 力扣（191. 位1的个数) https://leetcode-cn.com/problems/number-of-1-bits/
/// SWAR算法“计算汉明重量” https://baike.baidu.com/item/%E6%B1%89%E6%98%8E%E9%87%8D%E9%87%8F
pub fn hamming_weight_v1(n: u32) -> i32 {
    let mut i = n as i32;
    let const1 = 0x55555555;
    let const2 = 0x33333333;
    let const3 = 0x0F0F0F0F;
    let const4 = 0x01010101;

    i = (i & const1) + ((i >> 1) & const1);
    i = (i & const2) + ((i >> 2) & const2);
    i = (i & const3) + ((i >> 4) & const3);
    i = (i * const4) >> 24;

    i
}

const MASK1: u32 = 0x55555555;
const MASK2: u32 = 0x33333333;
const MASK3: u32 = 0x0F0F0F0F;
const MASK4: u32 = 0x01010101;
/// 力扣（191. 位1的个数)
pub fn hamming_weight(n: u32) -> i32 {
    let mut i = n;
    i = (i & MASK1) + ((i >> 1) & MASK1);
    i = (i & MASK2) + ((i >> 2) & MASK2);
    i = (i & MASK3) + ((i >> 4) & MASK3);
    i = (i * MASK4) >> 24;

    i as i32
}

/// 力扣（191. 位1的个数)
pub fn hamming_weight_v2(n: u32) -> i32 {
    format!("{:b}", n).chars().filter(|c| *c == '1').count() as i32
}

/// 力扣（191. 位1的个数)
pub fn hamming_weight_v3(n: u32) -> i32 {
    count_ones(n)
}

/// Brian Kernighan 算法
fn count_ones(n: u32) -> i32 {
    let mut ones = 0;
    let mut n = n;
    while n > 0 {
        n &= (n - 1);
        ones += 1;
    }
    ones
}

/// 计算数字number各个位上的数字的平方和
fn get_next(number: i32) -> i32 {
    let mut total_sum = 0;
    let mut num = number;
    while num > 0 {
        let d = num % 10;
        num /= 10;
        total_sum += d * d;
    }
    total_sum
}

// 计算数字number各个位上的数字的平方和
fn bit_square_sum(number: i32) -> i32 {
    let total_sum = number.to_string().chars().fold(0, |acc, x| {
        let y = (x as i32) - 48;
        acc + y * y
    });
    total_sum
}
use std::collections::HashSet;
// const cycle_number:HashSet<i32> = [4, 16, 37,58,89,145,42,20].iter().cloned().collect();
/// 力扣（202. 快乐数) https://leetcode-cn.com/problems/happy-number/
/// 不是快乐数的数称为不快乐数（unhappy number），所有不快乐数的数位平方和计算，最後都会进入 4 → 16 → 37 → 58 → 89 → 145 → 42 → 20 → 4 的循环中。
pub fn is_happy(n: i32) -> bool {
    let mut cycle_number = HashSet::new();
    cycle_number.insert(4);
    cycle_number.insert(16);
    cycle_number.insert(37);
    cycle_number.insert(58);
    cycle_number.insert(89);
    cycle_number.insert(145);
    cycle_number.insert(42);
    cycle_number.insert(20);

    let mut num = n;
    while num != 1 && !cycle_number.contains(&num) {
        num = get_next(num);
        // num = bit_square_sum(num);
    }

    num == 1
}

/// 力扣（202. 快乐数)
/// 快慢指针法、龟兔赛跑法
pub fn is_happy_v2(n: i32) -> bool {
    let mut slow_runner = n;
    let mut fast_runner = get_next(n);
    while fast_runner != 1 && slow_runner != fast_runner {
        slow_runner = get_next(slow_runner);
        fast_runner = get_next(fast_runner);
        fast_runner = get_next(fast_runner);
    }

    fast_runner == 1
}

fn is_prime(x: i32) -> bool {
    let mut i = 2;
    while i * i <= x {
        if x % i == 0 {
            return false;
        }
        i += 1;
    }
    true
}

/// 力扣（203. 移除链表元素) https://leetcode-cn.com/problems/remove-linked-list-elements/
pub fn remove_elements(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut dummy_head = ListNode::new(0);
    dummy_head.next = head;
    let mut nums = vec![];
    while let Some(mut node) = dummy_head.next.take() {
        if node.val != val {
            nums.push(node.val);
        }
        if let Some(next) = node.next.take() {
            dummy_head.next = Some(next);
        }
    }

    vec_to_list(&nums)
}

/// 力扣（203. 移除链表元素)
/// 虚拟头结点
pub fn remove_elements_v2(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut head = head;
    let mut root = Box::new(ListNode::new(0));
    //用于存储、修改当前值不为val的节点
    let mut current = &mut root;
    while let Some(mut node) = head.take() {
        head = node.next.take();
        if node.val != val {
            current.next = Some(node);
            current = current.next.as_mut().unwrap();
        }
    }
    root.next
}

/// 力扣（203. 移除链表元素)
/// 直接使用原来的链表来进行删除操作 FIXME take()会消耗掉Option中的值，以下代码不正确
pub fn remove_elements_v3(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut head = head;

    while let Some(mut node) = head.take() {
        if node.val == val {
            head = node.next.take();
        } else {
            head = Some(node);
            break;
        }
    }
    while let Some(node) = head.take().as_deref_mut() {
        if node.val == val {
            head = node.next.take();
            continue;
        } else {
            head = node.next.take();
        }
        if let Some(next) = node.next.take().as_deref_mut() {
            if next.val == val {
                if let Some(next2) = next.next.take() {
                    *next = *next2;
                }
            }
        }
    }

    head
}

/// 力扣（204. 计数质数) https://leetcode-cn.com/problems/count-primes/
pub fn count_primes(n: i32) -> i32 {
    let mut ans = 0;
    let mut i = 2;
    while i < n {
        if is_prime(i) {
            ans += 1;
        }
        i += 1;
    }
    ans
}

/// 力扣（204. 计数质数)
/// 方法二：厄拉多塞筛法（埃氏筛）
pub fn count_primes_v2(n: i32) -> i32 {
    let n = n as usize;
    let mut primes = vec![1; n];
    let mut ans = 0;
    let mut i = 2_usize;

    while i < n {
        if primes[i] == 1 {
            ans += 1;
        }
        if let Some(squar) = i.checked_mul(i) {
            if squar < n {
                let mut j = squar;
                while j < n {
                    primes[j] = 0;
                    j += i;
                }
            }
        }

        i += 1;
    }
    ans
}

/// 力扣（205. 同构字符串） https://leetcode-cn.com/problems/isomorphic-strings/
pub fn is_isomorphic(s: String, t: String) -> bool {
    let s_chars: Vec<char> = s.chars().collect();
    let t_chars: Vec<char> = t.chars().collect();

    let mut s2t = HashMap::new();
    let mut t2s = HashMap::new();
    let len = s_chars.len();
    for i in 0..len {
        let (x, y) = (s_chars[i], t_chars[i]);
        if (s2t.contains_key(&x) && s2t.get(&x) != Some(&y))
            || (t2s.contains_key(&y) && t2s.get(&y) != Some(&x))
        {
            return false;
        }
        s2t.insert(x, y);
        t2s.insert(y, x);
    }

    true
}

/// 力扣（205. 同构字符串）
pub fn is_isomorphic_v2(s: String, t: String) -> bool {
    let mut s_t = std::collections::HashMap::<char, char>::new();
    let mut t_s = std::collections::HashMap::<char, char>::new();
    // 提示：可以假设 s 和 t 长度相同。
    // if s.len() != t.len() {
    //     return false;
    // }
    for (a, b) in s.chars().zip(t.chars()) {
        match (s_t.get(&a), t_s.get(&b)) {
            (Some(&v1), Some(v2)) => {
                if v1 != b {
                    return false;
                }
            }
            (None, None) => {
                s_t.insert(a, b);
                t_s.insert(b, a);
            }
            _ => {
                return false;
            }
        }
    }
    true
}

/// 力扣（206. 反转链表） https://leetcode-cn.com/problems/reverse-linked-list/
/// 假设：head 对应的链表为：1 -> 2 -> 3 -> 4 -> 5 -> None
pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut head = head;
    let mut tail = None;
    while let Some(mut node) = head.take() {
        // head = node(1).next = node(2)
        head = node.next;
        // node(1).next = None
        node.next = tail;
        // tail = node(1)
        tail = Some(node);
    }
    tail
}

/// 力扣（217. 存在重复元素） https://leetcode-cn.com/problems/contains-duplicate/
/// 方法1： 排序
pub fn contains_duplicate(nums: Vec<i32>) -> bool {
    let mut new_nums = nums;
    new_nums.sort_unstable();
    let len = new_nums.len();
    for i in 1..len {
        if new_nums[i] == new_nums[i - 1] {
            return true;
        }
    }

    false
}

/// 力扣（217. 存在重复元素）
/// 方法2：哈希表
pub fn contains_duplicate_v2(nums: Vec<i32>) -> bool {
    use std::collections::HashMap;
    let mut counts_map = HashMap::<i32, i32>::new();
    for num in nums {
        match counts_map.get(&num) {
            Some(_) => {
                return true;
            }
            None => {
                counts_map.insert(num, 1);
            }
        }
    }

    false
}

/// 力扣（219. 存在重复元素 II） https://leetcode-cn.com/problems/contains-duplicate-ii/
pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
    use std::collections::HashSet;
    let mut nums_set = HashSet::new();
    let len = nums.len();
    for i in 0..len {
        if nums_set.contains(&nums[i]) {
            return true;
        }
        nums_set.insert(nums[i]);
        if nums_set.len() > (k as usize) {
            nums_set.remove(&nums[i - k as usize]);
        }
    }
    false
}

/// 力扣（231. 2的幂） https://leetcode-cn.com/problems/power-of-two/
pub fn is_power_of_two(n: i32) -> bool {
    if n <= 0 {
        return false;
    }
    if n == 1 {
        return true;
    }

    let mut m = n;
    loop {
        if m % 2 == 0 {
            m /= 2;
            if m <= 1 {
                break;
            }
        } else {
            return false;
        }
    }

    true
}

/// 力扣（231. 2的幂）
pub fn is_power_of_two_v2(n: i32) -> bool {
    if n == 0 {
        return false;
    }

    n & (n - 1) == 0
}
/// 力扣（258. 各位相加） https://leetcode-cn.com/problems/add-digits/
pub fn add_digits(num: i32) -> i32 {
    (num - 1) % 9 + 1
}
/// 力扣（263. 丑数）   https://leetcode-cn.com/problems/ugly-number/
/// 丑数 就是只包含质因数 2、3 和/或 5 的正整数。
/// 1 通常被视为丑数。
pub fn is_ugly(num: i32) -> bool {
    if num <= 0 {
        return false;
    }
    let mut num = num;
    for x in [2, 3, 5].iter() {
        while num > 1 && num % x == 0 {
            num /= x;
        }
    }
    num == 1
}

fn bad_version(n: usize, version: usize) -> Vec<bool> {
    let mut versions = vec![false; n];
    for index in version..n {
        versions[index - 1] = true;
    }
    versions
}

fn is_bad_version(version: i32) -> bool {
    // let mut versions = vec![false;100];
    VERSIONS[version as usize]
}

/// 力扣（242. 有效的字母异位词） https://leetcode-cn.com/problems/valid-anagram/
pub fn is_anagram(s: String, t: String) -> bool {
    use std::collections::HashMap;
    let s_chars: Vec<char> = s.chars().collect();
    let t_chars: Vec<char> = t.chars().collect();

    let mut s_map = HashMap::<char, i32>::new();
    let mut t_map = HashMap::<char, i32>::new();

    for ch in s_chars {
        if let Some(val) = s_map.get_mut(&ch) {
            *val += 1;
        } else {
            s_map.insert(ch, 1);
        }
    }

    for ch in t_chars {
        if let Some(val) = t_map.get_mut(&ch) {
            *val += 1;
        } else {
            t_map.insert(ch, 1);
        }
    }

    if s_map.len() != t_map.len() {
        return false;
    }

    for (key, val) in s_map.iter() {
        if !t_map.contains_key(&key) {
            return false;
        }
        if let Some(val2) = t_map.get(&key) {
            if val != val2 {
                return false;
            }
        }
    }
    true
}
/// 力扣（268. 丢失的数字） https://leetcode-cn.com/problems/missing-number/
pub fn missing_number(nums: Vec<i32>) -> i32 {
    use std::collections::HashSet;
    let len = nums.len();
    let mut nums_set = HashSet::with_capacity(len);
    nums.iter().for_each(|num| {
        nums_set.insert(*num as usize);
    });

    for num in 0..=len {
        if !nums_set.contains(&num) {
            return num as i32;
        }
    }

    -1
}

/// 力扣（268. 丢失的数字）
/// 方法：数学法
pub fn missing_number_v2(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let expect_sum = (len * (len + 1) / 2) as i32;
    let sum = nums.iter().fold(0, |sum, num| sum + *num);
    expect_sum - sum
}
/// 力扣（268. 丢失的数字）
/// 方法：位运算（异或）
pub fn missing_number_v3(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let mut missing = len;
    for (idx, num) in nums.iter().enumerate().take(len) {
        missing ^= idx ^ (*num as usize);
    }
    missing as i32
}

/// 力扣（278. 第一个错误的版本）  https://leetcode-cn.com/problems/first-bad-version/
// The API isBadVersion is defined for you.
// isBadVersion(versions:i32)-> bool;
// to call it use self.isBadVersion(versions)
pub fn first_bad_version(n: i32) -> i32 {
    let mut left = 1;
    let mut right = n;
    while left < right {
        let middle = left + (right - left) / 2;
        if is_bad_version(middle) {
            right = middle;
        } else {
            left = middle + 1;
        }
    }
    left
}

/// 力扣（283. 移动零） https://leetcode-cn.com/problems/move-zeroes/
pub fn move_zeroes(nums: &mut Vec<i32>) {
    let mut slow_index = 0;
    let len = nums.len();

    for fast_index in 0..len {
        if nums[fast_index] != 0 {
            nums[slow_index] = nums[fast_index];
            slow_index += 1;
        }
    }

    for num in nums.iter_mut().take(len).skip(slow_index) {
        *num = 0;
    }
}

/// 力扣（283. 移动零） https://leetcode-cn.com/problems/move-zeroes/
pub fn move_zeroes_v2(nums: &mut Vec<i32>) {
    let mut slow_index = 0;
    let mut fast_index = 0;
    let len = nums.len();
    while fast_index < len {
        if nums[fast_index] != 0 {
            nums.swap(slow_index, fast_index);
            slow_index += 1;
        }
        fast_index += 1;
    }
}

/// 力扣（290. 单词规律） https://leetcode-cn.com/problems/word-pattern/
///  与 力扣（205. 同构字符串）类似
pub fn word_pattern(pattern: String, s: String) -> bool {
    let pattern_chars = pattern.chars().collect::<Vec<_>>();
    let words = s.split(' ').collect::<Vec<_>>();
    let len = words.len();
    if pattern_chars.len() != len {
        return false;
    }
    let mut pattern_map = HashMap::<char, String>::new();
    let mut word_map = HashMap::<String, char>::new();
    for i in 0..len {
        match (pattern_map.get(&pattern_chars[i]), word_map.get(words[i])) {
            (Some(word), Some(ch)) => {
                if word != words[i] || *ch != pattern_chars[i] {
                    return false;
                }
            }
            (None, None) => {
                pattern_map.insert(pattern_chars[i], String::from(words[i]));
                word_map.insert(String::from(words[i]), pattern_chars[i]);
            }
            _ => {
                return false;
            }
        }
    }

    true
}

/// 力扣（292. Nim 游戏） https://leetcode-cn.com/problems/nim-game/
pub fn can_win_nim(n: i32) -> bool {
    n % 4 != 0
}

/// 力扣（326. 3的幂) https://leetcode-cn.com/problems/power-of-three/
pub fn is_power_of_three(n: i32) -> bool {
    n > 0 && 1162261467 % n == 0
}

/// 力扣（326. 3的幂)
pub fn is_power_of_three_v2(n: i32) -> bool {
    if n < 1 {
        return false;
    }
    let mut n = n;
    while n % 3 == 0 {
        n /= 3;
    }
    n == 1
}

/// 力扣（326. 3的幂)
pub fn is_power_of_three_v3(n: i32) -> bool {
    if n <= 0 {
        return false;
    }
    let mut n = n;
    while n > 1 {
        if n % 3 != 0 {
            return false;
        }
        n /= 3;
    }
    true
}

/// 力扣（338. 比特位计数） https://leetcode-cn.com/problems/counting-bits/
/// 与 力扣（191. 位1的个数) 类似
pub fn count_bits(n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut result = vec![0; n + 1];

    for (num, item) in result.iter_mut().enumerate().take(n + 1) {
        *item = num.count_ones() as i32;
    }

    result
}

/// 力扣（338. 比特位计数）
pub fn count_bits_v2(n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut result = vec![0; n + 1];
    for (num, item) in result.iter_mut().enumerate().take(n + 1).skip(1) {
        *item = count_ones(num as u32);
    }

    result
}

/// 力扣（338. 比特位计数）
/// 动态规划
pub fn count_bits_v3(n: i32) -> Vec<i32> {
    let n = n as usize;
    let mut result = vec![0; n + 1];
    let mut high_bit = 0;
    for num in 1..=n {
        if num & (num - 1) == 0 {
            high_bit = num;
        }
        result[num] = result[num - high_bit] + 1;
    }

    result
}

/// 力扣（345. 反转字符串中的元音字母） https://leetcode-cn.com/problems/reverse-vowels-of-a-string/
pub fn reverse_vowels(s: String) -> String {
    const VOWELS: [char; 10] = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'];
    //vowels.contains(&'a');
    let mut chars: Vec<char> = s.chars().collect();
    let mut vowels_chars: Vec<usize> = Vec::new();
    for (idx, ch) in chars.iter().enumerate() {
        if VOWELS.contains(ch) {
            vowels_chars.push(idx);
        }
    }

    let len = vowels_chars.len();
    if len > 1 {
        let mut left = 0;
        let mut right = len - 1;
        while left < right {
            chars.swap(vowels_chars[left], vowels_chars[right]);
            left += 1;
            right -= 1;
        }
        return chars.iter().collect();
    } else {
        return s;
    }
}

/// 力扣（345. 反转字符串中的元音字母）
pub fn reverse_vowels_v2(s: String) -> String {
    const VOWELS: [char; 10] = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'];
    //vowels.contains(&'a');
    let mut chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut left = 0;
    let mut right = len - 1;
    while left < right {
        while left < len && !VOWELS.contains(&chars[left]) {
            left += 1;
        }
        while right > 0 && !VOWELS.contains(&chars[right]) {
            right -= 1;
        }
        if left < right {
            chars.swap(left, right);
            left += 1;
            right -= 1;
        }
    }
    chars.iter().collect()
}

/// 力扣（345. 反转字符串中的元音字母）
pub fn reverse_vowels_v3(s: String) -> String {
    let mut chars = s.into_bytes();
    let len = chars.len();
    let mut left = 0;
    let mut right = len - 1;
    while left < right {
        while left < len && !is_vowel(chars[left]) {
            left += 1;
        }
        while right > 0 && !is_vowel(chars[right]) {
            right -= 1;
        }
        if left < right {
            chars.swap(left, right);
            left += 1;
            right -= 1;
        }
    }
    std::str::from_utf8(&chars).unwrap().to_string()
}

fn is_vowel(ch: u8) -> bool {
    match ch {
        b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U' => true,
        _ => false,
    }
}

/// 力扣（350. 两个数组的交集 II） https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/
/// 方法1： 排序 + 双指针
pub fn intersect(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    let len1 = nums1.len();
    let len2 = nums2.len();

    let mut less_sorted_nums: Vec<i32> = Vec::new();
    let mut greater_sorted_nums: Vec<i32> = Vec::new();
    let mut greater_len = 0;
    let mut less_len = 0;
    match len1.cmp(&len2) {
        Ordering::Greater => {
            greater_len = len1;
            less_len = len2;
            less_sorted_nums = nums2;
            less_sorted_nums.sort_unstable();
            greater_sorted_nums = nums1;
            greater_sorted_nums.sort_unstable();
        }
        Ordering::Equal | Ordering::Less => {
            greater_len = len2;
            less_len = len1;
            less_sorted_nums = nums1;
            less_sorted_nums.sort_unstable();
            greater_sorted_nums = nums2;
            greater_sorted_nums.sort_unstable();
        }
    }
    let mut intersect_vec = Vec::new();
    let mut i = 0;
    let mut j = 0;
    loop {
        if (j >= less_len || i >= greater_len) {
            break;
        }

        match greater_sorted_nums[i].cmp(&less_sorted_nums[j]) {
            Ordering::Equal => {
                intersect_vec.push(greater_sorted_nums[i]);
                i += 1;
                j += 1;
            }
            Ordering::Greater => {
                j += 1;
            }
            Ordering::Less => {
                i += 1;
            }
        }
    }

    intersect_vec
}

/// 力扣（349. 两个数组的交集） https://leetcode-cn.com/problems/intersection-of-two-arrays/
pub fn intersection(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    use std::collections::HashSet;
    let mut nums1_set = HashSet::new();
    let mut nums2_set = HashSet::new();
    nums1.iter().for_each(|num| {
        nums1_set.insert(*num);
    });
    nums2.iter().for_each(|num| {
        nums2_set.insert(*num);
    });

    nums1_set.intersection(&nums2_set).copied().collect()
}

/// 力扣（350. 两个数组的交集 II）
/// 方法2：哈希表
pub fn intersect_v2(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    if nums1.len() > nums2.len() {
        return intersect_v2(nums2, nums1);
    }

    use std::collections::HashMap;
    // nums1 较短的数组，nums2 较长的数组
    let mut counts_map = HashMap::new();
    for num in nums1 {
        match counts_map.get_mut(&num) {
            Some(count) => {
                *count += 1;
            }
            None => {
                counts_map.insert(num, 1);
            }
        }
    }

    let mut intersect_vec = Vec::with_capacity(nums2.len());

    for num in nums2 {
        if let Some(count) = counts_map.get_mut(&num) {
            if *count > 0 {
                intersect_vec.push(num);
                *count -= 1;
            }
            if *count <= 0 {
                counts_map.remove(&num);
            }
        }
    }

    intersect_vec
}

/// 力扣（342. 4的幂） https://leetcode-cn.com/problems/power-of-four/
pub fn is_power_of_four(n: i32) -> bool {
    n > 0 && (n & (n - 1)) == 0 && (n & 0x2aaaaaaa == 0)
}

/// 力扣（344. 反转字符串） https://leetcode-cn.com/problems/reverse-string/
pub fn reverse_string(s: &mut Vec<char>) {
    let len = s.len();
    if len > 1 {
        let mut i = 0;
        let half = len / 2;
        while i < half {
            s.swap(i, len - i - 1);
            i += 1;
        }
    }
}

/// 力扣（367. 有效的完全平方数) https://leetcode-cn.com/problems/valid-perfect-square/
pub fn is_perfect_square(num: i32) -> bool {
    if num == 1 {
        return true;
    }
    let mut left = 2;
    let mut right = num / 2;
    while left <= right {
        let x = left + (right - left) / 2;
        if let Some(guess_square) = x.checked_mul(x) {
            if guess_square == num {
                return true;
            }

            if guess_square > num {
                right = x - 1;
            } else {
                left = x + 1;
            }
        } else {
            // 过大
            right = x - 1;
        }
    }
    false
}

/// 力扣（383. 赎金信） https://leetcode-cn.com/problems/ransom-note/
pub fn can_construct(ransom_note: String, magazine: String) -> bool {
    // 记录杂志中各个字符出现的次数
    let mut chars_vec = vec![0; 26];
    let a = b'a';
    magazine.as_bytes().iter().for_each(|ch| {
        chars_vec[(*ch - a) as usize] += 1;
    });

    for ch in ransom_note.as_bytes().iter() {
        chars_vec[(*ch - a) as usize] -= 1;
        if chars_vec[(*ch - a) as usize] < 0 {
            return false;
        }
    }

    true
}

/// 力扣（387. 字符串中的第一个唯一字符） https://leetcode-cn.com/problems/first-unique-character-in-a-string/
///  方法一：使用哈希表存储频数
pub fn first_uniq_char(s: String) -> i32 {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    // 统计各个字符出现的次数
    use std::collections::HashMap;
    let mut counts_map = HashMap::<char, i32>::new();
    for ch in &chars {
        match counts_map.get_mut(&ch) {
            Some(count) => *count += 1,
            None => {
                counts_map.insert(*ch, 1);
            }
        };
    }

    for (i, &ch) in chars.iter().enumerate() {
        if let Some(&count) = counts_map.get(&ch) {
            if count == 1 {
                return i as i32;
            }
        }
    }

    -1
}

/// 力扣（387. 字符串中的第一个唯一字符）
/// 方法二：使用哈希表存储索引
pub fn first_uniq_char_v2(s: String) -> i32 {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    // 存储首次出现的下标
    use std::collections::HashMap;
    let mut indexs_map = HashMap::<char, i32>::new();
    for (i, &ch) in chars.iter().enumerate() {
        match indexs_map.get_mut(&ch) {
            Some(index) => {
                // 再次出现则将下标设置为-1
                *index = -1;
            }
            None => {
                // 首次出现存储下标
                indexs_map.insert(ch, i as i32);
            }
        };
    }

    // 下标不为-1且最小的值即为答案
    let mut first = len as i32;
    for index in indexs_map.values() {
        if *index != -1 && *index < first {
            first = *index;
        }
    }

    if first == len as i32 {
        first = -1;
    }

    first
}

/// 力扣（387. 字符串中的第一个唯一字符）
/// 方法三：队列
pub fn first_uniq_char_v3(s: String) -> i32 {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    // 存储首次出现的下标
    use std::collections::HashMap;
    use std::collections::VecDeque;
    // 存放各个字符及其首次出现的位置下标的元组
    let mut queue = VecDeque::<(char, i32)>::new();
    let mut indexs_map = HashMap::<char, i32>::new();
    for (i, &ch) in chars.iter().enumerate() {
        match indexs_map.get_mut(&ch) {
            Some(index) => {
                // 再次出现则将下标设置为-1
                *index = -1;
                // 只保留第一次出现的
                queue.retain(|x| x.0 != ch);
            }
            None => {
                // 首次出现存储下标
                indexs_map.insert(ch, i as i32);
                queue.push_back((ch, i as i32));
            }
        };
    }

    if queue.is_empty() {
        return -1;
    } else if let Some(only) = queue.pop_front() {
        return only.1;
    }
    -1
}

/// 力扣（389. 找不同） https://leetcode-cn.com/problems/find-the-difference/
pub fn find_the_difference(s: String, t: String) -> char {
    let mut chars_vec = vec![0; 26];
    let a = b'a' as usize;
    t.as_bytes().iter().for_each(|x| {
        chars_vec[(*x as usize) - a] += 1;
    });
    s.as_bytes().iter().for_each(|x| {
        chars_vec[(*x as usize) - a] -= 1;
    });

    for (index, count) in chars_vec.iter().enumerate() {
        if *count == 1 {
            return (index as u8 + b'a') as char;
        }
    }
    ' '
}

// TODO 400

/// 力扣（401. 二进制手表） https://leetcode-cn.com/problems/binary-watch/
pub fn read_binary_watch(turned_on: i32) -> Vec<String> {
    if turned_on < 0 || turned_on > 8 {
        return vec![];
    }
    //小时最多亮3盏灯[0,3]
    let mut hour_turn_on: Vec<Vec<String>> = vec![vec![]; 4];
    for hour in (0i32..=11i32) {
        hour_turn_on[hour.count_ones() as usize].push(format!("{}", hour));
    }
    //dbg!(hour_turn_on);
    //分钟最多亮5盏灯[0,5]
    let mut minute_turn_on: Vec<Vec<String>> = vec![vec![]; 6];
    for minute in (0i32..=59i32) {
        minute_turn_on[minute.count_ones() as usize].push(format!("{:02}", minute));
    }
    //dbg!(minute_turn_on);
    let mut result = Vec::new();

    // turned_on = hour + minute;
    for hour in (0i32..=3i32) {
        let mut minute = turned_on - hour;
        if minute >= 0 && minute <= 5 {
            for h in &hour_turn_on[hour as usize] {
                for m in &minute_turn_on[minute as usize] {
                    result.push(format!("{}:{}", h, m));
                }
            }
        } else {
            continue;
        }
    }

    result
}

static HEX_CHARS: [&str; 16] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f",
];
/// 力扣（405. 数字转换为十六进制数） https://leetcode-cn.com/problems/convert-a-number-to-hexadecimal/
pub fn to_hex(num: i32) -> String {
    match num.cmp(&0) {
        Ordering::Greater | Ordering::Less => {
            let mut ret = String::new();
            let mut num = num;
            let mut i = 7;
            while i >= 0 {
                let val = (num >> (4 * i)) & 0xf;
                if !ret.is_empty() || val > 0 {
                    ret.push_str(HEX_CHARS[val as usize]);
                }
                i -= 1;
            }
            ret
        }

        Ordering::Equal => "0".to_owned(),
    }
}

/// 力扣（412. Fizz Buzz） https://leetcode-cn.com/problems/fizz-buzz/
pub fn fizz_buzz(n: i32) -> Vec<String> {
    let len = n as usize;
    let mut result = Vec::<String>::with_capacity(len);
    for i in 1..=len {
        let divisible_by_3 = i % 3 == 0;
        let divisible_by_5 = i % 5 == 0;
        if divisible_by_3 && divisible_by_5 {
            result.push("FizzBuzz".to_string());
        } else if divisible_by_3 {
            result.push("Fizz".to_string());
        } else if divisible_by_5 {
            result.push("Buzz".to_string());
        } else {
            result.push(i.to_string());
        }
    }

    result
}

/// 力扣（412. Fizz Buzz）
pub fn fizz_buzz_v2(n: i32) -> Vec<String> {
    let len = n as usize;
    let mut result = Vec::<String>::with_capacity(len);
    for i in 1..=len {
        if i % 3 == 0 {
            if i % 5 == 0 {
                result.push("FizzBuzz".to_string());
            } else {
                result.push("Fizz".to_string());
            }
        } else if i % 5 == 0 {
            result.push("Buzz".to_string());
        } else {
            result.push(i.to_string());
        }
    }

    result
}

/// 力扣（415. 字符串相加） https://leetcode-cn.com/problems/add-strings/
pub fn add_strings(num1: String, num2: String) -> String {
    let mut i = (num1.len() - 1) as i32;
    let mut j = (num2.len() - 1) as i32;
    let mut add = 0;
    let mut ans = Vec::<u8>::new();
    let num1_chars = num1.into_bytes();
    let num2_chars = num2.into_bytes();

    while i >= 0 || j >= 0 || add != 0 {
        let x = if i >= 0 {
            num1_chars[i as usize] - b'0'
        } else {
            0
        };
        let y = if j >= 0 {
            num2_chars[j as usize] - b'0'
        } else {
            0
        };
        let result = x + y + add;
        ans.push((result % 10 + b'0') as u8);
        add = result / 10;
        i -= 1;
        j -= 1;
    }
    ans.reverse();
    String::from_utf8(ans).unwrap()
}

/// 力扣（448. 找到所有数组中消失的数字） https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/
pub fn find_disappeared_numbers(nums: Vec<i32>) -> Vec<i32> {
    let len = nums.len();
    let mut appear = vec![false; len + 1];
    let mut result = vec![];
    for num in nums {
        appear[num as usize] = true;
    }

    for i in 1..=len {
        if !appear[i] {
            result.push(i as i32);
        }
    }
    result
}

/// 力扣（461. 汉明距离） https://leetcode-cn.com/problems/hamming-distance/
pub fn hamming_distance(x: i32, y: i32) -> i32 {
    let z = x ^ y;
    z.count_ones() as i32
}

/// 力扣（461. 汉明距离）
/// 这里采用右移位，每个位置都会被移动到最右边。移位后检查最右位的位是否为 1 即可。检查最右位是否为 1，可以使用取模运算（i % 2）或者 AND 操作（i & 1），这两个操作都会屏蔽最右位以外的其他位。
pub fn hamming_distance_v2(x: i32, y: i32) -> i32 {
    let mut xor = x ^ y;
    let mut distance = 0;
    while xor != 0 {
        if xor % 2 == 1 {
            distance += 1;
        }
        xor >>= 1;
    }
    distance
}

/// 力扣（461. 汉明距离）
/// 布赖恩·克尼根算法
pub fn hamming_distance_v3(x: i32, y: i32) -> i32 {
    let mut xor = x ^ y;
    let mut distance = 0;
    while xor != 0 {
        distance += 1;
        xor = xor & (xor - 1);
    }
    distance
}

/// 力扣（476. 数字的补数） https://leetcode-cn.com/problems/number-complement/
pub fn find_complement(num: i32) -> i32 {
    let mut num = num;
    //最高位为1的位
    let mut high_bit = 0;
    for i in 1..=30 {
        if num >= (1 << i) {
            high_bit = i;
        } else {
            break;
        }
    }

    let mark = match (high_bit == 30) {
        true => 0x7fffffff,
        false => (1 << (high_bit + 1)) - 1,
    };

    num ^ mark
}

///力扣（485. 最大连续1的个数）https://leetcode-cn.com/problems/max-consecutive-ones/
pub fn find_max_consecutive_ones(nums: Vec<i32>) -> i32 {
    let mut max = 0;
    let mut max_temp = 0;
    for num in nums {
        if num == 0 {
            if max_temp > max {
                max = max_temp;
            }
            max_temp = 0;
        } else {
            max_temp += 1;
        }
    }

    if max_temp > max {
        max_temp
    } else {
        max
    }
}

/// 力扣（496. 下一个更大元素 I） https://leetcode-cn.com/problems/next-greater-element-i/
pub fn next_greater_element(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    use std::collections::HashMap;
    use std::collections::VecDeque;
    let mut map = HashMap::new();
    // VecDeque模拟单调栈（栈底最大，栈顶最小），front 栈底，back 栈顶
    let mut stack = VecDeque::new();
    for num in nums2.iter().rev() {
        while let Some(top) = stack.back() {
            if num > top {
                stack.pop_back();
            } else {
                break;
            }
        }

        let next_element = if stack.is_empty() {
            -1
        } else {
            *stack.back().unwrap()
        };

        map.insert(num, next_element);
        stack.push_back(*num);
    }

    let mut res = Vec::with_capacity(nums1.len());
    for num in nums1 {
        if let Some(&next_element) = map.get(&num) {
            res.push(next_element);
        }
    }

    res
}

/// TODO: 500

/// 力扣（561. 数组拆分 I） https://leetcode-cn.com/problems/array-partition-i/
pub fn array_pair_sum(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    if len % 2 != 0 {
        panic!("数组长度必须为偶数");
    }

    let mut nums_sort = nums;
    nums_sort.sort_unstable();

    let mut sum = 0;
    for i in 0..len / 2 {
        sum += nums_sort[2 * i];
    }

    sum
}

/// 力扣（704. 二分查找) https://leetcode-cn.com/problems/binary-search/
pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    // target在[left,right]中查找
    let len = nums.len();
    let mut left = 0;
    let mut right = len - 1;
    let mut pivot;
    while left <= right {
        pivot = left + (right - left) / 2;
        // 注意usize的范围和nums的下标范围
        if nums[pivot] == target {
            return pivot as i32;
        }
        if target < nums[pivot] {
            if pivot == 0 {
                break;
            }
            right = pivot - 1;
        } else {
            if pivot == len - 1 {
                break;
            }
            left = pivot + 1;
        }
    }
    -1
}

/// 力扣（704. 二分查找)
pub fn search_v2(nums: Vec<i32>, target: i32) -> i32 {
    use std::cmp::Ordering;
    // target在[left,right]中查找
    let mut left = 0;
    let mut right = (nums.len() - 1) as i32;
    while left <= right {
        let middle = (left + right) as usize / 2;
        match nums[middle].cmp(&target) {
            Ordering::Greater => {
                right = middle as i32 - 1;
            }
            Ordering::Less => {
                left = middle as i32 + 1;
            }
            Ordering::Equal => {
                return middle as i32;
            }
        }
    }
    -1
}

/// 力扣（704. 二分查找)
pub fn search_v3(nums: Vec<i32>, target: i32) -> i32 {
    // target在[left,right)中查找，由于rust下标usize的限制，推荐使用这种方式
    let mut left = 0;
    let mut right = nums.len();
    while left < right {
        let middle = left + (right - left) / 2;
        match nums[middle].cmp(&target) {
            Ordering::Greater => {
                right = middle;
            }
            Ordering::Less => {
                left = middle + 1;
            }
            Ordering::Equal => {
                return middle as i32;
            }
        }
    }
    -1
}

/// 力扣（724. 寻找数组的中心索引） https://leetcode-cn.com/problems/find-pivot-index/
pub fn pivot_index(nums: Vec<i32>) -> i32 {
    let mut sum = 0;
    for num in &nums {
        sum += *num;
    }
    let mut left_sum = 0;
    for (idx, num) in nums.iter().enumerate() {
        if *num + left_sum * 2 == sum {
            return idx as i32;
        }
        left_sum += *num;
    }
    -1
}

/// 力扣（747. 至少是其他数字两倍的最大数） https://leetcode-cn.com/problems/largest-number-at-least-twice-of-others/submissions/
pub fn dominant_index(nums: Vec<i32>) -> i32 {
    let mut idx = 0;
    let mut max = nums[idx];
    // 找出最大值及其下表
    for (i, num) in nums.iter().enumerate() {
        if *num > max {
            max = *num;
            idx = i;
        }
    }
    //println!("max:{},idx:{}",max,idx);
    // 找出第二大的数,最小的值为0
    let mut second_biggest = 0;
    for (i, num) in nums.iter().enumerate() {
        if idx != i && *num > second_biggest {
            second_biggest = *num;
        } else {
            continue;
        }
    }

    //println!("second_biggest:{}",second_biggest);
    if max >= 2 * second_biggest {
        return idx as i32;
    }
    -1
}

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

/// TODO 800

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

/// 力扣（867. 转置矩阵) https://leetcode-cn.com/problems/transpose-matrix/
pub fn transpose(matrix: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut transposed = Vec::<Vec<i32>>::new();

    let m = matrix.len();
    let n = matrix[0].len();
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

/// TODO 900

/// 力扣（977. 有序数组的平方）https://leetcode-cn.com/problems/squares-of-a-sorted-array/
pub fn sorted_squares(nums: Vec<i32>) -> Vec<i32> {
    let mut v: Vec<i32> = nums.iter().map(|x| x * x).collect();
    if nums[0] < 0 {
        v.sort_unstable();
    }
    v
}

/// 归并排序解法（参考官方java版本解法，主要区别在于i下标的范围）
/// 力扣（977. 有序数组的平方）https://leetcode-cn.com/problems/squares-of-a-sorted-array/
pub fn sorted_squares_v2(nums: Vec<i32>) -> Vec<i32> {
    let len = nums.len();
    if nums[0] >= 0 {
        let mut v = Vec::<i32>::with_capacity(len);
        for &num in nums.iter().take(len) {
            v.push(num * num);
        }
        return v;
    }

    if nums[len - 1] <= 0 {
        let mut v = Vec::<i32>::with_capacity(len);
        for &num in nums.iter().take(len) {
            v.push(num * num);
        }
        v.reverse();
        return v;
    }
    //将数组划分成两组，负数组和非负数组，然后使用归并排序得到排序结果。
    //1. 确定最后个非负数的位置negtive。
    let mut negtive = 0;
    for (i, &num) in nums.iter().enumerate().take(len) {
        if num < 0 {
            negtive = i;
        } else {
            break;
        }
    }

    //2. 定义两个指针，分别定为两个数组（nums[0..negtive],[negtive..len]）的下标。
    let mut i = negtive;
    let mut j = negtive + 1;
    let mut temp_i = i as i32;

    //3. 根据归并算法得到排序结果。
    let mut index = 0;
    let mut v = vec![0i32; len];
    while temp_i >= 0 || j < len {
        let x = nums[i] * nums[i];
        if temp_i < 0 {
            v[index] = nums[j] * nums[j];
            j += 1;
        } else if j == len || x < nums[j] * nums[j] {
            v[index] = x;
            temp_i -= 1;
            if temp_i >= 0 {
                i = temp_i as usize;
            }
        } else {
            v[index] = nums[j] * nums[j];
            j += 1;
        }
        index += 1;
    }
    v
}

/// 力扣（977. 有序数组的平方）
/// 方法三：双指针
pub fn sorted_squares_v3(nums: Vec<i32>) -> Vec<i32> {
    let len = nums.len();
    if len == 1 {
        return vec![nums[0] * nums[0]];
    }
    let mut ans = vec![0; len];
    let mut i = 0;
    let mut j = len - 1;
    let mut pos = j as i32;
    while i <= j {
        let square1 = nums[i] * nums[i];
        let square2 = nums[j] * nums[j];
        // 为了防止j自减溢出，当前后平方相同时把优先把square1写入ans
        if square1 >= square2 {
            ans[pos as usize] = square1;
            i += 1;
        } else {
            ans[pos as usize] = square2;
            j -= 1;
        }
        pos -= 1;
    }

    ans
}

/// TODO 1000

/// 力扣（1051. 高度检查器），https://leetcode-cn.com/problems/height-checker/
pub fn height_checker(heights: Vec<i32>) -> i32 {
    let len = heights.len();
    if len <= 1 {
        return 0;
    }
    let mut arr = vec![0; 101];
    for height in &heights {
        arr[*height as usize] += 1;
    }

    let mut count = 0;
    let mut j = 0;
    for (i, item) in arr.iter_mut().enumerate().take(101).skip(1) {
        while *item > 0 {
            if heights[j] != (i as i32) {
                count += 1;
            }
            j += 1;
            *item -= 1;
        }
    }
    count
}

/// 力扣（1486. 数组异或操作） https://leetcode-cn.com/problems/xor-operation-in-an-array/
pub fn xor_operation(n: i32, start: i32) -> i32 {
    (1..n).fold(start, |acc, i| acc ^ (start + 2 * i))
}

/// 力扣（1863. 找出所有子集的异或总和再求和）https://leetcode-cn.com/problems/sum-of-all-subset-xor-totals/
pub fn subset_xor_sum(nums: Vec<i32>) -> i32 {
    let mut xor_sum = 0;
    let n = nums.len();
    for num in nums {
        xor_sum |= num;
    }
    xor_sum << (n - 1)
}

/// 力扣（1863. 找出所有子集的异或总和再求和）
pub fn subset_xor_sum_v2(nums: Vec<i32>) -> i32 {
    let mut xor_sum = 0;
    let n = nums.len();
    let two_pow_n = 1 << n;
    for i in 0..two_pow_n {
        let mut temp = 0;
        for j in 0..n {
            if i & (1 << j) != 0 {
                temp ^= nums[j];
            }
        }
        xor_sum += temp;
    }
    xor_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test() {
        let nums = vec![2, 7, 2, 11];
        let result = two_sum(nums, 9);
        dbg!(result);

        let valid_string = String::from("(){{}");
        dbg!(is_valid(valid_string));

        let linked_list = vec_to_list_v2(&vec![1, 3, 5, 7, 9]);
        display(linked_list);

        let l = merge_two_lists(
            vec_to_list(&vec![1, 3, 5, 7, 9]),
            vec_to_list(&vec![2, 4, 6, 8, 10]),
        );
        display(l);

        let l = merge_two_lists(
            vec_to_list(&vec![1, 2, 4]),
            vec_to_list(&vec![1, 3, 4, 5, 6]),
        );
        display(l);

        dbg!(reverse(132));
        dbg!(reverse(-1999999999));

        let roman_numbers = String::from("MCMXCIV");
        dbg!(roman_to_int(roman_numbers));
        let roman_numbers_v2 = String::from("MCMXCIV");
        dbg!(roman_to_int_v2(roman_numbers_v2));

        let roman_numbers_v3 = String::from("MCMXCIV");
        dbg!(roman_to_int_v3(roman_numbers_v3));

        let sorted_nums = vec![1, 3, 5, 6];
        let target = 4;

        dbg!(search_insert(sorted_nums, target));

        dbg!(search_insert_v2(vec![1, 3, 5, 6], 7));

        let mut nums = vec![3, 2, 2, 3];
        let val = 3;

        let len = remove_element(&mut nums, val);

        let haystack = String::from("aaacaaab");
        let needle = String::from("aaab");
        dbg!(str_str(haystack, needle));

        let mut strs = Vec::new();
        strs.push(String::from("cdf"));
        //strs.push(String::from("acc"));
        strs.push(String::from("cd"));
        strs.push(String::from("cde"));
        //    strs.push(String::from("abscd"));
        dbg!(longest_common_prefix(strs));

        let count = climb_stairs(30);
        dbg!(count);

        dbg!(plus_one(vec![9, 1, 9]));

        let mut chars = Vec::<char>::new();
        chars.push('a');
        chars.push('b');
        chars.push('c');
        chars.push('d');
        //    chars.push('e');
        reverse_string(&mut chars);
        dbg!(chars);

        let a = String::from("0");
        let b = String::from("0");
        dbg!(add_binary(a, b));

        dbg!(generate(10));

        dbg!(get_row(33));

        let numbers = vec![2, 7, 11, 15];
        let target = 18;

        dbg!(two_sum2(numbers, target));

        let nums = vec![1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1];
        dbg!(find_max_consecutive_ones(nums));

        let mut nums = Vec::<i32>::new();
        nums.push(1);
        nums.push(4);
        nums.push(3);
        nums.push(2);
        dbg!(array_pair_sum(nums));

        dbg!(pivot_index(vec![1, 7, 3, 6, 5, 6]));

        dbg!(dominant_index(vec![2]));

        sorted_squares_v2(vec![-10000, -9999, -7, -5, 0, 0, 10000]);

        let sorted_vec = vec![-3, -2, 0, 1, 4, 5];
        let sorted_squares_vec = sorted_squares(sorted_vec);
        assert_eq!(sorted_squares_vec, vec![0, 1, 4, 9, 16, 25]);

        let sorted_vec_v2 = vec![-3, -2, 0, 1, 4, 5];
        let sorted_squares_vec_v2 = sorted_squares_v2(sorted_vec_v2);
        assert_eq!(sorted_squares_vec_v2, vec![0, 1, 4, 9, 16, 25]);

        let sorted_vec_v3 = vec![-3, 3, 4];
        let sorted_squares_vec_v3 = sorted_squares_v3(sorted_vec_v3);
        assert_eq!(sorted_squares_vec_v3, vec![9, 9, 16]);

        dbg!(is_palindrome(121));
        dbg!(is_palindrome(-121));
        dbg!(is_palindrome(10));
        dbg!(is_palindrome(1));
        dbg!(is_palindrome_v3(8848));

        assert_eq!(length_of_longest_substring("dvdf".to_string()), 3);
        assert_eq!(length_of_longest_substring("abcabcbb".to_string()), 3);
        assert_eq!(length_of_longest_substring("bbbbb".to_string()), 1);
        assert_eq!(length_of_longest_substring("pwwkew".to_string()), 3);
        assert_eq!(length_of_longest_substring("c".to_string()), 1);
        assert_eq!(length_of_longest_substring("au".to_string()), 2);

        // for ch in '0'..='z'{
        //     dbg!("{} {}",ch, ch as u8);
        // }

        let mut nums1: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let mut nums2: Vec<i32> = vec![2, 5, 6];
        merge(&mut nums1, 3, &mut nums2, 3);
        dbg!(nums1);

        let mut nums3: Vec<i32> = vec![7, 8, 9, 0, 0, 0];
        let mut nums4: Vec<i32> = vec![2, 5, 6];
        merge_v2(&mut nums3, 3, &mut nums4, 3);
        dbg!(nums3);

        // let res = longest_palindrome(String::from("banana"));
        // dbg!("longest_palindrome res:{}",res);

        let heights = vec![1, 2, 4, 5, 3, 3];

        dbg!(height_checker(heights));

        assert_eq!(my_sqrt(4), 2);
        assert_eq!(my_sqrt(8), 2);
    }

    #[test]
    fn no_pass() {
        dbg!(my_sqrt(2147395599));
        dbg!(my_sqrt_v2(2147395599));
        dbg!(my_sqrt_v2(256));
        let num = 2147395599f64;
        dbg!(num.sqrt().floor());
        dbg!(my_sqrt_v3(2147395599));
        dbg!(my_sqrt_v4(2147395599));

        dbg!("1e-7 is very small");

        assert_eq!(hamming_weight(15), hamming_weight(15));

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

        let nums = vec![1, 16, 218];
        for num in nums {
            assert_eq!(is_power_of_two(num), is_power_of_two_v2(num));
        }

        let mut nums = vec![0, 1, 0, 3, 0, 12, 14];
        move_zeroes(&mut nums);
        dbg!(nums);

        let mut nums2 = vec![0, 1, 0, 3, 0, 12, 14];
        move_zeroes_v2(&mut nums2);
        dbg!(nums2);

        let nums3 = vec![-2, 1, -3, 4, -1, 2, 1, -5, 4];
        let max_ans = max_sub_array(nums3);
        dbg!(max_ans);

        let ones = hamming_distance(1, 4);
        dbg!(ones);

        let distance = hamming_distance_v2(4, 255);
        dbg!(distance);

        let distance = hamming_distance_v3(4, 65535);
        dbg!(distance);

        let nums = vec![2, 2, 1, 1, 1, 2, 2];
        let major = majority_element(nums);
        dbg!(major);

        let nums2 = vec![6, 5, 5];
        let major2 = majority_element_v2(nums2);
        dbg!(major2);

        let nums3 = vec![-12, 6, 10, 11, 12, 13, 15];
        for num in nums3 {
            dbg!(is_ugly(num));
        }

        let rand_num = rand::random::<u16>() as i32;
        dbg!(is_happy(rand_num));
        dbg!(is_happy_v2(rand_num));

        dbg!(bit_square_sum(123456));

        dbg!(is_power_of_three(81 * 3));

        let ans = count_primes(10000);
        dbg!(ans);

        let ans = count_primes_v2(10000);
        dbg!(ans);

        let nums = vec![-1, 0, 3, 5, 9, 12];
        let target = 9;

        dbg!(search(nums, target));

        let nums = vec![1, 0, 3, 5, 9, 12];

        dbg!(search_v2(nums, 2));

        let mut nums = vec![1, 3, 3, 3, 5, 5, 9, 9, 9, 9];

        dbg!(remove_duplicates(&mut nums));

        let mut nums = vec![1, 3, 3, 3, 5, 5, 9, 9, 9, 9];

        dbg!(remove_duplicates_v2(&mut nums));

        let s = String::from("ab#c");
        let t = String::from("ad#c");

        dbg!(backspace_compare(s, t));

        let mut version0 = 0;
        for version in 0..100 {
            if (VERSIONS[version]) {
                dbg!(version);
                version0 = version;
                break;
            }
        }

        dbg!(first_bad_version(100));

        let prices = vec![7, 1, 5, 3, 6, 4];

        dbg!(max_profit(prices));

        dbg!(max_profit_v2(vec![7, 1, 5, 3, 6, 4]));

        let s = String::from("leetcode");
        let first_uniq_char_result = first_uniq_char(s);
        dbg!(first_uniq_char_result);

        let s = String::from("loveleetcodeverymuch");
        let first_uniq_char_v2_result = first_uniq_char_v2(s);
        dbg!(first_uniq_char_v2_result);

        let s = String::from("loveleetcodeverymuch");
        let first_uniq_char_v3_result = first_uniq_char_v3(s);
        dbg!(first_uniq_char_v3_result);

        let nums1 = vec![4, 9, 5, 1];
        let nums2 = vec![9, 2, 4, 10, 5];
        let intersect_result = intersect(nums1, nums2);
        dbg!(intersect_result);

        let nums1 = vec![4, 9, 5, 1];
        let nums2 = vec![9, 2, 4, 10, 5];
        let intersect_v2_result = intersect_v2(nums1, nums2);
        dbg!(intersect_v2_result);
    }

    #[test]
    fn test_200_plus() {
        let nums = vec![1, 2, 3, 4, 5];
        let head = vec_to_list(&nums);
        let tail = reverse_list(head);

        display(tail);

        let head = vec_to_list(&nums);
        let remove_elements_result = remove_elements(head, 3);
        display(remove_elements_result);

        let head = vec_to_list(&nums);
        let remove_elements_v2_result = remove_elements_v2(head, 3);
        display(remove_elements_v2_result);

        let head = vec_to_list(&[7, 6, 8, 7, 7, 7, 7, 7]);
        let remove_elements_v3_result = remove_elements_v3(head, 7);
        display(remove_elements_v3_result);

        dbg!(next_greater_element(vec![4, 1, 2], vec![1, 3, 4, 2]));

        dbg!(add_strings("11".to_string(), "123".to_string()));

        dbg!(read_binary_watch(7));
    }

    #[test]
    fn test_list_node() {
        let head = vec_to_list(&vec![1, 1, 2, 3, 3]);
        let unique_nodes = delete_duplicates(head);
        display(unique_nodes);

        dbg!(reverse_bits(43261596));

        let x = 43261596u32;
        // [u8;4]
        dbg!(x.to_be_bytes());
        // 16进制
        println!("{:08X}", x);
        // 2进制
        println!("{:032b}", x);

        let rev_x: String = format!("{:032b}", x).chars().rev().collect();
        dbg!(u32::from_str_radix(&rev_x, 2));
    }
}
