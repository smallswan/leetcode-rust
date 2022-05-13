//! 剑指 Offer（第 2 版）
//! https://leetcode-cn.com/problem-list/xb9nqhhg/

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_lcoff() {
        let mut c = 80;
        c >>= 1;
        println!("{}", c);
        println!("{}", c & 1);
        println!("{}", 1 & 1);
        println!("{}", 2 & 1);
        println!("{}", 3 & 1);
        println!("{}", 4 & 1);

        let nums = vec![2, 3, 1, 0, 2, 5, 3];
        let find_repeat_number_result = find_repeat_number(nums);
        dbg!(find_repeat_number_result);

        let nums = vec![2, 3, 1, 0, 2, 5, 3];
        let find_repeat_number_v3_result = find_repeat_number_v3(nums);
        dbg!(find_repeat_number_v3_result);

        let nums = vec![2, 3, 1, 0, 2, 5, 3];
        dbg!(get_least_numbers(nums, 4));

        let mut obj = MedianFinder::new();
        obj.add_num(1);
        obj.add_num(2);
        dbg!(obj.find_median());
        obj.add_num(3);
        dbg!(obj.find_median());

        dbg!(reverse_words("a good   example".to_string()));

        let nums = vec![1, 2];
        missing_number_v2(nums);

        dbg!(find_nth_digit(1000000000));
    }

    #[test]
    fn strings() {
        let s = String::from("We are happy.");
        dbg!(replace_space(s));

        let mut s_url_encode = String::from(" are you ok?");

        dbg!(s_url_encode.replace(" ", "%20"));
    }
}

/// 剑指 Offer 03. 数组中重复的数字 https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/
///  方法1：哈希集合
pub fn find_repeat_number(nums: Vec<i32>) -> i32 {
    use std::collections::HashSet;
    let mut nums_set = HashSet::<i32>::new();
    for num in nums {
        if nums_set.contains(&num) {
            return num;
        } else {
            nums_set.insert(num);
        }
    }

    // 题目没有指明如果没有重复的数字就返回-1，是测试出来的
    -1
}

/// 剑指 Offer 03. 数组中重复的数字
pub fn find_repeat_number_v2(nums: Vec<i32>) -> i32 {
    use std::collections::HashSet;
    let mut nums_set = HashSet::<i32>::new();
    for num in nums {
        if !nums_set.insert(num) {
            return num;
        }
    }

    // 题目没有指明如果没有重复的数字就返回-1，是测试出来的
    -1
}

/// 剑指 Offer 03. 数组中重复的数字
/// 方法2：原地交换
pub fn find_repeat_number_v3(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let mut new_nums = nums;
    let mut i = 0;
    while i < len {
        let num = new_nums[i] as usize;
        if num == i {
            i += 1;
            continue;
        }
        if new_nums[num] as usize == num {
            return num as i32;
        }
        new_nums.swap(i, num);
    }

    -1
}

/// 剑指 Offer 04. 二维数组中的查找 https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/
pub fn find_number_in2_d_array(matrix: Vec<Vec<i32>>, target: i32) -> bool {
    if matrix.is_empty() || matrix[0].is_empty() {
        return false;
    }
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut row = 0;
    let mut col = n - 1;
    let mut max_in_row = matrix[row][col];

    while row < m {
        match max_in_row.cmp(&target) {
            Ordering::Equal => return true,
            Ordering::Greater => {
                if col > 0 {
                    col -= 1;
                } else {
                    break;
                }
                max_in_row = matrix[row][col];
            }
            Ordering::Less => {
                if row < m - 1 {
                    row += 1;
                } else {
                    break;
                }
                max_in_row = matrix[row][col];
            }
        }
    }

    false
}

/// 剑指 Offer 05. 替换空格 https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/
pub fn replace_space(s: String) -> String {
    let original_len = s.len();
    if original_len == 0 {
        return "".into();
    }
    let num_of_blank = s.as_bytes().iter().filter(|&x| *x == b' ').count();
    if num_of_blank == 0 {
        return s;
    }
    let bytes = s.as_bytes();
    let new_len = original_len + num_of_blank * 2;
    let mut new_bytes: Vec<u8> = vec![0; new_len];
    let mut index_of_original = original_len - 1;
    let mut index_of_new = new_len - 1;
    while index_of_new >= index_of_original {
        if bytes[index_of_original] == b' ' {
            new_bytes[index_of_new] = b'0';
            new_bytes[index_of_new - 1] = b'2';
            new_bytes[index_of_new - 2] = b'%';
            index_of_new -= 3;
        } else {
            new_bytes[index_of_new] = bytes[index_of_original];
            if index_of_new == 0 {
                break;
            }
            index_of_new -= 1;
        }
        if index_of_original == 0 {
            break;
        }
        index_of_original -= 1;
    }
    //dbg!(index_of_new);
    //dbg!(index_of_original);

    String::from_utf8(new_bytes).unwrap()
}

/// 剑指 Offer 10- II. 青蛙跳台阶问题  https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/
/// 注意：本题与主站 70 题相同：https://leetcode-cn.com/problems/climbing-stairs/ ，不同之处在于需要对结果取模！！
pub fn num_ways(n: i32) -> i32 {
    if n <= 1 {
        return 1;
    } else if n == 2 {
        return 2;
    }

    let mut dp = vec![0; (n + 1) as usize];
    dp[1] = 1;
    dp[2] = 2;
    for i in 3..=n as usize {
        dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007;
    }

    dp[n as usize]
}

/// 剑指 Offer 10- II. 青蛙跳台阶问题
/// 动态规划，使用更少的内存
pub fn num_ways_v2(n: i32) -> i32 {
    let mut a = 1;
    let mut b = 1;
    let mut i = 0;
    while i < n {
        let sum = (a + b) % 1000000007;
        a = b;
        b = sum;
        i += 1;
    }
    a
}

use std::cmp::Ordering;
/// 剑指 Offer 11. 旋转数组的最小数字 https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/
pub fn min_array(numbers: Vec<i32>) -> i32 {
    let mut left = 0;
    let mut right = numbers.len() - 1;
    while left < right {
        let middle = left + (right - left) / 2;
        match numbers[middle].cmp(&numbers[right]) {
            Ordering::Greater => {
                left = middle + 1;
            }
            Ordering::Equal => {
                return *numbers.iter().min().unwrap();
            }
            Ordering::Less => {
                right = middle;
            }
        }
    }

    numbers[left]
}

use crate::solution::data_structures::lists::ListNode;
/// 剑指 Offer 18. 删除链表的节点 https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/
pub fn delete_node(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut dummy_head = Some(Box::new(ListNode {
        val: 0i32,
        next: head,
    }));

    let mut current = dummy_head.as_mut();
    while current.is_some() {
        if current.as_mut().unwrap().next.is_none()
            || current.as_mut().unwrap().next.as_mut().unwrap().val == val
        {
            break;
        }
        current = current.unwrap().next.as_mut();
    }

    if current.as_mut().unwrap().next.is_some() {
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

/// 剑指 Offer 18. 删除链表的节点
pub fn delete_node_v2(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut root = head;
    let mut head = &mut root;
    while let Some(node) = head {
        if node.val == val {
            *head = node.next.take();
            break;
        }
        head = &mut head.as_mut().unwrap().next;
    }
    root
}

/// 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面 https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/
pub fn exchange(nums: Vec<i32>) -> Vec<i32> {
    if nums.len() <= 1 {
        return nums;
    }
    let mut nums = nums;
    let mut i = 0;
    let mut j = nums.len() - 1;
    while i < j {
        while (i < j) && (nums[i] & 1) == 1 {
            i += 1;
        }
        while (i < j) && (nums[j] & 1) == 0 {
            j -= 1;
        }
        nums.swap(i, j);
    }

    nums
}

use std::cmp::Reverse;
use std::collections::BinaryHeap;
/// 剑指 Offer 40. 最小的k个数  https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/comments/
/// 方法1：小顶堆（大顶堆 + Reverse）
pub fn get_least_numbers(arr: Vec<i32>, k: i32) -> Vec<i32> {
    let len = arr.len();
    if len == 0 || (len as i32) == k {
        return arr;
    }
    if k == 0 {
        return vec![];
    }

    let mut nums = Vec::with_capacity(k as usize);
    let mut heap = BinaryHeap::new();
    for num in arr {
        heap.push(Reverse(num));
    }

    for _ in 0..k {
        if let Some(Reverse(curr)) = heap.pop() {
            nums.push(curr);
        }
    }

    nums
}

/// 剑指 Offer 40. 最小的k个数
/// 方法2：大顶堆
pub fn get_least_numbers_v2(arr: Vec<i32>, k: i32) -> Vec<i32> {
    let len = arr.len();
    if len == 0 || (len as i32) == k {
        return arr;
    }
    if k == 0 {
        return vec![];
    }
    let mut heap = BinaryHeap::new();
    for item in arr.iter().take(k as usize) {
        heap.push(*item);
    }

    for item in arr.iter().take(len).skip(k as usize) {
        if let Some(&top) = heap.peek() {
            if top > *item {
                heap.pop();
                heap.push(*item);
            }
        }
    }

    let mut nums = Vec::with_capacity(k as usize);
    for _ in 0..k {
        if let Some(val) = heap.pop() {
            nums.push(val);
        }
    }
    nums
}

/// 剑指 Offer 41. 数据流中的中位数  https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/
/// 注意：本题与主站 295 题相同：https://leetcode-cn.com/problems/find-median-from-data-stream/
struct MedianFinder {
    //max_heap来存储数据流中较小一半的值
    max_heap: BinaryHeap<i32>,
    //min_heap来存储数据流中较大一半的值
    min_heap: BinaryHeap<Reverse<i32>>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl MedianFinder {
    /** initialize your data structure here. */
    fn new() -> Self {
        MedianFinder {
            max_heap: BinaryHeap::new(),
            min_heap: BinaryHeap::<Reverse<i32>>::new(),
        }
    }

    fn add_num(&mut self, num: i32) {
        if self.max_heap.len() != self.min_heap.len() {
            self.min_heap.push(Reverse(num));
            if let Some(Reverse(peek)) = self.min_heap.pop() {
                self.max_heap.push(peek);
            }
        } else {
            self.max_heap.push(num);
            if let Some(peek) = self.max_heap.pop() {
                self.min_heap.push(Reverse(peek));
            }
        }
    }

    fn find_median(&self) -> f64 {
        if self.max_heap.len() != self.min_heap.len() {
            if let Some(Reverse(peek)) = self.min_heap.peek() {
                *peek as f64
            } else {
                0f64
            }
        } else {
            match (self.max_heap.peek(), self.min_heap.peek()) {
                (Some(peek1), Some(Reverse(peek2))) => (peek1 + (*peek2)) as f64 / 2.0f64,
                (_, _) => 0f64,
            }
        }
    }
}

/// 剑指 Offer 44. 数字序列中某一位的数字 https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/
pub fn find_nth_digit(n: i32) -> i32 {
    let (mut digit, mut start, mut count) = (1, 1_i64, 9_i64);
    let mut n = n as i64;
    while n > count {
        n -= count;
        digit += 1;
        start *= 10;
        count = digit * start * 9;
    }
    let num = start + (n - 1) / digit;
    let num = format!("{}", num);
    let index = ((n - 1) % digit) as usize;
    (num.chars().nth(index).unwrap() as i32) - ('0' as i32)
}

/// 剑指 Offer 45. 把数组排成最小的数 https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/
pub fn min_number(nums: Vec<i32>) -> String {
    let mut nums_str: Vec<String> = nums.iter().map(|&num| format!("{}", num)).collect();
    nums_str.sort_by(|x, y| {
        let xy = format!("{}{}", x, y).parse::<u64>().unwrap();
        let yx = format!("{}{}", y, x).parse::<u64>().unwrap();

        xy.cmp(&yx)
    });

    nums_str.iter().fold("".to_string(), |acc, num| acc + num)
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * let obj = MedianFinder::new();
 * obj.add_num(num);
 * let ret_2: f64 = obj.find_median();
 */
use std::collections::HashMap;
/// 剑指 Offer 50. 第一个只出现一次的字符  https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/
pub fn first_uniq_char(s: String) -> char {
    let mut map: HashMap<char, i32> = HashMap::new();
    for c in s.chars() {
        if let Some(count) = map.get_mut(&c) {
            *count += 1;
        } else {
            map.insert(c, 1);
        }
    }
    for c in s.chars() {
        if let Some(count) = map.get(&c) {
            if *count == 1 {
                return c;
            }
        };
    }

    ' '
}

/// 剑指 Offer 50. 第一个只出现一次的字符
pub fn first_uniq_char_v2(s: String) -> char {
    let mut map: HashMap<char, i32> = HashMap::new();
    for (i, c) in s.chars().enumerate() {
        match map.entry(c) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                e.insert(-1);
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(i as i32);
            }
        }
    }
    let mut first = s.len() as i32;
    for (key, &val) in map.iter() {
        if val != -1 && val < first {
            first = val;
        }
    }
    if first == s.len() as i32 {
        ' '
    } else {
        s.chars().skip(first as usize).take(1).next().unwrap()
    }
}

/// 剑指 Offer 53 - I. 在排序数组中查找数字 I https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/
/// 注意：本题与主站 34 题相同（仅返回值不同）：https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
pub fn search(nums: Vec<i32>, target: i32) -> i32 {
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
    if range[0] < 0 {
        0
    } else {
        range[1] - range[0] + 1
    }
}

/// 剑指 Offer 53 - II. 0～n-1中缺失的数字 https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/
/// 方法1：逐个遍历
pub fn missing_number(nums: Vec<i32>) -> i32 {
    for (idx, val) in nums.iter().enumerate() {
        if (idx as i32) != (*val) {
            return idx as i32;
        }
    }
    nums.len() as i32
}

/// 剑指 Offer 53 - II. 0～n-1中缺失的数字
/// 方法2：二分查找
pub fn missing_number_v2(nums: Vec<i32>) -> i32 {
    let (mut i, mut j) = (0, nums.len() - 1);
    while i <= j {
        let middle = (i + j) / 2;

        if nums[middle] == (middle as i32) {
            i = middle + 1;
        } else if middle > 0 {
            j = middle - 1;
        } else {
            break;
        }
    }

    i as i32
}

/// 剑指 Offer 56 - I. 数组中数字出现的次数 https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/
/// 方法1：分组异或
pub fn single_numbers(nums: Vec<i32>) -> Vec<i32> {
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

/// 剑指 Offer 57. 和为s的两个数字  https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut result = vec![];
    let (mut i, mut j) = (0, nums.len() - 1);
    while i < j {
        match (nums[i] + nums[j]).cmp(&target) {
            Ordering::Equal => {
                result.push(nums[i]);
                result.push(nums[j]);
                break;
            }
            Ordering::Greater => {
                j -= 1;
            }
            Ordering::Less => {
                i += 1;
            }
        }
    }
    result
}

/// 剑指 Offer 57 - II. 和为s的连续正数序列 https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/
/// 方法1：数学公式法，算法来源：https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/solution/shu-ju-jie-gou-he-suan-fa-hua-dong-chuan-74eb/
/// start,start+1,start+2,...,start+(n-1) = n * start + n*(n-1)/2 = target
/// 令 n*start = total，则 total = target - n * (n - 1) / 2
pub fn find_continuous_sequence(target: i32) -> Vec<Vec<i32>> {
    let mut result = vec![];
    let mut n = 2;
    loop {
        let total = target - n * (n - 1) / 2;
        if total <= 0 {
            break;
        }
        if total % n == 0 {
            let mut arr = vec![];
            let start = total / n;
            for i in 0..n {
                arr.push(start + i);
            }
            result.push(arr);
        }
        n += 1;
    }

    result.reverse();
    result
}

/// 剑指 Offer 58 - I. 翻转单词顺序 https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/
pub fn reverse_words(s: String) -> String {
    let mut words: Vec<&str> = s.split(' ').collect();
    let mut result = String::new();
    words.reverse();
    for word in words {
        // 注意：按照" "分割，结果中空字符串为""而不是" "
        if !word.is_empty() {
            result = format!("{} {}", result, word);
        }
    }
    result.trim().to_string()
}

/// 剑指 Offer 58 - I. 翻转单词顺序
pub fn reverse_words_v2(s: String) -> String {
    s.split_whitespace().rev().collect::<Vec<_>>().join(" ")
}

/// 剑指 Offer 58 - II. 左旋转字符串 https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/
pub fn reverse_left_words(s: String, n: i32) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    chars.rotate_left(n as usize);
    chars.iter().collect()
}

use std::collections::HashSet;
/// 剑指 Offer 61. 扑克牌中的顺子 https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/
/// 1.除大小王外，所有牌 无重复 ；
/// 2.设此 5 张牌中最大的牌为 max ，最小的牌为 min （大小王除外），则需满足：
pub fn is_straight(nums: Vec<i32>) -> bool {
    let mut reapt = HashSet::new();
    let (mut min, mut max) = (14, 0);
    for num in nums {
        if num == 0 {
            continue;
        }

        if reapt.contains(&num) {
            return false;
        } else {
            reapt.insert(num);
        }

        if num > max {
            max = num;
        }

        if num < min {
            min = num;
        }
    }

    max - min < 5
}

/// 剑指 Offer 62. 圆圈中最后剩下的数字  https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/
/// 约瑟夫环问题
pub fn last_remaining(n: i32, m: i32) -> i32 {
    let mut f = 0;
    let mut i = 2;
    while i != n + 1 {
        f = (m + f) % i;
        i += 1;
    }
    f
}
