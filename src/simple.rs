//！ 简单难度
//!  [简单]: https://leetcode-cn.com/problemset/all/?difficulty=EASY&page=1
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

/// 力扣（3. 无重复的字符串的最长子串）https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/submissions/
/// 方法一：滑动窗口
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
    let len = s.len();
    while i < len - 1 {
        let (len, next) = get_len(i, &s);
        if len > ret {
            ret = len;
        }
        i = next;
    }

    ret
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

/// 力扣（53. 最大子序和） https://leetcode-cn.com/problems/maximum-subarray/
/// 剑指 Offer 42. 连续子数组的最大和 https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/
/// 动态规划转移方程： f(i) = max{f(i−1)+nums[i],nums[i]}  
///  f(i) 代表以第 i 个数结尾的「连续子数组的最大和」
pub fn max_sub_array(nums: Vec<i32>) -> i32 {
    let mut prev = 0;
    let mut max_ans = nums[0];
    for x in nums {
        if prev > 0 {
            prev += x;
        } else {
            prev = x;
        }
        if prev > max_ans {
            max_ans = prev;
        }
    }
    max_ans
}

/// 力扣（53. 最大子序和）
pub fn max_sub_array_v2(nums: Vec<i32>) -> i32 {
    let len = nums.len();

    //dp[i] 表示以第 i 个元素结尾的最大子数组的和
    let mut dp = vec![0; len];
    dp[0] = nums[0];

    let mut max_ans = nums[0];
    for i in 1..len {
        dp[i] = max(dp[i - 1], nums[i]);
        max_ans = max(max_ans, dp[i]);
    }
    max_ans
}

/// 力扣（58. 最后一个单词的长度） https://leetcode-cn.com/problems/length-of-last-word/submissions/
/// 方法1：rsplitn()
pub fn length_of_last_word(s: String) -> i32 {
    let s_trim = s.trim_end();
    let words: Vec<&str> = s_trim.rsplitn(2, ' ').collect();
    words[0].len() as i32
}

/// 力扣（58. 最后一个单词的长度）
/// 方法2：双指针
pub fn length_of_last_word_v2(s: String) -> i32 {
    let chars: Vec<char> = s.chars().collect();
    let mut end = (chars.len() - 1) as i32;
    while end >= 0 && chars[end as usize] == ' ' {
        end -= 1;
    }
    if end < 0 {
        return 0;
    }

    let mut start = end;
    while start >= 0 && chars[start as usize] != ' ' {
        start -= 1;
    }

    end - start
}

use std::cell::RefCell;
use std::rc::Rc;
/// 力扣（70. 爬楼梯） https://leetcode-cn.com/problems/climbing-stairs/
/// 方法1：动态规划
pub fn climb_stairs(n: i32) -> i32 {
    if n <= 1 {
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
/// 剑指 Offer 63. 股票的最大利润  https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/
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

/// 力扣（169. 多数元素） https://leetcode-cn.com/problems/majority-element/
/// 剑指 Offer 39. 数组中出现次数超过一半的数字 https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/
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

/// 力扣（191. 位1的个数) https://leetcode-cn.com/problems/number-of-1-bits/
/// SWAR算法“计算汉明重量” https://baike.baidu.com/item/%E6%B1%89%E6%98%8E%E9%87%8D%E9%87%8F
pub fn hamming_weight_v1(n: u32) -> i32 {
    let mut i = n as i32;
    let const1 = 0x55555555;
    let const2 = 0x33333333;
    let const3 = 0x0F0F0F0F;
    let const4 = 0x01010101;

    // 奇数位 + 偶数位
    i = (i & const1) + ((i >> 1) & const1);
    i = (i & const2) + ((i >> 2) & const2);
    i = (i & const3) + ((i >> 4) & const3);
    // (i * 0x01010101)>>24 == ((i<<24)>>24) + ((i<<16)>>24) + ((i<<8)>>24) + ((i<<0)>>24)
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

/// 力扣（350. 两个数组的交集 II） https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/
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

// TODO 400

/// 力扣（401. 二进制手表） https://leetcode-cn.com/problems/binary-watch/
/// 方法1：空间换时间避免重复计算
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

/// 力扣（448. 找到所有数组中消失的数字）
pub fn find_disappeared_numbers_v2(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    let mut result = vec![];
    let len = nums.len();
    for i in 0..len {
        let x = ((nums[i] - 1) as usize) % len;
        nums[x] += (len as i32);
    }
    for i in 0..len {
        if nums[i] <= (len as i32) {
            result.push(i as i32 + 1);
        }
    }
    result
}

///力扣（485. 最大连续1的个数）https://leetcode-cn.com/problems/max-consecutive-ones/
pub fn find_max_consecutive_ones(nums: Vec<i32>) -> i32 {
    let mut max = 0;
    let mut len = 0;
    for num in nums {
        if num == 1 {
            len += 1;
            max = len.max(max);
        } else {
            len = 0;
        }
    }
    max
}

///力扣（485. 最大连续1的个数）
pub fn find_max_consecutive_ones_v2(nums: Vec<i32>) -> i32 {
    let ones_group = nums.as_slice().split(|&num| num == 0);
    ones_group.map(|ones| ones.len()).max().unwrap_or(0) as i32
}

/// TODO: 500

/// 520. 检测大写字母 https://leetcode-cn.com/problems/detect-capital/
pub fn detect_capital_use(word: String) -> bool {
    let mut word = word.chars();
    let first = word.next();
    if first.is_none() {
        return true;
    }
    let first = first.unwrap();

    if let Some(second) = word.next() {
        let res = word.try_fold(second, move |sd, x| {
            if sd.is_lowercase() && x.is_lowercase() {
                return Ok(sd);
            }

            if sd.is_uppercase() && x.is_uppercase() {
                return Ok(sd);
            }

            Err(())
        });

        if res.is_err() {
            return false;
        }
        if first.is_uppercase() {
            return true;
        }

        if first.is_lowercase() && second.is_lowercase() {
            return true;
        }

        false
    } else {
        true
    }
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
    // 找出最大值及其下标
    for (i, num) in nums.iter().enumerate() {
        if *num > max {
            max = *num;
            idx = i;
        }
    }
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

/// TODO 900

/// 力扣（977. 有序数组的平方）https://leetcode-cn.com/problems/squares-of-a-sorted-array/
/// 方法一：直接排序
pub fn sorted_squares(nums: Vec<i32>) -> Vec<i32> {
    let mut v: Vec<i32> = nums.iter().map(|x| x * x).collect();
    if nums[0] < 0 {
        v.sort_unstable();
    }
    v
}

/// 归并排序解法（参考官方java版本解法，主要区别在于i下标的范围）
/// 力扣（977. 有序数组的平方）
pub fn sorted_squares_v2(nums: Vec<i32>) -> Vec<i32> {
    let len = nums.len();
    // 非负数数组
    if nums[0] >= 0 {
        let mut v = Vec::<i32>::with_capacity(len);
        for &num in nums.iter().take(len) {
            v.push(num * num);
        }
        return v;
    }

    // 非正数数组
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
/// 同样地，我们可以使用两个指针分别指向位置 0 和 n−1，每次比较两个指针对应的数，选择较大的那个逆序放入答案并移动指针。
/// 这种方法无需处理某一指针移动至边界的情况，读者可以仔细思考其精髓所在。
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test() {
        let mut nums = vec![3, 2, 2, 3];
        let val = 3;

        let len = remove_element(&mut nums, val);

        let mut strs = Vec::new();
        strs.push(String::from("cdf"));
        //strs.push(String::from("acc"));
        strs.push(String::from("cd"));
        strs.push(String::from("cde"));
        //    strs.push(String::from("abscd"));
        dbg!(longest_common_prefix(strs));

        let count = climb_stairs(30);
        dbg!(count);

        let mut chars = Vec::<char>::new();
        chars.push('a');
        chars.push('b');
        chars.push('c');
        chars.push('d');
        //    chars.push('e');
        reverse_string(&mut chars);
        dbg!(chars);

        dbg!(generate(10));

        dbg!(get_row(33));

        let nums = vec![1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1];
        dbg!(find_max_consecutive_ones(nums));

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

        assert_eq!(length_of_longest_substring("dvdf".to_string()), 3);
        assert_eq!(length_of_longest_substring("abcabcbb".to_string()), 3);
        assert_eq!(length_of_longest_substring("bbbbb".to_string()), 1);
        assert_eq!(length_of_longest_substring("pwwkew".to_string()), 3);
        assert_eq!(length_of_longest_substring("c".to_string()), 1);
        assert_eq!(length_of_longest_substring("au".to_string()), 2);

        // for ch in '0'..='z'{
        //     dbg!("{} {}",ch, ch as u8);
        // }

        // let res = longest_palindrome(String::from("banana"));
        // dbg!("longest_palindrome res:{}",res);

        let heights = vec![1, 2, 4, 5, 3, 3];

        dbg!(height_checker(heights));
    }

    #[test]
    fn no_pass() {
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

        let mut nums = vec![0, 1, 0, 3, 0, 12, 14];
        move_zeroes(&mut nums);
        dbg!(nums);

        let mut nums2 = vec![0, 1, 0, 3, 0, 12, 14];
        move_zeroes_v2(&mut nums2);
        dbg!(nums2);

        let nums3 = vec![-2, 1, -3, 4, -1, 2, 1, -5, 4];
        let max_ans = max_sub_array(nums3);
        dbg!(max_ans);

        let nums = vec![2, 2, 1, 1, 1, 2, 2];
        let major = majority_element(nums);
        dbg!(major);

        let nums2 = vec![6, 5, 5];
        let major2 = majority_element_v2(nums2);
        dbg!(major2);

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
        let intersect_v2_result = intersect_v2(nums1, nums2);
        dbg!(intersect_v2_result);
    }

    #[test]
    fn test_200_plus() {
        dbg!(read_binary_watch(7));
    }

    #[test]
    fn test_list_node() {
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
