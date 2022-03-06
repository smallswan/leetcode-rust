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

        let nums = vec![3, 2];
        let result = majority_element(nums);
        dbg!("majority_element: {:?}", result);

        let digits = String::from("234");
        let combination = letter_combinations(digits);
        dbg!("combination1: {:?}", combination);

        let digits = String::from("234");
        let combination = letter_combinations_v2(digits);
        dbg!("combination2: {:?}", combination);
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
    }

    fn test_equal_substring() {
        dbg!(equal_substring("abcd".to_string(), "bcdf".to_string(), 3));
    }
}
