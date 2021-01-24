use std::collections::HashMap;

#[test]
fn unit_test() {
    let nums = vec![2, 7, 2, 11];
    let result = two_sum(nums, 9);
    println!("{:?}", result);

    // vec!["abc","abs","abd"
    let mut strs = Vec::new();
    strs.push(String::from("cdf"));
    //strs.push(String::from("acc"));
    strs.push(String::from("cd"));
    strs.push(String::from("cde"));
    //    strs.push(String::from("abscd"));
    println!("common:{}", longest_common_prefix(strs));

    let mut nums = vec![3, 2, 2, 3];
    let val = 3;

    let len = remove_element(&mut nums, val);

    let haystack = String::from("aaacaaab");
    let needle = String::from("aaab");
    println!("idx:{}", str_str(haystack, needle));

    println!("{:?}", plus_one(vec![9, 1, 9]));

    //vec![[1,2,3],[4,5,6],[7,8,9]]
    let mut matrix = Vec::<Vec<i32>>::new();
    matrix.push(vec![1, 2, 3, 4]);
    matrix.push(vec![5, 6, 7, 8]);
    matrix.push(vec![9, 10, 11, 12]);

    println!("{:?}", matrix);
    //    println!("{:?}",find_diagonal_order(matrix));

    println!("spiral_order: {:?}", spiral_order(matrix));

    let a = String::from("0");
    let b = String::from("0");
    println!("{}", add_binary(a, b));

    println!("{:?}", generate(10));

    println!("{:?}", get_row(33));

    let numbers = vec![2, 7, 11, 15];
    let target = 18;
    let result = two_sum2(numbers, target);
    println!("two_sum2 : {:?}", result);

    let mut chars = Vec::<char>::new();
    chars.push('a');
    chars.push('b');
    chars.push('c');
    chars.push('d');
    //    chars.push('e');
    reverse_string(&mut chars);
    println!("reverse_string:{:?}", chars);

    let nums = vec![1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1];
    let len = find_max_consecutive_ones(nums);
    println!("len:{}", len);

    let mut nums = Vec::<i32>::new();
    nums.push(1);
    nums.push(4);
    nums.push(3);
    nums.push(2);
    println!("sum:{}", array_pair_sum(nums));

    println!("index:{}", pivot_index(vec![1, 7, 3, 6, 5, 6]));

    println!("{}", dominant_index(vec![2]));

    let s = 7;
    let nums = vec![1, 2, 3, 4, 3, 7, 2, 2];
    let min_len = min_sub_array_len(s, nums);
    println!("min_len:{}", min_len);

    let s = String::from("LEETCODEISHIRING");
    let zz = convert(s, 4);
    println!("{}", zz);

    let mut rotate_vec = vec![1, 2];
    rotate(&mut rotate_vec, 1);

    println!("{:?}", rotate_vec);

    let roman_numbers = String::from("MCMXCIV");
    println!("roman_to_int()={}", roman_to_int(roman_numbers));
}

#[test]
fn simple() {
    let count = climb_stairs(30);
    println!("count:{}", count);

    let new_x = reverse(132);
    println!("new_x:{}", new_x);
    let new_x = reverse(-1999999999);
    println!("new_x:{}", new_x);

    let sorted_nums = vec![1, 3, 5, 6];
    let target = 4;
    let idx = search_insert(sorted_nums, target);
    println!("idx:{}", idx);

    let intervals: Vec<Vec<i32>> = vec![vec![1, 3], vec![2, 6], vec![8, 10], vec![15, 18]];
    let merge_intervals = merge(intervals);
    println!("{:?}", merge_intervals);

    // let res = longest_palindrome(String::from("banana"));
    // println!("longest_palindrome res:{}",res);

    let heights = vec![1, 2, 4, 5, 3, 3];
    let move_person = height_checker(heights);
    println!("height_checker move_person:{}", move_person);

    let ip = String::from("2001:0db8:85a3:0:0:8A2E:0370:7334:");

    let ret = valid_ip_address2(ip);
    println!("{}", ret);
}
///
pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
    let len = nums.len();
    let mut idx = 0;
    while idx < len {
        if target == nums[idx] || target < nums[idx] {
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

///
pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    if intervals.len() == 1 {
        return intervals;
    }
    let mut merge_vec = vec![vec![]; intervals.len()];
    merge_vec.clone_from_slice(&intervals);

    merge_vec
}

///
pub fn longest_palindrome(s: String) -> String {
    //    let s_chars = s.chars();
    let s_bytes = s.as_bytes();
    let mut res_bytes = vec!['#' as u8; 2 * s_bytes.len() + 1];

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

    let new_s = String::from_utf8(res_bytes).unwrap();
    new_s
}

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
    for i in 1..101 {
        while arr[i] > 0 {
            if heights[j] != (i as i32) {
                count += 1;
            }
            j += 1;
            arr[i] -= 1;
        }
    }
    count
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
        y = y / 10;
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

/// 力扣（13. 罗马数字转整数） https://leetcode-cn.com/problems/roman-to-integer/
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

    //println!("idx4:{}", idx); //这条表达式仅仅为了编译通过
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

/// 力扣（28. 实现 strStr()）  https://leetcode-cn.com/problems/implement-strstr/
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
                println!("不可能发生这种情况");
            }
        }
    }
    result
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

/// 力扣（118. 杨辉三角） https://leetcode-cn.com/problems/pascals-triangle/
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

use std::cmp::Ordering;

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
        // if sum < target {
        //     index1 += 1;
        //     continue;
        // } else if sum > target {
        //     index2 -= 1;
        //     continue;
        // } else {
        //     result.push((index1 + 1) as i32);
        //     result.push((index2 + 1) as i32);
        //     break;
        // }
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

/// 力扣（344. 反转字符串） https://leetcode-cn.com/problems/reverse-string/
pub fn reverse_string(s: &mut Vec<char>) {
    let len = s.len();
    if len > 1 {
        let mut i = 0;
        let half = len / 2;
        while i < half {
            // let x = s[i];
            // s[i] = s[len - i - 1];
            // s[len - i - 1] = x;
            s.swap(i, len - i - 1);
            i += 1;
        }
    }
}

/// 力扣（468. 验证IP地址）  https://leetcode-cn.com/problems/validate-ip-address/
use std::net::IpAddr;
pub fn valid_ip_address(ip: String) -> String {
    match ip.parse::<IpAddr>() {
        Ok(IpAddr::V4(x)) => {
            let array: Vec<Vec<char>> = ip.split('.').map(|x| x.chars().collect()).collect();
            for i in 0..array.len() {
                if (array[i][0] == '0' && array[i].len() > 1) {
                    return String::from("Neither");
                }
            }
            String::from("IPv4")
        }
        Ok(IpAddr::V6(_)) => {
            let array: Vec<Vec<char>> = ip.split(':').map(|x| x.chars().collect()).collect();
            for i in 0..array.len() {
                if array[i].len() == 0 {
                    return String::from("Neither");
                }
            }
            String::from("IPv6")
        }
        _ => String::from("Neither"),
    }
}

/// 使用分治法解 力扣（468. 验证IP地址）  https://leetcode-cn.com/problems/validate-ip-address/
pub fn valid_ip_address2(ip: String) -> String {
    if ip.chars().filter(|ch| *ch == '.').count() == 3 {
        //println!("valid_ipv4_address..");
        return valid_ipv4_address(ip);
    } else if ip.chars().filter(|ch| *ch == ':').count() == 7 {
        //println!("valid_ipv6_address..");
        return valid_ipv6_address(ip);
    } else {
        return String::from("Neither");
    }
}

fn valid_ipv4_address(ip: String) -> String {
    let array: Vec<Vec<char>> = ip.split('.').map(|x| x.chars().collect()).collect();
    for i in 0..array.len() {
        //Validate integer in range (0, 255):
        //1. length of chunk is between 1 and 3
        if array[i].len() == 0 || array[i].len() > 3 {
            return String::from("Neither");
        }
        //2. no extra leading zeros
        if (array[i][0] == '0' && array[i].len() > 1) {
            return String::from("Neither");
        }
        //3. only digits are allowed
        for ch in &array[i] {
            if !(*ch).is_digit(10) {
                return String::from("Neither");
            }
        }
        //4. less than 255
        let num_str: String = array[i].iter().collect();
        let num = num_str.parse::<u16>().unwrap();
        if num > 255 {
            return String::from("Neither");
        }
    }
    "IPv4".to_string()
}

fn valid_ipv6_address(ip: String) -> String {
    let array: Vec<Vec<char>> = ip.split(':').map(|x| x.chars().collect()).collect();
    for i in 0..array.len() {
        let num = &array[i];
        if num.len() == 0 || num.len() > 4 {
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

/// 力扣（561. 数组拆分 I） https://leetcode-cn.com/problems/array-partition-i/
pub fn array_pair_sum(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    if len % 2 != 0 {
        panic!("数组长度必须为偶数");
    }

    let mut nums_sort = Vec::<i32>::with_capacity(len);
    // for i in 0..len {
    //     nums_sort.push(nums[i]);
    // }
    for num in nums.iter().take(len) {
        nums_sort.push(*num);
    }
    nums_sort.sort();

    let mut sum = 0;
    for i in 0..len / 2 {
        sum += nums_sort[2 * i];
    }

    sum
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
