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

    println!("{:?}", spiral_order(matrix));

    let a = String::from("0");
    let b = String::from("0");
    println!("{}", add_binary(a, b));

    println!("{:?}", generate(10));
    let numbers = vec![2, 7, 11, 15];
    let target = 18;
    let result = two_sum2(numbers, target);
    println!("{:?}", result);

    let mut chars = Vec::<char>::new();
    chars.push('a');
    chars.push('b');
    chars.push('c');
    chars.push('d');
    //    chars.push('e');
    reverse_string(&mut chars);

    let nums = vec![1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1];
    let len = find_max_consecutive_ones(nums);
    println!("len:{}", len);

    let mut nums = Vec::<i32>::new();
    nums.push(1);
    nums.push(4);
    //nums.push(3);
    //nums.push(2);
    println!("sum:{}", array_pair_sum(nums));

    println!("index:{}", pivot_index(vec![1, 7, 3, 6, 5, 6]));

    println!("{}", dominant_index(vec![2]));

    let s = 7;
    let nums = vec![1,2,3,4,3,7,2,2];
    let min_len = min_sub_array_len(s,nums);
    println!("min_len:{}",min_len);
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

    println!("idx4:{}", idx); //这条表达式仅仅为了编译通过
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

        if s == false && t == false {
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
    let result = result.chars().rev().collect();
    result
}

/// 力扣（118. 杨辉三角） https://leetcode-cn.com/problems/pascals-triangle/
pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
    let rows = num_rows as usize;
    let mut result = Vec::<Vec<i32>>::with_capacity(rows);

    for row in 1..=rows {
        if row <= 2 {
            let r = vec![1; row];
            result.push(r);
        } else {
            if let Some(last_row) = result.last() {
                let mut r = vec![1; row];
                for i in 1..row - 1 {
                    r[i] = last_row[i - 1] + last_row[i];
                }
                result.push(r);
            }
        }
    }
    result
}

/// 力扣（167. 两数之和 II - 输入有序数组）https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/
pub fn two_sum2(numbers: Vec<i32>, target: i32) -> Vec<i32> {
    let mut result = Vec::<i32>::with_capacity(2);

    let mut index1 = 0;
    let mut index2 = numbers.len() - 1;
    while index2 >= 1 {
        let sum = numbers[index1] + numbers[index2];
        if sum < target {
            index1 += 1;
            continue;
        } else if sum > target {
            index2 -= 1;
            continue;
        } else {
            result.push((index1 + 1) as i32);
            result.push((index2 + 1) as i32);
            break;
        }
    }

    result
}

/// 力扣（344. 反转字符串） https://leetcode-cn.com/problems/reverse-string/
pub fn reverse_string(s: &mut Vec<char>) {
    let len = s.len();
    if len > 1 {
        let mut i = 0;
        let half = len / 2;
        while i < half {
            let x = s[i];
            s[i] = s[len - i - 1];
            s[len - i - 1] = x;
            i += 1;
        }
    }
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
        return max_temp;
    } else {
        return max;
    }
}

/// 力扣（561. 数组拆分 I） https://leetcode-cn.com/problems/array-partition-i/
pub fn array_pair_sum(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    if len % 2 != 0 {
        panic!("数组长度必须为偶数");
    }

    let mut nums_sort = Vec::<i32>::with_capacity(len);
    for i in 0..len {
        nums_sort.push(nums[i]);
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
///  力扣（209. 长度最小的子数组） https://leetcode-cn.com/problems/minimum-size-subarray-sum/
pub fn min_sub_array_len(s: i32, nums: Vec<i32>) -> i32 {
    let mut k = 0;
    let mut i = 0usize;
    let mut j = 0usize;
    let len = nums.len();
    let mut sum = 0;
    if len >= 1{
        sum = nums[0];
    }

    while i <= j && j < len {
        // 当 sum>=s 时，i++
        if sum >= s{
            if k == 0{
                k = j - i + 1;
            }else{
                let temp = j - i + 1;
                if temp < k{
                    k = temp;
                }
            }
            sum -= nums[i];
            i +=1;
        }else{
            // 当 sum<s 时，j++
            j +=1;
            if j < len{
                sum += nums[j];
            }else{
                break;
            }
        }
    }

    k as i32
}