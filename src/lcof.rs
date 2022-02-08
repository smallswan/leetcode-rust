//! 剑指Offer
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
