use std::cmp::min;

//！ 简单难度

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
        for j in i..s.len() {
            let index = s[j] as usize;
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

    // println!("{:?}", merge_vec);

    //
    let t1 = (len1 + len2) % 2;
    if t1 == 1 {
        return merge_vec[(len1 + len2) / 2] as f64;
    } else {
        let t2 = (len1 + len2) / 2;
        return ((merge_vec[t2 - 1] + merge_vec[t2]) as f64) / 2.0;
    }
}

/// 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
/// 这里的 "/" 表示整除
/// nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
/// nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
/// 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
/// 这样 pivot 本身最大也只能是第 k-1 小的元素
/// 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
/// 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
/// 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
fn kth_elem(nums1: &Vec<i32>, nums2: &Vec<i32>, k: usize) -> f64 {
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
/// 二分查找
pub fn find_median_sorted_arrays_v2(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let len1 = nums1.len();
    let len2 = nums2.len();

    let total_len = len1 + len2;
    if total_len % 2 == 1 {
        let medium_idx = total_len / 2;
        return kth_elem(&nums1, &nums2, medium_idx + 1);
    } else {
        let medium_idx1 = total_len / 2 - 1;
        let medium_idx2 = total_len / 2;
        return (kth_elem(&nums1, &nums2, medium_idx1 + 1)
            + kth_elem(&nums1, &nums2, medium_idx2 + 1))
            / 2.0;
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

/// 反转一半的数字
pub fn is_palindrome_v2(x: i32) -> bool {
    if x < 0 || (x % 10 == 0 && x != 0) {
        return false;
    }
    let mut y = x;
    let mut reverted_num = 0;
    while y > reverted_num {
        reverted_num = reverted_num * 10 + y % 10;
        y /= 10;
    }
    return y == reverted_num || y == reverted_num / 10;
}

/// 力扣（88. 合并两个有序数组） https://leetcode-cn.com/problems/merge-sorted-array/
pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    let mut m = m;
    let mut index: usize = 0;
    for i in 0..(n as usize) {
        while (index < m as usize) && nums1[index] <= nums2[i] {
            index += 1;
        }

        if index < (m as usize) {
            for j in (index + 1..nums1.len()).rev() {
                nums1[j] = nums1[j - 1];
            }
            m += 1;
        }
        nums1[index] = nums2[i];
        index += 1;
    }
}

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

    // println!("merge_v2 p2: {:?}",p2);

    for idx in 0..(p2 + 1) as usize {
        nums1[idx] = nums2[idx];
    }
}

#[test]
fn simple_test() {
    println!("{}", is_palindrome(121));
    println!("{}", is_palindrome(-121));
    println!("{}", is_palindrome(10));
    println!("{}", is_palindrome(1));

    assert_eq!(length_of_longest_substring("dvdf".to_string()), 3);
    assert_eq!(length_of_longest_substring("abcabcbb".to_string()), 3);
    assert_eq!(length_of_longest_substring("bbbbb".to_string()), 1);
    assert_eq!(length_of_longest_substring("pwwkew".to_string()), 3);
    assert_eq!(length_of_longest_substring("c".to_string()), 1);
    assert_eq!(length_of_longest_substring("au".to_string()), 2);

    // for ch in '0'..='z'{
    //     println!("{} {}",ch, ch as u8);
    // }

    let nums1: Vec<i32> = vec![1, 3];
    let nums2: Vec<i32> = vec![2];
    let median_num = find_median_sorted_arrays(nums1, nums2);
    println!("median_num v1:{}", median_num);

    let mut nums3: Vec<i32> = vec![1, 3];
    let mut nums4: Vec<i32> = vec![2];
    let median_num = find_median_sorted_arrays_v2(nums3, nums4);
    println!("median_num v2:{}", median_num);

    let mut nums1: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
    let mut nums2: Vec<i32> = vec![2, 5, 6];
    merge(&mut nums1, 3, &mut nums2, 3);
    println!("{:?}", nums1);

    let mut nums3: Vec<i32> = vec![7, 8, 9, 0, 0, 0];
    let mut nums4: Vec<i32> = vec![2, 5, 6];
    merge_v2(&mut nums3, 3, &mut nums4, 3);
    println!("{:?}", nums3);
}
