//! 双指针（two-pointers）
//! https://leetcode-cn.com/tag/two-pointers/problemset/

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

/// 42. 接雨水 https://leetcode-cn.com/problems/trapping-rain-water/
/// 方法：双指针
pub fn trap(height: Vec<i32>) -> i32 {
    let mut result = 0;
    let mut iter = height.into_iter();

    if let (Some(mut left), Some(mut right)) = (iter.next(), iter.next_back()) {
        'outer: loop {
            if left < right {
                for middle in &mut iter {
                    if middle < left {
                        result += left - middle;
                    } else {
                        left = middle;

                        continue 'outer;
                    }
                }
            } else {
                while let Some(middle) = iter.next_back() {
                    if middle < right {
                        result += right - middle;
                    } else {
                        right = middle;

                        continue 'outer;
                    }
                }
            }

            break;
        }
    }

    result
}

use std::cmp::max;
/// 42. 接雨水
/// 方法：双指针
pub fn trap_v2(height: Vec<i32>) -> i32 {
    let mut result = 0;
    let (mut left, mut right) = (0, height.len() - 1);
    let (mut left_max, mut right_max) = (0, 0);
    while left < right {
        left_max = max(left_max, height[left]);
        right_max = max(right_max, height[right]);
        if height[left] < height[right] {
            result += left_max - height[left];
            left += 1;
        } else {
            result += right_max - height[right];
            right -= 1;
        }
    }
    result
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
    }

    result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_() {
        let mut nums = vec![1, 3, 3, 3, 5, 5, 9, 9, 9, 9];

        dbg!(remove_duplicates(&mut nums));

        let mut nums = vec![1, 3, 3, 3, 5, 5, 9, 9, 9, 9];

        dbg!(remove_duplicates_v2(&mut nums));

        let mut nums = vec![4, 5, 2, 6, 3, 1];
        next_permutation(&mut nums);
        println!("nums: {:?}", nums);

        let mut nums1: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let mut nums2: Vec<i32> = vec![2, 5, 6];
        merge(&mut nums1, 3, &mut nums2, 3);
        dbg!(nums1);

        let mut nums3: Vec<i32> = vec![7, 8, 9, 0, 0, 0];
        let mut nums4: Vec<i32> = vec![2, 5, 6];
        merge_v2(&mut nums3, 3, &mut nums4, 3);
        dbg!(nums3);

        let nums1 = vec![4, 9, 5, 1];
        let nums2 = vec![9, 2, 4, 10, 5];
        let intersect_result = intersect(nums1, nums2);
        dbg!(intersect_result);
    }
    #[test]
    fn test_sum() {
        let numbers = vec![2, 7, 11, 15];
        let target = 18;

        dbg!(two_sum2(numbers, target));

        let heights = vec![1, 8, 6, 2, 5, 4, 8, 3, 7];
        dbg!("max area : {}", max_area(heights));

        let heights = vec![4, 3, 2, 1, 4];
        dbg!("max area : {}", max_area(heights));

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
    }

    #[test]
    fn test_reverse() {
        assert_eq!(trap(vec![4, 2, 0, 3, 2, 5]), 9);
        assert_eq!(trap_v2(vec![4, 2, 0, 3, 2, 5]), 9);
    }
}
