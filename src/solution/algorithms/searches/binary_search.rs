use std::cmp::Ordering;
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_search() {
        let nums = vec![-1, 0, 3, 5, 9, 12];
        let target = 9;

        dbg!(search(nums, target));

        let nums = vec![1, 0, 3, 5, 9, 12];

        dbg!(search_v2(nums, 2));
    }
}
