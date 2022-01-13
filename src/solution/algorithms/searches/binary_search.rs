use std::cmp::Ordering;
/// 33. 搜索旋转排序数组 https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
/// 方法一：二分查找
pub fn search_in_rotated_sorted_array(nums: Vec<i32>, target: i32) -> i32 {
    let len = nums.len();
    if len == 0 {
        return -1;
    }
    if len == 1 {
        if nums[0] == target {
            return 0;
        } else {
            return -1;
        }
    }

    let (mut left, mut right) = (0, len - 1);
    while left <= right {
        let mut middle = (left + right) / 2;
        if nums[middle] == target {
            return middle as i32;
        }
        if nums[0] <= nums[middle] {
            if nums[0] <= target && target < nums[middle] {
                right = middle - 1;
            } else {
                left = middle + 1;
            }
        } else {
            if nums[middle] < target && target <= nums[right] {
                left = middle + 1;
            } else {
                right = middle - 1;
            }
        }
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

/// 力扣（34. 在排序数组中查找元素的第一个和最后一个位置) https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
/// 先用二分查找算法找到target的下标，然后向左右两边继续查找
pub fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
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

    range
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
        let sorted_nums = vec![1, 3, 5, 6];
        let target = 4;
        let nums = vec![8, 8, 8, 8, 8, 8];
        let range = search_range(nums, 7);
        dbg!("range {:?}", range);

        dbg!(search_insert(sorted_nums, target));

        dbg!(search_insert_v2(vec![1, 3, 5, 6], 7));

        let nums = vec![-1, 0, 3, 5, 9, 12];
        let target = 9;

        dbg!(search(nums, target));

        let nums = vec![1, 0, 3, 5, 9, 12];

        dbg!(search_v2(nums, 2));
    }
}
