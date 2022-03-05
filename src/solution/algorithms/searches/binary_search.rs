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

    let t1 = (len1 + len2) % 2;
    if t1 == 1 {
        merge_vec[(len1 + len2) / 2] as f64
    } else {
        let t2 = (len1 + len2) / 2;
        ((merge_vec[t2 - 1] + merge_vec[t2]) as f64) / 2.0
    }
}

/// 力扣（4. 寻找两个正序数组的中位数）
/// 二分查找
pub fn find_median_sorted_arrays_v2(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let len1 = nums1.len();
    let len2 = nums2.len();

    let total_len = len1 + len2;
    if total_len % 2 == 1 {
        let medium_idx = total_len / 2;
        kth_elem(&nums1, &nums2, medium_idx + 1)
    } else {
        let medium_idx1 = total_len / 2 - 1;
        let medium_idx2 = total_len / 2;
        (kth_elem(&nums1, &nums2, medium_idx1 + 1) + kth_elem(&nums1, &nums2, medium_idx2 + 1))
            / 2.0
    }
}

use std::cmp::min;

/// 主要思路：要找到第 k (k>1) 小的元素，那么就取 pivot1 = nums1[k/2-1] 和 pivot2 = nums2[k/2-1] 进行比较
/// 这里的 "/" 表示整除
/// nums1 中小于等于 pivot1 的元素有 nums1[0 .. k/2-2] 共计 k/2-1 个
/// nums2 中小于等于 pivot2 的元素有 nums2[0 .. k/2-2] 共计 k/2-1 个
/// 取 pivot = min(pivot1, pivot2)，两个数组中小于等于 pivot 的元素共计不会超过 (k/2-1) + (k/2-1) <= k-2 个
/// 这样 pivot 本身最大也只能是第 k-1 小的元素
/// 如果 pivot = pivot1，那么 nums1[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums1 数组
/// 如果 pivot = pivot2，那么 nums2[0 .. k/2-1] 都不可能是第 k 小的元素。把这些元素全部 "删除"，剩下的作为新的 nums2 数组
/// 由于我们 "删除" 了一些元素（这些元素都比第 k 小的元素要小），因此需要修改 k 的值，减去删除的数的个数
fn kth_elem(nums1: &[i32], nums2: &[i32], k: usize) -> f64 {
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
        let nums1: Vec<i32> = vec![1, 3];
        let nums2: Vec<i32> = vec![2];
        let median_num = find_median_sorted_arrays(nums1, nums2);
        dbg!("median_num v1:{}", median_num);

        let mut nums3: Vec<i32> = vec![1, 3];
        let mut nums4: Vec<i32> = vec![2];
        let median_num = find_median_sorted_arrays_v2(nums3, nums4);
        dbg!("median_num v2:{}", median_num);

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
