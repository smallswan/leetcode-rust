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
        return merge_vec[(len1 + len2) / 2] as f64;
    } else {
        let t2 = (len1 + len2) / 2;
        return ((merge_vec[t2 - 1] + merge_vec[t2]) as f64) / 2.0;
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
        return kth_elem(&nums1, &nums2, medium_idx + 1);
    } else {
        let medium_idx1 = total_len / 2 - 1;
        let medium_idx2 = total_len / 2;
        return (kth_elem(&nums1, &nums2, medium_idx1 + 1)
            + kth_elem(&nums1, &nums2, medium_idx2 + 1))
            / 2.0;
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

#[test]
fn hard() {
    let nums1: Vec<i32> = vec![1, 3];
    let nums2: Vec<i32> = vec![2];
    let median_num = find_median_sorted_arrays(nums1, nums2);
    println!("median_num v1:{}", median_num);

    let mut nums3: Vec<i32> = vec![1, 3];
    let mut nums4: Vec<i32> = vec![2];
    let median_num = find_median_sorted_arrays_v2(nums3, nums4);
    println!("median_num v2:{}", median_num);
}
