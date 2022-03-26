use std::collections::HashMap;
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

/// 56. 合并区间 https://leetcode-cn.com/problems/merge-intervals/
pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut intervals = intervals;

    intervals.sort_unstable_by_key(|interval| interval[0]);

    let mut result = Vec::new();
    let mut iter = intervals.into_iter();

    result.push(iter.next().unwrap());

    for interval in iter {
        let previous_end = result.last_mut().unwrap().last_mut().unwrap();

        if interval[0] <= *previous_end {
            *previous_end = (*previous_end).max(interval[1]);
        } else {
            result.push(interval);
        }
    }

    result
}

/// 189. 轮转数组 https://leetcode-cn.com/problems/rotate-array/
pub fn rotate(nums: &mut Vec<i32>, k: i32) {
    let len = nums.len();
    nums.rotate_right(k as usize % len)
}

/// 303. 区域和检索 - 数组不可变 https://leetcode-cn.com/problems/range-sum-query-immutable/
struct NumArray {
    data: Vec<i32>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl NumArray {
    fn new(nums: Vec<i32>) -> Self {
        NumArray { data: nums }
    }

    fn sum_range(&self, left: i32, right: i32) -> i32 {
        let (mut left, mut right) = (left as usize, right as usize);
        let mut result = 0;
        while left <= right {
            result += self.data[left];
            left += 1;
        }
        result
    }
}

use std::cmp::Ordering;
/// 941. 有效的山脉数组  https://leetcode-cn.com/problems/valid-mountain-array/
pub fn valid_mountain_array(arr: Vec<i32>) -> bool {
    // slice pattern
    if let [first, second, ref rest @ ..] = *arr {
        if first < second {
            let mut prev = second;
            let mut iter = rest.iter().copied();

            loop {
                if let Some(num) = iter.next() {
                    match num.cmp(&prev) {
                        Ordering::Less => {
                            prev = num;

                            break;
                        }
                        Ordering::Equal => return false,
                        Ordering::Greater => prev = num,
                    }
                } else {
                    return false;
                }
            }

            for num in iter {
                if num < prev {
                    prev = num;
                } else {
                    return false;
                }
            }

            true
        } else {
            false
        }
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector() {
        let nums = vec![2, 7, 2, 11];
        let result = two_sum(nums, 9);
        dbg!(result);
    }
    #[test]
    fn test_rotate() {
        let mut nums = vec![1, 2, 3, 4, 5, 6, 7];
        let k = 3;
        rotate(&mut nums, k);
    }

    #[test]
    fn test_array() {
        let array = NumArray::new(vec![1, 2, 3, 4, 5, 6, 7]);
        dbg!(array.sum_range(2, 5));
    }
}
