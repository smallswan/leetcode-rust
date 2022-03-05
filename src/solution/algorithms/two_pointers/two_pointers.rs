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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_reverse() {
        assert_eq!(trap(vec![4, 2, 0, 3, 2, 5]), 9);
        assert_eq!(trap_v2(vec![4, 2, 0, 3, 2, 5]), 9);
    }
}
