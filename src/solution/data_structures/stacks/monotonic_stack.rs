//! 单调栈（monotonic stack）
//! https://leetcode-cn.com/tag/monotonic-stack/problemset/

use std::cmp::min;
use std::collections::VecDeque;
/// 42. 接雨水 https://leetcode-cn.com/problems/trapping-rain-water/
pub fn trap(height: Vec<i32>) -> i32 {
    let mut result = 0;
    let mut n = height.len();
    // 维护一个单调栈，单调栈存储的是下标，满足从栈底到栈顶的下标对应的数组 height 中的元素递减。
    let mut stack: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        while !stack.is_empty() && height[i] > height[*stack.back().unwrap()] {
            // stack单调递减，则height[top]为高度最低的地方，height[left]、height[top]、height[i]构成一个接雨水的区域
            let top = stack.pop_back().unwrap();
            if stack.is_empty() {
                break;
            }
            let left = *stack.back().unwrap();
            let curr_width = (i - left - 1) as i32;
            let curr_height = min(height[left], height[i]) - height[top];
            result += curr_width * curr_height;
        }
        stack.push_back(i);
    }

    result
}

/// 84. 柱状图中最大的矩形 https://leetcode-cn.com/problems/largest-rectangle-in-histogram/
pub fn largest_rectangle_area(heights: Vec<i32>) -> i32 {
    let mut result = 0;
    let mut stack_base = Vec::<(i32, i32)>::new();
    let mut stack_top = (-1, 0);

    for item in (0..).zip(heights) {
        loop {
            if item.1 <= stack_top.1 {
                if let Some(new_top) = stack_base.pop() {
                    result = result.max((item.0 - new_top.0 - 1) * stack_top.1);
                    stack_top = new_top;
                } else {
                    result = result.max(item.0 * stack_top.1);
                    stack_top = item;

                    break;
                }
            } else {
                stack_base.push(stack_top);
                stack_top = item;

                break;
            }
        }
    }

    let right = stack_top.0;
    while let Some(new_top) = stack_base.pop() {
        result = result.max((right - new_top.0) * stack_top.1);
        stack_top = new_top;
    }

    result
}

/// 496. 下一个更大元素 I https://leetcode-cn.com/problems/next-greater-element-i/
pub fn next_greater_element(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    use std::collections::HashMap;
    use std::collections::VecDeque;
    let mut map = HashMap::new();
    // VecDeque模拟单调栈（栈底最大，栈顶最小），front 栈底，back 栈顶
    let mut stack = VecDeque::new();
    for num in nums2.iter().rev() {
        while let Some(top) = stack.back() {
            if num > top {
                stack.pop_back();
            } else {
                break;
            }
        }

        let next_element = if stack.is_empty() {
            -1
        } else {
            *stack.back().unwrap()
        };

        map.insert(num, next_element);
        stack.push_back(*num);
    }

    let mut res = Vec::with_capacity(nums1.len());
    for num in nums1 {
        if let Some(&next_element) = map.get(&num) {
            res.push(next_element);
        }
    }

    res
}

/// 力扣（496. 下一个更大元素 I）
pub fn next_greater_element_v2(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    // Vec模拟单调栈（栈底最大，栈顶最小），first 栈底，last 栈顶
    let mut stack = Vec::new();
    for num in nums2.iter().rev() {
        while let Some(top) = stack.last() {
            if num > top {
                stack.pop();
            } else {
                break;
            }
        }

        let next_element = if stack.is_empty() {
            -1
        } else {
            *stack.last().unwrap()
        };

        map.insert(num, next_element);
        stack.push(*num);
    }

    let mut res = Vec::with_capacity(nums1.len());
    for num in nums1 {
        if let Some(&next_element) = map.get(&num) {
            res.push(next_element);
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_monotonic() {
        assert_eq!(trap(vec![4, 2, 0, 3, 2, 5]), 9);

        dbg!(next_greater_element(vec![4, 1, 2], vec![1, 3, 4, 2]));
    }
}
