//! 贪心算法（greedy）
//! https://leetcode-cn.com/tag/greedy/problemset/

/// 45. 跳跃游戏 II https://leetcode-cn.com/problems/jump-game-ii/
/// 方法一：反向查找出发位置
pub fn jump(nums: Vec<i32>) -> i32 {
    let mut position = nums.len() - 1;
    let mut steps = 0;
    while position > 0 {
        for i in 0..position {
            if i + (nums[i] as usize) >= position {
                position = i;
                steps += 1;
                break;
            }
        }
    }

    steps
}

use std::cmp::max;
/// 45. 跳跃游戏 II
/// 方法二：正向查找可到达的最大位置
pub fn jump_v2(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let mut end = 0;
    let mut max_position = 0;
    let mut steps = 0;
    for i in 0..len - 1 {
        max_position = max(max_position, i + (nums[i] as usize));
        if i == end {
            end = max_position;
            steps += 1;
        }
    }

    steps
}

/// 55. 跳跃游戏 https://leetcode-cn.com/problems/jump-game/
pub fn can_jump(nums: Vec<i32>) -> bool {
    let len = nums.len();
    let mut end = 0;
    let mut max_position = 0;
    for i in 0..len {
        if i <= max_position {
            max_position = max(max_position, i + (nums[i] as usize));
            if max_position >= len - 1 {
                return true;
            }
        }
    }

    false
}

/// 力扣（561. 数组拆分 I） https://leetcode-cn.com/problems/array-partition-i/
pub fn array_pair_sum(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    if len % 2 != 0 {
        panic!("数组长度必须为偶数");
    }

    let mut nums_sort = nums;
    nums_sort.sort_unstable();

    let mut sum = 0;
    for i in 0..len / 2 {
        sum += nums_sort[2 * i];
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let mut nums = Vec::<i32>::new();
        nums.push(1);
        nums.push(4);
        nums.push(3);
        nums.push(2);
        dbg!(array_pair_sum(nums));
    }
}
