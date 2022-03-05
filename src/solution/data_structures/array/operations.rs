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

/// 189. 轮转数组 https://leetcode-cn.com/problems/rotate-array/
pub fn rotate(nums: &mut Vec<i32>, k: i32) {
    let len = nums.len();
    nums.rotate_right(k as usize % len)
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
}
