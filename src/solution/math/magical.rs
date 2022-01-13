/// 453. 最小操作次数使数组元素相等 https://leetcode-cn.com/problems/minimum-moves-to-equal-array-elements/
pub fn min_moves(nums: Vec<i32>) -> i32 {
    let min = nums.iter().min().unwrap();
    let mut res = 0;
    nums.iter().fold(res, |acc, &num| {
        res += (num - min);
        res
    });
    res
}

/// 453. 最小操作次数使数组元素相等
pub fn min_moves_v2(nums: Vec<i32>) -> i32 {
    let mut min = nums[0];
    let mut sum = 0;
    let mut len = 0;
    for num in nums {
        sum += num;
        if num < min {
            min = num;
        }
        len += 1;
    }
    sum - min * len
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn magic() {
        for n in 1..=9 {
            dbg!(142857 * n);
        }
        let nums = vec![1, 2, 3];
        dbg!(min_moves(nums));
    }
}
