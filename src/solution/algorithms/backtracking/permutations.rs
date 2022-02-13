pub struct Solution;

impl Solution {
    /// 46. 全排列  https://leetcode-cn.com/problems/permutations/
    pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let len = nums.len();
        let mut result: Vec<Vec<i32>> = Vec::new();
        if len == 0 {
            return result;
        }
        let mut used_vec = vec![false; len];
        let mut path = Vec::<i32>::new();
        Solution::dfs(&nums, len, 0, &mut path, &mut used_vec, &mut result);
        result
    }

    /// 47. 全排列 II https://leetcode-cn.com/problems/permutations-ii/
    /// 代码来源：https://github.com/aylei/leetcode-rust/blob/master/src/solution/s0047_permutations_ii.rs
    pub fn permute_unique(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut nums = nums;
        nums.sort_unstable();
        Solution::permute_unique_backtrace(nums)
    }

    fn permute_unique_backtrace(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        if nums.len() <= 1 {
            return vec![nums];
        }
        let mut prev: Option<i32> = None;
        let mut res = Vec::new();
        for (i, &num) in nums.iter().enumerate() {
            if prev.is_some() && prev.unwrap() == num {
                continue;
            } else {
                prev = Some(num)
            }
            let mut sub = nums.clone();
            sub.remove(i);
            let mut permutations: Vec<Vec<i32>> = Solution::permute_unique_backtrace(sub)
                .into_iter()
                .map(|x| {
                    let mut x = x;
                    x.push(num);
                    x
                })
                .collect();
            res.append(&mut permutations);
        }
        res
    }

    // use std::collections::VecDeque;
    fn dfs(
        nums: &Vec<i32>,
        len: usize,
        dept: usize,
        path: &mut Vec<i32>,
        used_vec: &mut Vec<bool>,
        result: &mut Vec<Vec<i32>>,
    ) {
        if dept == len {
            let full_path: Vec<i32> = path.into_iter().map(|&mut num| num).collect();
            result.push(full_path);
            return;
        }

        for i in 0..len {
            if used_vec[i] {
                continue;
            }
            path.push(nums[i]);
            used_vec[i] = true;
            Solution::dfs(&nums, len, dept + 1, path, used_vec, result);
            path.pop();
            used_vec[i] = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn permute() {
        let nums = vec![1, 2, 3];
        dbg!(Solution::permute(nums));

        let nums = vec![1, 1, 2];
        dbg!(Solution::permute_unique(nums));
    }
}
