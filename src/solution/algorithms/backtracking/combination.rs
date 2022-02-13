pub struct Solution;

impl Solution {
    /// 39. 组合总和  https://leetcode-cn.com/problems/combination-sum/
    pub fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut res: Vec<Vec<i32>> = Vec::with_capacity(150);
        let mut v: Vec<i32> = Vec::with_capacity(150);
        // println!("\ncandidates: {:?} target: {}", candidates, target);
        Solution::combination_sum_backtrace(&candidates, 0, target, &mut res, &mut v);
        res
    }

    fn combination_sum_backtrace(
        candidates: &[i32],
        i: usize,
        target: i32,
        res: &mut Vec<Vec<i32>>,
        v: &mut Vec<i32>,
    ) {
        if i == candidates.len() {
            return;
        }
        if target == 0 {
            res.push(v.clone());
            return;
        }
        Solution::combination_sum_backtrace(candidates, i + 1, target, res, v);
        let d = candidates[i];
        if target >= d {
            v.push(d);

            Solution::combination_sum_backtrace(candidates, i, target - d, res, v);
            v.pop();
        }
    }

    /// 40. 组合总和 II https://leetcode-cn.com/problems/combination-sum-ii/
    pub fn combination_sum2(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        let mut seq = candidates;
        let mut res = Vec::new();
        seq.sort_unstable_by(|a, b| b.cmp(a));
        let mut vec = Vec::new();
        Solution::backtrack(&seq, target, vec, &mut res, 0);
        res
    }

    fn backtrack(
        seq: &Vec<i32>,
        target: i32,
        mut curr: Vec<i32>,
        result: &mut Vec<Vec<i32>>,
        start_idx: usize,
    ) {
        let mut i = start_idx;
        while i < seq.len() {
            let item = seq[i];
            if target - item < 0 {
                i += 1;
                continue;
            }
            let mut new_vec = curr.clone();
            new_vec.push(item);
            if target == item {
                result.push(new_vec);
            } else {
                Solution::backtrack(seq, target - item, new_vec, result, i + 1);
            }
            // skip duplicate result
            while i < seq.len() && seq[i] == item {
                i += 1;
            }
        }
    }

    /// 77. 组合 https://leetcode-cn.com/problems/combinations/
    pub fn combine(n: i32, k: i32) -> Vec<Vec<i32>> {
        let mut res: Vec<Vec<i32>> = Vec::new();
        Solution::combine_backtrack(1, n, k, vec![], &mut res);
        res
    }

    fn combine_backtrack(start: i32, end: i32, k: i32, curr: Vec<i32>, result: &mut Vec<Vec<i32>>) {
        if k < 1 {
            result.push(curr);
            return;
        }
        if end - start + 1 < k {
            // elements is not enough, return quickly
            return;
        }
        for i in start..end + 1 {
            let mut vec = curr.clone();
            vec.push(i);
            Solution::combine_backtrack(i + 1, end, k - 1, vec, result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn combination_sum() {
        let candidates = vec![2, 3, 5];
        let target = 8;
        dbg!(Solution::combination_sum(candidates, target));

        let candidates = vec![10, 1, 2, 7, 6, 1, 5];
        let target = 8;
        dbg!(Solution::combination_sum2(candidates, target));

        dbg!(Solution::combine(4, 2));
    }
}
