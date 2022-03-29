pub struct Solution;
use std::collections::HashSet;

impl Solution {
    /// 128. 最长连续序列 https://leetcode-cn.com/problems/longest-consecutive-sequence/
    pub fn longest_consecutive(nums: Vec<i32>) -> i32 {
        let nums = nums.into_iter().collect::<HashSet<_>>();
        let mut result = 0;

        for num in nums.iter().copied() {
            if !nums.contains(&(num - 1)) {
                let mut end = num + 1;

                while nums.contains(&end) {
                    end += 1;
                }

                result = result.max(end - num);
            }
        }

        result
    }
}

struct MyHashSet {
    data: Vec<Vec<i32>>,
}

/// 705. 设计哈希集合 https://leetcode-cn.com/problems/design-hashset/
impl MyHashSet {
    const BASE: i32 = 811;

    fn new() -> Self {
        MyHashSet {
            data: vec![vec![]; MyHashSet::BASE as usize],
        }
    }

    fn hash(key: i32) -> usize {
        (key % MyHashSet::BASE) as usize
    }

    fn add(&mut self, key: i32) {
        let h = MyHashSet::hash(key);
        match self.data[h].binary_search(&key) {
            Err(idx) => {
                self.data[h].insert(idx, key);
            }
            _ => {}
        }
    }

    fn remove(&mut self, key: i32) {
        let h = MyHashSet::hash(key);
        match self.data[h].binary_search(&key) {
            Ok(idx) => {
                self.data[h].remove(idx);
            }
            _ => {}
        }
    }

    fn contains(&self, key: i32) -> bool {
        self.data[MyHashSet::hash(key)].binary_search(&key).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_() {
        let nums = vec![0, 3, 7, 2, 5, 8, 4, 6, 0, 1];
        dbg!(Solution::longest_consecutive(nums));
    }

    #[test]
    fn operations() {
        let mut set = MyHashSet::new();
        set.add(1);
        set.add(2);
        set.contains(1);
        set.contains(3);
        set.add(2);
        set.contains(2);
        set.remove(2);
        set.contains(2);
    }
}
