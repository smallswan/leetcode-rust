pub struct Solution;
use std::collections::{HashMap, HashSet};

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

    /// 575. 分糖果 https://leetcode-cn.com/problems/distribute-candies/
    pub fn distribute_candies(candy_type: Vec<i32>) -> i32 {
        let half = candy_type.len() / 2;
        let mut unique_types = HashSet::with_capacity(half);

        for t in candy_type {
            if unique_types.insert(t) && unique_types.len() == half {
                break;
            }
        }

        unique_types.len() as _
    }

    /// 594. 最长和谐子序列 https://leetcode-cn.com/problems/longest-harmonious-subsequence/
    pub fn find_lhs(nums: Vec<i32>) -> i32 {
        let mut counts = HashMap::with_capacity(nums.len());

        for num in nums {
            counts
                .entry(num)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }

        counts
            .iter()
            .filter_map(|(num, low)| counts.get(&(num + 1)).map(|high| low + high))
            .max()
            .unwrap_or(0)
    }

    /// 599. 两个列表的最小索引总和 https://leetcode-cn.com/problems/minimum-index-sum-of-two-lists/
    pub fn find_restaurant(list1: Vec<String>, list2: Vec<String>) -> Vec<String> {
        let (list1, mut list2) = if list2.len() < list1.len() {
            (list2, list1)
        } else {
            (list1, list2)
        };

        let indices = list1
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect::<HashMap<_, _>>();

        let mut min_sum = usize::MAX;

        for (i, name) in list2.iter().enumerate() {
            if i <= min_sum {
                if let Some(j) = indices.get(name) {
                    let sum = i + j;

                    if sum < min_sum {
                        min_sum = sum;
                    }
                }
            }
        }

        let mut i = 0;

        list2.retain(|name| {
            let result = i <= min_sum && indices.get(name).map_or(false, |j| i + j == min_sum);

            i += 1;

            result
        });

        list2
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
