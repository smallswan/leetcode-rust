pub struct Solution;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::iter;

#[derive(Default)]
struct Node {
    children: [Option<Box<Node>>; 26],
    has_value: bool,
}

impl Solution {
    /// 127. 单词接龙 https://leetcode-cn.com/problems/word-ladder/
    pub fn ladder_length(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
        let mut graph = HashMap::new();
        for word in word_list.iter().map(String::as_bytes) {
            for i in 0..word.len() {
                let key = (&word[..i], &word[i + 1..]);

                match graph.entry(key) {
                    Entry::Vacant(entry) => {
                        entry.insert(vec![word]);
                    }
                    Entry::Occupied(entry) => {
                        entry.into_mut().push(word);
                    }
                }
            }
        }
        let mut queue = iter::once(begin_word.as_bytes()).collect::<VecDeque<_>>();
        let mut visited = iter::once(begin_word.as_bytes()).collect::<HashSet<_>>();
        let mut length = 1;

        loop {
            for _ in 0..queue.len() {
                let current = queue.pop_front().unwrap();

                if current == end_word.as_bytes() {
                    return length;
                }

                for i in 0..current.len() {
                    if let Some(nexts) = graph.get(&(&current[..i], &current[i + 1..])) {
                        for next in nexts {
                            if visited.insert(next) {
                                queue.push_back(next);
                            }
                        }
                    }
                }
            }

            if queue.is_empty() {
                break;
            }

            length += 1;
        }

        0
    }

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

    /// 139. 单词拆分 https://leetcode-cn.com/problems/word-break/
    pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
        let s = s.into_bytes();
        let mut root = Node::default();

        for word in word_dict {
            let mut node = &mut root;

            for c in word.bytes() {
                node = node.children[usize::from(c - b'a')].get_or_insert_with(Box::default);
            }

            node.has_value = true;
        }
        let mut cache = vec![false; s.len() + 1];

        cache[s.len()] = true;

        for i in (0..s.len()).rev() {
            let mut node = &root;

            for (j, c) in s.iter().enumerate().skip(i) {
                if let Some(child) = node.children[usize::from(c - b'a')].as_deref() {
                    if child.has_value && cache[j + 1] {
                        cache[i] = true;

                        break;
                    }

                    node = child;
                } else {
                    break;
                }
            }
        }

        cache[0]
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

    /// 645. 错误的集合 https://leetcode-cn.com/problems/set-mismatch/
    pub fn find_error_nums(nums: Vec<i32>) -> Vec<i32> {
        let mut extra_xor_missing = 0;
        let mut nums = nums;
        let mut i = 0;

        let extra = loop {
            let num = nums[i] & i32::MAX;
            let slot = &mut nums[(num - 1) as usize];

            extra_xor_missing ^= num ^ ((i + 1) as i32);
            i += 1;

            if *slot & i32::MIN == 0 {
                *slot |= i32::MIN;
            } else {
                break num;
            }
        };

        while let Some(num) = nums.get(i) {
            extra_xor_missing ^= (num & i32::MAX) ^ ((i + 1) as i32);
            i += 1;
        }

        vec![extra, extra_xor_missing ^ extra]
    }

    /// 697. 数组的度 https://leetcode-cn.com/problems/degree-of-an-array/
    pub fn find_shortest_sub_array(nums: Vec<i32>) -> i32 {
        let mut range = HashMap::with_capacity(nums.len());
        let mut max_frequency = 0;
        let mut min_range = 0;

        for (i, num) in (0..).zip(nums) {
            let (first, last, count) = *range
                .entry(num)
                .and_modify(|(_, last, count)| {
                    *last = i;
                    *count += 1;
                })
                .or_insert((i, i, 1));

            match count.cmp(&max_frequency) {
                Ordering::Less => {}
                Ordering::Equal => min_range = min_range.min(last - first + 1),
                Ordering::Greater => {
                    max_frequency = count;
                    min_range = last - first + 1;
                }
            }
        }

        min_range
    }

    fn dfs(node: &Node, base: &mut String, result: &mut String) {
        for (c, child) in (b'a'..).zip(&node.children) {
            if let Some(child) = child.as_deref() {
                if child.has_value {
                    base.push(c.into());
                    Self::dfs(child, base, result);
                    base.pop();
                }
            }
        }

        if base.len() > result.len() {
            result.replace_range(.., base);
        }
    }

    /// 720. 词典中最长的单词 https://leetcode-cn.com/problems/longest-word-in-dictionary/
    pub fn longest_word(words: Vec<String>) -> String {
        let mut root = Node {
            has_value: true,
            children: Default::default(),
        };

        for word in &words {
            let mut node = &mut root;

            for c in word.bytes() {
                node = node.children[usize::from(c - b'a')].get_or_insert_with(Box::default);
            }

            node.has_value = true;
        }

        let mut result = String::new();

        Self::dfs(&root, &mut String::new(), &mut result);

        result
    }

    /// 884. 两句话中的不常见单词 https://leetcode-cn.com/problems/uncommon-words-from-two-sentences/
    pub fn uncommon_from_sentences(s1: String, s2: String) -> Vec<String> {
        let mut states = HashMap::new();

        for s in [s1.as_str(), s2.as_str()] {
            for word in s.split(' ') {
                states
                    .entry(word)
                    .and_modify(|state| *state = true)
                    .or_insert(false);
            }
        }

        states
            .into_iter()
            .filter_map(|(key, value)| if value { None } else { Some(key.to_string()) })
            .collect()
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

        if let Err(idx) = self.data[h].binary_search(&key) {
            self.data[h].insert(idx, key);
        }
    }

    fn remove(&mut self, key: i32) {
        let h = MyHashSet::hash(key);
        if let Ok(idx) = self.data[h].binary_search(&key) {
            self.data[h].remove(idx);
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
        let begin_word: String = String::from("hit");
        let end_word: String = String::from("cog");
        let word_list: Vec<String> = vec![
            "hot".to_string(),
            "dot".to_string(),
            "dog".to_string(),
            "lot".to_string(),
            "log".to_string(),
            "cog".to_string(),
        ];
        dbg!(Solution::ladder_length(begin_word, end_word, word_list));

        let s = String::from("leetcode");
        let word_dict: Vec<String> = vec!["leet".to_string(), "code".to_string()];
        dbg!(Solution::word_break(s, word_dict));

        let nums = vec![0, 3, 7, 2, 5, 8, 4, 6, 0, 1];
        dbg!(Solution::longest_consecutive(nums));

        let s1 = String::from("this apple is sweet");
        let s2 = String::from("this apple is sour");

        let vec = Solution::uncommon_from_sentences(s1, s2);
        println!("{:?}", vec);
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
