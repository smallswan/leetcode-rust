use std::mem;
/// 72. 编辑距离 https://leetcode-cn.com/problems/edit-distance/
pub fn min_distance(word1: String, word2: String) -> i32 {
    let (word1, word2) = if word2.len() < word1.len() {
        (word2, word1)
    } else {
        (word1, word2)
    };

    let mut cache = (0..=word1.len() as _).rev().collect::<Box<_>>();

    for (prev_base, c2) in word2.as_bytes().iter().rev().enumerate() {
        let mut prev = prev_base as _;

        cache[word1.len()] = prev + 1;

        for (i, c1) in word1.as_bytes().iter().enumerate().rev() {
            let distance = if c1 == c2 {
                prev
            } else {
                cache[i].min(cache[i + 1]).min(prev) + 1
            };

            prev = mem::replace(&mut cache[i], distance);
        }
    }

    cache[0]
}