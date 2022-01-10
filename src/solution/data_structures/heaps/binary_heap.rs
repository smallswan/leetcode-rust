use std::collections::BinaryHeap;
/// 力扣（215. 数组中的第K个最大元素） https://leetcode-cn.com/problems/kth-largest-element-in-an-array/
pub fn find_kth_largest(nums: Vec<i32>, k: i32) -> i32 {
    let mut heap = BinaryHeap::from(nums);
    for _ in 1..k {
        heap.pop();
    }
    heap.pop().unwrap()
}

/// 1046. 最后一块石头的重量 https://leetcode-cn.com/problems/last-stone-weight/
pub fn last_stone_weight(stones: Vec<i32>) -> i32 {
    let mut heap = BinaryHeap::from(stones);
    loop {
        if let Some(rock1) = heap.pop() {
            if let Some(rock2) = heap.pop() {
                if rock1 > rock2 {
                    heap.push(rock1 - rock2);
                }
            } else {
                return rock1;
            }
        } else {
            return 0;
        }
    }
}
