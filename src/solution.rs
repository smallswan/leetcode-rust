use std::collections::HashMap;

#[test]
fn unit_test() {}

#[test]
fn simple() {
    let intervals: Vec<Vec<i32>> = vec![vec![1, 3], vec![2, 6], vec![8, 10], vec![15, 18]];
    let merge_intervals = merge(intervals);
    println!("{:?}", merge_intervals);
}

///
pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    if intervals.len() == 1 {
        return intervals;
    }
    let mut merge_vec = vec![vec![]; intervals.len()];
    merge_vec.clone_from_slice(&intervals);

    merge_vec
}
