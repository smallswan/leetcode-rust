#![allow(unused)]
mod dp;
mod solution;
mod spring2020;
fn main() {
    println!("LeetCode problems that I've solved in Rust");

    let mut tower = Vec::<Vec<i32>>::new();
    tower.push(vec![3]);
    tower.push(vec![1, 5]);
    tower.push(vec![8, 4, 3]);
    tower.push(vec![2, 6, 7, 9]);
    tower.push(vec![6, 2, 3, 5, 1]);

    let max = dp::number_tower(tower);
    println!("number_tower max = {:?}", max);
}
