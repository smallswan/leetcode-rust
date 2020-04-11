#![allow(unused)]
extern crate chrono;
use std::collections::BTreeSet;
mod solution;
fn main() {
    let nums = vec![2, 7, 2, 11];
    let result = solution::two_sum(nums, 9);
    println!("{:?}", result);

    let s = String::from("LEETCODEISHIRING");
    let zz = solution::convert(s,4);
    println!("{}",zz)
}
