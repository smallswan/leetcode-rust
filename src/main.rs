#![allow(unused)]
mod dp;
mod hard;
mod medium;
mod simple;

mod lcof;
mod solution;
mod spring2020;

#[macro_use]
extern crate lazy_static;

extern "C" {
    fn rand() -> i32;
}

fn main() {
    println!("LeetCode problems that I've solved in Rust");

    let ip = String::from("192.168.2.251");
    let nums: Vec<&str> = ip.split('.').collect();

    println!("{:?}", nums);
    for num in nums {
        println!("{} len is {}", num, num.len());
    }

    let rand = unsafe { rand() };

    println!("{}", rand);

    // rust 58 新特性：捕获格式字符串中的标识符
    let x: f32 = 2.0;
    let y = x.sin();
    println!("sin(2) is {y}");
}
