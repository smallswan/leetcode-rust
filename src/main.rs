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

    let index = solution::string::knuth_morris_pratt("Rust is a programming language empowering everyone to build reliable and efficient software".to_string(),"everyone".to_string());

    println!("{:?}", index);
}
