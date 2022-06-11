#![allow(unused)]
#![warn(clippy::needless_doctest_main)]
mod dp;
mod medium;
mod simple;

mod contest;
mod interview;
mod lcof;
mod practice;
pub mod solution;

#[macro_use]
extern crate lazy_static;

extern "C" {
    fn rand() -> i32;
}

#[inline]
pub fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
