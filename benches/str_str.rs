#[macro_use]
extern crate criterion;
extern crate leetcode_rust;

use criterion::Criterion;
use leetcode_rust::fibonacci;
use leetcode_rust::solution::string::str_str::{str_str, str_str_v3};
fn str_str_benchmark(c: &mut Criterion) {
    c.bench_function("str_str ", |b| {
        b.iter(|| str_str(String::from("aaacaaab"), String::from("aaab")))
    });
}

fn str_str_v3_benchmark(c: &mut Criterion) {
    c.bench_function("str_str_v3 ", |b| {
        b.iter(|| str_str_v3(String::from("aaacaaab"), String::from("aaab")))
    });
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(20)));
}

criterion_group!(
    str_str_bench,
    str_str_benchmark,
    str_str_v3_benchmark,
    criterion_benchmark
);
criterion_main!(str_str_bench);
