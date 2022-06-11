#[macro_use]
extern crate criterion;
extern crate leetcode_rust;

use criterion::Criterion;
use leetcode_rust::fibonacci;
use leetcode_rust::solution::string::str_str::str_str;
fn str_str_benchmark(c: &mut Criterion) {
    c.bench_function("str_str ", |b| {
        b.iter(|| str_str(String::from("aaacaaab"), String::from("aaab")))
    });
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(20)));
}

criterion_group!(str_str_bench, str_str_benchmark, criterion_benchmark);
criterion_main!(str_str_bench);
