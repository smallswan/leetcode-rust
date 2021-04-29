use std::collections::HashMap;

#[test]
fn unit_test() {
    let num = 123_456f64;
    println!("sqrt(x)={}", num.sqrt().floor());
}

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

use std::io;
///
/// 将16进制数字字符串&str转为10进制数字输出
#[test]
fn hua_wei_test() {
    loop {
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .expect("Failed to read line");
        let mut digtis = "";
        if line.trim().is_empty() {
            break;
        }

        if line.as_str().starts_with("0x") || line.as_str().starts_with("0X") {
            digtis = &line[2..];
        } else {
            digtis = line.as_str();
        }

        // 为了防止数字较大这里使用i128类型
        match i128::from_str_radix(digtis.trim(), 16) {
            Ok(num) => {
                println!("{}", num);
            }
            Err(e) => {
                println!("can't convert str : {}", e);
                break;
            }
        }
    }
}

#[test]
fn lamada_demo() {
    let vec = fibonacci(20);
    for num in vec {
        println!("{}", num);
    }
}

fn fibonacci(n: usize) -> Vec<u32> {
    let mut fib_vec = Vec::with_capacity(n);
    let mut idx = 0;
    if idx <= 2 {
        for _ in 0..2 {
            fib_vec.push(1);
            idx += 1;
        }
    }
    while idx >= 2 && idx < n {
        fib_vec.push(fib_vec[idx - 1] + fib_vec[idx - 2]);
        idx += 1;
    }
    fib_vec
}
