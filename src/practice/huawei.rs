//! 华为机试 https://www.nowcoder.com/exam/oj/ta?tpId=37  
//! ACM 模式：你的代码需要处理输入输出，请使用如下样例代码读取输入和打印输出：  
//! ```
//! use std::io::{self, *};
//!
//! fn main() {
//!     let stdin = io::stdin();
//!     unsafe {
//!         for line in stdin.lock().lines() {
//!             let ll = line.unwrap();
//!             let numbers: Vec<&str> = ll.split(" ").collect();
//!             let a = numbers[0].trim().parse::<i32>().unwrap_or(0);
//!             let b = numbers[1].trim().parse::<i32>().unwrap_or(0);
//!             print!("{}\n", a + b);
//!         }
//!     }
//! }
//! ```
//! # Examples
//! ```
//! use std::io;
//! pub fn main(){
//!     let mut input = String::new();
//!     io::stdin().read_line(&mut input).expect("expect a line");
//!
//!     let mut count = 0;
//!     for ch in input.trim_end().chars().rev(){
//!         if ch == ' ' {
//!             break;
//!         }
//!         count += 1;
//!     }
//!     println!("{}", count);
//! }
//! ```

use std::io;

/// HJ1 字符串最后一个单词的长度  https://www.nowcoder.com/practice/8c949ea5f36f422594b306a2300315da
pub fn length_of_last_word() {
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("expect a line");

    let mut count = 0;
    for ch in input.trim_end().chars().rev() {
        if ch == ' ' {
            break;
        }
        count += 1;
    }
    println!("{}", count);
}

/// HJ2 计算某字符出现次数  https://www.nowcoder.com/practice/a35ce98431874e3a820dbe4b2d0508b1
pub fn count_chars_insensitive() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");

    let mut line2 = String::new();
    io::stdin().read_line(&mut line2).expect("expect a line");
    let ch = line2.chars().next().unwrap();

    let mut count = 0;
    line1.chars().for_each(|c| {
        if ch.eq_ignore_ascii_case(&c) {
            count += 1;
        }
    });

    println!("{}", count);
}

/// HJ4 字符串分隔  https://www.nowcoder.com/practice/d9162298cb5a437aad722fccccaae8a7
use std::iter::FromIterator;
pub fn hj4_str_split() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");

    let line1 = line1.trim_end();
    let len = line1.len();

    if len % 8 == 0 {
        let chars: Vec<char> = line1.chars().collect::<Vec<_>>();
        let mut iter = chars.chunks(8);
        for chunk in iter {
            println!("{}", String::from_iter(chunk));
        }
    } else {
        let mut line2 = String::from(line1);
        for i in 0..(8 - (len % 8)) {
            line2.push('0');
        }
        let chars: Vec<char> = line2.chars().collect::<Vec<_>>();
        let mut iter = chars.chunks(8);
        for chunk in iter {
            println!("{}", String::from_iter(chunk));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huawei() {
        // length_of_last_word();
        // count_chars_insensitive();
        hj4_str_split();
    }
}
