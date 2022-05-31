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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huawei() {
        length_of_last_word();
    }
}
