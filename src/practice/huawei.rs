//! 华为机试 https://www.nowcoder.com/exam/oj/ta?tpId=37  
//! ACM 模式：你的代码需要处理输入输出，请使用如下样例代码读取输入和打印输出：  
//! ```
//!
//!
//!
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
//!
//! ```
//! # Examples
//! ```rust
//!
//!
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
//!
//! ```
//! 注意：
//! 1. 从终端输入的行包含换行符，需要使用```trim_end()```去掉！！！
//! 2. 标记单元测试为```#[ignore]```，避免在cargo test时阻塞。
//! 3. ```read_line(&mut line)```多次运行，line将不断追加字符串，可以使用```line.clear()```清除。
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

/// HJ3 明明的随机数  https://www.nowcoder.com/practice/3245215fffb84b7b81285493eae92ff0
use std::collections::BTreeSet;
pub fn hj3() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    let num = line1.trim_end().parse::<i32>().unwrap();
    let mut set = BTreeSet::new();
    for _ in 0..num {
        line1.clear();
        io::stdin().read_line(&mut line1).expect("expect a line");

        set.insert(line1.trim_end().parse::<i32>().unwrap());
    }

    set.iter().for_each(|n| println!("{}", n));
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

/// HJ5 进制转换  https://www.nowcoder.com/practice/8f3df50d2b9043208c5eed283d1d4da6?tpId=37
pub fn hj5() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    if line1.starts_with("0x") || line1.starts_with("0X") {
        let num = i32::from_str_radix(&line1.trim_end()[2..], 16).unwrap();
        println!("{}", num);
    } else {
        let num = i32::from_str_radix(line1.trim_end(), 16).unwrap();
        println!("{}", num);
    }
}

/// HJ6 质数因子 https://www.nowcoder.com/practice/196534628ca6490ebce2e336b47b3607?tpId=37
pub fn hj6() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    let mut num = line1.trim_end().parse::<i32>().unwrap();
    let sqrt = (num as f32).sqrt().floor() as i32;
    for i in 2..=sqrt {
        while num % i == 0 {
            println!("{}", i);
            num /= i;
        }
    }

    if num == 1 {
        println!();
    } else {
        println!("{}", num);
    }
}

/// HJ7 取近似值  https://www.nowcoder.com/practice/3ab09737afb645cc82c35d56a5ce802a?tpId=37
pub fn hj7() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    let pair: Vec<&str> = line1.trim_end().split('.').collect();
    if pair.len() == 2 {
        let integer = pair[0].parse::<i32>().unwrap();

        println!("{}", integer);
        let result = match pair[1].as_bytes().get(0) {
            Some(num) if (b'0'..=b'4').contains(num) => integer,
            Some(num) if (b'5'..=b'9').contains(num) => integer + 1,
            _ => unreachable!(),
        };

        println!("{}", result);
    }
}

/// HJ8 合并表记录 https://www.nowcoder.com/practice/de044e89123f4a7482bd2b214a685201?tpId=37
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
pub fn hj8() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    let num = line1.trim_end().parse::<i32>().unwrap();
    let mut map = BTreeMap::<i32, i32>::new();
    for _ in 0..num {
        line1.clear();
        io::stdin().read_line(&mut line1).expect("expect a line");

        let pair: Vec<i32> = line1
            .trim_end()
            .split(' ')
            .map(|s| s.parse::<i32>().unwrap())
            .collect();
        // match map.entry(pair[0]) {
        //     Entry::Vacant(entry) => {
        //         entry.insert(pair[1]);
        //     }
        //     Entry::Occupied(entry) => {
        //         *entry.into_mut() += pair[1];
        //     }
        // }
        map.entry(pair[0])
            .and_modify(|entry| *entry += pair[1])
            .or_insert(pair[1]);
    }

    for (key, value) in map.iter() {
        println!("{} {}", key, value);
    }
}

/// HJ9 提取不重复的整数 https://www.nowcoder.com/practice/253986e66d114d378ae8de2e6c4577c1?tpId=37
pub fn hj9() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    let mut num = line1.trim_end().parse::<i32>().unwrap();
    let mut nums = vec![];
    while num != 0 {
        if !nums.contains(&(num % 10)) {
            nums.push(num % 10);
        }
        num /= 10;
    }

    println!(
        "{}",
        nums.iter()
            .map(|n| format!("{}", n))
            .collect::<Vec<_>>()
            .concat()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_huawei() {
        // length_of_last_word();
        // count_chars_insensitive();
        // hj3();
        // hj4_str_split();
        // hj5();
        // hj6();
        // hj7();
        // hj8();
        hj9();
    }
}
