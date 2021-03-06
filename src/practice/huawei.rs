//! 华为机试 https://www.nowcoder.com/exam/oj/ta?tpId=37  
//! ACM 模式：你的代码需要处理输入输出，请使用如下样例代码读取输入和打印输出：  
//! ```
//!     use std::io;
//!     use std::io::BufRead;
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
//!     use std::io;
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

use std::collections::HashSet;
/// HJ10 字符个数统计  https://www.nowcoder.com/practice/eb94f6a5b2ba49c6ac72d40b5ce95f50?tpId=37
pub fn hj10() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    // let mut chars: Vec<i32> = vec![0; 128];
    let mut set = HashSet::new();
    line1.trim_end().as_bytes().iter().for_each(|&ch| {
        set.insert(ch);
    });

    println!("{}", set.len());
    // println!("{}", chars.iter().filter(|&&count| count > 0).count());
}

/// HJ11 数字颠倒  https://www.nowcoder.com/practice/ae809795fca34687a48b172186e3dafe?tpId=37
pub fn hj11() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");

    let mut new_line = String::new();
    line1
        .trim_end()
        .chars()
        .rev()
        .for_each(|ch| new_line.push(ch));
    println!("{}", new_line);

    // let new_line: String = String::from_iter(line1.trim_end().chars().rev().collect::<Vec<_>>());
    // println!("{}",new_line);
}

/// HJ12 字符串反转  https://www.nowcoder.com/practice/e45e078701ab4e4cb49393ae30f1bb04?tpId=37
pub fn hj12() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");

    let new_line = line1.trim_end().bytes().rev().collect::<Vec<_>>();
    println!("{}", String::from_utf8(new_line).unwrap());
}

/// HJ13 句子逆序  https://www.nowcoder.com/practice/48b3cb4e3c694d9da5526e6255bb73c3?tpId=37
pub fn hj13() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");

    for word in line1.trim_end().split(' ').rev() {
        println!("{}", word);
    }
}

/// HJ14 字符串排序  https://www.nowcoder.com/practice/5af18ba2eb45443aa91a11e848aa6723?tpId=37
pub fn hj14() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    let mut words = vec![];
    let mut num = line1.trim_end().parse::<i32>().unwrap();
    for _ in 0..num {
        line1.clear();
        io::stdin().read_line(&mut line1).expect("expect a line");
        words.push(line1.trim_end().to_owned());
    }

    words.sort_unstable();
    for word in words {
        println!("{}", word);
    }
}

/// HJ15 求int型正整数在内存中存储时1的个数 https://www.nowcoder.com/practice/440f16e490a0404786865e99c6ad91c9?tpId=37
pub fn hj15() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    let mut num = line1.trim_end().parse::<i32>().unwrap();

    println!("{}", num.count_ones());
}

/// HJ21 简单密码  https://www.nowcoder.com/practice/7960b5038a2142a18e27e4c733855dac?tpId=37
pub fn hj21() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");

    for byte in line1.trim_end().bytes() {
        match byte {
            b'A'..=b'Y' => print!("{}", (b'b' + byte - b'A') as char),
            b'Z' => print!("a"),
            b'0' => print!("0"),
            b'1' => print!("1"),
            b'a'..=b'c' => print!("2"),
            b'd'..=b'f' => print!("3"),
            b'g'..=b'i' => print!("4"),
            b'j'..=b'l' => print!("5"),
            b'm'..=b'o' => print!("6"),
            b'p'..=b's' => print!("7"),
            b't'..=b'v' => print!("8"),
            b'w'..=b'z' => print!("9"),
            _ => print!("{}", (byte as char)),
        }
    }
    println!();
}

/// HJ22 汽水瓶  https://www.nowcoder.com/practice/fe298c55694f4ed39e256170ff2c205f?tpId=37
pub fn hj22() {
    let mut line1 = String::new();
    io::stdin().read_line(&mut line1).expect("expect a line");
    let mut bottle = line1.trim_end().parse::<i32>().unwrap();
    while bottle != 0 {
        println!("{}", bottle / 2);
        line1.clear();
        io::stdin().read_line(&mut line1).expect("expect a line");
        bottle = line1.trim_end().parse::<i32>().unwrap();
    }
}

/// HJ46 截取字符串 https://www.nowcoder.com/practice/a30bbc1a0aca4c27b86dd88868de4a4a?tpId=37
pub fn hj46() {
    let mut line = String::new();
    io::stdin().read_line(&mut line).expect("expect a line");
    let new_line = line.trim_end().to_owned();
    line.clear();
    io::stdin().read_line(&mut line).expect("expect a line");
    let num = line.trim_end().parse::<usize>().unwrap();

    println!("{}", String::from(&new_line[..num]));
}

/// HJ58 输入n个整数，输出其中最小的k个  https://www.nowcoder.com/practice/69ef2267aafd4d52b250a272fd27052c?tpId=37
pub fn hj58() {
    let mut line = String::new();
    io::stdin().read_line(&mut line).expect("expect a line");
    let pair: Vec<usize> = line
        .trim_end()
        .split(' ')
        .map(|num| num.parse::<usize>().unwrap())
        .collect();

    line.clear();
    io::stdin().read_line(&mut line).expect("expect a line");
    let mut nums: Vec<i32> = line
        .trim_end()
        .split(' ')
        .map(|num| num.parse::<i32>().unwrap())
        .collect();

    nums.sort_unstable();

    for i in 0..pair[1] {
        print!("{} ", nums[i]);
    }
}

/// HJ101 输入整型数组和排序标识，对其元素按照升序或降序进行排序 https://www.nowcoder.com/practice/dd0c6b26c9e541f5b935047ff4156309?tpId=37
pub fn hj101() {
    let mut line = String::new();
    io::stdin().read_line(&mut line).expect("expect a line");
    let count = line.trim_end().parse::<usize>().unwrap();

    line.clear();
    io::stdin().read_line(&mut line).expect("expect a line");
    let mut nums: Vec<i32> = line
        .trim_end()
        .split(' ')
        .map(|num| num.parse::<i32>().unwrap())
        .collect();

    nums.sort_unstable();

    line.clear();
    io::stdin().read_line(&mut line).expect("expect a line");
    let asc = line.trim_end().parse::<usize>().unwrap();
    if asc == 0 {
        for num in nums {
            print!("{} ", num);
        }
    } else if asc == 1 {
        for num in nums.iter().rev() {
            print!("{} ", num);
        }
    }
}

/// 华为笔试
/// 题目大意：依次往栈中放入数据n，如果某一时刻栈顶的元素等于其以下连续的几个元素的总和，就需要将栈顶元素连同
/// 其以下连续的几个元素一同出栈，并将栈顶元素乘以2入栈。例如6,1,2,3。3 = 2 + 1，则需要将1 2 3出栈，将6（=3*2）入栈；
/// 由于栈顶的6以下的元素和为6，根据上述规则，所有元素都出栈，将12入栈，最终栈中的元素只有12。
use std::collections::VecDeque;
fn huawei3() {
    let stdin = io::stdin();
    let mut stack = VecDeque::new();
    let mut line = String::new();
    stdin.read_line(&mut line).expect("expect a line");
    let nums: Vec<i32> = line
        .trim_end()
        .split(' ')
        .map(|n| n.parse::<i32>().unwrap())
        .collect();
    stack.push_back(nums[0]);
    let len = nums.len();

    fn try_push(stack: &mut VecDeque<i32>, target: i32) {
        if stack.is_empty() {
            stack.push_back(target);
            return;
        }

        let (mut count, mut sum) = (0, 0);
        // stack.iter().rev().for_each(|num| {
        //     if sum < target {
        //         sum += num;
        //         count += 1;
        //     }
        // });

        //及时break 1
        let mut iter = stack.iter().rev();
        // loop {
        //     match iter.next() {
        //         None => break,
        //         Some(num) => {
        //             if sum < target {
        //                 sum += num;
        //                 count += 1;
        //             } else {
        //                 break;
        //             }
        //         }
        //     }
        // }

        // 及时break 2
        while let Some(num) = iter.next() {
            if sum < target {
                sum += num;
                count += 1;
            } else {
                break;
            }
        }

        if sum != target {
            stack.push_back(target);
            return;
        } else if sum == target {
            for k in 0..count {
                stack.pop_back();
            }
            let double = 2 * target;
            try_push(stack, double);
        }
    }

    for i in 1..len {
        try_push(&mut stack, nums[i]);
    }
    while let Some(value) = stack.pop_back() {
        print!("{} ", value);
    }
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
        // hj9();
        // hj10();
        // hj11();
        // hj12();

        // hj21();

        huawei3();

        // for i in (0..10).rev() {
        //     print!("{} ", i);
        // }
    }
}
