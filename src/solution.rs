use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_operations() {
        //(1)最原始直接基础的位操作方法。
        let mut byte: u8 = 0b0000_0000;
        println!("0b{:08b}", byte);
        byte |= 0b0000_1000; // Set a bit
        println!("0b{:08b}", byte);
        byte &= 0b1111_0111; // Unset a bit
        println!("0b{:08b}", byte);
        byte ^= 0b0000_1000; // Toggle a bit
        println!("0b{:08b}", byte);
        byte = !byte; // Flip all bits
        println!("0b{:08b}", byte);
        byte <<= 1; // shift left one bit
        println!("0b{:08b}", byte);
        byte >>= 1; // shift right one bit
        println!("0b{:08b}", byte);
        //特别提醒：rust为每一个数字类型都实现了大量方法，其中包括位操作方法！！！具体请参看下方链接！！！
        //https://doc.rust-lang.org/std/primitive.u8.html
        let mut rbyte: u8 = 0b1000_0000;
        rbyte = rbyte.rotate_left(1); // rotate left one bit
        println!("0b{:08b}", byte);
        //https://doc.rust-lang.org/std/#primitives
        rbyte = rbyte.rotate_right(1); // rotate right one bit
        println!("0b{:08b}", rbyte);
        bit_twiddling(0, 3);
        bit_twiddling(8, 3);
        //test bitwise operation macros
        // assert_eq!(eq1!(0b0000_1111, 0), true);
        // assert_eq!(eq0!(0b0000_1111, 4), true);
        // assert_eq!(set!(0b0000_1111, 0), 0x0f);
        // assert_eq!(clr!(0b0000_1111, 0), 0x0e);
    }

    //(2)定义为一个rust函数
    fn bit_twiddling(original: u8, bit: u8) {
        let mask = 1 << bit;
        println!(
            "Original: {:b}, Set: {:b}, Cleared: {:b}, Toggled: {:b}",
            original,
            original | mask,
            original & !mask,
            original ^ mask
        );
    }

    //define rust macro for bitwise operations
    //(3)定义为rust 宏
    #[macro_export]
    macro_rules! eq1 {
        ($n:expr, $b:expr) => {
            $n & (1 << $b) != 0
        };
    }

    #[macro_export]
    macro_rules! eq0 {
        ($n:expr, $b:expr) => {
            $n & (1 << $b) == 0
        };
    }

    #[test]
    fn lamada_demo() {
        let vec = fibonacci(20);
        for num in vec {
            print!("{} ", num);
        }
    }

    #[test]
    fn simple() {
        let num = 123_456f64;
        println!("sqrt(x)={}", num.sqrt().floor());

        let intervals: Vec<Vec<i32>> = vec![vec![1, 3], vec![2, 6], vec![8, 10], vec![15, 18]];
        let merge_intervals = merge(intervals);
        println!("{:?}", merge_intervals);

        use itertools::kmerge;
        for element in kmerge(vec![vec![1, 3], vec![2, 6], vec![8, 10], vec![15, 18]]) {
            print!("{} ", element);
        }
        println!();

        use itertools::cloned;
        let elements = vec![vec![1, 3], vec![2, 6], vec![8, 10], vec![15, 18]];
        let cloned_elements = cloned(&elements);

        for element in cloned_elements {
            println!("{:?}", element);
        }
    }

    #[test]
    fn rotate_reverse() {
        use std::collections::VecDeque;

        let mut buf: VecDeque<_> = (0..10).collect();

        buf.rotate_right(3);
        assert_eq!(buf, [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]);

        // https://doc.rust-lang.org/std/primitive.slice.html#method.rotate_right
        let mut nums: Vec<i32> = (0..10).collect();
        crate::medium::rotate(&mut nums, 3);
        assert_eq!(nums, [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]);

        let mut nums2: Vec<i32> = (0..10).collect();
        nums2.rotate_right(3);
        assert_eq!(nums2, [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]);
    }
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

fn fibonacci(n: usize) -> Vec<u32> {
    if n <= 2 {
        return vec![1; n];
    }
    let mut fib_vec = Vec::with_capacity(n);
    fib_vec.push(1);
    fib_vec.push(1);
    for index in 2..n {
        fib_vec.push(fib_vec[index - 1] + fib_vec[index - 2]);
    }
    fib_vec
}
