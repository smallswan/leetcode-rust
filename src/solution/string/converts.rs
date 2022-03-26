//! 字符串与其他类型之间的想换转换
/// 力扣（6. Z 字形变换） https://leetcode-cn.com/problems/zigzag-conversion/
pub fn convert(s: String, num_rows: i32) -> String {
    if num_rows == 1 {
        return s;
    }

    let mut result_vec = vec![vec![]; num_rows as usize];
    let mut row = 0usize;
    let mut direct_down = true;
    for ch in s.chars() {
        if row == 0 {
            direct_down = true;
        } else if row == (num_rows - 1) as usize {
            direct_down = false;
        }

        if let Some(row_vec) = result_vec.get_mut(row) {
            row_vec.push(ch);
        }

        if direct_down {
            row += 1;
        } else {
            row -= 1;
        }
    }

    let mut result_str_vec = Vec::<char>::new();
    for row_vec in result_vec {
        result_str_vec.extend_from_slice(&row_vec);
    }

    result_str_vec.iter().collect()
}
enum State {
    Init,
    ExpectNumber, // 已经碰到了+或者-，下一个字符必须是数字
    Number(i32),
}
/// 力扣（8. 字符串转整数（atoi）） https://leetcode-cn.com/problems/string-to-integer-atoi/
/// 方法1：DFA（Deterministic Finite Automaton，即确定有穷自动机）
/// 注意：剑指 Offer 67. 把字符串转换成整数 https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/
pub fn my_atoi(s: String) -> i32 {
    let mut state = State::Init;
    let mut neg = 1;
    for c in s.chars() {
        match c {
            ' ' => match state {
                State::Init => {}
                State::ExpectNumber => return 0,
                State::Number(n) => return neg * n,
            },
            '+' | '-' => match state {
                State::Init => {
                    state = State::ExpectNumber;
                    if c == '-' {
                        neg = -1;
                    }
                }
                State::ExpectNumber => return 0,
                State::Number(n) => return neg * n,
            },
            '0'..='9' => {
                let digit = c.to_digit(10).unwrap() as i32;
                match state {
                    State::Init | State::ExpectNumber => state = State::Number(digit),
                    State::Number(n) => {
                        match n.checked_mul(10).and_then(|x| x.checked_add(digit)) {
                            Some(number) => state = State::Number(number),
                            _ => {
                                return if neg < 0 {
                                    std::i32::MIN
                                } else {
                                    std::i32::MAX
                                }
                            }
                        }
                    }
                }
            }
            _ => match state {
                State::Init | State::ExpectNumber => return 0,
                State::Number(n) => return neg * n,
            },
        }
    }

    match state {
        State::Number(n) => neg * n,
        _ => 0,
    }
}

/// 力扣（12. 整数转罗马数字）  https://leetcode-cn.com/problems/integer-to-roman/
/// 贪心算法
pub fn int_to_roman(num: i32) -> String {
    let arr = vec![
        (1, "I"),
        (4, "IV"),
        (5, "V"),
        (9, "IX"),
        (10, "X"),
        (40, "XL"),
        (50, "L"),
        (90, "XC"),
        (100, "C"),
        (400, "CD"),
        (500, "D"),
        (900, "CM"),
        (1000, "M"),
    ];

    fn find(n: i32, arr: &[(i32, &'static str)]) -> (i32, &'static str) {
        for (value, s) in arr.iter().rev() {
            if n >= *value {
                return (*value, *s);
            }
        }
        unreachable!()
    }

    // 上次用除法，这次用减法
    let mut ret = "".to_string();
    let mut num = num;
    while num > 0 {
        let (v, s) = find(num, &arr);
        ret.push_str(s);
        num -= v;
    }

    ret
}

/// 力扣（12. 整数转罗马数字）
pub fn int_to_roman_v2(num: i32) -> String {
    let arr = vec![
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];

    fn find(n: i32, arr: &[(i32, &'static str)]) -> (i32, &'static str) {
        for (value, s) in arr {
            if n >= *value {
                return (*value, *s);
            }
        }
        unreachable!()
    }

    // 上次用除法，这次用减法
    let mut ret = "".to_string();
    let mut num = num;
    while num > 0 {
        let (v, s) = find(num, &arr);
        ret.push_str(s);
        num -= v;
    }

    ret
}

/// 力扣（468. 验证IP地址）  https://leetcode-cn.com/problems/validate-ip-address/
/// 使用标准库中的方法
use std::net::IpAddr;
pub fn valid_ip_address(ip: String) -> String {
    match ip.parse::<IpAddr>() {
        Ok(IpAddr::V4(x)) => {
            let array: Vec<Vec<char>> = ip.split('.').map(|x| x.chars().collect()).collect();
            for item in &array {
                if (item[0] == '0' && item.len() > 1) {
                    return String::from("Neither");
                }
            }
            String::from("IPv4")
        }
        Ok(IpAddr::V6(_)) => {
            let array: Vec<Vec<char>> = ip.split(':').map(|x| x.chars().collect()).collect();
            for item in array {
                if item.is_empty() {
                    return String::from("Neither");
                }
            }
            String::from("IPv6")
        }
        _ => String::from("Neither"),
    }
}

/// 力扣（468. 验证IP地址）  https://leetcode-cn.com/problems/validate-ip-address/
/// 使用分治法解
pub fn valid_ip_address2(ip: String) -> String {
    if ip.chars().filter(|ch| *ch == '.').count() == 3 {
        //println!("valid_ipv4_address..");
        // return valid_ipv4_address(ip);
        valid_ipv4_address_v2(ip)
    } else if ip.chars().filter(|ch| *ch == ':').count() == 7 {
        //println!("valid_ipv6_address..");
        // return valid_ipv6_address(ip);
        valid_ipv6_address_v2(ip)
    } else {
        String::from("Neither")
    }
}

fn valid_ipv4_address(ip: String) -> String {
    let array: Vec<Vec<char>> = ip.split('.').map(|x| x.chars().collect()).collect();
    for item in array {
        //Validate integer in range (0, 255):
        //1. length of chunk is between 1 and 3
        if item.is_empty() || item.len() > 3 {
            return String::from("Neither");
        }
        //2. no extra leading zeros
        if (item[0] == '0' && item.len() > 1) {
            return String::from("Neither");
        }
        //3. only digits are allowed
        for ch in &item {
            if !ch.is_digit(10) {
                return String::from("Neither");
            }
        }
        //4. less than 255
        let num_str: String = item.iter().collect();
        let num = num_str.parse::<u16>().unwrap();
        if num > 255 {
            return String::from("Neither");
        }
    }
    "IPv4".to_string()
}

fn valid_ipv4_address_v2(ip: String) -> String {
    // let array: Vec<Vec<char>> = ip.split('.').map(|x| x.chars().collect()).collect();
    let array: Vec<&str> = ip.split('.').collect();
    for item in array {
        let len = item.len();
        //Validate integer in range (0, 255):
        //1. length of chunk is between 1 and 3
        if len == 0 || len > 3 {
            return String::from("Neither");
        }
        //2. no extra leading zeros
        let mut chars = item.chars().peekable();
        // let first_char = chars.peek();

        if let Some(first) = chars.peek() {
            if *first == '0' && len > 1 {
                return String::from("Neither");
            }

            if !(*first).is_digit(10) {
                return String::from("Neither");
            }
        }
        //3. only digits are allowed

        for ch in chars {
            if !(ch).is_digit(10) {
                return String::from("Neither");
            }
        }
        //4. less than 255
        // let num_str: String = array[i].iter().collect();
        let num = item.parse::<u16>().unwrap();
        if num > 255 {
            return String::from("Neither");
        }
    }
    "IPv4".to_string()
}

fn valid_ipv6_address(ip: String) -> String {
    let array: Vec<Vec<char>> = ip.split(':').map(|x| x.chars().collect()).collect();
    for item in &array {
        let num = item;
        if num.is_empty() || num.len() > 4 {
            return String::from("Neither");
        }
        //2.
        for ch in num {
            if !(*ch).is_digit(16) {
                return String::from("Neither");
            }
        }
    }
    String::from("IPv6")
}

fn valid_ipv6_address_v2(ip: String) -> String {
    // let array: Vec<Vec<char>> = ip.split(':').map(|x| x.chars().collect()).collect();
    let array: Vec<&str> = ip.split(':').collect();
    for item in array {
        let len = item.len();
        if len == 0 || len > 4 {
            return String::from("Neither");
        }
        //2.
        for ch in item.chars() {
            if !(ch).is_digit(16) {
                return String::from("Neither");
            }
        }
    }
    String::from("IPv6")
}

/// 966. 元音拼写检查器 https://leetcode-cn.com/problems/vowel-spellchecker/
use std::collections::{HashMap, HashSet};
pub fn spellchecker(wordlist: Vec<String>, queries: Vec<String>) -> Vec<String> {
    let as_is = wordlist.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut capitalized = HashMap::with_capacity(wordlist.len());
    let mut vowels = HashMap::with_capacity(wordlist.len());

    for word in &wordlist {
        capitalized
            .entry(word.to_ascii_uppercase())
            .or_insert(word.as_str());

        vowels
            .entry(word.to_ascii_uppercase().replace(['E', 'I', 'O', 'U'], "A"))
            .or_insert(word.as_str());
    }

    let mut result = queries;

    for word in &mut result {
        if !as_is.contains(word.as_str()) {
            word.make_ascii_uppercase();

            if let Some(&original) = capitalized.get(word.as_str()) {
                word.clear();
                word.push_str(original);
            } else if let Some(&original) =
                vowels.get(word.replace(['E', 'I', 'O', 'U'], "A").as_str())
            {
                word.clear();
                word.push_str(original);
            } else {
                word.clear();
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medium() {
        let s = String::from("LEETCODEISHIRING");

        dbg!(convert(s, 4));

        dbg!(my_atoi(" -423456".to_string()));
        dbg!(my_atoi("4193 with words".to_string()));
        dbg!(my_atoi("words and 987".to_string()));
        dbg!(my_atoi("-91283472332".to_string()));

        dbg!("3999 to roman {}", int_to_roman(3999));
        dbg!("3999 to roman {}", int_to_roman_v2(3999));

        let ip = String::from("2001:0db8:85a3:0:0:8A2E:0370:7334");
        dbg!(valid_ip_address2(ip));
    }
}
