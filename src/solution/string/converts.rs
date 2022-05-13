//! 字符串与其他类型之间的想换转换

use std::cmp::Ordering;

pub struct Solution;

enum State {
    Init,
    ExpectNumber, // 已经碰到了+或者-，下一个字符必须是数字
    Number(i32),
}

#[derive(Default)]
struct Node {
    has_value: bool,
    children: [Option<Box<Node>>; 26],
}
impl Solution {
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

    /// 71. 简化路径 https://leetcode-cn.com/problems/simplify-path/
    pub fn simplify_path(path: String) -> String {
        let mut stack = Vec::new();

        for component in path.split('/') {
            match component {
                "" | "." => {}
                ".." => {
                    stack.pop();
                }
                component => stack.push(component),
            }
        }

        let mut result = String::from("/");
        let mut iter = stack.into_iter();

        if let Some(first) = iter.next() {
            result.push_str(first);

            for component in iter {
                result.push('/');
                result.push_str(component);
            }
        }

        result
    }

    /// 87. 扰乱字符串 https://leetcode-cn.com/problems/scramble-string/
    pub fn is_scramble(s1: String, s2: String) -> bool {
        let s1 = s1.into_bytes();
        let s2 = s2.into_bytes();
        let n = s1.len();
        let n_squared = n * n;
        let mut cache = vec![false; n_squared * n];
        let index = move |length, i, j| n_squared * (length - 1) + n * i + j;

        for i in 0..n {
            for j in 0..n {
                cache[n * i + j] = s1[i] == s2[j];
            }
        }

        for length in 2..=n {
            for i in 0..=n - length {
                for j in 0..=n - length {
                    cache[index(length, i, j)] = (1..length).any(|k| {
                        (cache[index(k, i, j)] && cache[index(length - k, i + k, j + k)])
                            || (cache[index(k, i, j + (length - k))]
                                && cache[index(length - k, i + k, j)])
                    });
                }
            }
        }

        cache[index(n, 0, 0)]
    }

    /// 165. 比较版本号 https://leetcode-cn.com/problems/compare-version-numbers/submissions/
    pub fn compare_version(version1: String, version2: String) -> i32 {
        let v1: Vec<&str> = version1.split('.').collect();
        let v2: Vec<&str> = version2.split('.').collect();
        let (len1, len2) = (v1.len(), v2.len());
        let mut i = 0;
        while i < len1 || i < len2 {
            let (mut x, mut y) = (0, 0);
            if i < len1 {
                x = v1[i].parse::<i32>().unwrap();
            }
            if i < len2 {
                y = v2[i].parse::<i32>().unwrap();
            }

            match x.cmp(&y) {
                Ordering::Greater => return 1,
                Ordering::Less => return -1,
                Ordering::Equal => i += 1,
            }
        }

        0
    }

    /// 力扣（468. 验证IP地址）  https://leetcode-cn.com/problems/validate-ip-address/
    /// 使用标准库中的方法
    pub fn valid_ip_address(ip: String) -> String {
        use std::net::IpAddr;
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
            Self::valid_ipv4_address_v2(ip)
        } else if ip.chars().filter(|ch| *ch == ':').count() == 7 {
            //println!("valid_ipv6_address..");
            // return valid_ipv6_address(ip);
            Self::valid_ipv6_address_v2(ip)
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

    fn trie_contains_word(mut root: &Node, word: &[u8]) -> bool {
        for c in word {
            if let Some(child) = root.children[usize::from(c - b'a')].as_deref() {
                root = child;
            } else {
                return false;
            }
        }
        root.has_value
    }

    fn is_concatenated_word(root: &Node, word: &[u8], cache: &mut Vec<bool>) -> bool {
        if word.len() < 2 {
            false
        } else {
            cache.reserve(word.len() + 1);
            cache.push(true);

            for end in 1..=word.len() {
                cache.push(cache[..end].iter().enumerate().any(|(start, &value)| {
                    value && Self::trie_contains_word(root, &word[start..end])
                }));
            }

            let result = *cache.last().unwrap();

            cache.clear();

            result
        }
    }

    fn trie_insert(mut root: &mut Node, word: &[u8]) {
        for c in word {
            root = root.children[usize::from(c - b'a')].get_or_insert_with(Box::default);
        }

        root.has_value = true;
    }

    /// 472. 连接词 https://leetcode-cn.com/problems/concatenated-words/
    /// 方法一：字典树 + 记忆化搜索
    pub fn find_all_concatenated_words_in_a_dict(words: Vec<String>) -> Vec<String> {
        let mut words = words;

        words.sort_by_key(String::len);

        //字典树
        let mut trie = Node::default();
        let mut cache = Vec::new();

        words.retain(|word| {
            let result = Self::is_concatenated_word(&trie, word.as_bytes(), &mut cache);

            Self::trie_insert(&mut trie, word.as_bytes());

            result
        });

        words
    }

    /// 482. 密钥格式化 https://leetcode-cn.com/problems/license-key-formatting/
    pub fn license_key_formatting(s: String, k: i32) -> String {
        let k = k as usize;
        let letters = s.bytes().filter(|&x| x != b'-').count();

        let mut iter = s.bytes().filter_map(|x| {
            if x == b'-' {
                None
            } else {
                Some(x.to_ascii_uppercase())
            }
        });

        let (first_group_size, rest_groups) = if letters % k == 0 {
            (k, (letters / k).saturating_sub(1))
        } else {
            (letters % k, letters / k)
        };

        let mut result = Vec::with_capacity(letters + rest_groups);

        result.extend(iter.by_ref().take(first_group_size));

        for _ in 0..rest_groups {
            result.push(b'-');
            result.extend(iter.by_ref().take(k));
        }

        String::from_utf8(result).unwrap()
    }

    fn get_row(c: u8) -> u8 {
        match c {
            b'Z' | b'X' | b'C' | b'V' | b'B' | b'N' | b'M' | b'z' | b'x' | b'c' | b'v' | b'b'
            | b'n' | b'm' => 3,
            b'A' | b'S' | b'D' | b'F' | b'G' | b'H' | b'J' | b'K' | b'L' | b'a' | b's' | b'd'
            | b'f' | b'g' | b'h' | b'j' | b'k' | b'l' => 2,
            _ => 0,
        }
    }

    /// 806. 写字符串需要的行数 https://leetcode-cn.com/problems/number-of-lines-to-write-string/
    pub fn number_of_lines(widths: Vec<i32>, s: String) -> Vec<i32> {
        let max_width = 100_u8;
        let mut remaining = max_width;
        let mut lines = 1;

        for c in s.bytes() {
            let width = widths[usize::from(c) - usize::from(b'a')] as u8;

            if let Some(new_remaining) = remaining.checked_sub(width) {
                remaining = new_remaining;
            } else {
                lines += 1;
                remaining = max_width - width;
            }
        }

        vec![lines, (max_width - remaining).into()]
    }

    /// 824. 山羊拉丁文 https://leetcode-cn.com/problems/goat-latin/
    pub fn to_goat_latin(sentence: String) -> String {
        let mut result = String::with_capacity(sentence.len() * 2);

        // map<i,word> => 闭包
        let mut iter = sentence
            .split_ascii_whitespace()
            .enumerate()
            .map(|(i, word)| {
                move |result: &mut String| {
                    match word.as_bytes()[0] {
                        b'A' | b'E' | b'I' | b'O' | b'U' | b'a' | b'e' | b'i' | b'o' | b'u' => {
                            result.push_str(word)
                        }
                        c => {
                            result.push_str(&word[1..]);
                            result.push(char::from(c));
                        }
                    }

                    result.push_str("maa");

                    for _ in 0..i {
                        result.push('a');
                    }
                }
            });

        iter.next().unwrap()(&mut result);

        for f in iter {
            result.push(' ');
            f(&mut result);
        }
        result
    }

    /// 500. 键盘行 https://leetcode-cn.com/problems/keyboard-row/
    pub fn find_words(words: Vec<String>) -> Vec<String> {
        let mut words = words;

        words.retain(|word| {
            let (&first, rest) = word.as_bytes().split_first().unwrap();
            let row = Self::get_row(first);

            rest.iter().all(|&c| Self::get_row(c) == row)
        });

        words
    }

    /// 966. 元音拼写检查器 https://leetcode-cn.com/problems/vowel-spellchecker/
    pub fn spellchecker(wordlist: Vec<String>, queries: Vec<String>) -> Vec<String> {
        use std::collections::{HashMap, HashSet};

        let as_is = wordlist.iter().map(String::as_str).collect::<HashSet<_>>();
        let mut capitalized = HashMap::with_capacity(wordlist.len());
        let mut vowels = HashMap::with_capacity(wordlist.len());

        for word in &wordlist {
            capitalized
                .entry(word.to_ascii_uppercase())
                .or_insert_with(|| word.as_str());

            vowels
                .entry(word.to_ascii_uppercase().replace(['E', 'I', 'O', 'U'], "A"))
                .or_insert_with(|| word.as_str());
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
}

/// 821. 字符的最短距离 https://leetcode-cn.com/problems/shortest-distance-to-a-character/
pub fn shortest_to_char(s: String, c: char) -> Vec<i32> {
    let c = c as u8;
    let mut result = vec![0; s.len()];
    let mut prev_position = i32::MIN;

    for (i, (distance, x)) in (0..).zip(result.iter_mut().zip(s.bytes())) {
        if x == c {
            prev_position = i;
        } else {
            *distance = i.saturating_sub(prev_position);
        }
    }

    prev_position = i32::MIN;

    for (i, (distance, x)) in (0..).zip(result.iter_mut().zip(s.bytes()).rev()) {
        if x == c {
            prev_position = i;
        } else {
            *distance = (*distance).min(i.saturating_sub(prev_position));
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

        dbg!(Solution::convert(s, 4));

        dbg!(Solution::my_atoi(" -423456".to_string()));
        dbg!(Solution::my_atoi("4193 with words".to_string()));
        dbg!(Solution::my_atoi("words and 987".to_string()));
        dbg!(Solution::my_atoi("-91283472332".to_string()));

        dbg!("3999 to roman {}", Solution::int_to_roman(3999));
        dbg!("3999 to roman {}", Solution::int_to_roman_v2(3999));

        let ip = String::from("2001:0db8:85a3:0:0:8A2E:0370:7334");
        dbg!(Solution::valid_ip_address2(ip));

        let license_key = String::from("5F3Z-2e-9-w");

        dbg!(Solution::license_key_formatting(license_key, 4));
    }

    #[test]
    fn distance() {
        let str = String::from("loveleetcode");
        let c = 'e';
        dbg!(shortest_to_char(str, c));
    }
}
