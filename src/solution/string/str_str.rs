//! 子字符串查找
//! 常见算法：KMP(Knuth-Morris-Pratt)算法； Rabin-Karp指纹字符串查找算法；

/// 力扣（28. 实现 strStr()）  https://leetcode-cn.com/problems/implement-strstr/
/// 当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。
pub fn str_str(haystack: String, needle: String) -> i32 {
    // 参考Java String.indexOf()的代码
    let source = haystack.as_bytes();
    let target = needle.as_bytes();

    let source_offset = 0usize;
    let source_count = source.len();
    let target_offset = 0usize;
    let target_count = target.len();
    let from_index = 0usize;
    if target_count == 0usize {
        return 0;
    }

    if target_count > source_count {
        return -1;
    }

    let first = target[target_offset];
    let max = source_offset + (source_count - target_count);

    let mut i = source_offset + from_index;
    while i <= max {
        // 首先匹配首字母
        while source[i] != first {
            i += 1;
            if i <= max {
                continue;
            } else {
                break;
            }
        }

        if i <= max {
            let mut j = i + 1;
            let end = j + target_count - 1;
            let mut k = target_offset + 1;
            // 匹配剩余的字符
            while j < end && source[j] == target[k] {
                j += 1;
                k += 1;
            }

            if j == end {
                return (i - source_offset) as i32;
            }
        }

        i += 1;
    }

    -1
}

/// 力扣（28. 实现 strStr()）
/// 系统内置方法
pub fn str_str_v2(haystack: String, needle: String) -> i32 {
    match haystack.find(&needle) {
        Some(index) => index as i32,
        None => return -1,
    }
}

/// 力扣（28. 实现 strStr()）
/// KMP(Knuth-Morris-Pratt)算法
/// 前缀函数，记作 π(i)，其定义如下：
/// 对于长度为 m 的字符串 s，其前缀函数 π(i)(0≤i<m) 表示 s 的子串 s[0:i] 的最长的相等的真前缀与真后缀的长度。特别地，如果不存在符合条件的前后缀，那么 π(i)=0。
pub fn str_str_v3(haystack: String, needle: String) -> i32 {
    let (m, n) = (needle.len(), haystack.len());
    if m == 0 {
        return 0;
    }
    let haystack_chars = haystack.chars().collect::<Vec<char>>();
    let needle_chars = needle.chars().collect::<Vec<char>>();
    let mut pi = vec![0; m];
    let (mut i, mut j) = (1, 0);
    while i < m {
        while j > 0 && (&needle_chars[i] != &needle_chars[j]) {
            j = pi[j - 1];
        }
        // 如果 s[i]=s[π(i−1)]，那么 π(i)=π(i−1)+1。
        if &needle_chars[i] == &needle_chars[j] {
            j += 1;
        }
        pi[i] = j;
        i += 1;
    }

    let (mut i, mut j) = (0, 0);
    while i < n {
        while j > 0 && (&haystack_chars[i] != &needle_chars[j]) {
            j = pi[j - 1];
        }
        if &haystack_chars[i] == &needle_chars[j] {
            j += 1;
        }
        if (j == m) {
            return (i - m + 1) as i32;
        }
        i += 1;
    }

    -1
}

/// https://github.com/TheAlgorithms/Rust/blob/master/src/string/knuth_morris_pratt.rs
pub fn knuth_morris_pratt(st: String, pat: String) -> Vec<usize> {
    if st.is_empty() || pat.is_empty() {
        return vec![];
    }

    let string = st.into_bytes();
    let pattern = pat.into_bytes();

    // build the partial match table
    let mut partial = vec![0];
    for i in 1..pattern.len() {
        let mut j = partial[i - 1];
        while j > 0 && pattern[j] != pattern[i] {
            j = partial[j - 1];
        }
        partial.push(if pattern[j] == pattern[i] { j + 1 } else { j });
    }

    // and read 'string' to find 'pattern'
    let mut ret = vec![];
    let mut j = 0;

    for (i, &c) in string.iter().enumerate() {
        while j > 0 && c != pattern[j] {
            j = partial[j - 1];
        }
        if c == pattern[j] {
            j += 1;
        }
        if j == pattern.len() {
            ret.push(i + 1 - j);
            j = partial[j - 1];
        }
    }

    ret
}

/// https://github.com/TheAlgorithms/Rust/blob/master/src/string/rabin_karp.rs
pub fn rabin_karp(target: String, pattern: String) -> Vec<usize> {
    // Quick exit
    if target.is_empty() || pattern.is_empty() || pattern.len() > target.len() {
        return vec![];
    }

    let string: String = (&pattern[0..pattern.len()]).to_string();
    let hash_pattern = hash(string.clone());
    let mut ret = vec![];
    for i in 0..(target.len() - pattern.len() + 1) {
        let s = (&target[i..(i + pattern.len())]).to_string();
        let string_hash = hash(s.clone());

        if string_hash == hash_pattern && s == string {
            ret.push(i);
        }
    }
    ret
}

fn hash(mut s: String) -> u16 {
    let prime: u16 = 101;
    let last_char = s
        .drain(s.len() - 1..)
        .next()
        .expect("Failed to get the last char of the string");
    let mut res: u16 = 0;
    for (i, &c) in s.as_bytes().iter().enumerate() {
        if i == 0 {
            res = (c as u16 * 256) % prime;
        } else {
            res = (((res + c as u16) % 101) * 256) % 101;
        }
    }
    (res + last_char as u16) % prime
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_string() {
        let haystack = String::from("aaacaaab");
        let needle = String::from("aaab");
        dbg!(str_str(haystack, needle));

        let index = knuth_morris_pratt("Rust is a programming language empowering everyone to build reliable and efficient software".to_string(),"everyone".to_string());
        println!("{:?}", index);
    }

    mod kmp {
        use super::*;
        #[test]
        fn each_letter_matches() {
            let index = knuth_morris_pratt("aaa".to_string(), "a".to_string());
            assert_eq!(index, vec![0, 1, 2]);
        }

        #[test]
        fn a_few_separate_matches() {
            let index = knuth_morris_pratt("abababa".to_string(), "ab".to_string());
            assert_eq!(index, vec![0, 2, 4]);
        }

        #[test]
        fn one_match() {
            let index =
                knuth_morris_pratt("ABC ABCDAB ABCDABCDABDE".to_string(), "ABCDABD".to_string());
            assert_eq!(index, vec![15]);
        }

        #[test]
        fn lots_of_matches() {
            let index = knuth_morris_pratt("aaabaabaaaaa".to_string(), "aa".to_string());
            assert_eq!(index, vec![0, 1, 4, 7, 8, 9, 10]);
        }

        #[test]
        fn lots_of_intricate_matches() {
            let index = knuth_morris_pratt("ababababa".to_string(), "aba".to_string());
            assert_eq!(index, vec![0, 2, 4, 6]);
        }

        #[test]
        fn not_found0() {
            let index = knuth_morris_pratt("abcde".to_string(), "f".to_string());
            assert_eq!(index, vec![]);
        }

        #[test]
        fn not_found1() {
            let index = knuth_morris_pratt("abcde".to_string(), "ac".to_string());
            assert_eq!(index, vec![]);
        }

        #[test]
        fn not_found2() {
            let index = knuth_morris_pratt("ababab".to_string(), "bababa".to_string());
            assert_eq!(index, vec![]);
        }

        #[test]
        fn empty_string() {
            let index = knuth_morris_pratt("".to_string(), "abcdef".to_string());
            assert_eq!(index, vec![]);
        }
    }

    mod rabin_karp {
        use super::*;

        ///  Tests
        #[test]
        fn hi_hash() {
            let hash_result = hash("hi".to_string());
            assert_eq!(hash_result, 65);
        }

        #[test]
        fn abr_hash() {
            let hash_result = hash("abr".to_string());
            assert_eq!(hash_result, 4);
        }

        #[test]
        fn bra_hash() {
            let hash_result = hash("bra".to_string());
            assert_eq!(hash_result, 30);
        }

        // Attribution to @pgimalac for his tests from Knuth-Morris-Pratt
        #[test]
        fn each_letter_matches() {
            let index = rabin_karp("aaa".to_string(), "a".to_string());
            assert_eq!(index, vec![0, 1, 2]);
        }

        #[test]
        fn a_few_separate_matches() {
            let index = rabin_karp("abababa".to_string(), "ab".to_string());
            assert_eq!(index, vec![0, 2, 4]);
        }

        #[test]
        fn one_match() {
            let index = rabin_karp("ABC ABCDAB ABCDABCDABDE".to_string(), "ABCDABD".to_string());
            assert_eq!(index, vec![15]);
        }

        #[test]
        fn lots_of_matches() {
            let index = rabin_karp("aaabaabaaaaa".to_string(), "aa".to_string());
            assert_eq!(index, vec![0, 1, 4, 7, 8, 9, 10]);
        }

        #[test]
        fn lots_of_intricate_matches() {
            let index = rabin_karp("ababababa".to_string(), "aba".to_string());
            assert_eq!(index, vec![0, 2, 4, 6]);
        }

        #[test]
        fn not_found0() {
            let index = rabin_karp("abcde".to_string(), "f".to_string());
            assert_eq!(index, vec![]);
        }

        #[test]
        fn not_found1() {
            let index = rabin_karp("abcde".to_string(), "ac".to_string());
            assert_eq!(index, vec![]);
        }

        #[test]
        fn not_found2() {
            let index = rabin_karp("ababab".to_string(), "bababa".to_string());
            assert_eq!(index, vec![]);
        }

        #[test]
        fn empty_string() {
            let index = rabin_karp("".to_string(), "abcdef".to_string());
            assert_eq!(index, vec![]);
        }
    }
}
