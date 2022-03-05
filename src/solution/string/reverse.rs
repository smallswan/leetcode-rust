/// 557. 反转字符串中的单词 III https://leetcode-cn.com/problems/reverse-words-in-a-string-iii/
pub fn reverse_words(s: String) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let (mut i, mut j, mut k) = (0, 0, 0);
    while k < len {
        if chars[k] == ' ' {
            j = k - 1;
            while i < j {
                chars.swap(i, j);
                i += 1;
                j -= 1;
            }
            i = k + 1;
            j = i;
        }
        k += 1;
    }
    if j != len - 1 {
        j = len - 1;
        while i < j {
            chars.swap(i, j);
            i += 1;
            j -= 1;
        }
    }

    chars.iter().collect()
}

/// 917. 仅仅反转字母 https://leetcode-cn.com/problems/reverse-only-letters/
pub fn reverse_only_letters(s: String) -> String {
    let mut s = s.into_bytes();
    let mut iter = s.iter_mut().filter(|&&mut c| c.is_ascii_alphabetic());

    while let Some(left) = iter.next() {
        if let Some(right) = iter.next_back() {
            std::mem::swap(left, right);
        } else {
            break;
        }
    }

    String::from_utf8(s).unwrap()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_reverse() {
        dbg!(reverse_words(String::from("Let's take LeetCode contest")));

        assert_eq!(
            reverse_only_letters("Test1ng-Leet=code-Q!".into()),
            "Qedo1ct-eeLg=ntse-T!".to_string()
        );
    }
}
