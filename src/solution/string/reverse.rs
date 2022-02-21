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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_reverse() {
        dbg!(reverse_words(String::from("Let's take LeetCode contest")));
    }
}
