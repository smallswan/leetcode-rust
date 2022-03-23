/// 434. 字符串中的单词数 https://leetcode-cn.com/problems/number-of-segments-in-a-string/
pub fn count_segments(s: String) -> i32 {
    let mut result = 0;
    let mut iter = s.bytes();

    while let Some(c) = iter.next() {
        if c != b' ' {
            result += 1;

            loop {
                if let Some(c) = iter.next() {
                    if c == b' ' {
                        break;
                    }
                } else {
                    return result;
                }
            }
        }
    }

    result
}
