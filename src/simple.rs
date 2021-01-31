/// 简单难度

/// 力扣（9. 回文数） https://leetcode-cn.com/problems/palindrome-number/
/// 数字转字符串
pub fn is_palindrome(x: i32) -> bool {
    // 负数不是回文数
    if x < 0 {
        return false;
    }
    let s = x.to_string();
    let arr = s.as_bytes();
    let len = arr.len();
    for (c1, c2) in (0..len / 2)
        .into_iter()
        .zip((len / 2..len).rev().into_iter())
    {
        if arr[c1] != arr[c2] {
            return false;
        }
    }
    true
}

/// 反转一半的数字
pub fn is_palindrome_v2(x: i32) -> bool {
    if x < 0 || (x % 10 == 0 && x != 0) {
        return false;
    }
    let mut y = x;
    let mut reverted_num = 0;
    while y > reverted_num {
        reverted_num = reverted_num * 10 + y % 10;
        y /= 10;
    }
    return y == reverted_num || y == reverted_num / 10;
}

#[test]
fn simple_test() {
    println!("{}", is_palindrome(121));
    println!("{}", is_palindrome(-121));
    println!("{}", is_palindrome(10));
    println!("{}", is_palindrome(1));
}
