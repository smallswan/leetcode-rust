/// 力扣编号大于1000的题目

/// 力扣（1486. 数组异或操作） https://leetcode-cn.com/problems/xor-operation-in-an-array/
pub fn xor_operation(n: i32, start: i32) -> i32 {
    (1..n).fold(start, |acc, i| acc ^ start + 2 * i as i32)
}
