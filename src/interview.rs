use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// 面试题 01.06. 字符串压缩 https://leetcode-cn.com/problems/compress-string-lcci/
pub fn compress_string(s: String) -> String {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut cnt = 1;
    let mut ch = chars[0];
    let mut ans = String::new();
    for item in chars.iter().take(len).skip(1) {
        if ch == *item {
            cnt += 1;
        } else {
            ans.push(ch);
            ans.push_str(&format!("{}", cnt));
            ch = *item;
            cnt = 1;
        }
    }

    ans.push(ch);
    ans.push_str(&format!("{}", cnt));

    if ans.len() >= len {
        s
    } else {
        ans
    }
}

/// 面试题 01.07. 旋转矩阵 https://leetcode-cn.com/problems/rotate-matrix-lcci/
pub fn rotate(matrix: &mut Vec<Vec<i32>>) {
    let n = matrix.len();

    for first in 0..n / 2 {
        let last = n - 1 - first;

        for i in first..last {
            let j = n - 1 - i;

            let temp = matrix[first][i];

            matrix[first][i] = matrix[j][first];
            matrix[j][first] = matrix[last][j];
            matrix[last][j] = matrix[i][last];
            matrix[i][last] = temp;
        }
    }
}

/// 面试题 17.14. 最小K个数 https://leetcode-cn.com/problems/smallest-k-lcci/
pub fn smallest_k(arr: Vec<i32>, k: i32) -> Vec<i32> {
    let len = arr.len();
    let mut heap = BinaryHeap::from(arr);
    //去掉 len - k个大的数字
    for i in 0..(len - (k as usize)) {
        heap.pop();
    }
    heap.into_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interview() {
        let arr = vec![1, 3, 5, 7, 2, 4, 6, 8];
        let k = 4;
        dbg!(smallest_k(arr, k));
    }
}
