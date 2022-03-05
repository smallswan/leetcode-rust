/// 42. 接雨水 https://leetcode-cn.com/problems/trapping-rain-water/
pub fn trap(height: Vec<i32>) -> i32 {
    let mut result = 0;
    let mut iter = height.into_iter();

    if let (Some(mut left), Some(mut right)) = (iter.next(), iter.next_back()) {
        'outer: loop {
            if left < right {
                for middle in &mut iter {
                    if middle < left {
                        result += left - middle;
                    } else {
                        left = middle;

                        continue 'outer;
                    }
                }
            } else {
                while let Some(middle) = iter.next_back() {
                    if middle < right {
                        result += right - middle;
                    } else {
                        right = middle;

                        continue 'outer;
                    }
                }
            }

            break;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_reverse() {
        assert_eq!(trap(vec![4, 2, 0, 3, 2, 5]), 9);
    }
}
