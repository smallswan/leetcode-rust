use std::collections::HashMap;
/// 力扣（1. 两数之和） https://leetcode-cn.com/problems/two-sum
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut nums_map = HashMap::<i32, i32>::new();
    for (idx, num) in nums.into_iter().enumerate() {
        let complement = target - num;

        let j = idx as i32;
        if let Some(idx) = nums_map.get(&complement) {
            return vec![*idx, j];
        }
        nums_map.insert(num, j);
    }
    vec![]
}

/// 56. 合并区间 https://leetcode-cn.com/problems/merge-intervals/
pub fn merge(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut intervals = intervals;

    intervals.sort_unstable_by_key(|interval| interval[0]);

    let mut result = Vec::new();
    let mut iter = intervals.into_iter();

    result.push(iter.next().unwrap());

    for interval in iter {
        let previous_end = result.last_mut().unwrap().last_mut().unwrap();

        if interval[0] <= *previous_end {
            *previous_end = (*previous_end).max(interval[1]);
        } else {
            result.push(interval);
        }
    }

    result
}

/// 163. 缺失的区间 https://leetcode-cn.com/problems/missing-ranges/
pub fn find_missing_ranges(nums: Vec<i32>, lower: i32, upper: i32) -> Vec<String> {
    fn find(a: i32, b: i32) -> Option<String> {
        match b - a {
            0 | 1 => None,
            2 => Some((a + 1).to_string()),
            _ => Some(format!("{}->{}", a + 1, b - 1)),
        }
    }

    let mut ans = vec![];
    //预处理，简化边界判断
    for window in [vec![lower - 1], nums, vec![upper + 1]].concat().windows(2) {
        if let Some(s) = find(window[0], window[1]) {
            ans.push(s);
        }
    }
    ans
}

/// 179. 最大数 https://leetcode-cn.com/problems/largest-number/
/// 拓展：剑指 Offer 45. 把数组排成最小的数 https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/
pub fn largest_number(nums: Vec<i32>) -> String {
    let mut nums_str: Vec<String> = nums.iter().map(|&num| format!("{}", num)).collect();
    nums_str.sort_by(|x, y| {
        let xy = format!("{}{}", x, y).parse::<u64>().unwrap();
        let yx = format!("{}{}", y, x).parse::<u64>().unwrap();

        yx.cmp(&xy)
    });

    if nums_str[0] == "0" {
        "0".to_string()
    } else {
        nums_str.iter().fold("".to_string(), |acc, num| acc + num)
    }
}

/// 189. 轮转数组 https://leetcode-cn.com/problems/rotate-array/
pub fn rotate(nums: &mut Vec<i32>, k: i32) {
    let len = nums.len();
    nums.rotate_right(k as usize % len)
}

/// 252. 会议室 https://leetcode-cn.com/problems/meeting-rooms/
pub fn can_attend_meetings(intervals: Vec<Vec<i32>>) -> bool {
    let len = intervals.len();
    if len <= 1 {
        return true;
    }
    let mut intervals = intervals;
    intervals.sort_by(|a, b| a[0].cmp(&b[0]));

    for i in 0..len - 1 {
        if intervals[i][1] > intervals[i + 1][0] {
            return false;
        }
    }

    true
}

/// 256. 粉刷房子 https://leetcode-cn.com/problems/paint-house/
/// 最优化原理
pub fn min_cost(costs: Vec<Vec<i32>>) -> i32 {
    let (mut first, mut second, mut third) = (-1, -1, -1);
    let len = costs.len();
    for cost in costs.iter().take(len) {
        if first == -1 {
            first = cost[0];
            second = cost[1];
            third = cost[2];
        } else {
            let t1 = (cost[0] + second).min(cost[0] + third);
            let t2 = (cost[1] + first).min(cost[1] + third);
            let t3 = (cost[2] + first).min(cost[2] + second);
            first = t1;
            second = t2;
            third = t3;
        }
    }

    first.min(second.min(third))
}

/// 303. 区域和检索 - 数组不可变 https://leetcode-cn.com/problems/range-sum-query-immutable/
struct NumArray {
    data: Vec<i32>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl NumArray {
    fn new(nums: Vec<i32>) -> Self {
        NumArray { data: nums }
    }

    fn sum_range(&self, left: i32, right: i32) -> i32 {
        let (mut left, mut right) = (left as usize, right as usize);
        let mut result = 0;
        while left <= right {
            result += self.data[left];
            left += 1;
        }
        result
    }
}

/// 495. 提莫攻击 https://leetcode-cn.com/problems/teemo-attacking/
pub fn find_poisoned_duration(time_series: Vec<i32>, duration: i32) -> i32 {
    let mut result = 0;
    let mut poisoned = i32::MIN;
    let mut healed = i32::MIN;

    for time_point in time_series {
        if time_point > healed {
            result += healed - poisoned;
            poisoned = time_point;
        }

        healed = time_point + duration;
    }

    result + healed - poisoned
}

/// 661. 图片平滑器 https://leetcode-cn.com/problems/image-smoother/
pub fn image_smoother(img: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let rows = img.len();
    let columns = img.first().map_or(0, Vec::len);
    let mut result = Vec::with_capacity(rows);

    for y in 0..rows {
        let mut output_row = Vec::with_capacity(columns);

        for x in 0..columns {
            let mut sum = 0;
            let mut count = 0;

            for input_row in &img[y.saturating_sub(1)..(y + 2).min(rows)] {
                for num in &input_row[x.saturating_sub(1)..(x + 2).min(columns)] {
                    sum += num;
                    count += 1;
                }
            }

            output_row.push(sum / count);
        }

        result.push(output_row);
    }

    result
}

/// 905. 按奇偶排序数组 https://leetcode-cn.com/problems/sort-array-by-parity/
pub fn sort_array_by_parity(nums: Vec<i32>) -> Vec<i32> {
    let mut nums = nums;
    let len = nums.len();
    let (mut left, mut right) = (0, len - 1);
    while left < right {
        if nums[left] % 2 > nums[right] % 2 {
            nums.swap(left, right);
        }
        if nums[left] % 2 == 0 {
            left += 1;
        }
        if nums[right] % 2 == 1 {
            right -= 1;
        }
    }
    nums
}

use std::cmp::Ordering;
/// 941. 有效的山脉数组  https://leetcode-cn.com/problems/valid-mountain-array/
pub fn valid_mountain_array(arr: Vec<i32>) -> bool {
    // slice pattern
    if let [first, second, ref rest @ ..] = *arr {
        if first < second {
            let mut prev = second;
            let mut iter = rest.iter().copied();

            loop {
                if let Some(num) = iter.next() {
                    match num.cmp(&prev) {
                        Ordering::Less => {
                            prev = num;

                            break;
                        }
                        Ordering::Equal => return false,
                        Ordering::Greater => prev = num,
                    }
                } else {
                    return false;
                }
            }

            for num in iter {
                if num < prev {
                    prev = num;
                } else {
                    return false;
                }
            }

            true
        } else {
            false
        }
    } else {
        false
    }
}

/// 961. 在长度 2N 的数组中找出重复 N 次的元素 https://leetcode-cn.com/problems/n-repeated-element-in-size-2n-array/
pub fn repeated_n_times(nums: Vec<i32>) -> i32 {
    let mut iter = nums.iter().copied().enumerate();

    loop {
        let (i, num) = iter.next().unwrap();

        for &prev in &nums[i.saturating_sub(3)..i] {
            if num == prev {
                return num;
            }
        }
    }
}

/// 1672. 最富有客户的资产总量 https://leetcode.cn/problems/richest-customer-wealth/
pub fn maximum_wealth(accounts: Vec<Vec<i32>>) -> i32 {
    accounts
        .iter()
        .map(|account| account.iter().sum())
        .max()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector() {
        let nums = vec![2, 7, 2, 11];
        let result = two_sum(nums, 9);
        dbg!(result);
    }
    #[test]
    fn test_rotate() {
        let mut nums = vec![1, 2, 3, 4, 5, 6, 7];
        let k = 3;
        rotate(&mut nums, k);
    }

    #[test]
    fn test_array() {
        let array = NumArray::new(vec![1, 2, 3, 4, 5, 6, 7]);
        dbg!(array.sum_range(2, 5));

        let time_series: Vec<i32> = vec![1, 4];
        let duration: i32 = 2;
        dbg!(find_poisoned_duration(time_series, duration));

        let wealth = vec![vec![1, 2, 3], vec![3, 2, 1]];

        assert_eq!(maximum_wealth(wealth), 6)
    }
}
