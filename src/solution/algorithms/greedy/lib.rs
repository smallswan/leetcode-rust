//! 贪心算法（greedy）
//! https://leetcode-cn.com/tag/greedy/problemset/

/// 45. 跳跃游戏 II https://leetcode-cn.com/problems/jump-game-ii/
/// 方法一：反向查找出发位置
pub fn jump(nums: Vec<i32>) -> i32 {
    let mut position = nums.len() - 1;
    let mut steps = 0;
    while position > 0 {
        for (i, num) in nums.iter().enumerate().take(position) {
            if i + (*num as usize) >= position {
                position = i;
                steps += 1;
                break;
            }
        }
    }

    steps
}

use std::cmp::max;
/// 45. 跳跃游戏 II
/// 方法二：正向查找可到达的最大位置
pub fn jump_v2(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    let mut end = 0;
    let mut max_position = 0;
    let mut steps = 0;
    for (i, num) in nums.iter().enumerate().take(len - 1) {
        max_position = max_position.max(i + (*num as usize));
        if i == end {
            end = max_position;
            steps += 1;
        }
    }

    steps
}

/// 55. 跳跃游戏 https://leetcode-cn.com/problems/jump-game/
pub fn can_jump(nums: Vec<i32>) -> bool {
    let len = nums.len();
    let mut end = 0;
    let mut max_position = 0;
    for (i, num) in nums.iter().enumerate().take(len) {
        if i <= max_position {
            max_position = max_position.max(i + (*num as usize));
            if max_position >= len - 1 {
                return true;
            }
        }
    }

    false
}

/// 力扣（121. 买卖股票的最佳时机）  https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
/// 暴力解法，容易超时
pub fn max_profit(prices: Vec<i32>) -> i32 {
    let len = prices.len();
    if len <= 1 {
        return 0;
    }
    let mut buy_day = 0;
    let mut sale_day = 1;
    let mut max_profit = 0;
    while buy_day < len - 1 {
        while sale_day < len {
            let profit = prices[sale_day] - prices[buy_day];
            if profit > 0 {
                max_profit = max(max_profit, profit);
            }
            sale_day += 1;
        }
        buy_day += 1;
        sale_day = buy_day + 1;
    }

    max_profit
}

/// 力扣（121. 买卖股票的最佳时机）
/// 剑指 Offer 63. 股票的最大利润  https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/
/// 最低价格、最大利润
pub fn max_profit_v2(prices: Vec<i32>) -> i32 {
    let len = prices.len();
    let mut min_prince = i32::MAX;
    let mut max_profit = 0;
    for price in prices {
        if price < min_prince {
            min_prince = price;
        } else if price - min_prince > max_profit {
            max_profit = price - min_prince;
        }
    }

    max_profit
}

/// 122. 买卖股票的最佳时机 II https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
/// 方法：贪心算法
pub fn max_profit_ii(prices: Vec<i32>) -> i32 {
    let mut buy = i32::MIN;
    let mut sell = 0;

    for price in prices {
        buy = buy.max(sell - price);
        sell = sell.max(buy + price);
    }

    sell
}

/// 122. 买卖股票的最佳时机 II
/// 方法：动态规划
pub fn max_profit_ii_v2(prices: Vec<i32>) -> i32 {
    let len = prices.len();
    let mut dp = Vec::with_capacity(len);
    // (profit1,profit2) = (第 i 天交易完后手里没有股票的最大利润,表示第 i 天交易完后手里持有一支股票的最大利润)
    dp.push((0, -prices[0]));
    for i in 1..len {
        let profit1 = max(dp[i - 1].0, dp[i - 1].1 + prices[i]);
        let profit2 = max(dp[i - 1].1, dp[i - 1].0 - prices[i]);
        dp.push((profit1, profit2));
    }

    dp[len - 1].0
}

/// 134. 加油站 https://leetcode-cn.com/problems/gas-station/
pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
    let len = gas.len();
    let mut i = 0;
    while i < len {
        let (mut sum_of_gas, mut sum_of_cost) = (0, 0);
        let mut count = 0;
        while count < len {
            let j = (i + count) % len;
            sum_of_gas += gas[j];
            sum_of_cost += cost[j];
            if sum_of_cost > sum_of_gas {
                break;
            }
            count += 1;
        }

        if count == len {
            return i as i32;
        } else {
            i += count + 1;
        }
    }

    -1
}

use std::cmp::Ordering;
fn triangular(x: i32) -> i32 {
    x * (x + 1) / 2
}

fn triangular_2(x: i32, y: i32) -> i32 {
    (x * (x + 1) + y * (y + 1)) / 2
}

fn going_down(first: i32, rest: &[i32], up_length: i32, down_length: i32, result: &mut i32) {
    if let Some((&second, rest)) = rest.split_first() {
        match second.cmp(&first) {
            Ordering::Less => going_down(second, rest, up_length, down_length + 1, result),
            Ordering::Equal => {
                *result += triangular_2(up_length, down_length) + up_length.max(down_length) + 1;

                going_up(second, rest, 0, result);
            }
            Ordering::Greater => {
                *result += triangular_2(up_length, down_length) + up_length.max(down_length);

                going_up(second, rest, 1, result);
            }
        }
    } else {
        *result += triangular_2(up_length, down_length) + up_length.max(down_length) + 1;
    }
}

fn going_up(first: i32, rest: &[i32], up_length: i32, result: &mut i32) {
    if let Some((&second, rest)) = rest.split_first() {
        match second.cmp(&first) {
            Ordering::Less => going_down(second, rest, up_length, 1, result),
            Ordering::Equal => {
                *result += triangular(up_length) + up_length + 1;

                going_up(second, rest, 0, result);
            }
            Ordering::Greater => going_up(second, rest, up_length + 1, result),
        }
    } else {
        *result += triangular(up_length) + up_length + 1;
    }
}

/// 135. 分发糖果 https://leetcode-cn.com/problems/candy/
pub fn candy(ratings: Vec<i32>) -> i32 {
    let mut result = 0;
    let (&first, rest) = ratings.split_first().unwrap();

    going_up(first, rest, 0, &mut result);

    result
}

/// 力扣（561. 数组拆分 I） https://leetcode-cn.com/problems/array-partition-i/
pub fn array_pair_sum(nums: Vec<i32>) -> i32 {
    let len = nums.len();
    if len % 2 != 0 {
        panic!("数组长度必须为偶数");
    }

    let mut nums_sort = nums;
    nums_sort.sort_unstable();

    let mut sum = 0;
    for i in 0..len / 2 {
        sum += nums_sort[2 * i];
    }

    sum
}

/// 605. 种花问题 https://leetcode-cn.com/problems/can-place-flowers/
pub fn can_place_flowers(flowerbed: Vec<i32>, n: i32) -> bool {
    if n == 0 {
        true
    } else {
        let mut n = n;
        let (&first, rest) = flowerbed.split_first().unwrap();
        let mut prev = (0, first);

        for &num in rest {
            if (prev, num) == ((0, 0), 0) {
                if n == 1 {
                    return true;
                }

                n -= 1;

                prev = (1, 0);
            } else {
                prev = (prev.1, num);
            }
        }

        prev == (0, 0) && n == 1
    }
}

/// 605. 种花问题
/// 方法：“跳格子” https://leetcode-cn.com/problems/can-place-flowers/solution/fei-chang-jian-dan-de-tiao-ge-zi-jie-fa-nhzwc/
pub fn can_place_flowers_v2(flowerbed: Vec<i32>, n: i32) -> bool {
    let len = flowerbed.len();
    let mut i = 0;
    let mut n = n;
    while i < len && n > 0 {
        if flowerbed[i] == 1 {
            //当前位置有花，下一个可能种植花的地方为i+2
            i += 2;
        } else if i == len - 1 || flowerbed[i + 1] == 0 {
            // 走到这里，说明flowerbed[i]=0，如果当前位置是最后一个位置或者下一个位置也无花，则可以种植
            n -= 1;
            if n == 0 {
                return true;
            }
            i += 2;
        } else {
            // 当前位置无花，但是由于下一个位置有花，所以不能种花，二下一个可能种花的地方就是i+3
            i += 3;
        }
    }

    n == 0
}

/// 680. 验证回文字符串 Ⅱ https://leetcode-cn.com/problems/valid-palindrome-ii/
/// 注：相同的题目有： 银联-01. 回文链表 https://leetcode-cn.com/contest/cnunionpay-2022spring/problems/D7rekZ/
pub fn valid_palindrome(s: String) -> bool {
    fn is_palindrome(mut iter: impl DoubleEndedIterator<Item = impl Eq>) -> bool {
        while let (Some(left), Some(right)) = (iter.next(), iter.next_back()) {
            if left != right {
                return false;
            }
        }

        true
    }
    let mut iter = s.bytes();
    while let (Some(left), Some(right)) = (iter.next(), iter.next_back()) {
        if left != right {
            let mut iter_2 = iter.clone();

            if let Some(left_2) = iter.next() {
                return (left_2 == right && is_palindrome(iter))
                    || (iter_2.next_back() == Some(left) && is_palindrome(iter_2));
            }

            break;
        }
    }

    true
}

/// 860. 柠檬水找零 https://leetcode-cn.com/problems/lemonade-change/
pub fn lemonade_change(bills: Vec<i32>) -> bool {
    let mut changes: Vec<(i32, i32)> = vec![(5, 0), (10, 0), (20, 0)];
    let len = bills.len();
    for bill in bills.iter().take(len) {
        match bill {
            5 => {
                changes[0].1 += 1;
            }
            10 => {
                if changes[0].1 > 0 {
                    changes[0].1 -= 1;
                } else {
                    return false;
                }
                changes[1].1 += 1;
            }
            20 => {
                // 找15元零钱: 10+5; 5+5+5
                let mut change = 15;
                if (5 * changes[0].1 + 10 * changes[1].1) >= 15 {
                    if changes[1].1 > 0 && changes[0].1 > 0 {
                        changes[0].1 -= 1;
                        changes[1].1 -= 1;
                    } else if changes[0].1 >= 3 {
                        changes[0].1 -= 3;
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            _ => (),
        }
    }

    true
}

/// 942. 增减字符串匹配 https://leetcode.cn/problems/di-string-match/
/// 贪心算法
pub fn di_string_match(s: String) -> Vec<i32> {
    let len = s.len();
    let mut result = Vec::with_capacity(len + 1);
    let mut min = 0;
    let mut max = len as i32;

    for c in s.bytes() {
        if c == b'D' {
            result.push(max);
            max -= 1;
        } else {
            result.push(min);
            min += 1;
        }
    }

    result.push(min);

    result
}

/// 976. 三角形的最大周长 https://leetcode-cn.com/problems/largest-perimeter-triangle/
/// 我们假设三角形的边长 a,b,c 满足 a≤b≤c，那么这三条边组成面积不为零的三角形的充分必要条件为 a+b>c
pub fn largest_perimeter(nums: Vec<i32>) -> i32 {
    let mut nums = nums;
    nums.sort_unstable();
    let mut i = nums.len() - 1;
    while i >= 2 {
        if nums[i - 2] + nums[i - 1] > nums[i] {
            return nums[i - 2] + nums[i - 1] + nums[i];
        }
        i -= 1;
    }
    0
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let mut nums = Vec::<i32>::new();
        nums.push(1);
        nums.push(4);
        nums.push(3);
        nums.push(2);
        dbg!(array_pair_sum(nums));

        let ratings = vec![1, 2, 2];
        dbg!(candy(ratings));
    }

    #[test]
    fn test_stock() {
        let prices = vec![7, 1, 5, 3, 6, 4];

        dbg!(max_profit(prices));

        dbg!(max_profit_v2(vec![7, 1, 5, 3, 6, 4]));

        dbg!(max_profit_ii(vec![7, 1, 5, 3, 6, 4]));
        dbg!(max_profit_ii_v2(vec![7, 1, 5, 3, 6, 4]));
    }
}
