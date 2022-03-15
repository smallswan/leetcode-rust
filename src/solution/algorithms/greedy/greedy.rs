//! 贪心算法（greedy）
//! https://leetcode-cn.com/tag/greedy/problemset/

/// 45. 跳跃游戏 II https://leetcode-cn.com/problems/jump-game-ii/
/// 方法一：反向查找出发位置
pub fn jump(nums: Vec<i32>) -> i32 {
    let mut position = nums.len() - 1;
    let mut steps = 0;
    while position > 0 {
        for i in 0..position {
            if i + (nums[i] as usize) >= position {
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
    for i in 0..len - 1 {
        max_position = max(max_position, i + (nums[i] as usize));
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
    for i in 0..len {
        if i <= max_position {
            max_position = max(max_position, i + (nums[i] as usize));
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
