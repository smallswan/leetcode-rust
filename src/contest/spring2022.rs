use crate::solution::data_structures::lists::ListNode;

///  银联-01. 回文链表 https://leetcode-cn.com/contest/cnunionpay-2022spring/problems/D7rekZ/
///  暴力解法，竞赛时超时了
pub fn is_palindrome(head: Option<Box<ListNode>>) -> bool {
    let mut vec = Vec::new();
    let mut head = &head;
    while head.is_some() {
        vec.push(head.as_ref().unwrap().val);
        head = &(head.as_ref().unwrap().next);
    }

    let len = vec.len();
    for i in 0..len {
        let mut new_vec = Vec::with_capacity(len - 1);
        for j in 0..len {
            if j != i {
                new_vec.push(vec[j]);
            }
        }

        if is_palindrome_vec(&new_vec) {
            return true;
        }
    }
    //
    fn is_palindrome_vec(data: &Vec<i32>) -> bool {
        let (mut i, mut j) = (0, data.len() - 1);
        while i < j {
            if data[i] == data[j] {
                i += 1;
                j -= 1;
            } else {
                return false;
            }
        }
        true
    }

    false
}

/// 判断迭代器中的元素是否是回文
fn is_palindrome_iter(mut iter: impl DoubleEndedIterator<Item = impl Eq>) -> bool {
    while let (Some(left), Some(right)) = (iter.next(), iter.next_back()) {
        if left != right {
            return false;
        }
    }

    true
}

/// 银联-01. 回文链表
/// 贪心算法:
/// 在允许最多删除一个字符的情况下，同样可以使用双指针，通过贪心实现。初始化两个指针 low和 high分别指向字符串的第一个字符和最后一个字符。每次判断两个指针指向的字符是否相同，如果相同，则更新指针，将 low 加 1，high 减 1，然后判断更新后的指针范围内的子串是否是回文字符串。
/// 如果两个指针指向的字符不同，则两个字符中必须有一个被删除，此时我们就分成两种情况：即删除左指针对应的字符，留下子串 s[low+1:high]，或者删除右指针对应的字符，留下子串 s[low:high−1]。
/// 当这两个子串中至少有一个是回文串时，就说明原始字符串删除一个字符之后就以成为回文串。
pub fn is_palindrome_v2(head: Option<Box<ListNode>>) -> bool {
    let mut vec = Vec::new();
    let mut head = &head;
    while head.is_some() {
        vec.push(head.as_ref().unwrap().val);
        head = &(head.as_ref().unwrap().next);
    }

    let mut iter = vec.iter();
    while let (Some(left), Some(right)) = (iter.next(), iter.next_back()) {
        if left != right {
            let mut iter_2 = iter.clone();

            if let Some(left_2) = iter.next() {
                return (left_2 == right && is_palindrome_iter(iter))
                    || (iter_2.next_back() == Some(left) && is_palindrome_iter(iter_2));
            }

            break;
        }
    }

    true
}

/// 银联-02. 优惠活动系统 https://leetcode-cn.com/contest/cnunionpay-2022spring/problems/kDPV0f/
use std::collections::HashMap;
struct DiscountSystem {
    activities: HashMap<i32, Activity>,
}

struct Activity {
    consume_records: Vec<i32>,
    act_id: i32,
    price_limit: i32,
    discount: i32,
    number: i32,
    user_limit: i32,
}
/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl DiscountSystem {
    fn new() -> Self {
        DiscountSystem {
            activities: HashMap::new(),
        }
    }

    fn add_activity(
        &mut self,
        act_id: i32,
        price_limit: i32,
        discount: i32,
        number: i32,
        user_limit: i32,
    ) {
        let activity = Activity {
            //consume_records[user_id]记录参加活动的次数
            consume_records: vec![0; 1001],
            act_id,
            price_limit,
            discount,
            number,
            user_limit,
        };
        self.activities.insert(act_id, activity);
    }

    fn remove_activity(&mut self, act_id: i32) {
        self.activities.remove(&act_id);
    }

    fn consume(&mut self, user_id: i32, cost: i32) -> i32 {
        if self.activities.is_empty() {
            cost
        } else {
            let mut max_discount = 0;
            let mut min_act_id = i32::MAX;

            self.activities.values().for_each(|value| {
                let sum = value.consume_records.iter().sum::<i32>();
                if cost >= value.price_limit && sum < value.number {
                    if value.consume_records[(user_id as usize)] < value.user_limit {
                        if value.discount > max_discount {
                            // 若同时满足多个优惠活动时，则优先参加优惠减免最大的活动
                            max_discount = value.discount;
                            min_act_id = value.act_id;
                        } else if value.discount == max_discount {
                            // 相同折扣优先使用act_id小的
                            if value.act_id < min_act_id {
                                min_act_id = value.act_id;
                            }
                        }
                    }
                }
            });

            if min_act_id < i32::MAX {
                if let Some(value) = self.activities.get_mut(&min_act_id) {
                    value.consume_records[(user_id as usize)] += 1;

                    return cost - value.discount;
                }
            }

            cost
        }
    }
}

/**
 * Your DiscountSystem object will be instantiated and called as such:
 * let obj = DiscountSystem::new();
 * obj.add_activity(actId, priceLimit, discount, number, userLimit);
 * obj.remove_activity(actId);
 * let ret_3: i32 = obj.consume(userId, cost);
 */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solution::data_structures::lists;
    #[test]
    fn unionpay() {
        dbg!(is_palindrome_iter(vec![1, 2, 3, 4, 4, 3, 2, 1].into_iter()));
        dbg!(is_palindrome_iter(vec!["s", "o", "s"].into_iter()));

        let head = lists::vec_to_list(&vec![1, 2, 3, 1]);
        dbg!(is_palindrome(head));

        let head = lists::vec_to_list(&vec![1, 2, 3, 1]);
        dbg!(is_palindrome_v2(head));
    }

    #[test]
    fn discount() {
        let mut system = DiscountSystem::new();
        // 创建编号 1 的优惠活动，单笔消费原价不小于 10 时，可享受 6 的减免，优惠活动共有 3 个名额，每个用户最多参与该活动 2 次
        system.add_activity(1, 10, 6, 3, 2);

        // 创建编号 2 的优惠活动，单笔消费原价不小于 15 时，可享受 8 的减免，优惠活动共有 8 个名额，每个用户最多参与该活动 2 次
        system.add_activity(2, 15, 8, 8, 2);

        dbg!(system.consume(101, 13)); // 用户 101 消费了 13，仅满足优惠活动 1 条件，返回实际支付 13 - 6 = 7用户 101 参加 1 次活动 1，活动 1 剩余 2 个名额
        dbg!(system.consume(101, 8)); // 用户 101 消费了 8，不满足任何活动，返回支付原价 8
        system.remove_activity(2); // 结束编号为 2 的活动
        dbg!(system.consume(101, 17)); // 用户 101 消费了 17，满足优惠活动 1 条件，返回实际支付 17 - 6 = 11用户 101 参加 2 次活动 1，活动 1 剩余 1 个名额
        dbg!(system.consume(101, 11)); // 用户 101 消费了 11，满足优惠活动 1 条件，但已达到参加次数限制，返回支付原价 11
        dbg!(system.consume(102, 16)); // 用户 102 消费了 16，满足优惠活动 1 条件，返回实际支付 16 - 6 = 10用户 102 参加 1 次活动 1，活动 1 无剩余名额
        dbg!(system.consume(102, 21)); // 用户 102 消费了 21，满足优惠活动 1 条件，但活动 1 已无剩余名额，返回支付原价 21
    }

    // ["DiscountSystem","consume","consume","addActivity","consume","consume","addActivity","consume","consume"]
    //[[],[4,69],[4,82],[5,45,19,3,2],[6,63],[6,95],[3,38,15,3,3],[6,70],[8,49]]
    #[test]
    fn test_discount2() {
        let mut system = DiscountSystem::new();
        dbg!(system.consume(4, 69));
        dbg!(system.consume(4, 82));
        system.add_activity(5, 45, 19, 3, 2);
        dbg!(system.consume(6, 63));
        dbg!(system.consume(6, 95));
        system.add_activity(3, 38, 15, 3, 3);
        dbg!(system.consume(6, 70));
        dbg!(system.consume(8, 49));
    }
}
