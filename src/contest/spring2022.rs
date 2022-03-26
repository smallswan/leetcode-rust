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

fn is_palindrome_iter(mut iter: impl DoubleEndedIterator<Item = impl Eq>) -> bool {
    while let (Some(left), Some(right)) = (iter.next(), iter.next_back()) {
        if left != right {
            return false;
        }
    }

    true
}

/// 银联-01. 回文链表
/// 贪心算法
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solution::data_structures::lists;
    #[test]
    fn unionpay() {
        let head = lists::vec_to_list(&vec![1, 2, 3, 1]);
        dbg!(is_palindrome(head));

        let head = lists::vec_to_list(&vec![1, 2, 3, 1]);
        dbg!(is_palindrome_v2(head));
    }
}
