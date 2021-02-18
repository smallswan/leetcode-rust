/// 中等难度

///  力扣（2. 两数相加） https://leetcode-cn.com/problems/add-two-numbers/
pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut l1 = l1;
    let mut l2 = l2;
    let mut head = None;

    let mut left = 0;

    loop {
        if l1.is_none() && l2.is_none() && left == 0 {
            break;
        }

        let mut sum = left;
        if let Some(n) = l1 {
            sum += n.val;
            l1 = n.next;
        }
        if let Some(n) = l2 {
            sum += n.val;
            l2 = n.next;
        }

        left = sum / 10;
        let mut node = ListNode::new(sum % 10);
        node.next = head;
        head = Some(Box::new(node));
    }

    // reverse the list
    let mut tail = None;
    while let Some(mut n) = head.take() {
        head = n.next;
        n.next = tail;
        tail = Some(n);
    }
    tail
}

// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

// 计算以第i个字符为中点的字符串的最长回文，从中间往两边检查，复杂度为O(N)。
// 返回值：(最长回文的起点、最长回文的终点)
fn longest_of(i: usize, s: &[u8]) -> (usize, usize) {
    // 先检查奇数长度的字符串，中点只有一个，就是第i个字符
    let mut ret1 = (i, i);
    for (f, t) in (0..=i).rev().zip(i..s.len()) {
        if s[f] != s[t] {
            break;
        }
        ret1 = (f, t);
    }

    // 再检查偶数长度的字符串，中点有两个，此处指定为第i、i+1个
    let mut ret2 = (i, i);
    for (f, t) in (0..=i).rev().zip(i + 1..s.len()) {
        if s[f] != s[t] {
            break;
        }
        ret2 = (f, t);
    }

    return if ret2.1 - ret2.0 > ret1.1 - ret1.0 {
        ret2
    } else {
        ret1
    };
}

/// 力扣（5. 最长回文子串） https://leetcode-cn.com/problems/longest-palindromic-substring/solution/zui-chang-hui-wen-zi-chuan-by-leetcode-solution/
/// 中心拓展算法
pub fn longest_palindrome(s: String) -> String {
    if s.is_empty() {
        return "".to_string();
    }

    let mut range = (0, 0);
    let s = s.as_bytes();
    // 遍历每个字符，找出以当前字符为中点的最长回文字符串
    for i in 0..s.len() {
        let r = longest_of(i, &s);
        if r.1 - r.0 > range.1 - range.0 {
            range = r;
        }
    }
    return std::str::from_utf8(&s[range.0..=range.1])
        .unwrap()
        .to_string();
}

enum State {
    Init,
    ExpectNumber,// 已经碰到了+或者-，下一个字符必须是数字
    Number(i32),
}

/// 力扣（8. 字符串转整数（atoi）） https://leetcode-cn.com/problems/string-to-integer-atoi/
pub fn my_atoi(s: String) -> i32 {
    let mut state = State::Init;
    let mut neg = 1;
    for c in s.chars() {
        match c {
            ' ' => match state {
                State::Init => {}
                State::ExpectNumber => return 0,
                State::Number(n) => return neg * n,
            },
            '+' | '-' => match state {
                State::Init => {
                    state = State::ExpectNumber;
                    if c == '-' {
                        neg = -1;
                    }
                }
                State::ExpectNumber => return 0,
                State::Number(n) => return neg * n,
            },
            '0'..='9' => {
                let digit = c.to_digit(10).unwrap() as i32;
                match state {
                    State::Init | State::ExpectNumber => state = State::Number(digit),
                    State::Number(n) => {
                        match n.checked_mul(10).and_then(|x| x.checked_add(digit)) {
                            Some(number) => state = State::Number(number),
                            _ => {
                                return if neg < 0 {
                                    std::i32::MIN
                                } else {
                                    std::i32::MAX
                                }
                            }
                        }
                    }
                }
            }
            _ => match state {
                State::Init | State::ExpectNumber => return 0,
                State::Number(n) => return neg * n,
            },
        }
    }

    match state {
        State::Number(n) => return neg * n,
        _ => return 0,
    }
}

/// 力扣（210. 课程表II），https://leetcode-cn.com/problems/course-schedule-ii/
pub fn find_order(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::VecDeque;

    let u_num_courses = num_courses as usize;

    // 存储有向图
    let mut edges = vec![vec![]; u_num_courses];
    // 存储答案
    let mut result = vec![0; u_num_courses];

    // 答案下标
    let mut index = 0usize;

    // 存储每个节点（课程）的入度，indeg[1]即为其他课程选修前必须选修课程1的总次数
    let mut indeg = vec![0; u_num_courses];

    for info in prerequisites {
        edges[info[1] as usize].push(info[0]);
        indeg[info[0] as usize] += 1;
    }

    let mut queue = VecDeque::<i32>::new();
    // 将所有入度为 0 的节点放入队列中
    for course in 0..u_num_courses {
        if indeg[course] == 0 {
            queue.push_back(course as i32);
        }
    }

    while !queue.is_empty() {
        if let Some(u) = queue.pop_back() {
            result[index] = u;
            index += 1;

            if let Some(edge_vec) = edges.get(u as usize) {
                for v in edge_vec {
                    indeg[*v as usize] -= 1;
                    if indeg[*v as usize] == 0 {
                        queue.push_back(*v);
                    }
                }
            }
        }
    }

    if index != u_num_courses {
        return vec![];
    } else {
        return result;
    }
}

#[test]
fn medium() {
    use super::*;

    let mut prerequisites = Vec::<Vec<i32>>::new();
    prerequisites.push(vec![1, 0]);
    prerequisites.push(vec![2, 0]);
    prerequisites.push(vec![3, 1]);
    prerequisites.push(vec![3, 2]);

    let result = find_order(4, prerequisites);
    println!("result-> : {:?}", result);

    let result = find_order(2, vec![]);
    println!("{:?}", result);

    let l1 = ListNode {
        val: 2,
        next: Some(Box::new(ListNode {
            val: 4,
            next: Some(Box::new(ListNode::new(3))),
        })),
    };

    let l2 = ListNode {
        val: 5,
        next: Some(Box::new(ListNode {
            val: 6,
            next: Some(Box::new(ListNode::new(4))),
        })),
    };

    let result = add_two_numbers(Some(Box::new(l1)), Some(Box::new(l2)));
    println!("{:?}", result);

    println!("{}", my_atoi(" -423456".to_string()));
    println!("{}", my_atoi("4193 with words".to_string()));
    println!("{}", my_atoi("words and 987".to_string()));
    println!("{}", my_atoi("-91283472332".to_string()));
}
