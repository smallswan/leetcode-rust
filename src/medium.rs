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
}
