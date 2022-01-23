use std::cmp::Ordering;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        ListNode { val, next: None }
    }
}

impl Ord for ListNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.val.cmp(&self.val)
    }
}

impl PartialOrd for ListNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 将数组转为链表（从尾到头构建）
pub fn vec_to_list(v: &[i32]) -> Option<Box<ListNode>> {
    let mut head = None;
    for i in v.iter().rev() {
        let mut node = ListNode::new(*i);
        node.next = head;
        head = Some(Box::new(node));
    }
    head
}

/// 将数组转为链表（从头到尾构建）
pub fn vec_to_list_v2(v: &[i32]) -> Option<Box<ListNode>> {
    let mut dummy_head = Box::new(ListNode::new(0));
    let mut head = &mut dummy_head;
    for i in v {
        head.next = Some(Box::new(ListNode::new(*i)));
        head = head.next.as_mut().unwrap();
    }
    dummy_head.next
}

/// 打印链表的值
pub fn display(l: Option<Box<ListNode>>) {
    let mut head = &l;
    while head.is_some() {
        print!("{}, ", head.as_ref().unwrap().val);
        head = &(head.as_ref().unwrap().next);
    }
    println!();
}

///  力扣（2. 两数相加） https://leetcode-cn.com/problems/add-two-numbers/
pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut l1 = l1;
    let mut l2 = l2;
    let mut head = None;

    // 进位
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
    // 8->0->7 => 7->0->8
    let mut tail = None;
    while let Some(mut n) = head.take() {
        head = n.next;
        n.next = tail;
        tail = Some(n);
    }
    tail
}

/// 力扣(19. 删除链表的倒数第 N 个结点) https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let mut dummy_head = Some(Box::new(ListNode {
        val: 0i32,
        next: head,
    }));
    //链表长度（不包括虚拟节点的长度）
    let mut len = 0;
    {
        let mut current = &dummy_head.as_ref().unwrap().next;
        while current.is_some() {
            len += 1;
            current = &(current.as_ref().unwrap().next);
        }
    }
    {
        let steps = len - n;
        let mut current = dummy_head.as_mut();
        for _ in 0..steps {
            current = current.unwrap().next.as_mut();
        }
        // current 被删除元素的前一个元素， next 为 被删除元素的后一个元素
        let next = current
            .as_mut()
            .unwrap()
            .next
            .as_mut()
            .unwrap()
            .next
            .clone();
        current.as_mut().unwrap().next = next;
    }
    dummy_head.unwrap().next
}

/// 力扣（21. 合并两个有序链表) https://leetcode-cn.com/problems/merge-two-sorted-lists/
pub fn merge_two_lists(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    match (l1, l2) {
        (Some(v1), None) => Some(v1),
        (None, Some(v2)) => Some(v2),
        (Some(mut v1), Some(mut v2)) => {
            if v1.val < v2.val {
                let n = v1.next.take();
                v1.next = self::merge_two_lists(n, Some(v2));
                Some(v1)
            } else {
                let n = v2.next.take();
                v2.next = self::merge_two_lists(Some(v1), n);
                Some(v2)
            }
        }
        _ => None,
    }
}

/// 力扣（23. 合并K个升序链表） https://leetcode-cn.com/problems/merge-k-sorted-lists/
pub fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    let mut result = Vec::new();
    for node in lists.iter() {
        let mut head = node;
        while head.is_some() {
            let value = head.as_ref().unwrap().val;
            result.push(value);
            head = &(head.as_ref().unwrap().next);
        }
    }

    result.sort();

    let mut head = None;
    for i in result.iter().rev() {
        let mut node = ListNode::new(*i);
        node.next = head;
        head = Some(Box::new(node));
    }
    head
}

/// 力扣（23. 合并K个升序链表）
pub fn merge_k_lists_v2(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    use std::collections::BinaryHeap;
    let mut result = BinaryHeap::new();
    for node in lists.iter() {
        let mut head = node;
        while head.is_some() {
            let value = head.as_ref().unwrap().val;
            result.push(value);
            head = &(head.as_ref().unwrap().next);
        }
    }

    let mut head = None;
    for i in result.into_sorted_vec().iter().rev() {
        let mut node = ListNode::new(*i);
        node.next = head;
        head = Some(Box::new(node));
    }
    head
}

/// 力扣（23. 合并K个升序链表）
/// 方法3：小顶堆
pub fn merge_k_lists_v3(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut dummy_head = Box::new(ListNode::new(0));
    let mut pans = &mut dummy_head;
    let mut result = BinaryHeap::new();
    for node in lists {
        let mut head = &node;
        while let Some(node) = head {
            result.push(Reverse(node.val));
            head = &(node.next);
        }
    }

    while let Some(Reverse(num)) = result.pop() {
        pans.next = Some(Box::new(ListNode::new(num)));
        pans = pans.next.as_mut().unwrap();
    }
    dummy_head.next
}

/// 24. 两两交换链表中的节点 https://leetcode-cn.com/problems/swap-nodes-in-pairs/
pub fn swap_pairs(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    if head.is_none() {
        return None;
    }

    let mut dummy_head = Box::new(ListNode::new(0));
    let mut new_head = &mut dummy_head;

    let (mut left, mut right) = (&head, &(head.as_ref().unwrap().next));
    while left.is_some() && right.is_some() {
        new_head.next = Some(Box::new(ListNode::new(right.as_ref().unwrap().val)));
        new_head = new_head.next.as_mut().unwrap();

        new_head.next = Some(Box::new(ListNode::new(left.as_ref().unwrap().val)));
        new_head = new_head.next.as_mut().unwrap();

        left = &(right.as_ref().unwrap().next);
        if left.is_none() {
            break;
        }
        right = &(left.as_ref().unwrap().next);
    }
    if let Some(next) = left.as_ref().take() {
        new_head.next = Some(Box::new(ListNode::new(left.as_ref().unwrap().val)));
        new_head = new_head.next.as_mut().unwrap();
    }

    dummy_head.next
}

/// 25. K 个一组翻转链表 https://leetcode-cn.com/problems/reverse-nodes-in-k-group/
pub fn reverse_k_group(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let mut dummy_head = Some(Box::new(ListNode { val: 0, next: head }));
    let mut head = dummy_head.as_mut();
    'outer: loop {
        let mut start = head.as_mut().unwrap().next.take();
        if start.is_none() {
            break 'outer;
        }
        let mut end = start.as_mut();
        for _ in 0..(k - 1) {
            end = end.unwrap().next.as_mut();
            if end.is_none() {
                head.as_mut().unwrap().next = start;
                break 'outer;
            }
        }
        let mut tail = end.as_mut().unwrap().next.take();
        // BEFORE: head -> start -> 123456... -> end   -> tail
        // AFTER:  head -> end   -> ...654321 -> start -> tail
        let end = reverse(start, tail);
        head.as_mut().unwrap().next = end;
        for _ in 0..k {
            head = head.unwrap().next.as_mut()
        }
    }
    dummy_head.unwrap().next
}

#[inline(always)]
fn reverse(mut head: Option<Box<ListNode>>, tail: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut prev = tail;
    let mut current = head;
    while let Some(mut current_node_inner) = current {
        let mut next = current_node_inner.next.take();
        current_node_inner.next = prev.take();
        prev = Some(current_node_inner);
        current = next;
    }
    prev
}

/// 力扣（83. 删除排序链表中的重复元素) https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/
pub fn delete_duplicates(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut dummy_head = Box::new(ListNode::new(0));
    let mut dummy_node = &mut dummy_head;
    let mut node = &head;
    while node.is_some() {
        let value = node.as_ref().unwrap().val;
        node = &(node.as_ref().unwrap().next);
        if node.is_none() || value != node.as_ref().unwrap().val {
            dummy_node.next = Some(Box::new(ListNode::new(value)));
            dummy_node = dummy_node.next.as_mut().unwrap();
        }
    }

    dummy_head.next
}

/// 92. 反转链表 II  https://leetcode-cn.com/problems/reverse-linked-list-ii/
pub fn reverse_between(
    head: Option<Box<ListNode>>,
    left: i32,
    right: i32,
) -> Option<Box<ListNode>> {
    let mut dummy_tail = Box::new(ListNode::new(0));
    let mut tail = &mut dummy_tail;
    let mut fast = &head;
    let mut slow = &head;
    let mut prev = 0;
    while fast.is_some() {
        prev += 1;
        if prev >= left && prev <= right {
            let val = fast.as_ref().unwrap().val;
            tail.next = Some(Box::new(ListNode::new(val)));
            tail = tail.next.as_mut().unwrap();
        }
        fast = &(fast.as_ref().unwrap().next);
    }

    let mut dummy_head = Box::new(ListNode::new(0));
    let mut new_head = &mut dummy_head;
    let mut reverse_list = reverse_list(dummy_tail.next);
    prev = 0;
    while slow.is_some() {
        prev += 1;

        if prev < left || prev > right {
            let val = slow.as_ref().unwrap().val;
            new_head.next = Some(Box::new(ListNode::new(val)));
            new_head = new_head.next.as_mut().unwrap();
            slow = &(slow.as_ref().unwrap().next);
        }

        if prev == left {
            while let Some(mut node) = reverse_list.take() {
                new_head.next = Some(Box::new(ListNode::new(node.val)));

                new_head = new_head.next.as_mut().unwrap();

                prev += 1;
                slow = &(slow.as_ref().unwrap().next);
                reverse_list = node.next.take();
            }
        }
    }

    dummy_head.next
}

/// 力扣（203. 移除链表元素) https://leetcode-cn.com/problems/remove-linked-list-elements/
pub fn remove_elements(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut dummy_head = ListNode::new(0);
    dummy_head.next = head;
    let mut nums = vec![];
    while let Some(mut node) = dummy_head.next.take() {
        if node.val != val {
            nums.push(node.val);
        }
        if let Some(next) = node.next.take() {
            dummy_head.next = Some(next);
        }
    }

    vec_to_list(&nums)
}

/// 力扣（203. 移除链表元素)
/// 虚拟头结点
pub fn remove_elements_v2(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut head = head;
    let mut root = Box::new(ListNode::new(0));
    //用于存储、修改当前值不为val的节点
    let mut current = &mut root;
    while let Some(mut node) = head.take() {
        head = node.next.take();
        if node.val != val {
            current.next = Some(node);
            current = current.next.as_mut().unwrap();
        }
    }
    root.next
}

/// 力扣（203. 移除链表元素)
/// 直接使用原来的链表来进行删除操作 FIXME take()会消耗掉Option中的值，以下代码不正确
pub fn remove_elements_v3(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
    let mut head = head;

    while let Some(mut node) = head.take() {
        if node.val == val {
            head = node.next.take();
        } else {
            head = Some(node);
            break;
        }
    }
    while let Some(node) = head.take().as_deref_mut() {
        if node.val == val {
            head = node.next.take();
            continue;
        } else {
            head = node.next.take();
        }
        if let Some(next) = node.next.take().as_deref_mut() {
            if next.val == val {
                if let Some(next2) = next.next.take() {
                    *next = *next2;
                }
            }
        }
    }

    head
}

/// 力扣（206. 反转链表） https://leetcode-cn.com/problems/reverse-linked-list/
/// 假设：head 对应的链表为：1 -> 2 -> 3 -> 4 -> 5 -> None
pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut head = head;
    let mut tail = None;
    while let Some(mut node) = head.take() {
        // head = node(1).next = node(2)
        head = node.next;
        // node(1).next = None
        node.next = tail;
        // tail = node(1)
        tail = Some(node);
    }
    tail
}

/// 力扣（707. 设计链表) https://leetcode-cn.com/problems/design-linked-list/
/**
 * Your MyLinkedList object will be instantiated and called as such:
 * let obj = MyLinkedList::new();
 * let ret_1: i32 = obj.get(index);
 * obj.add_at_head(val);
 * obj.add_at_tail(val);
 * obj.add_at_index(index, val);
 * obj.delete_at_index(index);
 */
pub struct MyLinkedList {
    len: usize,
    pub root: Option<Box<ListNode>>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl MyLinkedList {
    /** Initialize your data structure here. */
    fn new() -> Self {
        MyLinkedList {
            len: 0usize,
            root: None,
        }
    }

    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    fn get(&self, index: i32) -> i32 {
        if self.root.is_none() || index < 0 || index as usize >= self.len {
            -1
        } else {
            let mut current_index = 0;

            let mut current = &self.root;
            while current.is_some() {
                if current_index == index {
                    return current.as_ref().unwrap().val;
                }
                current_index += 1;
                current = &(current.as_ref().unwrap().next);
            }

            -1
        }
    }

    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    fn add_at_head(&mut self, val: i32) {
        if self.root.is_none() {
            self.root = Some(Box::new(ListNode::new(val)));
            self.len = 1;
        } else {
            let next = self.root.take();
            self.root = Some(Box::new(ListNode { val, next }));
            self.len += 1;
        }
    }

    /** Append a node of value val to the last element of the linked list. */
    fn add_at_tail(&mut self, val: i32) {
        if self.root.is_none() {
            self.root = Some(Box::new(ListNode::new(val)));
            self.len = 1;
        } else {
            let mut current = &mut self.root;
            while current.is_some() {
                let next = &current.as_ref().unwrap().next;
                //判断当前元素current是否为最后一个元素
                if next.is_none() {
                    if let Some(last) = current.as_deref_mut() {
                        last.next = Some(Box::new(ListNode::new(val)));
                        self.len += 1;
                        break;
                    }
                }

                current = &mut current.as_deref_mut().unwrap().next;
            }
        }
    }

    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    fn add_at_index(&mut self, index: i32, val: i32) {
        if index < 0 || index > self.len as i32 {
            eprintln!("invalid index :{}", index);
            return;
        }
        if index == 0 {
            self.add_at_head(val);
            return;
        } else if self.root.is_some() && index == self.len as i32 {
            self.add_at_tail(val);
            return;
        }
        let mut current_index = 0;
        let mut current = &mut self.root;
        while current.is_some() {
            if (current_index == index - 1) {
                let next = current.as_mut().unwrap().next.take();

                let node = Some(Box::new(ListNode { val, next }));

                current.as_mut().unwrap().next = node;
                self.len += 1;
                break;
            }
            current = &mut current.as_deref_mut().unwrap().next;

            current_index += 1;
        }
    }

    /** Delete the index-th node in the linked list, if the index is valid. */
    fn delete_at_index(&mut self, index: i32) {
        if index < 0 || index >= self.len as i32 {
            eprintln!("invalid index :{}", index);
            return;
        }

        if index == 0 {
            let mut current = &mut self.root;
            if current.is_some() {
                *current = current.as_mut().unwrap().next.take();
                self.len -= 1;
            }
        }

        // 遍历到 index - 1的位置，则删除index位置的元素，并将index+1位置以后的元素添加到当前元素的后面
        let mut current_index = 0;
        let mut current = &mut self.root;
        while current.is_some() {
            if current_index == index - 1 {
                let next_next = current.as_mut().unwrap().next.as_mut().unwrap().next.take();
                current.as_mut().unwrap().next = next_next;
                self.len -= 1;
                break;
            }
            current_index += 1;
            current = &mut current.as_deref_mut().unwrap().next;
        }
    }
}

/// 剑指 Offer 06. 从尾到头打印链表 https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/
/// 反转数组
pub fn reverse_print(head: Option<Box<ListNode>>) -> Vec<i32> {
    let mut res = Vec::new();
    let mut next = &head;
    while next.is_some() {
        res.push(next.as_ref().unwrap().val);
        next = &(next.as_ref().unwrap().next);
    }
    res.reverse();
    res
}

/// 剑指 Offer 22. 链表中倒数第k个节点  https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/
pub fn get_kth_from_end(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
    let mut fast = &head;
    let mut slow = &head;
    for i in 0..k {
        if fast.is_some() {
            fast = &(fast.as_ref().unwrap().next);
        }
    }
    while fast.is_some() {
        fast = &(fast.as_ref().unwrap().next);
        slow = &(slow.as_ref().unwrap().next);
    }
    if slow.is_none() {
        None
    } else {
        let mut dummy_head = Box::new(ListNode::new(0));
        let mut new_head = &mut dummy_head;
        while slow.is_some() {
            new_head.next = Some(Box::new(ListNode::new(slow.as_ref().unwrap().val)));
            new_head = new_head.next.as_mut().unwrap();
            slow = &(slow.as_ref().unwrap().next);
        }

        dummy_head.next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add_linked_list() {
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
        dbg!("{:?}", result);
    }

    #[test]
    fn test_linked_list_() {
        //use MyLinkedList;
        let mut linked_list = MyLinkedList::new();
        linked_list.add_at_head(7);
        linked_list.add_at_head(6);
        linked_list.add_at_head(8);
        linked_list.add_at_tail(9);
        dbg!(linked_list.get(2));
        dbg!(linked_list.get(3));

        linked_list.add_at_index(2, 10);

        dbg!(linked_list.get(2));
        dbg!(linked_list.get(3));
        dbg!(linked_list.get(4));

        linked_list.delete_at_index(0);
        dbg!(linked_list.get(0));
        dbg!(linked_list.get(3));

        linked_list.delete_at_index(1);
        dbg!(linked_list.get(0));
        dbg!(linked_list.get(1));
        dbg!(linked_list.get(2));

        linked_list.delete_at_index(3);
    }

    /// ["MyLinkedList","addAtHead","addAtHead","addAtHead","addAtIndex","deleteAtIndex","addAtHead","addAtTail","get","addAtHead","addAtIndex","addAtHead"]
    /// [[],[7],[2],[1],[3,0],[2],[6],[4],[4],[4],[5,0],[6]]
    #[test]
    fn test_linked_list_fn_add_at_head() {
        //use MyLinkedList;
        let mut linked_list = MyLinkedList::new();
        linked_list.add_at_head(7);
        linked_list.add_at_head(2);
        linked_list.add_at_head(1);
        linked_list.add_at_index(3, 0);
        linked_list.delete_at_index(2);
        linked_list.add_at_head(6);
        linked_list.add_at_tail(4);
        let val = linked_list.get(4);
        dbg!(val);
        linked_list.add_at_head(4);
        linked_list.add_at_index(5, 0);
        linked_list.add_at_head(6);
    }

    //["MyLinkedList","addAtIndex","addAtIndex","addAtIndex","get"]
    //[[],[0,10],[0,20],[1,30],[0]]
    #[test]
    fn test_linked_list_fn_add_at_index() {
        //use MyLinkedList;
        let mut linked_list = MyLinkedList::new();
        linked_list.delete_at_index(0);
        linked_list.add_at_index(0, 10);
        linked_list.add_at_index(0, 20);
        linked_list.add_at_index(0, 30);
        dbg!(linked_list.get(0));

        linked_list.add_at_tail(40);
        linked_list.add_at_tail(50);
        linked_list.add_at_tail(60);
        dbg!(linked_list.get(4));
        dbg!(linked_list.get(5));
        dbg!(linked_list.get(6));

        linked_list.delete_at_index(3);
        dbg!(linked_list.get(3));
    }
    #[test]
    fn linked_list() {
        let linked_list = vec_to_list_v2(&vec![3, 2, 4]);

        let swap_linked_list = swap_pairs(linked_list);
        display(swap_linked_list);
    }

    #[test]
    fn test_k_merge() {
        // lists: Vec<Option<Box<ListNode>>
        let node1 = vec_to_list(&vec![1, 2, 3]);
        let node2 = vec_to_list(&vec![1, 3, 4]);
        let node3 = vec_to_list(&vec![2, 6]);
        let lists = vec![node1, node2, node3];
        let res = merge_k_lists(lists);
        display(res);

        let node4 = vec_to_list(&vec![1, 2, 3]);
        let node5 = vec_to_list(&vec![1, 3, 4]);
        let node6 = vec_to_list(&vec![2, 6]);
        let lists2 = vec![node4, node5, node6];

        let res2 = merge_k_lists_v2(lists2);
        display(res2);

        let node7 = vec_to_list(&vec![1, 2, 3]);
        let node8 = vec_to_list(&vec![1, 3, 4]);
        let node9 = vec_to_list(&vec![2, 6]);
        let lists3 = vec![node7, node8, node9];
        let res3 = merge_k_lists_v3(lists3);

        display(res3);
    }

    #[test]
    fn linked_list2() {
        let linked_list = vec_to_list_v2(&vec![1, 2, 3, 4, 5]);
        let result = reverse_between(linked_list, 2, 4);
        display(result);

        let linked_list = vec_to_list_v2(&vec![1, 3, 5, 7, 9]);
        display(linked_list);

        let l = merge_two_lists(
            vec_to_list(&vec![1, 3, 5, 7, 9]),
            vec_to_list(&vec![2, 4, 6, 8, 10]),
        );
        display(l);

        let l = merge_two_lists(
            vec_to_list(&vec![1, 2, 4]),
            vec_to_list(&vec![1, 3, 4, 5, 6]),
        );
        display(l);

        let nums = vec![1, 2, 3, 4, 5];
        let head = vec_to_list(&nums);
        let tail = reverse_list(head);

        display(tail);

        let head = vec_to_list(&nums);
        let remove_elements_result = remove_elements(head, 3);
        display(remove_elements_result);

        let head = vec_to_list(&nums);
        let remove_elements_v2_result = remove_elements_v2(head, 3);
        display(remove_elements_v2_result);

        let head = vec_to_list(&[7, 6, 8, 7, 7, 7, 7, 7]);
        let remove_elements_v3_result = remove_elements_v3(head, 7);
        display(remove_elements_v3_result);
        let head = vec_to_list(&vec![1, 1, 2, 3, 3]);
        let unique_nodes = delete_duplicates(head);
        display(unique_nodes);
    }

    #[test]
    fn reverse() {
        let head = vec_to_list(&vec![1, 2, 3, 4, 5]);
        let reverse_head = reverse_k_group(head, 3);
        display(reverse_head);
    }
}
