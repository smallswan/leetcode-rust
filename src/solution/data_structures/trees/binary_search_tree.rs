//! 二叉搜索树
//! https://leetcode-cn.com/tag/binary-search-tree/problemset/

use super::binary_tree::TreeNode;
use std::cell::RefCell;
use std::rc::Rc;

use crate::solution::data_structures::lists::list_to_vec;
use crate::solution::data_structures::lists::ListNode;

/// 173. 二叉搜索树迭代器 https://leetcode-cn.com/problems/binary-search-tree-iterator/
pub struct BSTIterator {
    stack: Vec<Rc<RefCell<TreeNode>>>,
}

impl BSTIterator {
    fn new(mut root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut stack = Vec::new();

        while let Some(node) = root {
            root = node.borrow().left.clone();

            stack.push(node);
        }

        Self { stack }
    }

    fn next(&mut self) -> i32 {
        let node = self.stack.pop().unwrap();
        let node_ref = node.borrow();
        let mut root = node_ref.right.clone();

        while let Some(node) = root {
            root = node.borrow().left.clone();

            self.stack.push(node);
        }

        node_ref.val
    }

    fn has_next(&self) -> bool {
        !self.stack.is_empty()
    }
}

pub struct Solution;
use std::cmp::Ordering;
impl Solution {
    /// 将有序数组转为二叉搜索树
    fn bst_helper(nums: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
        if nums.is_empty() {
            return None;
        }
        Some(Rc::new(RefCell::new(TreeNode {
            val: nums[nums.len() / 2],
            left: Self::bst_helper(&nums[0..(nums.len() / 2)]),
            right: Self::bst_helper(&nums[(nums.len() / 2 + 1)..]),
        })))
    }

    /// 98. 验证二叉搜索树 https://leetcode-cn.com/problems/validate-binary-search-tree/
    pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        Self::traverse(root, None, None)
    }

    fn traverse(root: Option<Rc<RefCell<TreeNode>>>, min: Option<i32>, max: Option<i32>) -> bool {
        match root {
            None => true,
            Some(root) => {
                let val = root.borrow().val;
                let mut valid = true;
                if let Some(min) = min {
                    valid = valid && min < val;
                }
                if let Some(max) = max {
                    valid = valid && val < max;
                }
                if !valid {
                    return false;
                }
                let left = root.borrow().left.as_ref().map(|rc| rc.clone());
                let right = root.borrow().right.as_ref().map(|rc| rc.clone());
                if !Self::traverse(left, min, Some(val)) {
                    return false;
                }
                Self::traverse(right, Some(val), max)
            }
        }
    }

    /// 108. 将有序数组转换为二叉搜索树  https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/
    pub fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        Self::bst_helper(&nums[..])
    }

    fn iterate_tree(root: Option<&RefCell<TreeNode>>, f: &mut impl FnMut(i32)) {
        if let Some(node) = root {
            let node_ref = node.borrow();

            Self::iterate_tree(node_ref.left.as_deref(), f);
            f(node_ref.val);
            Self::iterate_tree(node_ref.right.as_deref(), f);
        }
    }

    fn iterate_chunks(root: Option<&RefCell<TreeNode>>, f: &mut impl FnMut(i32, usize)) {
        let mut prev = None;
        let mut length = 0;

        Self::iterate_tree(root, &mut |val| {
            if let Some(prev_val) = prev {
                if val == prev_val {
                    length += 1;
                } else {
                    f(prev_val, length);

                    prev = Some(val);
                    length = 1;
                }
            } else {
                prev = Some(val);
                length = 1;
            }
        });

        f(prev.unwrap(), length);
    }

    /// 109. 有序链表转换二叉搜索树 https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/
    pub fn sorted_list_to_bst(head: Option<Box<ListNode>>) -> Option<Rc<RefCell<TreeNode>>> {
        let nums: Vec<i32> = list_to_vec(head);
        Self::bst_helper(&nums[..])
    }

    /// 501. 二叉搜索树中的众数 https://leetcode-cn.com/problems/find-mode-in-binary-search-tree/
    pub fn find_mode(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut max_length = 0;
        let mut max_length_count = 0;

        Self::iterate_chunks(
            root.as_deref(),
            &mut |_, length| match length.cmp(&max_length) {
                Ordering::Less => {}
                Ordering::Equal => max_length_count += 1,
                Ordering::Greater => {
                    max_length = length;
                    max_length_count = 1;
                }
            },
        );

        let mut result = Vec::with_capacity(max_length_count);

        Self::iterate_chunks(root.as_deref(), &mut |val, length| {
            if length == max_length {
                result.push(val);
            }
        });

        result
    }
}

/// 剑指 Offer 54. 二叉搜索树的第k大节点 https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/
pub fn kth_largest(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> i32 {
    fn dfs(root: Option<Rc<RefCell<TreeNode>>>, result: &mut i32, over: &mut bool, k: &mut i32) {
        if *over {
            return;
        }

        if let Some(node) = root {
            dfs(node.borrow_mut().right.take(), result, over, k);

            *k -= 1;
            if *k == 0 {
                *result = node.borrow_mut().val;
                *over = true;
                return;
            }

            dfs(node.borrow_mut().left.take(), result, over, k);
        } else {
            return;
        }
    }

    let mut result = 0;
    let mut over = false;
    let mut k = k;
    dfs(root, &mut result, &mut over, &mut k);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trees() {
        let nums = vec![-10, -3, 0, 5, 9];
        Solution::sorted_array_to_bst(nums);
    }
}
