//! 二叉搜索树
//! https://leetcode-cn.com/tag/binary-search-tree/problemset/

use super::binary_tree::TreeNode;
use std::cell::RefCell;
use std::rc::Rc;
/// 将有序数组转为二叉搜索树
fn bst_helper(nums: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
    if nums.is_empty() {
        return None;
    }
    Some(Rc::new(RefCell::new(TreeNode {
        val: nums[nums.len() / 2],
        left: bst_helper(&nums[0..(nums.len() / 2)]),
        right: bst_helper(&nums[(nums.len() / 2 + 1)..]),
    })))
}

/// 98. 验证二叉搜索树 https://leetcode-cn.com/problems/validate-binary-search-tree/
pub fn is_valid_bst(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    traverse(root, None, None)
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
            if !traverse(left, min, Some(val)) {
                return false;
            }
            traverse(right, Some(val), max)
        }
    }
}

/// 108. 将有序数组转换为二叉搜索树  https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/
pub fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    bst_helper(&nums[..])
}

use crate::solution::data_structures::lists::list_to_vec;
use crate::solution::data_structures::lists::ListNode;

/// 109. 有序链表转换二叉搜索树 https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/
pub fn sorted_list_to_bst(head: Option<Box<ListNode>>) -> Option<Rc<RefCell<TreeNode>>> {
    let nums: Vec<i32> = list_to_vec(head);
    bst_helper(&nums[..])
}

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
        sorted_array_to_bst(nums);
    }
}
