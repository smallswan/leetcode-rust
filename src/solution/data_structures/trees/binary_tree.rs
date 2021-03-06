//! 二叉树
//! https://leetcode-cn.com/tag/binary-tree/problemset/

use std::cell::RefCell;
use std::cmp::max;
use std::collections::VecDeque;
use std::rc::Rc;
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }

    /// 树的深度：也称为树的高度，树中所有结点的层次最大值称为树的深度
    pub fn get_height(root: &Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs(root: &Option<Rc<RefCell<TreeNode>>>) -> i32 {
            match root {
                None => 0,
                Some(node) => {
                    let node = node.borrow_mut();
                    1 + max(dfs(&node.left), dfs(&node.right))
                }
            }
        }
        dfs(root)
    }
}

/// 94. 二叉树的中序遍历  https://leetcode-cn.com/problems/binary-tree-inorder-traversal/
/// 中序遍历：左中右
pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    fn traverse(root: Option<Rc<RefCell<TreeNode>>>, counter: &mut Vec<i32>) {
        if let Some(node) = root {
            traverse(node.borrow_mut().left.take(), counter);
            counter.push(node.borrow_mut().val);
            traverse(node.borrow_mut().right.take(), counter);
        }
    }

    let mut counter = vec![];
    traverse(root, &mut counter);
    counter
}

/// 100. 相同的树 https://leetcode-cn.com/problems/same-tree/
pub fn is_same_tree(p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
    match (p, q) {
        (Some(x), Some(y)) => {
            let mut x_b = x.borrow_mut();
            let mut y_b = y.borrow_mut();
            if x_b.val != y_b.val {
                return false;
            }

            is_same_tree(x_b.left.take(), y_b.left.take())
                && is_same_tree(x_b.right.take(), y_b.right.take())
        }
        (None, None) => true,
        (_, _) => false,
    }
}

fn symmetric(l: Option<Rc<RefCell<TreeNode>>>, r: Option<Rc<RefCell<TreeNode>>>) -> bool {
    match (l.as_ref(), r.as_ref()) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            if x.borrow_mut().val != y.borrow_mut().val {
                return false;
            }
            let (mut x_b, mut y_b) = (x.borrow_mut(), y.borrow_mut());
            // 左 = 右，右 = 左
            symmetric(x_b.left.take(), y_b.right.take())
                && symmetric(x_b.right.take(), y_b.left.take())
        }
        (_, _) => false,
    }
}

/// 力扣（101. 对称二叉树)  https://leetcode-cn.com/problems/symmetric-tree/
/// 注意：本题与主站 101 题相同：https://leetcode-cn.com/problems/symmetric-tree/
pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    if root.is_none() {
        return true;
    }

    let mut r_b = root.as_ref().unwrap().borrow_mut();
    let (l, r) = (r_b.left.take(), r_b.right.take());
    symmetric(l, r)
}

/// 102. 二叉树的层序遍历 https://leetcode-cn.com/problems/binary-tree-level-order-traversal/
/// 剑指 Offer 32 - II. 从上到下打印二叉树 II https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/
pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut ans = vec![];

    let mut queue = VecDeque::new();
    queue.push_back(root);

    while !queue.is_empty() {
        let size = queue.len();
        let mut level = vec![];
        for _ in 0..size {
            if let Some(x) = queue.pop_front().flatten() {
                let node = x.borrow();
                level.push(node.val);
                queue.push_back(node.left.clone());
                queue.push_back(node.right.clone());
            }
        }
        if !level.is_empty() {
            ans.push(level);
        }
    }
    ans
}

/// 103. 二叉树的锯齿形层序遍历 https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/
pub fn zigzag_level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut res = Vec::new();
    let mut current_level = 0;
    if root.is_none() {
        return res;
    }
    let mut deq = VecDeque::new();
    deq.push_back((0, root));
    let mut vec = Vec::new();
    while !deq.is_empty() {
        if let Some((level, Some(node))) = deq.pop_front() {
            deq.push_back((level + 1, node.borrow().left.clone()));
            deq.push_back((level + 1, node.borrow().right.clone()));
            if level > current_level {
                if current_level % 2 == 1 {
                    vec.reverse();
                }
                res.push(vec);
                vec = Vec::new();
                current_level = level;
            }
            vec.push(node.borrow().val);
        }
    }
    if !vec.is_empty() {
        if current_level % 2 == 1 {
            vec.reverse();
        }
        res.push(vec)
    }
    res
}

/// 104. 二叉树的最大深度 https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/
/// 剑指 Offer 55 - I. 二叉树的深度 https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/
pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut max_depth = 0;
    fn traverse(n: Option<Rc<RefCell<TreeNode>>>, max_depth: &mut i32, parent_depth: i32) {
        if n.is_none() {
            return;
        }

        if parent_depth + 1 > *max_depth {
            *max_depth = parent_depth + 1;
        }

        let mut n_b = n.as_ref().unwrap().borrow_mut();
        let (mut l, mut r) = (n_b.left.clone(), n_b.right.clone());
        traverse(l, max_depth, parent_depth + 1);
        traverse(r, max_depth, parent_depth + 1);
    }
    traverse(root, &mut max_depth, 0);
    max_depth
}

fn build_tree_helper(preorder: &[i32], inorder: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
    if preorder.is_empty() {
        return None;
    }
    let root_idx = inorder.iter().position(|&v| v == preorder[0]).unwrap();
    Some(Rc::new(RefCell::new(TreeNode {
        val: preorder[0],
        left: build_tree_helper(&preorder[1..root_idx + 1], &inorder[0..root_idx]),
        right: build_tree_helper(&preorder[root_idx + 1..], &inorder[root_idx + 1..]),
    })))
}

/// 105. 从前序与中序遍历序列构造二叉树  https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
/// 注意：剑指 Offer 07. 重建二叉树 https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/
pub fn build_tree(preorder: Vec<i32>, inorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    build_tree_helper(&preorder[..], &inorder[..])
}

/// 106. 从中序与后序遍历序列构造二叉树 https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
pub fn build_tree_106(inorder: Vec<i32>, postorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
    fn build_tree_helper(postorder: &[i32], inorder: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
        if postorder.is_empty() {
            return None;
        }
        let root_idx = inorder
            .iter()
            .position(|v| v == postorder.last().unwrap())
            .unwrap();
        Some(Rc::new(RefCell::new(TreeNode {
            val: *postorder.last().unwrap(),
            left: build_tree_helper(&postorder[0..root_idx], &inorder[0..root_idx]),
            right: build_tree_helper(
                &postorder[root_idx..postorder.len() - 1],
                &inorder[root_idx + 1..],
            ),
        })))
    }

    build_tree_helper(&postorder[..], &inorder[..])
}

/// 107. 二叉树的层序遍历 II  https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/
pub fn level_order_bottom(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
    let mut res = Vec::new();
    let mut current_level = 0;
    if root.is_none() {
        return res;
    }
    let mut deq = VecDeque::new();
    deq.push_back((0, root));
    let mut vec = Vec::new();
    while !deq.is_empty() {
        if let Some((level, Some(node))) = deq.pop_front() {
            deq.push_back((level + 1, node.borrow().left.clone()));
            deq.push_back((level + 1, node.borrow().right.clone()));
            if level > current_level {
                res.push(vec);
                vec = Vec::new();
                current_level = level;
            }
            vec.push(node.borrow().val);
        }
    }
    if !vec.is_empty() {
        res.push(vec)
    }
    res.reverse();
    res
}

/// 110. 平衡二叉树 https://leetcode-cn.com/problems/balanced-binary-tree/
/// 剑指 Offer 55 - II. 平衡二叉树 https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/
pub fn is_balanced(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn balanced_helper(root: Option<&Rc<RefCell<TreeNode>>>) -> Option<i32> {
        if let Some(node) = root {
            let pair = (
                balanced_helper(node.borrow().left.as_ref()),
                balanced_helper(node.borrow().right.as_ref()),
            );
            match pair {
                (Some(left), Some(right)) => {
                    if i32::abs(left - right) < 2 {
                        Some(i32::max(left, right) + 1)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            Some(0)
        }
    }
    balanced_helper(root.as_ref()).is_some()
}

/// 111. 二叉树的最小深度 https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/
pub fn min_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    match root {
        None => 0,
        Some(mut node) => {
            let ll = node.borrow_mut().left.take();
            let rr = node.borrow_mut().right.take();
            match (ll, rr) {
                (None, None) => 1,
                (Some(l), Some(r)) => {
                    let lc = min_depth(Some(l));
                    let rc = min_depth(Some(r));
                    let min = if lc < rc { lc } else { rc };
                    min + 1
                }
                (Some(l), None) => 1 + min_depth(Some(l)),
                (_, Some(r)) => 1 + min_depth(Some(r)),
            }
        }
    }
}

/// 112. 路径总和 https://leetcode-cn.com/problems/path-sum/
pub fn has_path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> bool {
    if let Some(mut node) = root {
        let mut sum = target_sum;
        let mut stack = Vec::new();

        loop {
            let (val, left, right) = {
                let node_ref = node.borrow();

                (node_ref.val, node_ref.left.clone(), node_ref.right.clone())
            };

            match (left, right) {
                (None, None) => {
                    if val == sum {
                        return true;
                    } else if let Some((next_node, next_sum)) = stack.pop() {
                        node = next_node;
                        sum = next_sum;
                    } else {
                        break;
                    }
                }
                (None, Some(child)) | (Some(child), None) => {
                    node = child;
                    sum -= val;
                }
                (Some(left), Some(right)) => {
                    node = left;
                    sum -= val;

                    stack.push((right, sum));
                }
            }
        }
    }

    false
}

fn max_path_sum_helper(root: Option<&RefCell<TreeNode>>, result: &mut i32) -> i32 {
    root.map(RefCell::borrow)
        .as_deref()
        .map_or(i32::MIN, |root| {
            let line_sum_1 = max_path_sum_helper(root.left.as_deref(), result);
            let line_sum_2 = max_path_sum_helper(root.right.as_deref(), result);

            *result = (*result).max(root.val + line_sum_1.max(0) + line_sum_2.max(0));

            line_sum_1.max(line_sum_2).max(0) + root.val
        })
}

/// 124. 二叉树中的最大路径和 https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/
pub fn max_path_sum(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut result = i32::MIN;

    max_path_sum_helper(root.as_deref(), &mut result);

    result
}

/// 144. 二叉树的前序遍历 https://leetcode-cn.com/problems/binary-tree-preorder-traversal/
/// 前序遍历：中左右
pub fn preorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut ans = vec![];

    if root.is_none() {
        return ans;
    }
    // 进栈
    let mut stack = vec![root];
    while !stack.is_empty() {
        // stack pop的值会自动包装 Option，需要调用 flatten 打平
        let node = stack.pop().flatten().unwrap();
        // 通过 Rc 的borrow 获取 Ref<TreeNode> 节点
        let node = node.borrow_mut();
        // 访问节点值
        ans.push(node.val);
        // 右子树进栈
        if let Some(ref right) = node.right {
            stack.push(Some(right.clone()));
        }
        // 左子树进栈
        if let Some(ref left) = node.left {
            stack.push(Some(left.clone()));
        }
    }
    ans
}

/// 145. 二叉树的后序遍历 https://leetcode-cn.com/problems/binary-tree-postorder-traversal/
/// 后序遍历：左右中
pub fn postorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    fn traverse(root: Option<Rc<RefCell<TreeNode>>>, counter: &mut Vec<i32>) {
        if let Some(node) = root {
            traverse(node.borrow_mut().left.take(), counter);
            traverse(node.borrow_mut().right.take(), counter);
            counter.push(node.borrow_mut().val);
        }
    }

    let mut counter = vec![];
    traverse(root, &mut counter);
    counter
}

/// 199. 二叉树的右视图 https://leetcode-cn.com/problems/binary-tree-right-side-view/
pub fn right_side_view(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut result = Vec::new();

    if let Some(node) = root {
        let mut queue = VecDeque::from(vec![node]);

        loop {
            for _ in 1..queue.len() {
                let node = queue.pop_front().unwrap();
                let node = node.borrow();

                if let Some(left) = node.left.clone() {
                    queue.push_back(left);
                }

                if let Some(right) = node.right.clone() {
                    queue.push_back(right);
                }
            }

            let node = queue.pop_front().unwrap();
            let node = node.borrow();

            result.push(node.val);

            if let Some(left) = node.left.clone() {
                queue.push_back(left);
            }
            if let Some(right) = node.right.clone() {
                queue.push_back(right);
            }

            if queue.is_empty() {
                break;
            }
        }
    }

    result
}

/// 222. 完全二叉树的节点个数 https://leetcode-cn.com/problems/count-complete-tree-nodes/
pub fn count_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if root.is_none() {
        return 0;
    }
    let mut r_b = root.as_ref().unwrap().borrow_mut();
    let (mut l, mut r) = (r_b.left.take(), r_b.right.take());
    count_nodes(l) + count_nodes(r) + 1
}

/// 222. 完全二叉树的节点个数
pub fn count_nodes_v2(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if let Some(root) = root {
        let left_height = count_height(root.borrow().left.clone());
        let right_height = count_height(root.borrow().right.clone());
        if left_height == right_height {
            (1 << left_height) + count_nodes(root.borrow().right.clone())
        } else {
            (1 << right_height) + count_nodes(root.borrow().left.clone())
        }
    } else {
        0
    }
}

fn count_height(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if let Some(node) = root {
        1 + count_height(node.borrow().left.clone())
    } else {
        0
    }
}

/// 226. 翻转二叉树 https://leetcode-cn.com/problems/invert-binary-tree/
pub fn invert_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    if let Some(node) = root.clone() {
        invert_tree(node.borrow_mut().right.clone());
        invert_tree(node.borrow_mut().left.clone());
        let left = node.borrow_mut().left.clone();
        let right = node.borrow_mut().right.clone();
        node.borrow_mut().left = right;
        node.borrow_mut().right = left;
    }
    root
}

/// 404. 左叶子之和 https://leetcode-cn.com/problems/sum-of-left-leaves/
pub fn sum_of_left_leaves(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn helper_left(node: &TreeNode) -> i32 {
        match (node.left.as_deref(), node.right.as_deref()) {
            (None, None) => node.val,
            (None, Some(right)) => helper_right(&right.borrow()),
            (Some(left), None) => helper_left(&left.borrow()),
            (Some(left), Some(right)) => {
                helper_left(&left.borrow()) + helper_right(&right.borrow())
            }
        }
    }

    fn helper_right(node: &TreeNode) -> i32 {
        match (node.left.as_deref(), node.right.as_deref()) {
            (None, None) => 0,
            (None, Some(right)) => helper_right(&right.borrow()),
            (Some(left), None) => helper_left(&left.borrow()),
            (Some(left), Some(right)) => {
                helper_left(&left.borrow()) + helper_right(&right.borrow())
            }
        }
    }
    root.map_or(0, |node| helper_right(&node.borrow()))
}

/// 543. 二叉树的直径 https://leetcode-cn.com/problems/diameter-of-binary-tree/
pub fn diameter_of_binary_tree(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn diameter_of_binary_tree_helper(root: &TreeNode) -> (i32, i32) {
        match (root.left.clone(), root.right.clone()) {
            (None, None) => (0, 1),
            (None, Some(child)) | (Some(child), None) => {
                let (diameter, height) = diameter_of_binary_tree_helper(&child.borrow());

                (diameter.max(height), height + 1)
            }
            (Some(left), Some(right)) => {
                let (left_diameter, left_height) = diameter_of_binary_tree_helper(&left.borrow());
                let (right_diameter, right_height) =
                    diameter_of_binary_tree_helper(&right.borrow());

                (
                    left_diameter
                        .max(right_diameter)
                        .max(left_height + right_height),
                    left_height.max(right_height) + 1,
                )
            }
        }
    }

    root.map_or(0, |root| diameter_of_binary_tree_helper(&root.borrow()).0)
}

/// 617. 合并二叉树 https://leetcode-cn.com/problems/merge-two-binary-trees/
pub fn merge_trees(
    mut root1: Option<Rc<RefCell<TreeNode>>>,
    root2: Option<Rc<RefCell<TreeNode>>>,
) -> Option<Rc<RefCell<TreeNode>>> {
    fn merge_to(target: &mut Option<Rc<RefCell<TreeNode>>>, tree: Option<Rc<RefCell<TreeNode>>>) {
        if let Some(tree) = tree {
            if let Some(target) = target {
                let mut target = target.borrow_mut();
                let mut tree = tree.borrow_mut();

                target.val += tree.val;

                merge_to(&mut target.left, tree.left.take());
                merge_to(&mut target.right, tree.right.take());
            } else {
                *target = Some(tree);
            }
        }
    }

    merge_to(&mut root1, root2);

    root1
}

/// 655. 输出二叉树 https://leetcode-cn.com/problems/print-binary-tree/
pub fn print_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<String>> {
    // 二叉树高度
    let height = TreeNode::get_height(&root);
    // 满二叉树的宽度
    let width = (1 << height) - 1;
    let mut ans = vec![vec!["".to_string(); width as usize]; height as usize];

    // dfs 搜索
    fn dfs(
        ans: &mut Vec<Vec<String>>,
        node: &Option<Rc<RefCell<TreeNode>>>,
        deep: usize,
        lo: usize,
        hi: usize,
    ) {
        if let Some(x) = node {
            let node = x.borrow();
            let mid = lo + (hi - lo) / 2;
            ans[deep][mid] = x.borrow().val.to_string();
            dfs(ans, &node.left, deep + 1, lo, mid);
            dfs(ans, &node.right, deep + 1, mid + 1, hi);
        }
    }

    dfs(&mut ans, &root, 0usize, 0usize, width as usize);
    // 将所有字符连起来
    ans
}

/// 965. 单值二叉树 https://leetcode-cn.com/problems/univalued-binary-tree/
pub fn is_unival_tree(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    fn helper(node: Option<&RefCell<TreeNode>>, expected: i32) -> bool {
        node.map_or(true, |node| {
            let node = node.borrow();

            node.val == expected
                && helper(node.left.as_deref(), expected)
                && helper(node.right.as_deref(), expected)
        })
    }

    root.map_or(true, |root| {
        let root = root.borrow();

        helper(root.left.as_deref(), root.val) && helper(root.right.as_deref(), root.val)
    })
}

/// 968. 监控二叉树 https://leetcode-cn.com/problems/binary-tree-cameras/
struct State {
    directly: u32,
    indirectly: u32,
    children_only: u32,
}
pub fn min_camera_cover(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    fn helper(node: Option<&RefCell<TreeNode>>) -> State {
        node.map_or(
            State {
                directly: u32::MAX / 2,
                indirectly: 0,
                children_only: 0,
            },
            |node| {
                let node = node.borrow();
                let left_state = helper(node.left.as_deref());
                let left_monitored = left_state.directly.min(left_state.indirectly);
                let right_state = helper(node.right.as_deref());
                let right_monitored = right_state.directly.min(right_state.indirectly);

                State {
                    directly: left_monitored.min(left_state.children_only)
                        + right_monitored.min(right_state.children_only)
                        + 1,
                    indirectly: (left_state.directly + right_monitored)
                        .min(right_state.directly + left_monitored),
                    children_only: (left_state.indirectly + right_state.indirectly),
                }
            },
        )
    }

    let state = helper(root.as_deref());

    state.directly.min(state.indirectly) as _
}

/// 剑指 Offer 27. 二叉树的镜像  https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/
pub fn mirror_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    match root {
        None => None,
        Some(node) => {
            let mut tmp_node = node.borrow_mut();
            if tmp_node.left.is_none() && tmp_node.right.is_none() {
                return Some(node.clone());
            }
            let node1 = tmp_node.left.clone();
            tmp_node.left = tmp_node.right.clone();
            tmp_node.right = node1;

            if tmp_node.left.is_some() {
                mirror_tree(tmp_node.left.clone());
            }
            if tmp_node.right.is_some() {
                mirror_tree(tmp_node.right.clone());
            }
            Some(node.clone())
        }
    }
}

/// 剑指 Offer 27. 二叉树的镜像
pub fn mirror_tree_v2(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
    if root.is_none() {
        return root;
    }
    let mut root = root.unwrap();
    let mut left = root.borrow_mut().left.take();
    let mut right = root.borrow_mut().right.take();
    root.borrow_mut().left = mirror_tree_v2(right);
    root.borrow_mut().right = mirror_tree_v2(left);
    Some(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trees() {
        let node = TreeNode {
            val: 119,
            left: Some(Rc::new(RefCell::new(TreeNode::new(110)))),
            right: Some(Rc::new(RefCell::new(TreeNode::new(120)))),
        };
        let root = Rc::new(RefCell::new(node));

        let vec = inorder_traversal(Some(root));
        assert_eq!(vec, vec![110, 119, 120]);

        let node = TreeNode {
            val: 119,
            left: Some(Rc::new(RefCell::new(TreeNode::new(110)))),
            right: Some(Rc::new(RefCell::new(TreeNode::new(120)))),
        };
        let root = Rc::new(RefCell::new(node));

        let vec = preorder_traversal(Some(root));
        assert_eq!(vec, vec![119, 110, 120]);

        let node = TreeNode {
            val: 119,
            left: Some(Rc::new(RefCell::new(TreeNode::new(110)))),
            right: Some(Rc::new(RefCell::new(TreeNode::new(120)))),
        };
        let root = Rc::new(RefCell::new(node));

        let vec = postorder_traversal(Some(root));
        assert_eq!(vec, vec![110, 120, 119]);

        let twenty = TreeNode {
            val: 20,
            left: Some(Rc::new(RefCell::new(TreeNode::new(15)))),
            right: Some(Rc::new(RefCell::new(TreeNode::new(7)))),
        };

        let three = TreeNode {
            val: 3,
            left: Some(Rc::new(RefCell::new(TreeNode::new(9)))),
            right: Some(Rc::new(RefCell::new(twenty))),
        };

        let tree = level_order(Some(Rc::new(RefCell::new(three))));
        assert_eq!(tree, vec![vec![3], vec![9, 20], vec![15, 7]]);

        let twenty = TreeNode {
            val: 20,
            left: Some(Rc::new(RefCell::new(TreeNode::new(15)))),
            right: Some(Rc::new(RefCell::new(TreeNode::new(7)))),
        };

        let three = TreeNode {
            val: 3,
            left: Some(Rc::new(RefCell::new(TreeNode::new(9)))),
            right: Some(Rc::new(RefCell::new(twenty))),
        };

        let demo_tree = Some(Rc::new(RefCell::new(three)));
        let is_balanced = is_balanced(demo_tree.clone());
        println!(
            "is_balanced : {is_balanced}, height:{}",
            TreeNode::get_height((&demo_tree))
        );

        let four = TreeNode {
            val: 4,
            left: Some(Rc::new(RefCell::new(TreeNode::new(1)))),
            right: Some(Rc::new(RefCell::new(TreeNode::new(2)))),
        };

        let six = TreeNode {
            val: 6,
            left: Some(Rc::new(RefCell::new(TreeNode::new(7)))),
            right: Some(Rc::new(RefCell::new(TreeNode::new(8)))),
        };

        let five = TreeNode {
            val: 5,
            left: Some(Rc::new(RefCell::new(four))),
            right: Some(Rc::new(RefCell::new(six))),
        };
        let root = Rc::new(RefCell::new(five));
        //TODO 为什么以下三条语句调整顺序，第三条语句总是输出不正确呢？
        dbg!(preorder_traversal(Some(root.clone())));
        dbg!(inorder_traversal(Some(root.clone())));
        dbg!(postorder_traversal(Some(root.clone())));
    }
}
