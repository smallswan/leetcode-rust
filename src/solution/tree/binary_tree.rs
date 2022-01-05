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
        dfs(&root)
    }
}

/// 94. 二叉树的中序遍历  https://leetcode-cn.com/problems/binary-tree-inorder-traversal/
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

/// 100. 相同的树 https://leetcode-cn.com/problems/same-tree/
pub fn is_same_tree(p: Option<Rc<RefCell<TreeNode>>>, q: Option<Rc<RefCell<TreeNode>>>) -> bool {
    match (p, q) {
        (Some(x), Some(y)) => {
            let mut x_b = x.borrow_mut();
            let mut y_b = y.borrow_mut();
            if x_b.val != y_b.val {
                return false;
            }

            return is_same_tree(x_b.left.take(), y_b.left.take())
                && is_same_tree(x_b.right.take(), y_b.right.take());
        }
        (None, None) => return true,
        (_, _) => return false,
    }
    return false;
}

fn symmetric(l: Option<Rc<RefCell<TreeNode>>>, r: Option<Rc<RefCell<TreeNode>>>) -> bool {
    match (l.as_ref(), r.as_ref()) {
        (None, None) => return true,
        (Some(x), Some(y)) => {
            if x.borrow_mut().val != y.borrow_mut().val {
                return false;
            }
            let (mut x_b, mut y_b) = (x.borrow_mut(), y.borrow_mut());
            // 左 = 右，右 = 左
            return symmetric(x_b.left.take(), y_b.right.take())
                && symmetric(x_b.right.take(), y_b.left.take());
        }
        (_, _) => return false,
    }
}

/// 力扣（101. 对称二叉树)  https://leetcode-cn.com/problems/symmetric-tree/
pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
    if root.is_none() {
        return true;
    }

    let mut r_b = root.as_ref().unwrap().borrow_mut();
    let (l, r) = (r_b.left.take(), r_b.right.take());
    return symmetric(l, r);
}

/// 102. 二叉树的层序遍历 https://leetcode-cn.com/problems/binary-tree-level-order-traversal/
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
    deq.push_back((0, root.clone()));
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
    return max_depth;
}

/// 144. 二叉树的前序遍历 https://leetcode-cn.com/problems/binary-tree-preorder-traversal/
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

/// 222. 完全二叉树的节点个数 https://leetcode-cn.com/problems/count-complete-tree-nodes/
pub fn count_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    if root.is_none() {
        return 0;
    }
    let mut r_b = root.as_ref().unwrap().borrow_mut();
    let (mut l, mut r) = (r_b.left.take(), r_b.right.take());
    return count_nodes(l) + count_nodes(r) + 1;
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
