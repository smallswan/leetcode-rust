/**     
 * Your MinStack object will be instantiated and called as such:
 * let obj = MinStack::new();
 * obj.push(val);
 * obj.pop();
 * let ret_3: i32 = obj.top();
 * let ret_4: i32 = obj.get_min();
 */

/// 155. 最小栈 https://leetcode-cn.com/problems/min-stack/
/// 剑指 Offer 30. 包含min函数的栈 https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/
struct MinStack {
    data: Vec<i32>,
    min: Vec<i32>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl MinStack {
    fn new() -> Self {
        MinStack {
            data: Vec::new(),
            min: Vec::new(),
        }
    }

    fn push(&mut self, val: i32) {
        self.data.push(val);
        if self.min.is_empty() || val <= self.min() {
            self.min.push(val);
        }
    }

    fn pop(&mut self) {
        if let Some(v) = self.data.last() {
            if *v == self.min() {
                self.min.pop();
            }
        }
        self.data.pop();
    }

    fn top(&self) -> i32 {
        if let Some(v) = self.data.last() {
            return *v;
        }
        0
    }

    fn min(&self) -> i32 {
        if let Some(v) = self.min.last() {
            return *v;
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_stack() {
        let mut stack = MinStack::new();
        stack.push(i32::MIN);
        stack.push(i32::MAX);
        stack.push(142857);

        dbg!(stack.min());
        dbg!(stack.top());
        stack.pop();
        dbg!(stack.min());
        dbg!(stack.top());
    }
}
