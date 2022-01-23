struct MyHashSet {
    // FIXME 由于Rust所有权的限制，链表（LinkedList）remove较难实现，这里暂时使用Vec代替
    data: Vec<Vec<i32>>,
}

const BASE: i32 = 769;
/// 705. 设计哈希集合 https://leetcode-cn.com/problems/design-hashset/
impl MyHashSet {
    fn new() -> Self {
        let data = vec![Vec::<i32>::new(); BASE as usize];
        return MyHashSet { data };
    }

    #[inline(always)]
    fn hash(key: i32) -> usize {
        (key % 769) as usize
    }

    fn add(&mut self, key: i32) {
        let h = MyHashSet::hash(key);
        let mut iter = self.data[h].iter();
        while let Some(item) = iter.next() {
            if *item == key {
                return;
            }
        }
        self.data[h].push(key);
    }

    fn remove(&mut self, key: i32) {
        let h = MyHashSet::hash(key);
        let mut iter = self.data[h].iter_mut();
        while let Some(item) = iter.next() {
            if *item == key {
                //FIXME 题目中提示: 0 <= key <= 10^6，这里相当于逻辑删除。
                *item = i32::MIN;
                return;
            }
        }
    }

    fn contains(&self, key: i32) -> bool {
        let h = MyHashSet::hash(key);
        let mut iter = self.data[h].iter();
        while let Some(item) = iter.next() {
            if *item == key {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operations() {
        let mut set = MyHashSet::new();
        set.add(1);
        set.add(2);
        set.contains(1);
        set.contains(3);
        set.add(2);
        set.contains(2);
        set.remove(2);
        set.contains(2);
    }
}
