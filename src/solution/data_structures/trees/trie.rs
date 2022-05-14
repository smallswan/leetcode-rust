//! Trie(前缀树，字典树)

/// 208. 实现 Trie (前缀树) https://leetcode.cn/problems/implement-trie-prefix-tree/
#[derive(Default)]
struct Trie {
    root: Node,
}

#[derive(Default)]
struct Node {
    end: bool,
    //word 和 prefix 仅由小写英文字母组成
    children: [Option<Box<Node>>; 26],
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl Trie {
    fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, word: String) {
        let mut node = &mut self.root;
        for c in word.as_bytes() {
            let index = (c - b'a') as usize;
            let next = &mut node.children[index];
            node = next.get_or_insert_with(Box::<Node>::default)
        }
        node.end = true;
    }

    fn search(&self, word: String) -> bool {
        self.word_node(&word).map_or(false, |node| node.end)
    }

    fn starts_with(&self, prefix: String) -> bool {
        self.word_node(&prefix).is_some()
    }

    // 前缀字符串
    // wps: word_prefix_string
    fn word_node(&self, wps: &str) -> Option<&Node> {
        let mut node = &self.root;
        for c in wps.as_bytes() {
            let index = (c - b'a') as usize;
            match &node.children[index] {
                None => return None,
                Some(next) => node = next.as_ref(),
            }
        }

        Some(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie() {
        let mut trie = Trie::new();
        trie.insert("apple".to_string());
        assert!(trie.search("apple".into()));
        assert!(!trie.search("app".to_string()));
        assert!(trie.starts_with("app".to_string()));
        trie.insert("app".to_string());
        trie.search("app".to_string());
        assert!(trie.search("app".to_string()));
    }
}
