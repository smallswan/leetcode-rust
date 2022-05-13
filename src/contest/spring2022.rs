use crate::solution::data_structures::lists::ListNode;

///  中国银联专场竞赛
///  银联-01. 回文链表 https://leetcode-cn.com/contest/cnunionpay-2022spring/problems/D7rekZ/
///  暴力解法，竞赛时超时了
pub fn is_palindrome(head: Option<Box<ListNode>>) -> bool {
    let mut vec = Vec::new();
    let mut head = &head;
    while head.is_some() {
        vec.push(head.as_ref().unwrap().val);
        head = &(head.as_ref().unwrap().next);
    }

    let len = vec.len();
    for i in 0..len {
        let mut new_vec = Vec::with_capacity(len - 1);
        for (j, item) in vec.iter().enumerate().take(len) {
            if j != i {
                new_vec.push(*item);
            }
        }

        if is_palindrome_vec(&new_vec) {
            return true;
        }
    }
    //
    fn is_palindrome_vec(data: &[i32]) -> bool {
        let (mut i, mut j) = (0, data.len() - 1);
        while i < j {
            if data[i] == data[j] {
                i += 1;
                j -= 1;
            } else {
                return false;
            }
        }
        true
    }

    false
}

/// 判断迭代器中的元素是否是回文
fn is_palindrome_iter(mut iter: impl DoubleEndedIterator<Item = impl Eq>) -> bool {
    while let (Some(left), Some(right)) = (iter.next(), iter.next_back()) {
        if left != right {
            return false;
        }
    }

    true
}

/// 银联-01. 回文链表
/// 贪心算法:
/// 在允许最多删除一个字符的情况下，同样可以使用双指针，通过贪心实现。初始化两个指针 low和 high分别指向字符串的第一个字符和最后一个字符。每次判断两个指针指向的字符是否相同，如果相同，则更新指针，将 low 加 1，high 减 1，然后判断更新后的指针范围内的子串是否是回文字符串。
/// 如果两个指针指向的字符不同，则两个字符中必须有一个被删除，此时我们就分成两种情况：即删除左指针对应的字符，留下子串 s[low+1:high]，或者删除右指针对应的字符，留下子串 s[low:high−1]。
/// 当这两个子串中至少有一个是回文串时，就说明原始字符串删除一个字符之后就以成为回文串。
pub fn is_palindrome_v2(head: Option<Box<ListNode>>) -> bool {
    let mut vec = Vec::new();
    let mut head = &head;
    while head.is_some() {
        vec.push(head.as_ref().unwrap().val);
        head = &(head.as_ref().unwrap().next);
    }

    let mut iter = vec.iter();
    while let (Some(left), Some(right)) = (iter.next(), iter.next_back()) {
        if left != right {
            let mut iter_2 = iter.clone();

            if let Some(left_2) = iter.next() {
                return (left_2 == right && is_palindrome_iter(iter))
                    || (iter_2.next_back() == Some(left) && is_palindrome_iter(iter_2));
            }

            break;
        }
    }

    true
}

/// 银联-02. 优惠活动系统 https://leetcode-cn.com/contest/cnunionpay-2022spring/problems/kDPV0f/
use std::collections::HashMap;
struct DiscountSystem {
    activities: HashMap<i32, Activity>,
}

struct Activity {
    consume_records: Vec<i32>,
    act_id: i32,
    price_limit: i32,
    discount: i32,
    number: i32,
    user_limit: i32,
}
/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl DiscountSystem {
    fn new() -> Self {
        DiscountSystem {
            activities: HashMap::new(),
        }
    }

    fn add_activity(
        &mut self,
        act_id: i32,
        price_limit: i32,
        discount: i32,
        number: i32,
        user_limit: i32,
    ) {
        let activity = Activity {
            //consume_records[user_id]记录参加活动的次数
            consume_records: vec![0; 1001],
            act_id,
            price_limit,
            discount,
            number,
            user_limit,
        };
        self.activities.insert(act_id, activity);
    }

    fn remove_activity(&mut self, act_id: i32) {
        self.activities.remove(&act_id);
    }

    fn consume(&mut self, user_id: i32, cost: i32) -> i32 {
        if self.activities.is_empty() {
            cost
        } else {
            let mut max_discount = 0;
            let mut min_act_id = i32::MAX;

            self.activities.values().for_each(|value| {
                let sum = value.consume_records.iter().sum::<i32>();
                if cost >= value.price_limit
                    && sum < value.number
                    && value.consume_records[(user_id as usize)] < value.user_limit
                {
                    if value.discount > max_discount {
                        // 若同时满足多个优惠活动时，则优先参加优惠减免最大的活动
                        max_discount = value.discount;
                        min_act_id = value.act_id;
                    } else if value.discount == max_discount {
                        // 相同折扣优先使用act_id小的
                        if value.act_id < min_act_id {
                            min_act_id = value.act_id;
                        }
                    }
                }
            });

            if min_act_id < i32::MAX {
                if let Some(value) = self.activities.get_mut(&min_act_id) {
                    value.consume_records[(user_id as usize)] += 1;

                    return cost - value.discount;
                }
            }

            cost
        }
    }
}

///  招商银行-01. 文本编辑程序设计  https://leetcode-cn.com/contest/cmbchina-2022spring/problems/fWcPGC/
pub fn delete_text(article: String, index: i32) -> String {
    let mut bytes: Vec<u8> = article.bytes().collect();
    let len = bytes.len();
    let index = index as usize;
    if bytes[index] != b' ' {
        let (mut left, mut right) = (index, index);
        while left > 0 && bytes[left] != b' ' {
            bytes[left] = b'-';
            left -= 1;
        }

        if bytes[left] != b' ' {
            bytes[left] = b'-';
        }

        while right < len && bytes[right] != b' ' {
            bytes[right] = b'-';
            right += 1;
        }

        let new_bytes: Vec<u8> = bytes.into_iter().filter(|&c| c != b'-').collect();
        let new_article = String::from_utf8(new_bytes).unwrap();

        return new_article.split_whitespace().collect::<Vec<_>>().join(" ");
    }

    article
}

/// 招商银行-02. 公园规划 https://leetcode-cn.com/contest/cmbchina-2022spring/problems/ReWLAw/
pub fn num_flowers(roads: Vec<Vec<i32>>) -> i32 {
    let len = roads.len();
    let mut edges = vec![0; len + 1];
    for road in roads {
        edges[road[0] as usize] += 1;
        edges[road[1] as usize] += 1;
    }

    1 + (*edges.iter().max().unwrap())
}

/// [力扣杯]春季编程大赛 https://leetcode-cn.com/contest/season/2022-spring
/// 1. 宝石补给 https://leetcode-cn.com/contest/season/2022-spring/problems/WHnhjV/
pub fn give_gem(gem: Vec<i32>, operations: Vec<Vec<i32>>) -> i32 {
    let mut gem = gem;
    for operation in operations {
        let half = gem[operation[0] as usize] / 2;
        gem[operation[0] as usize] -= half;
        gem[operation[1] as usize] += half;
    }
    let (mut max, mut min) = (-1, 10000);
    for g in gem {
        max = max.max(g);
        min = min.min(g);
    }

    max - min
}

/// 2. 烹饪料理 https://leetcode-cn.com/contest/season/2022-spring/problems/UEcfPD/
pub fn perfect_menu(
    materials: Vec<i32>,
    cookbooks: Vec<Vec<i32>>,
    attribute: Vec<Vec<i32>>,
    limit: i32,
) -> i32 {
    let len = cookbooks.len();
    let mut n = 2i32.pow(len as u32) - 1;
    // cook[i]表示使用i菜谱
    let mut cook = vec![0; len];
    let mut max_x = 0;
    while n > 0 {
        let mut m = n;
        for c in cook.iter_mut().rev() {
            *c = (m & 1);
            m >>= 1;
        }
        let (mut sum_x, mut sum_y) = (0, 0);
        let mut is_enough_cook = true;
        let mut need_meterials = vec![0; 5];
        for (i, c) in cook.iter().enumerate() {
            if *c == 1 {
                need_meterials[0] += cookbooks[i][0];
                need_meterials[1] += cookbooks[i][1];
                need_meterials[2] += cookbooks[i][2];
                need_meterials[3] += cookbooks[i][3];
                need_meterials[4] += cookbooks[i][4];

                if need_meterials[0] <= materials[0]
                    && need_meterials[1] <= materials[1]
                    && need_meterials[2] <= materials[2]
                    && need_meterials[3] <= materials[3]
                    && need_meterials[4] <= materials[4]
                {
                    sum_x += attribute[i][0];
                    sum_y += attribute[i][1];
                    continue;
                } else {
                    is_enough_cook = false;
                    break;
                }
            }
        }

        if is_enough_cook && sum_y >= limit {
            max_x = max_x.max(sum_x);
        }

        n -= 1;
    }

    if max_x > 0 {
        max_x
    } else {
        -1
    }
}

/// 2022招银网络笔试（第一题）
/// 题目大意：给定一些列购买理财产品的时间（格式为"小时:分钟:秒"）数组times，按照秒、分钟、小时升序排序，
///          即当秒相同时，分钟小的排在前面，同理当分钟相同时，小时数小的排在前面
pub fn time_sort(times: Vec<String>) -> Vec<String> {
    let mut result = Vec::<String>::with_capacity(times.len());
    times.into_iter().for_each(|time| {
        let hms: Vec<&str> = time.split(':').collect();
        //重新构造时间格式为"秒:分钟:小时"，方便后续的排序
        result.push(format!("{}:{}:{}", hms[2], hms[1], hms[0]));
    });

    result.sort_unstable();

    result.iter_mut().for_each(|time| {
        let smh: Vec<&str> = time.split(":").collect();
        //恢复原有的格式
        *time = format!("{}:{}:{}", smh[2], smh[1], smh[0]);
    });

    result
}

/// 2022招银网络笔试（第二题）
/// 题目大意：求闭区间[l,r]中，数字x（范围[0,9]）在各个数字中出现的次数的总和
pub fn count_num(l: i32, r: i32, x: i32) -> i32 {
    let mut count = 0;

    // 将数字num转为字符串，再统计数字x出现的次数
    fn count_x(num: i32, x: i32) -> i32 {
        let mut count = 0;
        let x_byte = (x as u8) + b'0';
        let num_str = format!("{}", num);
        num_str
            .into_bytes()
            .iter()
            .filter(|&byte| *byte == x_byte)
            .count() as i32
    }

    // 采用除10求余的方式得到数字num各个位上的数字，然后进行统计
    fn count_x_v2(num: i32, x: i32) -> i32 {
        if num == 0 && x == 0 {
            return 1;
        }
        let mut count = 0;
        let mut num = num;
        while num != 0 {
            let rem = num % 10;
            if rem == x {
                count += 1;
            }
            num /= 10;
        }

        count
    }

    for i in l..=r {
        // count += count_x(i, x);
        count += count_x_v2(i, x);
    }

    count
}

use std::collections::HashSet;
use std::iter::FromIterator;
/// 第 290 场周赛(华为) 6041 6042 6043
/// 6041. 多个数组求交集 https://leetcode-cn.com/problems/intersection-of-multiple-arrays/
pub fn intersection(nums: Vec<Vec<i32>>) -> Vec<i32> {
    let len = nums.len();
    let first = nums[0].clone();
    let mut set: HashSet<i32> = HashSet::from_iter(first);
    for item in nums.iter().take(len).skip(1) {
        //仅保留当前数组中存在的
        set.retain(|num| item.contains(num));
    }

    //排序
    let mut result: Vec<i32> = set.into_iter().collect();
    result.sort_unstable();

    result
}

/// 6042. 统计圆内格点数目 https://leetcode-cn.com/problems/count-lattice-points-inside-a-circle/
pub fn count_lattice_points(circles: Vec<Vec<i32>>) -> i32 {
    let len = circles.len();
    let mut count = 0;
    for x in 0..=200 {
        for y in 0..=200 {
            for circle in circles.iter().take(len) {
                //let circle = circles[c].clone();
                let (a, b, c) = ((circle[0] - x).abs(), (circle[1] - y).abs(), circle[2]);
                if a * a + b * b <= c * c {
                    count += 1;
                }
            }
        }
    }

    count
}

///  6043. 统计包含每个点的矩形数目 https://leetcode-cn.com/contest/weekly-contest-290/problems/count-number-of-rectangles-containing-each-point/
pub fn count_rectangles(rectangles: Vec<Vec<i32>>, points: Vec<Vec<i32>>) -> Vec<i32> {
    let len = points.len();
    let mut count = vec![0; len];
    for i in 0..len {
        //let point = points[i];
        let mut acc = 0;
        for rectangle in &rectangles {
            if rectangle[0] >= points[i][0] && rectangle[1] >= points[i][1] {
                acc += 1;
            }
        }
        count[i] = acc;
    }

    count
}

/// 6043. 统计包含每个点的矩形数目  https://leetcode-cn.com/contest/weekly-contest-290/problems/count-number-of-rectangles-containing-each-point/
/// 1 <= hi, yj <= 100
pub fn count_rectangles_v2(rectangles: Vec<Vec<i32>>, points: Vec<Vec<i32>>) -> Vec<i32> {
    let len = points.len();
    let mut count = vec![0; len];
    // 桶排序
    let mut rectangles_groups = vec![vec![]; 101];

    // 按照高度分组（桶）
    for rectangle in &rectangles {
        let high = rectangle[1];
        rectangles_groups[high as usize].push(rectangle[0]);
    }

    // 每个组（桶）中又按照长度排序
    rectangles_groups.iter_mut().for_each(|group| {
        group.sort_unstable();
    });

    for i in 0..len {
        let mut acc = 0;
        let (x, y) = (points[i][0], points[i][1]);
        for j in y..=100 {
            for &l in rectangles_groups[j as usize].iter().rev() {
                if l >= x {
                    acc += 1;
                } else {
                    break;
                }
            }
        }

        count[i] = acc;
    }

    count
}

/// 6051. 统计是给定字符串前缀的字符串数目 https://leetcode-cn.com/problems/count-prefixes-of-a-given-string/
pub fn count_prefixes(words: Vec<String>, s: String) -> i32 {
    let s_bytes = s.as_bytes();
    words
        .iter()
        .filter(|word| s_bytes.starts_with(word.as_bytes()))
        .count() as i32
}

///  6052. 最小平均差  https://leetcode-cn.com/contest/biweekly-contest-77/problems/minimum-average-difference/
pub fn minimum_average_difference(nums: Vec<i32>) -> i32 {
    let mut len = 0;
    if len == 1 {
        return 0;
    }
    let mut sum = 0;
    for num in &nums {
        len += 1;
        sum += num;
    }

    let mut acc = 0;
    let mut min_avg_diff = i32::MAX;
    let mut min_i = i32::MAX;
    for i in 0..len {
        acc += nums[i];
        let avg1 = acc / (i + 1) as i32;
        let count2 = if i == len - 1 {
            1
        } else {
            (len - i) as i32 - 1
        };
        let avg2 = (sum - acc) / count2;

        let diff = (avg1 - avg2).abs();

        if diff < min_avg_diff {
            min_avg_diff = diff;
            min_i = i as i32;
            if diff == 0 {
                return i as i32;
            }
        }
    }

    min_i
}

/// 6053. 统计网格图中没有被保卫的格子数 https://leetcode-cn.com/problems/count-unguarded-cells-in-the-grid/submissions/
pub fn count_unguarded(m: i32, n: i32, guards: Vec<Vec<i32>>, walls: Vec<Vec<i32>>) -> i32 {
    let (m, n) = (m as usize, n as usize);
    let mut grid = vec![vec![b'0'; n]; m];
    for wall in walls {
        grid[wall[0] as usize][wall[1] as usize] = b'W';
    }
    for guard in &guards {
        grid[guard[0] as usize][guard[1] as usize] = b'G';
    }

    let mut guard_cnt = 0;
    for guard in &guards {
        //行
        let row = guard[0] as usize;
        let col = guard[1] as usize;
        let (mut row_t, mut col_t) = (row, col);

        while row_t > 0 {
            row_t -= 1;
            if grid[row_t][col] == b'0' {
                grid[row_t][col] = b'1';
            } else if grid[row_t][col] == b'1' {
                continue;
            } else {
                break;
            }
        }

        row_t = row;
        row_t += 1;
        while row_t < m {
            if grid[row_t][col] == b'0' {
                grid[row_t][col] = b'1';
                row_t += 1;
            } else if grid[row_t][col] == b'1' {
                row_t += 1;
                continue;
            } else {
                break;
            }
        }

        //列
        col_t = col;
        while col_t > 0 {
            col_t -= 1;
            if grid[row][col_t] == b'0' {
                grid[row][col_t] = b'1';
            } else if grid[row][col_t] == b'1' {
                continue;
            } else {
                break;
            }
        }
        col_t = col;
        col_t += 1;
        while col_t < n {
            if grid[row][col_t] == b'0' {
                grid[row][col_t] = b'1';
                col_t += 1;
            } else if grid[row][col_t] == b'1' {
                col_t += 1;
                continue;
            } else {
                break;
            }
        }
    }

    for row in grid {
        guard_cnt += row.iter().filter(|&byte| *byte == b'0').count() as i32;
    }

    guard_cnt
}

use std::collections::BTreeMap;
/// 2251. 花期内花的数目 https://leetcode-cn.com/problems/number-of-flowers-in-full-bloom/
pub fn full_bloom_flowers(flowers: Vec<Vec<i32>>, persons: Vec<i32>) -> Vec<i32> {
    let n = persons.len();
    let mut ans = vec![0; n];
    let mut map = BTreeMap::<i32, i32>::new();
    for flower in flowers {
        let (a, b) = (flower[0], flower[1]);
        if let Some(start) = map.get_mut(&a) {
            *start += 1;
        } else {
            map.insert(a, 1);
        }

        if let Some(end) = map.get_mut(&(b + 1)) {
            *end -= 1;
        } else {
            map.insert(b + 1, -1);
        }
    }

    let mut sum = 0;
    let mut id = vec![0];

    map.insert(0, 0);
    for (key, value) in map.iter_mut() {
        *value += sum;
        sum = *value;
        id.push(*key);
    }

    fn binary_search(a: &[i32], target: i32) -> usize {
        let n = a.len();
        let (mut l, mut r) = (0, n);
        while l < r {
            let middle = (l + r) / 2;
            if target >= a[middle] {
                l = middle + 1;
            } else {
                r = middle;
            }
        }

        if l == 0 {
            l
        } else {
            l - 1
        }
    }

    for i in 0..n {
        let k = binary_search(&id, persons[i]);
        if let Some(&count) = map.get(&(id[k] as i32)) {
            ans[i] = count;
        }
    }

    ans
}

/**
 * Your DiscountSystem object will be instantiated and called as such:
 * let obj = DiscountSystem::new();
 * obj.add_activity(actId, priceLimit, discount, number, userLimit);
 * obj.remove_activity(actId);
 * let ret_3: i32 = obj.consume(userId, cost);
 */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solution::data_structures::lists;

    #[test]
    fn test_spring() {
        let materials: Vec<i32> = vec![3, 2, 4, 1, 2];
        let cookbooks: Vec<Vec<i32>> = vec![
            vec![1, 1, 0, 1, 2],
            vec![2, 1, 4, 0, 0],
            vec![3, 2, 4, 1, 0],
        ];
        let attribute: Vec<Vec<i32>> = vec![vec![3, 2], vec![2, 4], vec![7, 6]];
        let limit: i32 = 5;
        dbg!(perfect_menu(materials, cookbooks, attribute, limit));

        let flowers = vec![vec![1, 10], vec![3, 3]];
        let persons = vec![3, 3, 2];
        dbg!(full_bloom_flowers(flowers, persons));
    }

    #[test]
    fn cmbchina() {
        delete_text("Singing dancing in the rain".to_string(), 10);
    }

    #[test]
    fn cmbnt() {
        let times = vec![
            "12:30:10".to_string(),
            "12:15:10".to_string(),
            "11:20:14".to_string(),
        ];
        dbg!(time_sort(times));

        let (l, r, x) = (2, 22, 2);
        dbg!(count_num(l, r, x));

        let (l, r, x) = (1, 999999, 9);
        dbg!(count_num(l, r, x));

        let (l, r, x) = (1, 20, 0);
        dbg!(count_num(l, r, x));
    }

    #[test]
    fn unionpay() {
        dbg!(is_palindrome_iter(vec![1, 2, 3, 4, 4, 3, 2, 1].into_iter()));
        dbg!(is_palindrome_iter(vec!["s", "o", "s"].into_iter()));

        let head = lists::vec_to_list(&vec![1, 2, 3, 1]);
        dbg!(is_palindrome(head));

        let head = lists::vec_to_list(&vec![1, 2, 3, 1]);
        dbg!(is_palindrome_v2(head));
    }

    #[test]
    fn discount() {
        let mut system = DiscountSystem::new();
        // 创建编号 1 的优惠活动，单笔消费原价不小于 10 时，可享受 6 的减免，优惠活动共有 3 个名额，每个用户最多参与该活动 2 次
        system.add_activity(1, 10, 6, 3, 2);

        // 创建编号 2 的优惠活动，单笔消费原价不小于 15 时，可享受 8 的减免，优惠活动共有 8 个名额，每个用户最多参与该活动 2 次
        system.add_activity(2, 15, 8, 8, 2);

        dbg!(system.consume(101, 13)); // 用户 101 消费了 13，仅满足优惠活动 1 条件，返回实际支付 13 - 6 = 7用户 101 参加 1 次活动 1，活动 1 剩余 2 个名额
        dbg!(system.consume(101, 8)); // 用户 101 消费了 8，不满足任何活动，返回支付原价 8
        system.remove_activity(2); // 结束编号为 2 的活动
        dbg!(system.consume(101, 17)); // 用户 101 消费了 17，满足优惠活动 1 条件，返回实际支付 17 - 6 = 11用户 101 参加 2 次活动 1，活动 1 剩余 1 个名额
        dbg!(system.consume(101, 11)); // 用户 101 消费了 11，满足优惠活动 1 条件，但已达到参加次数限制，返回支付原价 11
        dbg!(system.consume(102, 16)); // 用户 102 消费了 16，满足优惠活动 1 条件，返回实际支付 16 - 6 = 10用户 102 参加 1 次活动 1，活动 1 无剩余名额
        dbg!(system.consume(102, 21)); // 用户 102 消费了 21，满足优惠活动 1 条件，但活动 1 已无剩余名额，返回支付原价 21
    }

    // ["DiscountSystem","consume","consume","addActivity","consume","consume","addActivity","consume","consume"]
    //[[],[4,69],[4,82],[5,45,19,3,2],[6,63],[6,95],[3,38,15,3,3],[6,70],[8,49]]
    #[test]
    fn test_discount2() {
        let mut system = DiscountSystem::new();
        dbg!(system.consume(4, 69));
        dbg!(system.consume(4, 82));
        system.add_activity(5, 45, 19, 3, 2);
        dbg!(system.consume(6, 63));
        dbg!(system.consume(6, 95));
        system.add_activity(3, 38, 15, 3, 3);
        dbg!(system.consume(6, 70));
        dbg!(system.consume(8, 49));
    }

    #[test]
    fn test_biweekly_contest_77() {
        let (m, n) = (8, 9);
        let guards = vec![vec![5, 8], vec![5, 5], vec![4, 6], vec![0, 5], vec![6, 5]];
        let walls = vec![vec![4, 1]];
        dbg!(count_unguarded(m, n, guards, walls));
    }
}
