#![allow(unused)]

use std::collections::BTreeSet;

fn main() {
    //let total = solution(320);
    //println!("total:{}",total);

    let numbers = vec![2, 7, 11, 15];
    let target = 18;
//    let result = two_sum(numbers, target);

//    println!("{:?}", result);

    let mut nums = vec![3,2,2,3];
    let val = 3;

    let len = remove_element(&mut nums,val);

    println!("len:{}",len);

    let nums = vec![1,1,0,1,1,1,1,0,1,1,1];
    let len = find_max_consecutive_ones(nums);
    println!("len:{}",len);
}

/// 50.神奇数字在哪里(https://developer.aliyun.com/coding/50)
///
fn solution(n: usize) -> usize {
    let bits = n.to_string().len();
    if bits >= 3 {
        let numbers = generate_array(bits);
        let mut last_result = BTreeSet::<String>::new();
        let mut temp_result = Vec::<Vec<char>>::new();
        for number in numbers {
            let bytes = number.as_bytes();
            perm(
                &mut bytes.iter().map(|x| (*x as char)).collect(),
                0,
                number.len(),
                &mut temp_result,
            );

            for result in &temp_result {
                let num: String = result.iter().collect();
                let num_usize = num.parse::<usize>().unwrap();
                if num_usize <= n {
                    last_result.insert(num);
                }
            }

            temp_result.clear();
        }

        return last_result.len();
    }
    return 0;
}

fn generate_array(bits: usize) -> Vec<String> {
    let mut result_list = Vec::<String>::new();
    if bits >= 3 {
        result_list.push(String::from("123"));

        let mut n = bits;
        if bits > 9 {
            n = 9;
        }
        let mut temp_list = Vec::<String>::new();
        let mut temp_set = BTreeSet::<String>::new();
        temp_set.insert(String::from("123"));
        if bits > 3 {
            for _ in 4..=n {
                let mut iterator = temp_set.iter();
                while let Some(temp_str) = iterator.next() {
                    let len = temp_str.len();
                    let temp_str_array = temp_str.as_bytes();
                    let mut temp_str1_array = Vec::<char>::with_capacity(len + 1);
                    temp_str1_array.push('1');
                    let mut temp_str2_array = Vec::<char>::with_capacity(len + 1);
                    let mut temp_str3_array = Vec::<char>::with_capacity(len + 1);
                    for ch in temp_str_array {
                        temp_str1_array.push(*ch as char);
                        temp_str2_array.push(*ch as char);
                        temp_str3_array.push(*ch as char);
                    }
                    temp_str2_array.push('2');
                    temp_str2_array.sort();
                    temp_str3_array.push('3');

                    let str1: String = temp_str1_array.into_iter().collect();
                    let str2: String = temp_str2_array.into_iter().collect();
                    let str3: String = temp_str3_array.into_iter().collect();

                    temp_list.push(str1);
                    temp_list.push(str2);
                    temp_list.push(str3);
                }

                temp_set.clear();
                for i in 0..temp_list.len() {
                    if let Some(str) = temp_list.get(i) {
                        temp_set.insert(str.to_string());
                        result_list.push(str.to_string());
                    }
                }
                temp_list.clear();
            }
        }
    }
    return result_list;
}

fn perm(array: &mut Vec<char>, k: usize, length: usize, result: &mut Vec<Vec<char>>) {
    if k == length - 1 {
        result.push(array.to_owned());
    } else {
        for i in k..length {
            if is_swap(array, k, i) {
                swap(array, k, i);
                perm(array, k + 1, length, result);
                swap(array, k, i);
            }
        }
    }
}

fn is_swap(array: &Vec<char>, begin: usize, end: usize) -> bool {
    for i in begin..end {
        if array[i] == array[end] {
            return false;
        }
    }
    return true;
}

fn swap(array: &mut Vec<char>, k: usize, i: usize) {
    let temp = array[k];
    array[k] = array[i];
    array[i] = temp;
}

use std::collections::HashMap;
/// 力扣--167. 两数之和 II - 输入有序数组（https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/）
pub fn two_sum(numbers: Vec<i32>, target: i32) -> Vec<i32> {
    let mut result = Vec::<i32>::with_capacity(2);

    let mut index1 = 0;
    let mut index2 = numbers.len() - 1;
    while index2 >= 1 {
        let sum = numbers[index1] + numbers[index2];
        if sum < target {
            index1 += 1;
            continue;
        } else if sum > target {
            index2 -= 1;
            continue;
        } else {
            result.push((index1 + 1) as i32);
            result.push((index2 + 1) as i32);
            break;
        }
    }

    result
}

///  力扣-- 27. 移除元素（https://leetcode-cn.com/problems/remove-element/）
pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
    let mut k = 0;
    for i in 0..nums.len(){
        if nums[i] != val{
            nums[k] = nums[i];
            k+=1;
        }
    }
    k as i32
}

///力扣 --485. 最大连续1的个数（https://leetcode-cn.com/problems/max-consecutive-ones/）
pub fn find_max_consecutive_ones(nums: Vec<i32>) -> i32 {
    let mut max = 0;
    let mut max_temp = 0;
    for num in nums {
        if num == 0{
            if max_temp > max{
                max = max_temp;
            }
            max_temp = 0;
        }else{
            max_temp+=1;
        }
    }

    if max_temp > max{
        return max_temp;
    }else{
        return max;
    }
}