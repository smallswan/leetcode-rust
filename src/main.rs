#![allow(unused)]
mod dp;
mod medium;
mod simple;
mod solution;
mod solution1000;
mod spring2020;

extern "C" {
    fn rand() -> i32;
}

fn main() {
    println!("LeetCode problems that I've solved in Rust");

    let mut tower = Vec::<Vec<i32>>::new();
    tower.push(vec![3]);
    tower.push(vec![1, 5]);
    tower.push(vec![8, 4, 3]);
    tower.push(vec![2, 6, 7, 9]);
    tower.push(vec![6, 2, 3, 5, 1]);

    let max = dp::number_tower(tower);
    println!("number_tower max = {:?}", max);

    let ip = String::from("192.168.2.251");
    let nums: Vec<&str> = ip.split(".").collect();

    println!("{:?}", nums);
    for num in nums {
        println!("{} len is {}", num, num.len());
    }

    let rand = unsafe { rand() };

    println!("{}", rand);

    // let sorted_vec_v2 = vec![-3, -2, 0, 1, 4, 5];
    // let sorted_squares_vec_v2 = solution::sorted_squares_v2(sorted_vec_v2);
    // assert_eq!(sorted_squares_vec_v2, vec![0, 1, 4, 9, 16, 25]);

    // let sorted_vec_all_postive = vec![0, 1, 4, 5, 8];

    // assert_eq!(
    //     solution::sorted_squares_v2(sorted_vec_all_postive),
    //     vec![0, 1, 16, 25, 64]
    // );

    solution::sorted_squares_v2(vec![-10000, -9999, -7, -5, 0, 0, 10000]);
}
