/// 数组最大连续子序列和
pub fn max_continue_array_sum(array: &[i32]) -> i32 {
    let len = array.len();
    let mut max = array[0];
    let mut sum = array[0];
    for i in 1..len {
        sum = if sum + array[i] > array[i] {
            sum + array[i]
        } else {
            array[i]
        };

        if sum >= max {
            max = sum;
        }
    }
    max
}

/// 4、数字塔从上到下所有路径中和最大的路径
pub fn number_tower(tower: Vec<Vec<i32>>) -> i32 {
    let mut max = 0;
    let mut dp = Vec::<Vec<i32>>::with_capacity(tower.len());
    dp.push(vec![tower[0][0]]);
    for i in 1..tower.len() {
        dp.push(vec![0; tower[i].len()]);
        let mut j = 0;
        while j <= i {
            println!("i:{},j:{}", i, j);
            println!("{:?}", dp);
            if j == 0 {
                let temp = dp[i - 1][j] + tower[i][j];
                dp[i][j] = temp;
            } else if j == i {
                dp[i][j] = dp[i - 1][j - 1] + tower[i][j];
            } else {
                let temp = if dp[i - 1][j - 1] > dp[i - 1][j] {
                    dp[i - 1][j - 1] + tower[i][j]
                } else {
                    dp[i - 1][j] + tower[i][j]
                };
                dp[i][j] = temp;
            }

            max = if dp[i][j] > max { dp[i][j] } else { max };

            j += 1;
        }
    }
    println!("{:?}", dp);

    max
}

#[test]
fn dp_demo() {
    let array = vec![6, -1, 3, -4, -6, 9, 2, -2, 5];
    let sum = max_continue_array_sum(&array);
    println!("max_continue_array_sum sum = {:?}", sum);
}
