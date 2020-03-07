use std::collections::BTreeSet;

fn main() {
    println!("Hello, world!");

    let mut test_char_array = vec!['1', '2'];

    let flag = is_swap(&test_char_array,0,1);
    println!("is_swap :{}",flag);
    swap(&mut test_char_array,0,1);
    println!("{:?}",test_char_array);
    //let result = Vec::<String>::new();

    let mut result = Vec::<Vec<char>>::new();
    perm(&mut test_char_array,0,2,&mut result);
    println!("{:?}",result);

    generate_array(9);

//    let mut iter = numbers.iter();
//    while let Some(x) = iter.next(){
//        println!("{}", x);
//    }

    let total = solution(320);
    println!("total:{}",total);
}

fn solution(n : usize) -> usize{
    let bits = n.to_string().len();
    println!("bits:{}",bits);
    if bits >= 3{
        let numbers = generate_array(bits);
//        let mut last_result = BTreeSet::<String>::new();
        let mut last_result = BTreeSet::<String>::new();
        let mut temp_result = Vec::<Vec<char>>::new();
        for number in numbers {
            println!("------{}------", number);
            let bytes = number.as_bytes();
            //let mut chars = bytes.to_ascii_lowercase();

            perm(&mut bytes.iter().map(|x| (*x as char)).collect(),0,number.len(),&mut temp_result);

            for result in &temp_result{
                //println!("{:?}",result);
                let num : String = result.iter().collect();
                let num_usize = num.parse::<usize>().unwrap();
                if  num_usize <= n{
                    println!("{:?}",num_usize);
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

fn perm( array : &mut Vec<char>,k :usize,length:usize,result: &mut Vec<Vec<char>>){
    if k == length - 1{
       result.push(array.to_owned());
    }else{
        for i in k..length{
            if is_swap(array, k, i) {
                swap(array, k, i);
                perm(array, k + 1, length, result);
                swap(array, k, i);
            }
        }
    }

}

fn is_swap( array : &Vec<char>, begin : usize,end :usize) -> bool{
    for i in begin..end{
        if array[i] == array[end]{
            return false;
        }
    }
    return true;
}

fn swap( array :  &mut Vec<char>, k : usize,i :usize){
    let temp = array[k];
    array[k] = array[i];
    array[i] = temp;
}
