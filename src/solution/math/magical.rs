#[cfg(test)]
mod tests {

    #[test]
    fn magic() {
        for n in 1..=9 {
            dbg!(142857 * n);
        }
    }
}
