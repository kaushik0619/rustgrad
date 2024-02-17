mod rustgrad;
use crate::rustgrad::helpers::DeepFlatten;
use std::collections::HashMap;
fn main() {
        // Create some example maps
        let map1: HashMap<&str, i32> = [("a", 1), ("b", 2)].iter().cloned().collect();
        let map2: HashMap<&str, i32> = [("e", 3), ("c", 4)].iter().cloned().collect();
        let map3: HashMap<&str, i32> = [("d", 5)].iter().cloned().collect();

        // Merge the maps
        let merged_map = rustgrad::helpers::merge_maps(vec![map1.clone(), map2.clone(), map3.clone()]);

        // Check the expected result
        let expected_result: HashMap<&str, i32> = [("a", 1), ("b", 3), ("c", 4), ("d", 5)]
            .iter()
            .cloned()
            .collect();

        assert_eq!(merged_map, expected_result);

}