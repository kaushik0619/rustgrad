mod rustgrad;
use crate::rustgrad::helpers::DeepFlatten;
use std::collections::HashMap;
use std::path::Path;
fn main() {
        // Create some example maps
        let cache_dir = "/your/cache/dir";  // Replace with your actual cache directory

        let cache_path = Path::new(cache_dir)
            .join("tinygrad")
            .join("cache.db");
    
        let absolute_path = cache_path.canonicalize().expect("Failed to get absolute path");
    
        println!("Absolute Path: {:?}", absolute_path);
        let mut map = std::collections::BTreeMap::new();
        map.insert("x".to_string(), 1.0);
        map.insert("y".to_string(), 2.0);
    
        let serialized = serde_pickle::to_vec(&map, Default::default()).unwrap();
        let deserialized = serde_pickle::from_slice(&serialized, Default::default()).unwrap();
    assert_eq!(map, deserialized);
}
