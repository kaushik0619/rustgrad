mod rustgrad;
use std::collections::BTreeMap;
use rustgrad
fn main() {
    let mut map = BTreeMap::new();
    map.insert("x".to_string(), 1.0);
    map.insert("y".to_string(), 2.0);

    // Serialize the map into a pickle stream.
    // The second argument are serialization options.
    let serialized = serde_pickle::to_vec(&map, Default::default()).unwrap();

    // Deserialize the pickle stream back into a map.
    // Because we compare it to the original `map` below, Rust infers
    // the type of `deserialized` and lets serde work its magic.
    // The second argument are additional deserialization options.
    let deserialized = serde_pickle::from_slice(&serialized, Default::default()).unwrap();
    assert_eq!(map, deserialized);
}
