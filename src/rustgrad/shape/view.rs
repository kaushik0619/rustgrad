use cached::proc_macro::cached;
use itertools::enumerate;

#[cached]
fn canonicalize_strides(shape: Vec<isize>, strides: Vec<isize>) -> Vec<isize> {
    shape
        .iter()
        .zip(strides)
        .map(|(s, st)| if *s == 1 { 0 } else { st })
        .collect()
}

#[cached]
fn strides_for_shape(shape: Vec<isize>) -> Vec<isize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let strides: Vec<_> = shape[1..]
        .iter()
        .rev()
        .scan(1, |state, &val| {
            *state *= val;
            Some(*state)
        })
        .collect();

    canonicalize_strides(shape, strides.iter().rev().cloned().collect())
}

#[cached]
fn merge_dims(shape: Vec<isize>, strides: Vec<isize>, mask: Option<Vec<Vec<isize>>>) -> Vec<Vec<isize>>{
    if shape.is_empty(){
        return vec![vec![]];
    }
    assert!(shape.len() == strides.len());

    let ret = if strides.len() == 1{
        vec![shape[0], shape[0], shape[0]]
    } else {
        vec![0]
    };
    let merging = if mask.is_some(){
        mask.unwrap()[0][1] - mask.unwrap()[0][0] == 1 && strides[0] == 0
    } else{
        strides[0] == 0 && shape[0] == 1
    };
    for (i, (sh, st)) in enumerate(shape[1..].iter().zip(strides[1..])){

    }

}