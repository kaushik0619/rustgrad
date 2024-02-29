use cached::proc_macro::cached;

#[cached]
fn canonicalize_strides(shape: Vec<isize>, strides: Vec<isize>) -> Vec<isize> {
    shape.iter()
        .zip(strides)
        .map(|(s, st)| if *s == 1 { 0 } else { st })
        .collect()
}

#[cached]
fn strides_for_shape(shape: Vec<isize>) -> Vec<isize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let strides: Vec<_> = shape[1..].iter().rev().scan(1, |state, &val| {
        *state *= val;
        Some(*state)
    }).collect();

    canonicalize_strides(shape, strides.iter().rev().cloned().collect())
}

