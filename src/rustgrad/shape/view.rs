use core::panic;

use cached::proc_macro::cached;
use num::ToPrimitive;

use super::sym::BTypes;

fn canonicalize_strides(shape: &[isize], strides: &[isize]) -> Vec<isize> {
    shape
        .iter()
        .zip(strides)
        .map(|(s, st)| if *s == 1 { 0 } else { st.clone() })
        .collect()
}


fn strides_for_shape(shape: &[isize]) -> Vec<isize> {
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
    
    canonicalize_strides(shape, &strides.iter().rev().cloned().collect::<Vec<isize>>())
}

fn merge_dims(shape: &[isize], strides: &[isize], mask: Option<&[(isize, isize)]>) -> Vec<(isize, isize, isize)> {
    // Merge contiguous subparts or zero strided dims.
    // ret = Vec[(merged_dims, stride, merged dims w/o zero stride), ...]
    if shape.is_empty() {
        return Vec::new();
    }
    assert_eq!(shape.len(), strides.len());
    let mut ret = vec![(shape[0], strides[0], if strides[0] == 0 { shape[0] } else { 0 })];
    // wrt merging zero strided dimensions
    let mut merging = strides[0] == 0 && (mask.map_or(false, |m| m[0].1 - m[0].0 == 1) || shape[0] == 1);
    for (i, (&sh, &st)) in shape[1..].iter().zip(strides[1..].iter()).enumerate() {
        if sh == 1 {
            continue;
        }
        if merging || ret.last().unwrap().1 == sh * st { // mergeable
            let (merged_dims, _, _) = ret.last_mut().unwrap();
            *merged_dims *= sh;
            let (_, _, merged_dims_without_zero_stride) = ret.last_mut().unwrap();
            *merged_dims_without_zero_stride = if merging { sh } else { *merged_dims_without_zero_stride * sh };
        } else {
            ret.push((sh, st, if st == 0 { sh } else { 0 })); // begin new
        }
        // merging ends with either non-zero strided dim or zero strided dim with mask range > 1
        merging = st == 0 && (mask.map_or(false, |m| m[i + 1].1 - m[i + 1].0 == 1) || sh == 1);
    }
    ret
}

fn reshape_mask(view: &View, new_shape: &[BTypes]) -> (Option<Vec<(BTypes, BTypes)>>, bool) {
    if view.mask.is_none() {
        return (view.mask.clone(), false);
    }
    if view.mask.clone().unwrap().into_iter().any(|m|{
        match &m{
            (BTypes::Int(_), BTypes::Int(_)) => true,
            (_, _ )=> false
        }
    } ) {
        return (view.mask.clone(), true);
    }
    let mut new_mask: Vec<(BTypes, BTypes)> = Vec::new();
    let m = view.mask.clone().unwrap();
    let mut r_masks = m.iter().rev();
    let mut r_shape = view.shape.iter().rev();
    let mut r_new_shape = new_shape.iter().rev();
    let mut curr_stride = BTypes::Int(1.0);
    let mut old_dim = r_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
    let mut new_dim = r_new_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
    let mut mask = r_masks.next().unwrap_or(&(BTypes::Int(0.0), BTypes::Int(1.0))).clone();
    if &(&mask.1 - &mask.0) < &BTypes::Int(1.0) {
        return (Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); new_shape.len()]), false);
    }

    while new_mask.len() < new_shape.len() {
        let (ref l, ref r) = mask;
        let next_stride = &new_dim * &curr_stride;

        if &old_dim >= &next_stride {
            if &old_dim == &next_stride {
                new_mask.push((l.floordiv(&curr_stride, true), &(r - &BTypes::Int(1.0)).floordiv(&curr_stride, true) + &BTypes::Int(1.0)));
                curr_stride = BTypes::Int(1.0);
                old_dim = r_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
                new_dim = r_new_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
                mask = r_masks.next().unwrap_or(&(BTypes::Int(0.0), BTypes::Int(1.0))).clone();
                if &mask.1 - &mask.0 < BTypes::Int(1.0) {
                    return (Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); new_shape.len()]), false);
                }
            } else {
                if (l % &next_stride != BTypes::Int(0.0) || r % &next_stride != BTypes::Int(0.0)) && l.floordiv(&next_stride, true) != (r - &BTypes::Int(1.0)).floordiv(&next_stride, true) {
                    return (view.mask.clone(), true);
                }
                new_mask.push(((l % &next_stride).floordiv(&curr_stride, true), &(&(r - &BTypes::Int(1.0)) % &next_stride).floordiv(&curr_stride, true)+ &BTypes::Int(1.0)));
                curr_stride = next_stride;
                new_dim = r_new_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
            }
        } else {
            let next_mask = r_masks.next().unwrap_or(&(BTypes::Int(0.0), BTypes::Int(1.0)));
            if mask != (BTypes::Int(0.0), old_dim.clone()) && &next_mask.1 - &next_mask.0 != BTypes::Int(1.0) {
                return (view.mask.clone(), true);
            }
            mask = (&(&next_mask.0 * &old_dim) + &l, &(&(&next_mask.1 - &BTypes::Int(1.0)) * &old_dim) + &r);
            old_dim = &old_dim * r_shape.next().unwrap_or(&BTypes::Int(1.0));
        }
    }

    for mask in r_masks {
        if mask != &(BTypes::Int(0.0), BTypes::Int(1.0)) {
            return (Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); new_shape.len()]), false);
        }
    }

    (Some(new_mask.into_iter().rev().collect()), false)
}
struct View{
    shape: Vec<BTypes>,
    strides: Vec<BTypes>,
    offset: BTypes,
    mask: Option<Vec<(BTypes, BTypes)>>,
    contiguous: bool
}