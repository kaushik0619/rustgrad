use core::panic;
use std::{ops::Deref, rc::Rc};

use cached::proc_macro::cached;
use itertools::{any, Itertools};
use num::ToPrimitive;

use super::sym::{BTypes, NodeMethods, NodeTypes};

fn canonicalize_strides(shape: &[BTypes], strides: &[BTypes]) -> Vec<BTypes> {
    shape
        .iter()
        .zip(strides)
        .map(|(s, st)| if *s == BTypes::Int(1.0) { BTypes::Int(0.0) } else { st.clone() })
        .collect()
}


fn strides_for_shape(shape: &[BTypes]) -> Vec<BTypes> {
    if shape.is_empty() {
        return Vec::new();
    }

    let strides: Vec<BTypes> = shape[1..]
        .iter()
        .rev()
        .scan(1, |state, val| {
            let s = &BTypes::Int(state.clone().to_f64().unwrap()) * &val;
            Some(s)
        })
        .collect();
    
    canonicalize_strides(shape, &strides.iter().rev().cloned().collect::<Vec<BTypes>>())
}

// fn merge_dims(shape: &[isize], strides: &[isize], mask: Option<&[(isize, isize)]>) -> Vec<(isize, isize, isize)> {
//     // Merge contiguous subparts or zero strided dims.
//     // ret = Vec[(merged_dims, stride, merged dims w/o zero stride), ...]
//     if shape.is_empty() {
//         return Vec::new();
//     }
//     assert_eq!(shape.len(), strides.len());
//     let mut ret = vec![(shape[0], strides[0], if strides[0] == 0 { shape[0] } else { 0 })];
//     // wrt merging zero strided dimensions
//     let mut merging = strides[0] == 0 && (mask.map_or(false, |m| m[0].1 - m[0].0 == 1) || shape[0] == 1);
//     for (i, (&sh, &st)) in shape[1..].iter().zip(strides[1..].iter()).enumerate() {
//         if sh == 1 {
//             continue;
//         }
//         if merging || ret.last().unwrap().1 == sh * st { // mergeable
//             let (merged_dims, _, _) = ret.last_mut().unwrap();
//             *merged_dims *= sh;
//             let (_, _, merged_dims_without_zero_stride) = ret.last_mut().unwrap();
//             *merged_dims_without_zero_stride = if merging { sh } else { *merged_dims_without_zero_stride * sh };
//         } else {
//             ret.push((sh, st, if st == 0 { sh } else { 0 })); // begin new
//         }
//         // merging ends with either non-zero strided dim or zero strided dim with mask range > 1
//         merging = st == 0 && (mask.map_or(false, |m| m[i + 1].1 - m[i + 1].0 == 1) || sh == 1);
//     }
//     ret
// }

// fn reshape_mask(view: &View, new_shape: &[BTypes]) -> (Option<Vec<(BTypes, BTypes)>>, bool) {
//     if view.mask.is_none() {
//         return (view.mask.clone(), false);
//     }
//     if view.mask.clone().unwrap().into_iter().any(|m|{
//         match &m{
//             (BTypes::Int(_), BTypes::Int(_)) => true,
//             (_, _ )=> false
//         }
//     } ) {
//         return (view.mask.clone(), true);
//     }
//     let mut new_mask: Vec<(BTypes, BTypes)> = Vec::new();
//     let m: Vec<(BTypes, BTypes)> = view.mask.clone().unwrap();
//     let mut r_masks: std::iter::Rev<std::slice::Iter<'_, (BTypes, BTypes)>> = m.iter().rev();
//     let mut r_shape: std::iter::Rev<std::slice::Iter<'_, BTypes>> = view.shape.iter().rev();
//     let mut r_new_shape: std::iter::Rev<std::slice::Iter<'_, BTypes>> = new_shape.iter().rev();
//     let mut curr_stride: BTypes = BTypes::Int(1.0);
//     let mut old_dim: BTypes = r_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
//     let mut new_dim: BTypes = r_new_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
//     let mut mask: (BTypes, BTypes) = r_masks.next().unwrap_or(&(BTypes::Int(0.0), BTypes::Int(1.0))).clone();
//     if &(&mask.1 - &mask.0) < &BTypes::Int(1.0) {
//         return (Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); new_shape.len()]), false);
//     }

//     while new_mask.len() < new_shape.len() {
//         let (ref l, ref r) = mask;
//         let next_stride: BTypes = &new_dim * &curr_stride;

//         if &old_dim >= &next_stride {
//             if &old_dim == &next_stride {
//                 new_mask.push((l.floordiv(&curr_stride, true), &(r - &BTypes::Int(1.0)).floordiv(&curr_stride, true) + &BTypes::Int(1.0)));
//                 curr_stride = BTypes::Int(1.0);
//                 old_dim = r_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
//                 new_dim = r_new_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
//                 mask = r_masks.next().unwrap_or(&(BTypes::Int(0.0), BTypes::Int(1.0))).clone();
//                 if &mask.1 - &mask.0 < BTypes::Int(1.0) {
//                     return (Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); new_shape.len()]), false);
//                 }
//             } else {
//                 if (l % &next_stride != BTypes::Int(0.0) || r % &next_stride != BTypes::Int(0.0)) && l.floordiv(&next_stride, true) != (r - &BTypes::Int(1.0)).floordiv(&next_stride, true) {
//                     return (view.mask.clone(), true);
//                 }
//                 new_mask.push(((l % &next_stride).floordiv(&curr_stride, true), &(&(r - &BTypes::Int(1.0)) % &next_stride).floordiv(&curr_stride, true)+ &BTypes::Int(1.0)));
//                 curr_stride = next_stride;
//                 new_dim = r_new_shape.next().unwrap_or(&BTypes::Int(1.0)).clone();
//             }
//         } else {
//             let next_mask: &(BTypes, BTypes) = r_masks.next().unwrap_or(&(BTypes::Int(0.0), BTypes::Int(1.0)));
//             if mask != (BTypes::Int(0.0), old_dim.clone()) && &next_mask.1 - &next_mask.0 != BTypes::Int(1.0) {
//                 return (view.mask.clone(), true);
//             }
//             mask = (&(&next_mask.0 * &old_dim) + &l, &(&(&next_mask.1 - &BTypes::Int(1.0)) * &old_dim) + &r);
//             old_dim = &old_dim * r_shape.next().unwrap_or(&BTypes::Int(1.0));
//         }
//     }

//     for mask in r_masks {
//         if mask != &(BTypes::Int(0.0), BTypes::Int(1.0)) {
//             return (Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); new_shape.len()]), false);
//         }
//     }

//     (Some(new_mask.into_iter().rev().collect()), false)
// }

fn merge_dims(shape: &[isize], strides: &[isize], mask: Option<&[(isize, isize)]>) -> Vec<(isize, isize, isize)> {
    if shape.is_empty() {
        return Vec::new();
    }
    
    let mut ret = Vec::with_capacity(shape.len());
    ret.push((shape[0], strides[0], if strides[0] == 0 { shape[0] } else { 0 }));

    let mut merging = strides[0] == 0 && (mask.map_or(false, |m| m[0].1 - m[0].0 == 1) || shape[0] == 1);

    for (i, (&sh, &st)) in shape[1..].iter().zip(strides[1..].iter()).enumerate() {
        if sh == 1 {
            continue;
        }
        if merging || ret.last().unwrap().1 == sh * st {
            let (merged_dims, _, merged_dims_without_zero_stride) = ret.last_mut().unwrap();
            *merged_dims *= sh;
            *merged_dims_without_zero_stride = if merging { sh } else { *merged_dims_without_zero_stride * sh };
        } else {
            ret.push((sh, st, if st == 0 { sh } else { 0 }));
        }
        merging = st == 0 && (mask.map_or(false, |m| m[i + 1].1 - m[i + 1].0 == 1) || sh == 1);
    }
    
    ret
}

fn reshape_mask(view: &View, new_shape: &[BTypes]) -> (Option<Vec<(BTypes, BTypes)>>, bool) {
    if let Some(mask) = &view.mask {
        if mask.iter().any(|(l, r)| match (l, r) {
            (BTypes::Int(_), BTypes::Int(_)) => true,
            _ => false,
        }) {
            return (view.mask.clone(), true);
        }

        let mut new_mask: Vec<(BTypes, BTypes)> = Vec::with_capacity(new_shape.len());
        let mut r_masks: std::iter::Rev<std::slice::Iter<'_, (BTypes, BTypes)>> = mask.iter().rev();
        let mut r_shape: std::iter::Rev<std::slice::Iter<'_, BTypes>> = view.shape.iter().rev();
        let mut r_new_shape: std::iter::Rev<std::slice::Iter<'_, BTypes>> = new_shape.iter().rev();
        let mut curr_stride: BTypes = BTypes::Int(1.0);
        let mut old_dim: BTypes = r_shape.next().cloned().unwrap_or(BTypes::Int(1.0));
        let mut new_dim: BTypes = r_new_shape.next().cloned().unwrap_or(BTypes::Int(1.0));

        if &(&mask.last().unwrap().1 - &mask.last().unwrap().0) < &BTypes::Int(1.0) {
            return (Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); new_shape.len()]), false);
        }

        while let Some((l, r)) = r_masks.next() {
            let next_stride: BTypes = &new_dim * &curr_stride;

            if &old_dim >= &next_stride {
                if &old_dim == &next_stride {
                    new_mask.push((l.floordiv(&curr_stride, true), &(r - &BTypes::Int(1.0)).floordiv(&curr_stride, true) + &BTypes::Int(1.0)));
                    curr_stride = BTypes::Int(1.0);
                    old_dim = r_shape.next().cloned().unwrap_or(BTypes::Int(1.0));
                    new_dim = r_new_shape.next().cloned().unwrap_or(BTypes::Int(1.0));
                    if &(&mask.last().unwrap().1 - &mask.last().unwrap().0) < &BTypes::Int(1.0) {
                        return (Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); new_shape.len()]), false);
                    }
                } else {
                    if (l % &next_stride != BTypes::Int(0.0) || r % &next_stride != BTypes::Int(0.0)) && l.floordiv(&next_stride, true) != (r - &BTypes::Int(1.0)).floordiv(&next_stride, true) {
                        return (view.mask.clone(), true);
                    }
                    new_mask.push(((l % &next_stride).floordiv(&curr_stride, true), &(&(r - &BTypes::Int(1.0)) % &next_stride).floordiv(&curr_stride, true) + &BTypes::Int(1.0)));
                    curr_stride = next_stride;
                    new_dim = r_new_shape.next().cloned().unwrap_or(BTypes::Int(1.0));
                }
            } else {
                let next_mask = r_masks.next().unwrap_or(&(BTypes::Int(0.0), BTypes::Int(1.0)));
                if l != &BTypes::Int(0.0) && &next_mask.1 - &next_mask.0 != BTypes::Int(1.0) {
                    return (view.mask.clone(), true);
                }
                let l_term = &(&next_mask.0 * &old_dim);
                let r_term = &(&(&next_mask.1 - &BTypes::Int(1.0)) * &old_dim) + &r;
                new_mask.push((l_term + l, r_term));
                old_dim = &old_dim * &r_shape.next().cloned().unwrap_or(BTypes::Int(1.0));
            }
        }

        return (Some(new_mask.into_iter().rev().collect()), false);
    }

    (view.mask.clone(), false)
}

fn un1d(shape: &[BTypes], offs: &BTypes) -> Vec<BTypes> {
    let strides: Vec<BTypes> = strides_for_shape(shape);
    let mut result: Vec<BTypes> = Vec::new();
    let mut offs_n = offs.clone();
    if strides.is_empty() {
        result.push(BTypes::Int(0.0));
    } else {
        for stride in &strides {
            let here: BTypes = offs_n.floordiv(stride, true);
            result.push(here.clone());
            offs_n = offs - &(&here * stride);
        }
    }

    result
}
struct View{
    shape: Vec<BTypes>,
    strides: Vec<BTypes>,
    offset: BTypes,
    mask: Option<Vec<(BTypes, BTypes)>>,
    contiguous: bool
}

impl View{
    fn size(&self) -> f64{
        let ret:BTypes = self.shape.iter().map(|x|{
            match x{
                BTypes::Node(n) => NodeTypes::max(n.clone().deref()).unwrap(),
                BTypes::Int(_) => x.clone()
            }
        }).product();

        match ret{
            BTypes::Int(i) => i,
            BTypes::Node(_) => panic!("{}", format!("{:?} is not int", ret))
        }
    }

    // fn create(shape: &[BTypes], strides:Option<&[BTypes]>, offset: BTypes, mask: Option<Vec<(BTypes, BTypes)>>) -> View{
    //     let mut strides_n: Vec<BTypes>;
    //     let mut offset_n = offset;
    //     let mut mask_n = mask;
    //     if let None = strides{
    //         strides_n = strides_for_shape(shape);
    //     }else{
    //         strides_n = canonicalize_strides(shape, strides.unwrap());
    //     }
    //     if let Some(m) = &mask_n{
    //         if m.iter().zip(shape.iter()).all(|(x, y)|{
    //             *x == (BTypes::Int(0.0), y.clone())
    //         }){
    //             mask_n = None;
    //         }
    //     }
    //     let contiguous = offset_n == BTypes::Int(0.0) && mask_n.is_none() && strides == Some(&strides_for_shape(shape));
    //     let m = mask_n.clone().unwrap();
    //     let elim =  m.iter().map(|(b,e)|{
    //         !(&(b + &BTypes::Int(1.0)) < e)
    //     });
    //     if mask_n.is_some() && elim.clone().any(|x|x){
    //         if m.iter().any(|(b, e)|{
    //             !(b < e)
    //         }){
    //             strides_n = vec![BTypes::Int(0.0); shape.len()];
    //             offset_n = BTypes::Int(0.0);
    //             mask_n = Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); shape.len()]);
    //         }

    //         offset_n = &offset_n + &elim.clone().enumerate().map(|(i, e)|{
    //             if e{
    //                 &strides_n[i] * &(mask_n.clone().unwrap()[i].0)
    //             } else {
    //                 BTypes::Int(0.0)
    //             }
    //         }).sum();

    //         strides_n = strides_n.iter().zip(elim).map(|(st, e)|{
    //             if e{
    //                 BTypes::Int(0.0)
    //             } else{
    //                 st.clone()
    //             }
    //         }).collect();
    //     }
    //     View{shape: shape.to_vec(), strides: strides_n, offset: offset_n, mask: mask_n, contiguous: contiguous}
    // }
    fn create(shape: &[BTypes], strides: Option<&[BTypes]>, offset: BTypes, mask: Option<Vec<(BTypes, BTypes)>>) -> View {
        let mut strides_n: Vec<BTypes> = strides.unwrap_or(&strides_for_shape(shape)).to_vec();
    
        let mask_n = mask.map(|m| {
            if m.iter().zip(shape.iter()).all(|(x, y)| *x == (BTypes::Int(0.0), y.clone())) {
                None
            } else {
                Some(m)
            }
        }).flatten();
    
        let contiguous = offset == BTypes::Int(0.0) && mask_n.is_none() && strides == Some(&strides_for_shape(shape));
    
        if let Some(mask_ref) = &mask_n {
            if mask_ref.iter().any(|(b, e)| !(b < e)) {
                strides_n = vec![BTypes::Int(0.0); shape.len()];
                return View {
                    shape: shape.to_vec(),
                    strides: strides_n,
                    offset: BTypes::Int(0.0),
                    mask: Some(vec![(BTypes::Int(0.0), BTypes::Int(0.0)); shape.len()]),
                    contiguous: true,
                };
            }
    
            let elim = mask_ref.iter().map(|(b, e)| !(&(b + &BTypes::Int(1.0)) < e));
    
            let offset_n = elim.clone().enumerate().fold(offset, |acc, (i, e)| {
                if e {
                    &acc + &(&strides_n[i] * &mask_ref[i].0)
                } else {
                    acc
                }
            });
    
            strides_n = strides_n.iter().zip(elim).map(|(st, e)| {
                if e {
                    BTypes::Int(0.0)
                } else {
                    st.clone()
                }
            }).collect();
    
            return View {
                shape: shape.to_vec(),
                strides: strides_n.to_vec(),
                offset: offset_n,
                mask: mask_n,
                contiguous: contiguous,
            };
        }
    
        View {
            shape: shape.to_vec(),
            strides: strides_n.to_vec(),
            offset: offset,
            mask: mask_n,
            contiguous: contiguous,
        }
    }

    fn vars(&self) -> Rc<NodeTypes> {
        let mut flatten_mask = Vec::new();
        if let Some(v) = &self.mask {
            flatten_mask.extend(v.iter().flat_map(|(x, y)| vec![x.clone(), y.clone()]));
        }
    
        let mut vec = Vec::new();
        vec.extend_from_slice(&self.shape);
        vec.extend_from_slice(&self.strides);
        vec.push(self.offset.clone());
        vec.extend(flatten_mask);
    
        let f = if let BTypes::Node(n) = vec[0].clone(){
            Some(n)
        }else{
            None
        };
        NodeTypes::ands(&vec.iter().flat_map(|x| {
            if let BTypes::Node(n) = x {
                n.clone().vars()
            } else {
                Vec::new()
            }
        }).collect())
    }
}