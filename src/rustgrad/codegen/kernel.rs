use std::{any::Any, borrow::BorrowMut, collections::{HashMap, HashSet}, fmt::{Debug, Display}, ops::Deref, rc::Rc, sync::Arc};

use itertools::{iproduct, Itertools};
use lazy_static::lazy_static;

use crate::rustgrad::{device::Device, dtype::{self, DTypes, ImageDType, DTYPES_DICT}, helpers::{dedup, flatten, get_contraction}, ops::{get_lazyop_info, BufferOps, BufferTypes, FlopCounter, Items, LazyOp, MemBuffer, Op}, shape::{shapetracker::ShapeTracker, sym::{BTypes, NodeTypes}, view::strides_for_shape}};

lazy_static!{
    pub static ref TENSOR_CORES: Arc<HashMap<String, Vec<TensorCore>>> = {
        let mut hm = HashMap::new();
        hm.insert(String::from("METAL"), vec![
            TensorCore{dims: vec![8, 8, 8], dtype_out:DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(), dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(), wmma_func: String::from("__metal_wmma<float2,simdgroup_float8x8,float2>"), threads: vec![(0,2), (1,4), (0,2),(1,2)], thread_local_sizes: vec![vec![2], vec![2], vec![2]],         thread_local_aliases: vec![
                vec![vec![4], vec![0], vec![2], vec![0], vec![-1, 1, 3], vec![0]],
                vec![vec![0], vec![3], vec![0], vec![1], vec![2, 4], vec![-1]],
                vec![vec![4], vec![3], vec![2], vec![1], vec![0], vec![-1]],
            ]},
            TensorCore {
                dims: vec![8, 8, 8],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(),
                threads: vec![(0, 2), (1, 4), (0, 2), (1, 2)],
                thread_local_sizes: vec![vec![2], vec![2], vec![2]],
                thread_local_aliases: vec![
                    vec![vec![4], vec![0], vec![2], vec![0], vec![-1, 1, 3], vec![0]],
                    vec![vec![0], vec![3], vec![0], vec![1], vec![2, 4], vec![-1]],
                    vec![vec![4], vec![3], vec![2], vec![1], vec![0], vec![-1]],
                ],
                wmma_func: "__metal_wmma<half2,simdgroup_float8x8,float2>".to_string(),
            },
            TensorCore {
                dims: vec![8, 8, 8],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                threads: vec![(0, 2), (1, 4), (0, 2), (1, 2)],
                thread_local_sizes: vec![vec![2], vec![2], vec![2]],
                thread_local_aliases: vec![
                    vec![vec![4], vec![0], vec![2], vec![0], vec![-1, 1, 3], vec![0]],
                    vec![vec![0], vec![3], vec![0], vec![1], vec![2, 4], vec![-1]],
                    vec![vec![4], vec![3], vec![2], vec![1], vec![0], vec![-1]],
                ],
                wmma_func: "__metal_wmma<half2,simdgroup_half8x8,half2>".to_string(),
            }
        ]);
        hm.insert(String::from("HSA"), vec![
            TensorCore {
                dims: vec![16, 16, 16],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(),
                threads: vec![(0, 16), (1, 2)],
                thread_local_sizes: vec![vec![16], vec![16], vec![8]],
                thread_local_aliases: vec![
                    vec![vec![0], vec![0], vec![-1], vec![1]],
                    vec![vec![0], vec![1], vec![-1], vec![0]],
                    vec![vec![0], vec![1], vec![0], vec![2, -1]],
                ],
                wmma_func: "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32".to_string(),
            },
            TensorCore {
                dims: vec![16, 16, 16],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                threads: vec![(0, 16), (1, 2)],
                thread_local_sizes: vec![vec![16], vec![16], vec![8]],
                thread_local_aliases: vec![
                    vec![vec![0], vec![0], vec![-1], vec![1]],
                    vec![vec![0], vec![1], vec![-1], vec![0]],
                    vec![vec![0], vec![1], vec![0], vec![2, -1]],
                ],
                wmma_func: "__hip_wmma_f16_f16".to_string(),
            }
        ]);

        hm.insert(String::from("CUDA"), vec![
            TensorCore {
                dims: vec![8, 16, 16],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(),
                threads: vec![(0, 2), (0, 2), (1, 2), (1, 2), (0, 2)],
                thread_local_sizes: vec![vec![2, 2, 2], vec![2, 2], vec![2, 2]],
                thread_local_aliases: vec![
                    vec![
                        vec![0],
                        vec![-2],
                        vec![5],
                        vec![0],
                        vec![0],
                        vec![-1, 1, 2, -3],
                        vec![3, 4],
                    ],
                    vec![
                        vec![5],
                        vec![0],
                        vec![0],
                        vec![4],
                        vec![3],
                        vec![-1, 1, 2, -2],
                        vec![0],
                    ],
                    vec![
                        vec![2],
                        vec![-2],
                        vec![5],
                        vec![1],
                        vec![-1],
                        vec![0],
                        vec![3, 4],
                    ],
                ],
                wmma_func: "__cuda_mma_m16n8k16_f16_f32".to_string(),
            }
        ]);

        Arc::new(hm)
    };
}
#[derive(PartialEq, PartialOrd, Debug)]
enum OptOps{
    TC,
    UPCAST,
    UPCASTMID,
    UNROLL,
    LOCAL,
    GROUP,
    GROUPTOP,
    NOLOCALS,
    PADTO
}

#[derive(PartialEq, PartialOrd, Clone)]
struct Opt{
    op: OptOps,
    axis: Option<isize>,
    amt: Option<isize>
}

impl Debug for Opt{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Opt(op={:?}, axis={:?}, amt={:?})", self.op, self.axis, self.amt)
    }
}

impl Display for Opt{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Opt(op={:?}, axis={:?}, amt={:?})", self.op, self.axis, self.amt)
    }
}

#[derive(Clone)]
struct TensorCore{
    dims: Vec<isize>,
    dtype_in: DTypes,
    dtype_out: DTypes,
    threads: Vec<(isize, isize)>,
    thread_local_aliases: Vec<Vec<Vec<isize>>>,
    thread_local_sizes: Vec<Vec<isize>>,
    wmma_func: String
}

impl ToString for TensorCore{
    fn to_string(&self) -> String {
        format!("tensor_core<{:?}, {:?}, {:?}>", self.dims, self.dtype_in, self.dtype_out)
    }
}

impl TensorCore{
    fn num_threads(&self) -> usize{
        self.threads.len()
    }
    fn num_upcasts(&self) -> usize{
        self.thread_local_aliases[0].len() - self.num_threads()
    }
}

#[derive(Clone)]
struct TensorCoreOptions{
    bufs: (isize, isize),
    axes: Vec<isize>,
    axes_exist: Vec<bool>
}

impl TensorCoreOptions{
    fn fix_axes(&mut self, removed_axis: isize){
        for (to_dim, &exist) in self.axes_exist.iter().enumerate(){
            if exist{
                if removed_axis < self.axes[to_dim]{
                    self.axes[to_dim] -= 1;
                }else if removed_axis == self.axes[to_dim]{
                    self.axes_exist[to_dim] = false;
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Hash, Clone)]
pub struct LocalBuffer{
    name: String,
    size: usize,
    dtype: DTypes,
    realized: Option<bool>
}
impl LocalBuffer{
    fn new(name: String, size: usize, dtype:DTypes) -> LocalBuffer{
        LocalBuffer { name, size, dtype, realized: None }
    }
}

impl ToString for LocalBuffer{
    fn to_string(&self) -> String {
        return format!("localbuffer<{}[{}]>", self.name, self.size)
    }
}

#[derive(Clone)]
struct LinearizerOptions{
    device: String,
    suffix: String,
    supports_float4: bool,
    has_local: bool,
    has_shared: bool,
    hash_tensor_cores: bool,
    global_max: Option<Vec<usize>>,
    local_max: Option<Vec<usize>>,
    shared_max: usize
}

impl Default for LinearizerOptions{
    fn default() -> Self {
        LinearizerOptions { device: String::new(), suffix: String::new(), supports_float4: true, has_local: true, has_shared: true, hash_tensor_cores: false, global_max: None, local_max: None, shared_max: 32768 }
    }
}

#[derive(Clone)]
struct Kernel{
    opts: Option<LinearizerOptions>,
    ast: Vec<Rc<LazyOp>>,
    lazyops: Vec<Rc<LazyOp>>,
    info: FlopCounter,
    reduceop: Option<Rc<LazyOp>>,
    outbufs: Vec<Option<Items>>,
    vars: Vec<Rc<NodeTypes>>,
    bufs: Vec<BufferTypes>,
    earlybufs: Vec<Option<Items>>,
    full_buf_index: isize,
    sts: Vec<Rc<ShapeTracker>>,
    applied_opts: Vec<Opt>,
    group_for_reduces: isize,
    upcasted: isize,
    local_dims: isize,
    local_alias: HashMap<isize, LocalBuffer>,
    tensor_core: Option<TensorCore>,
    tensor_core_opts: Option<TensorCoreOptions>,
    dont_use_locals: bool,
    applied_opts_cache: Option<Vec<Opt>>
}

impl Kernel{
    fn new(ast: Vec<Rc<LazyOp>>, opts: Option<LinearizerOptions>) -> Kernel{
        let opts_n = {
            opts.or({
                let device = Device.index(&Device.default());
                if let Some(d) = device.compiler{
                    Some(d.linearizer_opts)
                }else{
                    None
                }
            })
        };

        assert!(ast.iter().all(|op|{
            Op::BufferOps(BufferOps::STORE) == op.op
        }));

        assert!(ast.iter().map(|op|{
            if let Some(Items::Buffer(b)) = &op.arg{
                match b{
                    BufferTypes::ConstBuffer(c) => c.st.size(),
                    BufferTypes::MemBuffer(m) => m.st.size(),
                    BufferTypes::LocalBuffer(_) => panic!()
                }
            }else{
                panic!()
            }
        }).collect::<HashSet<usize>>().len() == 1);

        let ast_n = ast;
        let lazyops = ast_n.iter().map(|op|op.lazyops()).flatten().collect_vec();
        let info = get_lazyop_info(ast_n[0].clone());
        let reduceops = {
            let redops = lazyops.iter().filter_map(|x|{
                if let Op::ReduceOps(_) = x.op{
                    Some(x.clone())
                }else{
                    None
                }
            }).collect_vec();
            assert!(dedup(redops).len() <= 1);
            if !redops.is_empty(){
                Some(redops[0])
            }else{
                None
            }
        };

        let outbufs = ast_n.iter().map(|x| x.clone().arg).collect_vec();
        let vars = ast_n.iter().map(|x|x.vars()).flatten().collect_vec();

        let mut bufs: Vec<BufferTypes>  = outbufs.clone().into_iter().filter_map(|x|{
            if let Some(Items::Buffer(b)) = x{
                Some(b)
            }else{
                None
            }
        }).collect_vec();

        bufs.extend(lazyops.iter().filter_map(|x|{
            if let Op::BufferOps(BufferOps::CONST) = &x.op{
                if let Some(Items::Buffer(b)) = &x.arg{
                    return Some(b.clone())
                }
            }else if let Op::BufferOps(BufferOps::LOAD) = &x.op{
                if let Some(Items::Buffer(b)) = &x.arg{
                    return Some(b.clone())
                }
            }
            None
        }));

        let earlybufs = {
            if let Some(r) = &reduceops{
                r.lazyops().iter().filter_map(|x|{
                    if let Op::BufferOps(_) = &x.op{
                        Some(x.arg.clone())
                    }else{
                        None
                    }
                }).collect_vec()
            }else{
                vec![]
            }
        };

        let full_buf_index = {
            if !earlybufs.is_empty(){
                bufs.iter().position(|x| &Some(Items::Buffer(x.clone()))==&earlybufs[0])
            }else{
                Some(0)
            }
        };

        let sts = {
           bufs.iter().filter_map(|x|{
            if let BufferTypes::MemBuffer(m) = x{
                Some(m.st.clone())
            } else if let BufferTypes::ConstBuffer(c) = x{
                Some(c.st.clone())
            }else{
                None
            }
           }).collect_vec()
        };

        let reduce = full_shape(&sts, full_buf_index.unwrap_or(0)).iter().zip(Self::output_shape(&sts).iter()).enumerate().collect_vec();
        let mut premute = reduce.iter().filter_map(|(i, (s, n))|{
            if s == n{
                Some(i.clone())
            }else{
                None
            }
        }).collect_vec();

        premute.extend(reduce.into_iter().filter_map(|(i, (s, n))|{
            if s!=n{
                Some(i)
            }else{
                None
            }
        }));

        reshape_and_permute(&mut sts,None, Some(premute));



        let appiled_opts = vec![];
        let group_for_reduces:isize = 0;
        let upcasted: isize = 0;
        let local_dims: isize = 0;
        let local_alias: HashMap<isize, LocalBuffer> = HashMap::new();
        let tensor_core = None;
        let tensor_core_opts = None;
        let dont_use_locals = false;

        let mut k = Kernel{
            opts: opts_n,
            ast: ast_n,
            lazyops,
            info,
            reduceop: reduceops,
            outbufs,
            vars,
            bufs,
            earlybufs,
            full_buf_index: full_buf_index.unwrap() as isize,
            sts,
            applied_opts: appiled_opts,
            group_for_reduces,
            upcasted,
            local_dims,
            local_alias,
            tensor_core,
            tensor_core_opts,
            dont_use_locals,
            applied_opts_cache:None
        };
        k.simplify_ones();
        k.simplify_merge_adjacent();


    }


    fn full_shape(&self) -> Vec<BTypes>{
        self.sts[self.full_buf_index.clone() as usize].clone().shape()
    }

    fn output_shape(&self) -> Vec<BTypes>{
        self.sts[0].clone().shape()
    }

    fn reshape_and_permute<T>(&mut self, new_shape_fxn: Option<T>, axis: Option<Vec<usize>>)
        where
            T: Fn(Vec<BTypes>) -> Vec<BTypes>
    {
        let mut new_sts = vec![];
        self.sts.into_iter().for_each(|mut st|{
            if let Some(x) = new_shape_fxn{
                st = *st.reshape(&x(st.shape())).borrow_mut()
            }
            if let Some(x) = axis{
                st = st.permute(&x.into_iter().map(|y| y as isize).collect_vec())
            }
            new_sts.push(st)
        });
        self.sts = new_sts;
    }

    fn shape_len(&self) -> usize{
        self.sts[0].shape().len()
    }

    fn simplify_ones(&mut self) -> bool{
        if self.shape_len() == 0{
            return false
        }
        let all_ones = self.full_shape().into_iter().map(|s| s==BTypes::Int(1)).collect_vec();
        self.local_dims -= all_ones[self.first_reduce() - self.local_dims as usize .. self.first_reduce()].iter().sum();
        self.upcasted -= all_ones[self.shape_len() -self.upcasted.clone() as usize ..].iter().sum();
        self.reshape_and_permute(Some(|shape: Vec<BTypes>|{
            shape.into_iter().enumerate().filter_map(|(i, x)|{
                if !all_ones[i]{
                    Some(x)
                }else{
                    None
                }
            }).collect_vec()
        }), None);

        all_ones.into_iter().any(|x| x)
    }

    fn simplify_merge_adjacent(&mut self){
        if self.shape_len() == 0{
            return 
        }
        let mut shapes = self.sts.iter().map(|x|x.shape()).collect_vec();
        let mut strides = self.sts.iter().map(|x| x.real_strides(false)).collect_vec();
        match &self.bufs[0]{
            BufferTypes::ConstBuffer(c) => {
                if let DTypes::ImageDType(d) = &c.dtype{
                    let base_shape = d.shape.clone();
                    let shape_idx_groups = get_contraction(&self.output_shape(), &base_shape.iter().map(|x|BTypes::Int(x.clone())).collect_vec());
                    let mut special_strides: Vec<BTypes> = vec![];
                    if let Some(s_i_g) =  shape_idx_groups{
                        s_i_g.into_iter().enumerate().for_each(|(i, g)|{
                            let shape_piece = g.iter().map(|x|self.output_shape()[*x]).collect_vec();
                            assert!(shape_piece.iter().fold(BTypes::Int(1), |acc, i|{
                                &acc*i
                            }) == base_shape[i]);
                            special_strides.extend(strides_for_shape(&shape_piece));
                        })
                    }
                    shapes.push(self.output_shape());
                    strides.push(special_strides.into_iter().map(|x|Some(x)).collect_vec())
                }
            },
            BufferTypes::MemBuffer(c) => {
                if let DTypes::ImageDType(d) = &c.dtype{
                    let base_shape = d.shape.clone();
                    let shape_idx_groups = get_contraction(&self.output_shape(), &base_shape.iter().map(|x|BTypes::Int(x.clone())).collect_vec());
                    let mut special_strides: Vec<BTypes> = vec![];
                    if let Some(s_i_g) =  shape_idx_groups{
                        s_i_g.into_iter().enumerate().for_each(|(i, g)|{
                            let shape_piece = g.iter().map(|x|self.output_shape()[*x]).collect_vec();
                            assert!(shape_piece.iter().fold(BTypes::Int(1), |acc, i|{
                                &acc*i
                            }) == base_shape[i]);
                            special_strides.extend(strides_for_shape(&shape_piece));
                        })
                    }
                    shapes.push(self.output_shape());
                    strides.push(special_strides.into_iter().map(|x|Some(x)).collect_vec())
                }
            },
            BufferTypes::LocalBuffer(c) => {
                if let DTypes::ImageDType(d) = &c.dtype{
                    let base_shape = d.shape.clone();
                    let shape_idx_groups = get_contraction(&self.output_shape(), &base_shape.iter().map(|x|BTypes::Int(x.clone())).collect_vec());
                    let mut special_strides: Vec<BTypes> = vec![];
                    if let Some(s_i_g) =  shape_idx_groups{
                        s_i_g.into_iter().enumerate().for_each(|(i, g)|{
                            let shape_piece = g.iter().map(|x|self.output_shape()[*x]).collect_vec();
                            assert!(shape_piece.iter().fold(BTypes::Int(1), |acc, i|{
                                &acc*i
                            }) == base_shape[i]);
                            special_strides.extend(strides_for_shape(&shape_piece));
                        })
                    }
                    shapes.push(self.output_shape());
                    strides.push(special_strides.into_iter().map(|x|Some(x)).collect_vec())
                }
            }
        }
        let mut rets = (0..shapes.len()).map(|j|vec![(shapes[j][0], strides[j][0])]).collect_vec();
        (1..shapes[0].len()).for_each(|i|{
            let mut can_merge = vec![];
            (0..shapes.len()).for_each(|j|{
                can_merge.push(
                    if strides[j][i].is_some() &&
                        ((strides[j][i].unwrap() != 0 && rets[j].last().unwrap().1 == Some(&shapes[j][i] * &strides[j][i].unwrap()))
                            || (strides[j][i].unwrap() == 0 && rets[j].last().unwrap().1 == Some(BTypes::Int(0))))
                {
                    true
                } else {
                    false
                });
            });
            let mergable = can_merge.into_iter().all(|x|x) && i != self.first_reduce();

            (0..shapes.len()).for_each(|j|{
                if mergable{
                    rets[j][rets[j].len()-1] = (&rets[j][rets[j].len()-1].0 * &shapes[j][i], strides[j][i])
                }else{
                    rets[j].push((shapes[j][i], strides[j][i]))
                }
            });
        });
        rets[..self.sts.len()].into_iter().enumerate().for_each(|(i, x)|{
            self.sts[i] = self.sts[i].reshape(&x.into_iter().map(|y|y.0).collect_vec())
        })
    }

    fn first_reduce(&self) -> usize{
        let mut result = vec![];
        for (x, y) in self.sts[0].shape()[..self.shape_len() - self.upcasted.clone() as usize].iter().take(self.shape_len()-self.upcasted.clone() as usize).zip(self.full_shape().iter().take(self.shape_len() - self.upcasted.clone() as usize)){
            result.push(x != y)
        };
        result.into_iter().position(|x| x==true).unwrap()
    }

    fn copy(&self) -> Self{
        let mut ret = self.clone();
        ret.bufs = ret.bufs.into_iter().filter_map(|x|{
            if let BufferTypes::LocalBuffer(l) = &x{
                Some(x)
            }else{
                None
            }
        }).collect_vec();
        ret.sts = ret.sts[..ret.bufs.len()].to_vec();
        ret.applied_opts_cache = None;
        ret
    }

    fn membufs(&self) -> Vec<MemBuffer>{
        self.bufs.iter().filter_map(|x|{
            if let BufferTypes::MemBuffer(m) = x{
                Some(m.clone())
            }else{
                None
            }
        }).collect_vec()
    }
    fn shape_offsets(&self, i: usize) -> Vec<BTypes>{
        if self.upcasted > 0{
            iproduct!(self.sts[i].shape()[self.shape_len() - self.upcasted.clone() as usize..].into_iter().step_by(self.sts[i].shape()[self.shape_len() - self.upcasted.clone() as usize..].len() - 1).map(|x| *x)).collect_vec()
        }else{
            vec![]
        }
    }

    fn float4_axis(&self, i:usize) -> Vec<isize>{
        self.sts[i].unit_stride_axes().iter().filter_map(|x|{
            if x >= &(self.shape_len()-self.upcasted.clone() as usize) && self.sts[i].shape()[x]%4 == 0{
                Some(x - self.shape_len() - self.upcasted.clone())
            }else{
                None
            }
        }).collect_vec()
    }

    fn upcasted_axis(&self, i: usize) -> Vec<(BTypes, Option<BTypes>, bool)> {
        let shape_len = self.shape_len();
        let sts_i_shape = self.sts[i].shape();
        let sts_i_real_strides = self.sts[i].real_strides(false);
        let full_shape = &self.full_shape();
        let upcasted = self.upcasted.clone();
    
        (0..upcasted).map(|idx| {
            let shape_idx = shape_len - upcasted as usize + idx as usize;
            let full_shape_idx = shape_len - upcasted as usize + idx as usize;
            (
                sts_i_shape[shape_idx],
                sts_i_real_strides[shape_idx],
                sts_i_shape[shape_idx] != full_shape[full_shape_idx]
            )
        }).collect()
    }

    // fn acc_offsets(&self, i: usize) -> Vec<isize>{
    //     if &self.upcasted == &0{
    //         return vec![0]
    //     }
    //     let upcased_i = self.upcasted_axis(i.clone());
    //     //potential bug here
    //     let acc_strides = strides_for_shape(&upcased_i.iter().step_by(upcased_i.len() -1).map(|(s, _, r)|{
    //         if r.clone(){
    //             BTypes::Int(1)
    //         }else{
    //             s.clone()
    //         }
    //     }).collect_vec()).into_iter().enumerate().map(|(i, x)|{
    //         x*(1-upcased_i.iter().step_by(upcased_i.len() - 1).cloned().collect_vec()[i].2)
    //     }).collect_vec();
    // }

    fn acc_offsets(&self, i: usize) -> Vec<isize> {
        if self.upcasted == 0 {
            return vec![0];
        }
        
        let upcasted_i = self.upcasted_axis(i);
        let acc_strides: Vec<i32> = upcasted_i.iter()
            .enumerate()
            .map(|(i, x)| x * (1 - upcasted_i.iter().rev().map(|x| x.2).nth(i).unwrap_or(0)))
            .collect();
        
        let shape = upcasted_i.iter().rev().map(|x| if x.2 { 1 } else { x.0 }).collect::<Vec<_>>();
        let strides_for_shape: Vec<BTypes> = strides_for_shape(shape);
        
        let product_iter = iproduct!(upcasted_i.iter().rev(), 0..);
        let offsets: Vec<isize> = product_iter
            .map(|(x, _)| {
                let inner_offsets: Vec<Vec<isize>> = upcasted_i.iter().enumerate()
                    .map(|(i, x)| (0..{
                        match &x.0{
                            BTypes::Int(i) => *i,
                            BTypes::Node(_) => panic!()
                        }
                    }).map(|y| y * acc_strides[i]).collect::<Vec<isize>>())

                    .collect_vec();
                inner_offsets
            })
            .flatten()
            .map(|x| x.iter().sum())
            .collect();
        
        offsets
    }
}

    fn full_shape(sts: &Vec<Rc<ShapeTracker>>, full_buf_index: usize) -> Vec<BTypes>{
        sts[full_buf_index].clone().shape()
    }

    fn output_shape(sts: &Vec<Rc<ShapeTracker>>) -> Vec<BTypes>{
        return sts[0].clone().shape()
    }

    fn reshape_and_permute<T>(sts: &mut Vec<Rc<ShapeTracker>>, new_shape_fxn: Option<T>, axis: Option<Vec<usize>>)
        where
            T: Fn(Vec<BTypes>) -> BTypes
    {
        let mut new_sts = vec![];
        sts.into_iter().for_each(|mut st|{
            if let Some(x) = new_shape_fxn{
                st = st.reshape(&vec![x(st.shape())]).borrow_mut();
            }
            if let Some(x) = axis{
                st = st.permute(&x.into_iter().map(|y| y as isize).collect_vec()).borrow_mut()
            }
            new_sts.push(st.clone())
        });
        *sts = new_sts;
    }

    fn shape_len(sts: &Vec<Rc<ShapeTracker>>) -> usize{
        sts[0].shape().len()
    }