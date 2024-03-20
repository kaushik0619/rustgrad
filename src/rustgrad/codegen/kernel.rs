use std::{any::Any, collections::HashMap, fmt::{Debug, Display}, rc::Rc, sync::Arc};

use lazy_static::lazy_static;

use crate::rustgrad::{device::Device, dtype::{self, DTypes, DTYPES_DICT}, ops::{BufferTypes, FlopCounter, Items, LazyOp}, shape::{shapetracker::ShapeTracker, sym::NodeTypes}};

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

#[derive(PartialEq, PartialOrd)]
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

struct Kernel{
    opts: Option<LinearizerOptions>,
    ast: Vec<Rc<LazyOp>>,
    lazyops: Vec<Rc<LazyOp>>,
    info: FlopCounter,
    reduceop: Rc<LazyOp>,
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
        Kernel{
            opts: {
                opts.or({
                    let device: Compiled = Device[Device::DEFAULT()];
                    device.compiler.li
                }).and({
                })
            }
        }
    }
}