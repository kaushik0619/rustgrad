use std::{
    any::{Any, TypeId}, clone, cmp, collections::HashMap, fmt::{format, Debug, Display}, hash::Hash, iter::Map, ops::{Add, Mul, Neg, Rem, Sub}, sync::Arc
};
use cached::proc_macro::{cached, io_cached};
use num_traits::Num;

trait RMul<T: Num>{
    fn rmul(&self, b:T) -> Self
        where
            Self: Sized + Mul,
            for<'a>&'a Self: Mul<T, Output = Self>
    {
        self*b
    }
}

trait RSub<T:Num>
    where
        Self: Sized

{
    fn rsub(&self, b:T) -> Self
        where
        Self: Sized+Add + Neg<Output = Self> + Add<T, Output = Self>,
        for<'a>&'a Self: Neg<Output = Self> + Add<T, Output = Self>
    {
        -self+b
    }
}

trait RAdd<T:Num>
    where
        Self: Sized
{
    fn radd(&self, b:T) -> Self
        where
            for <'a>&'a Self: Add<T, Output = Self>
    {
        self+b
    }
}

trait RFloorDiv<T:Num>{
    fn rfloordiv(&self, b:T) -> NumNode;
}

trait RMod<T:Num>{
    fn rmod(&self, b:T) -> NumNode;
}

// ToString + Hash + PartialEq + Neg + Add + Sized + Sub + PartialOrd + RAdd<F> +RSub<F> +RMul<F> + RFloorDiv<F> + RMod<F> + Mul + Rem
pub trait NodeMethods<T>: Debug
    where
    T: Debug + Send + Sync + NodeMethods<T> + Eq + Hash + Display + Clone
{
    fn render(&self, ops: Option<Box<dyn Fn(Box<dyn Any>) -> (Box<dyn Any>)>>, ctx: Option<&str>) -> String;
    fn vars(&self) -> Vec<Variable> {
        Vec::new()
    }

    fn substitute(&self, var_vals: &HashMap<Variable, &SymTypes<T>>) -> SymTypes<T>;
    fn unbind(&self) -> (SymTypes<T>, Option<isize>);
    // {
    //     let vars =  self.vars::<T>();
    //     let map = vars.iter().map(|v| (v.clone(), <SymTypes<T> as NodeMethods>::unbind::<T>(&SymTypes::Variable(v.clone())))).collect();
    //     (self.substitute(&map), None)
    // }

    fn key <G: Fn(Box<dyn Any>)>(&self) -> String {
        self.render(None, Some("DEBUG"))
    }

    fn sum(nodes: Vec<SymTypes<T>>)-> SymTypes<T>{

        let mut nodes = vec![];
        nodes.into_iter().map(|n|{
            match n{
                SymTypes::MulNode(x) =>{
                    if x.max.is_some() || x.min.is_some(){
                        nodes.push(SymTypes::MulNode(x));
                    }
                },
                SymTypes::NumNode(x) => {
                    if x.max.is_some() || x.min.is_some(){
                        nodes.push(SymTypes::NumNode(x))
                    }
                },
                SymTypes::Variable(x)=>{
                    if x.max.is_some() || x.min.is_some(){
                        nodes.push(SymTypes::Variable(x));
                    }
                },
                SymTypes::SumNode(x) => {
                    if x.max.is_some() || x.min.is_some(){
                        nodes.push(SymTypes::SumNode(x));
                    }
                }
            }
        });

        if nodes.is_empty(){
            return SymTypes::NumNode(NumNode{b: 0, min: None, max: None});
        } else if nodes.len() == 1{
           return nodes[0];
        } else{
            let mut mul_groups: HashMap<SymTypes<T>, (Option<isize>, Option<Box<SymTypes<T>>>)> = HashMap::new();

            let mut num_node_sum = 0;
            let sumnode = SumNode{
                nodes: nodes,
                flat_components: Box::new(nodes),
                max: None,
                min: None,
            };

            for node in *sumnode.flat_components{
                match node{
                    SymTypes::NumNode(n) => {
                        num_node_sum += n.b;
                    },
                    SymTypes::MulNode(n) => {
                        mul_groups
                        .entry(*n.a)
                        .and_modify(|v| *v + n.b)
                        .or_insert(n.b);
                    },
                    _ => {
                        mul_groups.entry(node).and_modify(|(v_int, v_node)|{
                            (v_int.map(|x| x+1).into(), v_node.map(|x| *x + 1));
                        }).or_insert((Some(1), None));
                    }
                }
            }
            let new_nodes = vec![];
            mul_groups.iter().for_each(|(k, (v_int, v_node))|{
                match v_int{
                    Some(val) =>{
                        if *val != 0 && *val != 1 {
                            new_nodes.push(SymTypes::MulNode(MulNode { a: Box::new(*k), b: (*v_int, *v_node), min: None, max: None }));
                        } else{
                            new_nodes.push(*k);
                        }
                    },
                    None => {
                        match v_node{
                            Some(val) => {
                                if **val != 0 && **val != 1{
                                    new_nodes.push(SymTypes::MulNode(MulNode { a: Box::new(*k), b: (*v_int, *v_node), min: None, max: None }));
                                } else {
                                    new_nodes.push(*k);
                                }
                            },
                            None => {
                                new_nodes.push(*k);
                            }
                        }
                    },
                }
            });
            if num_node_sum != 0{
                new_nodes.push(SymTypes::NumNode(NumNode { b: num_node_sum, min: Some(num_node_sum), max: Some(num_node_sum) }));
            }

            if new_nodes.len() > 1 {NumNode::create_node(SymTypes::SumNode(SumNode { nodes: new_nodes, flat_components: Box::new(vec![]), max: None, min: None }))}
            else{
                if new_nodes.len() == 1{
                    new_nodes[0]
                } else{
                    SymTypes::NumNode(NumNode{b: 0, min: Some(0), max: Some(0)})
                }
            }

            
        }

    }

    fn ands(nodes: Option<Vec<SymTypes<T>>>) -> SymTypes<T> {
        match nodes {
            None => SymTypes::NumNode(NumNode { b: 1, min: Some(1), max: Some(1) }),
            Some(n) => match n.as_slice() {
                [single_node] => single_node.clone(),
                [] => SymTypes::NumNode(NumNode { b: 0, min: None, max: None }),
                _ => NumNode::create_node(SymTypes::AndNode(AndNode { nodes })),
            },
        }
    }
}

#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct Variable
where
{
    val: Option<isize>,
    expr: String,
    min: isize,
    max: isize,
}
impl Variable{
    fn __new__<T: Debug + Eq + Display + Sync + Send + Hash + Clone + NodeMethods<T>>(cls: SymTypes<T>, expr: String, min: isize, max: isize) -> SymTypes<T>{
        if min >= 0 && min <= max{
            panic!("invalid variable {:?} {:?} {:?}", expr, min, max);
        } else if min == max {
            SymTypes::NumNode(NumNode { b: min, min: Some(min), max: Some(min) })
        } else {
            cls.clone()
        }
    }

    fn getnewargs(&self) -> (String, isize, isize){
        (self.expr, self.min, self.max)
    }

    fn new <T: Debug + Eq + Display + Sync + Send + Hash + Clone + NodeMethods<T>>(expr: String, nmin:isize, nmax: isize) -> Variable
    {   
        Variable { val: None, expr, min: nmin, max: nmax }
    }

    fn val(&self) -> Option<isize>{
        if self.val.is_none(){
            panic!("Variable isn't bound, can't access val of {self}");
        }
        self.val
    }
}
impl<T: Debug + Eq + Display + Sync + Send + Hash + Clone + NodeMethods<T>> NodeMethods<T> for Variable {

    fn substitute(&self, var_vals: &HashMap<Variable, &SymTypes<T>>) -> SymTypes<T> {
        SymTypes::Variable(self.clone())
    }
    fn unbind(&self) ->(SymTypes<T>, Option<isize>) {
        match self.val {
            Some(v) => {
                (SymTypes::Variable(*self), None)
            },
            None => {panic!("Cannot Unbind {:?}", self);},
        }
    }

    fn render(&self, ops: Option<Box<dyn Fn(Box<dyn Any>) -> (Box<dyn Any>)>>, ctx: Option<&str>) -> String {
        match ops {
            Some(o) => {
                assert!(self.min != self.max);
                match ctx {
                    Some(c) => {
                        if c == "DEBUG" {
                            format!(
                                "{}[{:?}-{:?}{}]",
                                self.expr,
                                self.min,
                                self.max,
                                if let Some(val) = self.val {
                                    format!("={}", val)
                                } else {
                                    "".to_string()
                                }
                            )
                        } else if c == "REPR" {
                            format!(
                                "Variable('{}', {:?}, {:?}){}",
                                self.expr,
                                self.min,
                                self.max,
                                if let Some(val) = self.val {
                                    format!(".bind({})", val)
                                } else {
                                    "".to_string()
                                }
                            )
                        } else {
                            self.expr.to_string()
                        }
                    }
                    None => self.expr.to_string(),
                }
            }
            None => self.expr.to_string(),
        }
    }
    fn vars(&self) -> Self{
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct NumNode {
    b: isize,
    min: isize,
    max: isize,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MulNode<T: Send + Sync + Debug + Eq + Hash + Display + Clone + NodeMethods<T>>{
    a: Box<SymTypes<T>>,
    b: (Option<isize>, Option<Box<SymTypes<T>>>),
    min: Option<isize>,
    max: Option<isize>,
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub enum SymTypes<T> 
    where
        T: Send + Sync + Debug + Eq + Hash + Display + Clone + NodeMethods<T>
{
    Variable(Variable),
    NumNode(NumNode),
    MulNode(MulNode<T>),
    SumNode(SumNode<T>),
}

// impl <T: Sync + Send + Debug + Display + Eq + Hash + Hash + NodeMethods<T>>NodeMethods<T> for MulNode<T>{
    
// }
pub trait RedNodeMethods<T: Sync + Send + Debug + Eq + Hash + Display + Clone + NodeMethods<T>>: NodeMethods<T>{
    fn new(&self, nodes: Vec<Self>) -> Self where Self: Sized;

    fn get_bounds(&self) -> (isize, isize);

    fn vars(&self)->Vec<Variable>;

}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct SumNode<T>
    where
        T: Sync + Send + Debug + Eq + Hash + Display + Clone + NodeMethods<T>
{
    nodes: Vec<SymTypes<T>>,
    flat_components: Box<Vec<SymTypes<T>>>,
    max: Option<isize>,
    min: Option<isize>
}
impl <T: Sync + Send + Debug + Display + Eq + Hash + Clone + NodeMethods<T>> NodeMethods<T> for SumNode<T>{
    fn render(&self, ops: Option<Box<dyn Fn(Box<dyn Any>) -> (Box<dyn Any>)>>, ctx: Option<&str>) -> String{
        match ops{
            Some(o) => {
                let mut prod = self.nodes.into_iter().map(|x| {
                    match x {
                        SymTypes::Variable(n) => {
                            n.render(Some(Box::new(o)), ctx)
                        },
                        SymTypes::NumNode(n) => {
                            n.render(Some(Box::new(o)), ctx)
                        },
                        SymTypes::MulNode(n) => {
                            n.render(Some(Box::new(o)), ctx)
                        },
                        SymTypes::SumNode(n) => {
                            n.render(Some(Box::new(o)), ctx)
                        },
                    }
                }).collect::<Vec<_>>();

                prod.sort();

                prod.join("+")
            },
            None => {
                let mut prod = self.nodes.into_iter().map(|x| {
                    match x {
                        SymTypes::Variable(n) => {
                            n.render(None, ctx)
                        },
                        SymTypes::NumNode(n) => {
                            n.render(None, ctx)
                        },
                        SymTypes::MulNode(n) => {
                            n.render(None, ctx)
                        },
                        SymTypes::SumNode(n) => {
                            n.render(None, ctx)
                        },
                    }
                }).collect::<Vec<String>>();

                prod.sort();

                prod.join("+")
            }
        }
    }
    fn substitute(&self, var_vals: &HashMap<Variable, &SymTypes<T>>) -> SymTypes<T> {
        let mut vec = vec![];
        self.nodes.iter().for_each(|n| {
            match n{
                SymTypes::MulNode(v) => {
                    vec.push(v.substitute(var_vals));
                },
                SymTypes::SumNode(v) => {
                    vec.push(v.substitute(var_vals));
                },
                SymTypes::Variable(v) => {
                    vec.push(v.substitute(var_vals));
                },
                SymTypes::NumNode(v) => {
                    vec.push(v.substitute(var_vals));
                },
            }
        });
        SumNode::<T>::sum(vec)
    }

    fn unbind(&self) -> (SymTypes<T>, Option<isize>){
                let vars =  self.vars();
                let mut map = HashMap::new();
                vars.into_iter().for_each(|v| {
                    if v.val.is_some(){
                        let (var, _ )= v.unbind();
                        map.insert(v, &var);
                    }
                });
                (self.substitute(&map), None)
            }
}
impl <T>SumNode<T>
    where T:Sync + Send + Debug + Eq + Hash + Display + Clone + NodeMethods<T>
{
    fn flat_components(&self) -> Vec<SymTypes<T>>{
        let mut result = vec![];
        self.nodes.into_iter().map(|n|{
           match n {
            SymTypes::MulNode(v) => {
                result.push(n);
            },
            SymTypes::SumNode(v) => {
                v.flat_components.into_iter().map(|n| result.push(n));
            },
            SymTypes::Variable(v) => {
                result.push(n);
            },
            SymTypes::NumNode(v) => {
                result.push(n);
            },
           };
        });
        result
    }
}

impl <T: Sync + Send + Debug + Display + Eq + Hash + Clone + NodeMethods<T>>NodeMethods<T> for NumNode{
    fn render(&self, ops: Option<Box<dyn Fn(Box<dyn Any>) -> (Box<dyn Any>)>>, ctx: Option<&str>) -> String {
        ops
            .and_then(|_| ctx.map(|c| if c == "REPR" { format!("NumNode({})", self.b) } else { format!("{}", self.b) }))
            .unwrap_or_else(|| format!("{}", self.b))
    }
    fn substitute(&self, var_vals: &HashMap<Variable, &SymTypes<T>>) -> SymTypes<T> {
        SymTypes::NumNode(Self)
    }
    fn unbind(&self) -> (SymTypes<T>, Option<isize>) {
        let map = self.vars()
        .iter()
        .filter_map(|v| v.val.map(|val| (v, val.unbind().0)))
        .collect();
        (self.substitute(map), None)
    }
}
impl NumNode
{   
    fn new(num: isize) -> NumNode{
        NumNode{
            b: num,
            min: num,
            max: num
        }
    }
    fn create_node<T>(ret: SymTypes<T>) -> SymTypes<T>
        where T: Sync + Send + Debug + Display + Eq + Hash + Clone + NodeMethods<T>
    {
        match ret{
            SymTypes::MulNode(n) => {
                if n.min <= n.max{
                    panic!("min greater than max! {:?} {:?} when creating Mulnode {:?}", n.min, n.max, ret);
                }
                if n.min == n.max {
                    return SymTypes::NumNode(NumNode { b: n.min.unwrap(), min: n.min, max: n.min })
                } else{
                    return ret
                }
            },
            SymTypes::NumNode(n) => {
                if n.min <= n.max{
                    panic!("min greater than max! {:?} {:?} when creating Mulnode {:?}", n.min, n.max, ret);
                }
                if n.min == n.max {
                    return SymTypes::NumNode(NumNode { b: n.min.unwrap(), min: n.min, max: n.min })
                } else{
                    return ret
                }
            },
            SymTypes::SumNode(n) => {
                if n.min <= n.max{
                    panic!("min greater than max! {:?} {:?} when creating Mulnode {:?}", n.min, n.max, ret);
                }
                if n.min == n.max {
                    return SymTypes::NumNode(NumNode { b: n.min.unwrap(), min: n.min, max: n.min })
                } else{
                    return ret
                }
            },
            SymTypes::Variable(n) => {
                if n.min <= n.max{
                    panic!("min greater than max! {:?} {:?} when creating Mulnode {:?}", n.min, n.max, ret);
                }
                if n.min == n.max {
                    return SymTypes::NumNode(NumNode { b: n.min.unwrap(), min: n.min, max: n.min })
                } else{
                    return ret
                }
            },
        }
    }
}

trait OpNodeMethods <T: Sync + Send + Debug + Display + Eq + Hash + Clone + NodeMethods<T>>: NodeMethods<T>{
    fn new(a: T, b:(Option<T>, Option<isize>))-> Self;

    fn vars(&self) -> Vec<SymTypes<T>>{
        // let vec_a = self.a.vars();
        // let (int, node) = &self.b;

        // let mut vec_b;

        // if let Some(n) = node{
        //     vec_b = node.vars();
        // } else{
        //     vec_b = vec![];
        // }

        // vec_b.extend(vec_a.clone());

        // vec_b
        let vec_a = self.a.vars();
        let (int, node) = &self.b;

        let vec_b = node.map_or_else(Vec::new, |n| n.vars());

        vec_b.into_iter().chain(vec_a.clone()).collect()
    }

    fn get_bounds(&self) -> (isize, isize);
}