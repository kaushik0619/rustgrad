use std::{
    any::{Any, TypeId}, cmp, collections::HashMap, fmt::{format, Debug, Display}, hash::Hash, ops::{Add, Mul, Neg, Rem, Sub}, sync::Arc
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
trait NodeMethods<F: Num>: Debug{
    fn render<T: Send + Sync + Debug>(&self, ops: Option<HashMap<SymTypes<T>, &dyn Any>>, ctx: Option<&str>) -> String;
    fn vars<T: Sync + Send + Debug>(&self) -> Vec<Variable<T>> {
        Vec::new()
    }

    fn substitute<T: Sync + Send + Debug>(&self, var_vals: &HashMap<Variable<T>, (&Variable<T>, Option<isize>)>) -> &Self {
        self
    }

    fn unbind<T>(&self) -> (&Self, Option<isize>)
    where
        T: std::marker::Sync + std::marker::Send + Debug + Eq + Display + Num,
        Self: std::fmt::Debug + Sized,
        Variable<T>: std::hash::Hash + std::fmt::Debug + Clone + Display,
    {
        let vars =  self.vars::<T>();
        let map = vars.iter().map(|v| (v.clone(), <Variable<T> as NodeMethods<F>>::unbind::<T>(v))).collect();
        (self.substitute(&map), None)
    }

    fn key<T: Send + Sync + Debug>(&self) -> String {
        self.render::<T>(None, Some("DEBUG"))
    }

    fn sum<T: Send + Sync + Debug>(nodes: Vec<SymTypes<T>>)-> SymTypes<T>{

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
                }
            }
        });

        if nodes.is_empty(){
            return SymTypes::NumNode(NumNode{b: 0, min: None, max: None});
        } else if nodes.len() == 1{
           return nodes[0];
        } else{
            let mut mul_groups: HashMap<SymTypes<T>, isize> = HashMap::new();

            let mut num_node_sum = 0;

            for node in 
        }

    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct Variable<T>
where
    T: Sync + Send + std::fmt::Debug,
{
    val: Option<isize>,
    expr: T,
    min: Option<isize>,
    max: Option<isize>,
}

impl<T: Debug + Display + Sync + Send, F: Num> NodeMethods::<F> for Variable<T> {
    fn unbind<G>(&self) -> (&Variable<T>, Option<isize>) {
        match self.val {
            Some(v) => {
                (self.to_owned(), None)
            },
            None => {panic!("Cannot Unbind {:?}", self);},
        }
    }

    fn render<H: Send + Sync + Debug>(&self, ops: Option<HashMap<SymTypes<H>, &dyn Any>>, ctx: Option<&str>) -> String {
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
}

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct NumNode {
    b: isize,
    min: Option<isize>,
    max: Option<isize>,
}

#[derive(Eq, PartialEq, Hash)]
pub struct MulNode {
    b: isize,
    min: Option<isize>,
    max: Option<isize>,
}

#[derive(Eq, PartialEq, Hash)]
pub enum SymTypes<T> 
    where
        T: Send + Sync + Debug
{
    Variable(Variable<T>),
    NumNode(NumNode),
    MulNode(MulNode)
}