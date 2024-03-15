use core::panic;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::{Mul, Rem, Sub};
use std::rc::Weak;
use std::{any::Any, collections::HashMap, fmt::Debug, ops::Deref, rc::Rc};

use num::integer::gcd;
use num::{Num, ToPrimitive};

use crate::rustgrad::helpers::partition;
type N = Rc<NodeTypes>;
pub enum Context {
    DEBUG,
    REPR,
}
#[derive(Debug, Clone)]
pub struct Variable {
    expr: String,
    min: Option<f64>,
    max: Option<BTypes>,
    pub val: Option<f64>,
    ptr: RefCell<Option<Weak<NodeTypes>>>,
}
#[derive(Clone)]
pub enum BTypes {
    Node(N),
    Int(f64),
}
#[derive(Debug, Clone)]
pub struct OpNode {
    a: N,
    b: BTypes,
    min: Option<f64>,
    max: Option<BTypes>,
    ptr: RefCell<Option<Weak<NodeTypes>>>,
}
#[derive(Debug, Clone)]
pub struct NumNode {
    b: f64,
    min: Option<f64>,
    max: Option<BTypes>,
    ptr: RefCell<Option<Weak<NodeTypes>>>,
}
#[derive(Debug, Clone)]
pub struct RedNode {
    nodes: Vec<N>,
    min: Option<f64>,
    max: Option<BTypes>,
    ptr: RefCell<Option<Weak<NodeTypes>>>,
}
#[derive(Debug, Clone)]
pub enum NodeTypes {
    Variable(Variable),
    NumNode(NumNode),
    LtNode(OpNode),
    MulNode(OpNode),
    DivNode(OpNode),
    ModNode(OpNode),
    SumNode(RedNode),
    AndNode(RedNode),
}
pub trait NodeMethods {
    fn render(&self, ops: &Option<Box<dyn Any>>, ctx: &Option<Context>) -> String;

    fn vars(&self) -> Vec<N>;

    fn substitute(&self, var_vals: &HashMap<N, N>) -> N;

    fn unbind(&self) -> (N, Option<f64>);

    //to be cached
    fn key(&self) -> String;

    //to be cached
    fn hash(&self) -> u64;

    fn sum(nodes: &Vec<N>) -> N;

    fn ands(nodes: &Vec<N>) -> N;

    fn min(&self) -> Option<f64>;
    fn max(&self) -> Option<BTypes>;

    fn flat_components(&self) -> Vec<N>;

    fn ptr(&self) -> N;

    fn get_bounds(&self) -> (Option<f64>, Option<BTypes>);
}

impl NodeMethods for NodeTypes {
    fn render(&self, ops: &Option<Box<dyn Any>>, ctx: &Option<Context>) -> String {
        match self {
            NodeTypes::Variable(v) => ops
                .as_ref()
                .and_then(|_| {
                    ctx.as_ref().map(|c| {
                        if let Context::DEBUG = c {
                            format!(
                                "{}[{:?}-{:?}{}]",
                                &v.expr,
                                self.min().unwrap(),
                                self.max().unwrap(),
                                v.val
                                    .as_ref()
                                    .map_or_else(|| String::new(), |val| format!("={}", val))
                            )
                        } else if let Context::REPR = c {
                            format!(
                                "Variable('{}', {:?}, {:?}){}",
                                &v.expr,
                                self.min().unwrap(),
                                self.max().unwrap(),
                                v.val
                                    .as_ref()
                                    .map_or_else(|| String::new(), |val| format!(".bind({})", val))
                            )
                        } else {
                            v.expr.clone()
                        }
                    })
                })
                .unwrap_or_else(|| v.expr.clone()),
            NodeTypes::NumNode(n) => ops
                .as_ref()
                .and_then(|_| {
                    ctx.as_ref().map(|c| {
                        if let Context::REPR = c {
                            format!("NumNode({})", &n.b)
                        } else {
                            format!("{}", &n.b)
                        }
                    })
                })
                .unwrap_or_else(|| format!("{}", n.b)),

            NodeTypes::LtNode(l) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                format!("{} / {:?}", l.a.clone().render(ops, ctx), l.b.clone())
            }

            NodeTypes::MulNode(m) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                render_mulnode(m, ops, ctx)
            }

            NodeTypes::ModNode(md) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                format!("{} % {:?}", md.a.clone().render(ops, ctx), &md.b)
            }

            NodeTypes::SumNode(s) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                let mut vec: Vec<String> =
                    s.nodes.iter().map(|x| x.clone().render(ops, ctx)).collect();
                vec.sort();
                format!("{}", vec.join("+"))
            }
            NodeTypes::AndNode(a) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                let mut vec: Vec<String> =
                    a.nodes.iter().map(|x| x.clone().render(ops, ctx)).collect();
                vec.sort();
                format!("{}", vec.join(" and "))
            }
            NodeTypes::DivNode(d) => {
                assert!(self.min().unwrap() != self.max().unwrap());
                format!("{} // {:?}", d.a.render(ops, ctx), &d.b)
            }
        }
    }
    fn vars(&self) -> Vec<N> {
        match self {
            NodeTypes::Variable(_) => vec![self.ptr()],
            NodeTypes::LtNode(n)
            | NodeTypes::DivNode(n)
            | NodeTypes::ModNode(n)
            | NodeTypes::MulNode(n) => {
                let mut result = n.a.clone().vars();
                if let BTypes::Node(b_n) = &n.b {
                    result.extend(b_n.clone().vars());
                }
                result
            }
            NodeTypes::SumNode(n) | NodeTypes::AndNode(n) => {
                n.nodes.iter().flat_map(|x| x.clone().vars()).collect()
            }
            _ => vec![],
        }
    }
    fn substitute(&self, var_vals: &HashMap<N, N>) -> N {
        match self {
            NodeTypes::Variable(_) => var_vals
                .get(&self.ptr())
                .cloned()
                .unwrap_or_else(|| self.ptr()),
            NodeTypes::NumNode(_) => self.ptr(),
            NodeTypes::LtNode(l) => match &l.b {
                BTypes::Int(i) => l.a.clone().substitute(var_vals).n2i_lt(i),
                BTypes::Node(n) => {
                    l.a.clone()
                        .substitute(var_vals)
                        .n2n_lt(n.clone().substitute(var_vals).deref())
                }
            },
            NodeTypes::MulNode(m) => match &m.b {
                BTypes::Int(i) => m.a.clone().substitute(var_vals).clone().deref() * i,
                BTypes::Node(n) => {
                    m.a.clone().substitute(var_vals).clone().deref()
                        * n.clone().substitute(var_vals).clone().deref()
                }
            },
            NodeTypes::DivNode(d) => d.a.clone().substitute(var_vals).floordiv(&d.b, true),
            NodeTypes::ModNode(m) => match &m.b {
                BTypes::Int(i) => m.a.clone().substitute(var_vals).deref() % i,
                BTypes::Node(n) => m.a.clone().substitute(var_vals).deref() % n.clone().deref(),
            },
            NodeTypes::SumNode(s) => Self::sum(
                &s.nodes
                    .iter()
                    .map(|n| n.clone().substitute(var_vals))
                    .collect(),
            ),
            NodeTypes::AndNode(a) => Self::ands(
                &a.nodes
                    .iter()
                    .map(|n| n.clone().substitute(var_vals))
                    .collect(),
            ),
        }
    }
    fn unbind(&self) -> (N, Option<f64>) {
        match self {
            NodeTypes::Variable(v) => {
                assert!(v.val.is_some());
                (self.ptr(), v.val.clone())
            }
            _ => {
                let mut map = HashMap::new();
                self.vars().iter().for_each(|v| match v.clone().deref() {
                    NodeTypes::Variable(var) => {
                        if var.val.is_some() {
                            map.insert(v.clone().ptr(), v.unbind().0).unwrap();
                        }
                    }
                    _ => {}
                });
                (self.substitute(&map), None)
            }
        }
    }

    fn key(&self) -> String {
        self.render(&None, &Some(Context::DEBUG))
    }

    fn hash(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.key().hash(&mut s);
        s.finish()
    }

    fn sum(nodes: &Vec<N>) -> N {
        let mut nd = vec![];
        nodes.into_iter().for_each(|n| {
            if !n.clone().max().is_none() || !n.clone().min().is_none() {
                nd.push(n.clone());
            }
        });

        if nd.is_empty() {
            return NumNode::init(&0.0);
        }
        if nd.len() == 1 {
            return nd[0].clone();
        }

        let mut mul_groups: HashMap<N, BTypes> = HashMap::new();
        let mut num_node_sum = 0.0;

        for nodes in &NodeTypes::new_sum(&nd).flat_components() {
            match nodes.clone().deref() {
                NodeTypes::NumNode(n) => {
                    num_node_sum += n.b;
                }
                NodeTypes::MulNode(n) => match &n.b {
                    BTypes::Node(b_n) => {
                        let getter: &BTypes =
                            mul_groups.get(&n.a).unwrap_or_else(|| &BTypes::Int(0.0));

                        match getter {
                            BTypes::Int(i) => {
                                mul_groups
                                    .insert(n.a.clone(), BTypes::Node(b_n.clone().deref() + i));
                            }
                            BTypes::Node(bb_n) => {
                                mul_groups.insert(
                                    n.a.clone(),
                                    BTypes::Node(bb_n.clone().deref() + b_n.clone().deref()),
                                );
                            }
                        }
                    }
                    BTypes::Int(i) => {
                        let getter: &BTypes =
                            mul_groups.get(&n.a).unwrap_or_else(|| &BTypes::Int(0.0));
                        match getter {
                            BTypes::Int(ii) => {
                                mul_groups.insert(n.a.clone(), BTypes::Int(ii + i));
                            }
                            BTypes::Node(bb_n) => {
                                mul_groups
                                    .insert(n.a.clone(), BTypes::Node(bb_n.clone().deref() + i));
                            }
                        }
                    }
                },

                _ => {
                    let getter: &BTypes =
                        mul_groups.get(nodes).unwrap_or_else(|| &BTypes::Int(0.0));

                    match getter {
                        BTypes::Int(i) => {
                            mul_groups.insert(nodes.clone(), BTypes::Int(i + 1.0));
                        }
                        BTypes::Node(n) => {
                            mul_groups
                                .insert(nodes.clone(), BTypes::Node(n.clone().deref() + &1.0));
                        }
                    }
                }
            }
        }
        let mut new_nodes: Vec<N> = vec![];
        mul_groups.into_iter().for_each(|(a, b_sum)| match &b_sum {
            BTypes::Int(i) => {
                if i != &0.0 {
                    if i != &1.0 {
                        new_nodes.push(NodeTypes::new_mul(a, b_sum));
                    } else {
                        new_nodes.push(a);
                    }
                }
            }
            BTypes::Node(n) => {
                if n.clone().deref() != &0.0 {
                    if n.clone().deref() != &1.0 {
                        new_nodes.push(NodeTypes::new_mul(a, b_sum));
                    } else {
                        new_nodes.push(a);
                    }
                }
            }
        });

        if !num_node_sum.is_nan() {
            new_nodes.push(NumNode::init(&num_node_sum).ptr());
        }

        if new_nodes.len() > 1 {
            create_node(NodeTypes::new_sum(&new_nodes))
        } else if new_nodes.len() == 1 {
            create_node(new_nodes[0].clone())
        } else {
            create_node(NumNode::init(&0.0))
        }
    }

    fn ptr(&self) -> N {
        match self {
            NodeTypes::DivNode(n)
            | NodeTypes::LtNode(n)
            | NodeTypes::ModNode(n)
            | NodeTypes::MulNode(n) => n.ptr.borrow().as_ref().unwrap().upgrade().unwrap(),
            NodeTypes::AndNode(n) | NodeTypes::SumNode(n) => {
                n.ptr.borrow().as_ref().unwrap().upgrade().unwrap()
            }
            NodeTypes::Variable(n) => n.ptr.borrow().as_ref().unwrap().upgrade().unwrap(),
            NodeTypes::NumNode(n) => n.ptr.borrow().as_ref().unwrap().upgrade().unwrap(),
        }
    }

    fn min(&self) -> Option<f64> {
        match self {
            NodeTypes::DivNode(n)
            | NodeTypes::LtNode(n)
            | NodeTypes::ModNode(n)
            | NodeTypes::MulNode(n) => n.min.clone(),
            NodeTypes::AndNode(n) | NodeTypes::SumNode(n) => n.min.clone(),
            NodeTypes::Variable(n) => n.min.clone(),
            NodeTypes::NumNode(n) => n.min.clone(),
        }
    }

    fn max(&self) -> Option<BTypes> {
        match self {
            NodeTypes::DivNode(n)
            | NodeTypes::LtNode(n)
            | NodeTypes::ModNode(n)
            | NodeTypes::MulNode(n) => Some(n.max.clone()?.sint_infer().clone()),
            NodeTypes::AndNode(n) | NodeTypes::SumNode(n) => {
                Some(n.max.clone()?.sint_infer().clone())
            }
            NodeTypes::Variable(n) => Some(n.max.clone()?.sint_infer().clone()),
            NodeTypes::NumNode(n) => Some(n.max.clone()?.sint_infer().clone()),
        }
    }

    fn flat_components(&self) -> Vec<N> {
        match self {
            NodeTypes::SumNode(s) => {
                let mut result = vec![];
                s.nodes.iter().for_each(|x| match x.clone().deref() {
                    NodeTypes::SumNode(_) => result.extend(x.clone().flat_components()),
                    _ => result.push(x.clone()),
                });
                result
            }
            _ => {
                panic!("Not for {:?}", self)
            }
        }
    }

    fn ands(nodes: &Vec<N>) -> N {
        if nodes.is_empty() {
            return NumNode::init(&0.0);
        }
        if nodes.len() == 0 {
            return nodes[0].clone();
        }
        let mut tmp = vec![];
        nodes.iter().for_each(|n| {
            if n.min().unwrap() != n.max().unwrap() {
                tmp.push(n.clone())
            }
        });

        if tmp.len() > 1 {
            return create_node(NodeTypes::new_and(&tmp));
        } else if tmp.len() == 1 {
            return create_node(tmp[0].clone());
        } else {
            create_node(NumNode::init(&1.0))
        }
    }

    fn get_bounds(&self) -> (Option<f64>, Option<BTypes>) {
        match self {
            NodeTypes::LtNode(l) => {
                if &l.b == l.a.clone().deref() {
                    return (Some(0.0), Some(BTypes::Int(0.0)));
                }
                match &l.b {
                    BTypes::Int(i) => {
                        if &l.a.clone().max().unwrap() < i {
                            return (Some(1.0), Some(BTypes::Int(1.0)));
                        } else if i <= &l.a.clone().min().unwrap() {
                            return (Some(0.0), Some(BTypes::Int(0.0)));
                        } else {
                            return (Some(0.0), Some(BTypes::Int(1.0)));
                        }
                    }
                    BTypes::Node(n) => {
                        if l.a.clone().max().unwrap() < n.clone().min().unwrap() {
                            return (Some(1.0), Some(BTypes::Int(1.0)));
                        } else if n.clone().max().unwrap() <= l.a.clone().min().unwrap() {
                            return (Some(0.0), Some(BTypes::Int(0.0)));
                        } else {
                            return (Some(0.0), Some(BTypes::Int(0.0)));
                        }
                    }
                }
            }
            NodeTypes::MulNode(m) => {
                assert!(m.a.clone().min().unwrap() >= 0.0);
                // match m.a.clone().max().unwrap(){
                //     BTypes::Int(i) => {
                //         match &m.b{
                //             BTypes::Int(ii) => {
                //                 if ii>= &0.0{
                //                     return (Some(m.a.clone().min().unwrap() * ii), Some(BTypes::Int(i*ii)))
                //                 } else {
                //                     return (Some(i * ii), Some(BTypes::Int(m.a.clone().min().unwrap() * ii)))
                //                 }
                //             }
                //             BTypes::Node(ii) => {
                //                 if ii.clone().n2i_ge(&0.0).deref().into(){
                //                     return (Some(m.a.clone().min().unwrap() * ii.min().unwrap()), Some(BTypes::Node(ii.clone().deref() * &i)))
                //                 } else {
                //                     return (Some(i * ii.clone().min().unwrap()), Some(BTypes::Node(ii.clone().deref() * &m.a.min().unwrap())))
                //                 }
                //             }
                //         }
                //     }
                //     BTypes::Node(i) => {
                //         match &m.b{
                //             BTypes::Int(ii) => {
                //                 if ii>= &0.0{
                //                     return (Some(m.a.clone().min().unwrap() * ii), Some(BTypes::Int(i*ii)))
                //                 } else {
                //                     return (Some(i * ii), Some(BTypes::Int(m.a.clone().min().unwrap() * ii)))
                //                 }
                //             }
                //         }
                //     }
                // }

                match (m.a.clone().max().unwrap(), &m.b){
                    (BTypes::Int(i), BTypes::Int(ii)) => {
                        if ii >= &0.0{
                            return (Some(&m.a.clone().min().unwrap() * ii), Some(BTypes::Int(i * ii)))
                        } else{
                            return (Some(i * ii), Some(BTypes::Int(m.a.clone().min().unwrap() * ii)))
                        }
                    }
                    (BTypes::Node(i), BTypes::Int(ii)) => {
                        if i.clone().min().unwrap() >= 0.0{
                            return (Some(&m.a.clone().min().unwrap() * ii), Some(BTypes::Node(i.clone().deref() * ii)))
                        } else{
                            // return (Some(i.clone().deref() * ii), Some(BTypes::Int(m.a.clone().min().unwrap() * ii)))
                            panic!("Sint has int max values and a tuple of int, sint has to be infered");
                        }
                    }
                    (BTypes::Int(i), BTypes::Node(ii)) => {
                        if ii.clone().min().unwrap() >= 0.0{
                            match ii.clone().max().unwrap(){
                                BTypes::Int(iii) => {
                                    return (Some(m.a.clone().min().unwrap() * ii.clone().min().unwrap()), Some(BTypes::Int(i * iii)))
                                }
                                BTypes::Node(n) => {
                                    return (Some(m.a.clone().min().unwrap() * ii.clone().min().unwrap()), Some(BTypes::Node(n.clone().deref()*&i)))
                                }
                            }
                        }
                        else{
                            match ii.clone().max().unwrap(){
                                BTypes::Int(iii) => {
                                    return (Some(i * ii.clone().min().unwrap()), Some(BTypes::Int(m.a.clone().min().unwrap() * iii)))
                                }
                                BTypes::Node(n) => {
                                    return (Some(i * ii.clone().min().unwrap()), Some(BTypes::Node(n.clone().deref() * &m.a.clone().min().unwrap())))
                                }
                            }
                        }
                    }
                    (BTypes::Node(i), BTypes::Node(ii)) => {
                        if ii.clone().min().unwrap() >= 0.0{
                            match ii.clone().max().unwrap(){
                                BTypes::Int(iii) => {
                                    return (Some(m.a.clone().min().unwrap() * ii.clone().min().unwrap()), Some(BTypes::Node(i.clone().deref() * &iii)))
                                }
                                BTypes::Node(n) => {
                                    return (Some(m.a.clone().min().unwrap() * ii.clone().min().unwrap()), Some(BTypes::Node(n.clone().deref()*i.clone().deref())))
                                }
                            }
                        }
                        else{
                            // match ii.clone().max().unwrap(){
                            //     BTypes::Int(iii) => {
                            //         return (Some(i.clone().deref() * &ii.clone().min().unwrap()), Some(BTypes::Int(m.a.clone().min().unwrap() * iii)))
                            //     }
                            //     BTypes::Node(n) => {
                            //         return (Some(i.clone().deref() * &ii.clone().min().unwrap()), Some(BTypes::Node(n.clone().deref() * &m.a.clone().min().unwrap())))
                            //     }
                            // }

                            panic!("Sint has int max values and a tuple of int, sint has to be infered")
                        }
                    }
                }
            }
            NodeTypes::DivNode(d) => {
                if let BTypes::Int(i) = &d.b{
                    assert!(d.a.clone().min().unwrap() >= 0.0);
                    match d.a.clone().max().unwrap(){
                        BTypes::Int(ii) => {
                            return (Some((d.a.clone().min().unwrap() / i).floor()), Some(BTypes::Int((ii/i).floor())))
                        }
                        BTypes::Node(n) => {
                            return (Some((d.a.clone().min().unwrap() / i).floor()), Some(BTypes::Node(n.clone().floordiv(&d.b, true))))
                        }
                    }
                }
                panic!()
            }
            NodeTypes::ModNode(m) => {
                if let BTypes::Int(i) = &m.b{
                    assert!(m.a.clone().min().unwrap() >= 0.0);
                    match &m.a.clone().max().unwrap(){
                        BTypes::Int(ii) => {
                            if (ii - &m.a.clone().min().unwrap() >= i.clone()) || (&m.a.clone().min().unwrap() != ii && m.a.clone().min().unwrap() % i >= ii % i){
                                return (Some(0.0), Some(BTypes::Int(i - 1.0)))
                            }
                            return (Some(m.a.clone().min().unwrap() % i), Some(BTypes::Int(ii % i)))
                        }
                        BTypes::Node(ii) => {
                            if ((ii.clone().deref() - &m.a.clone().min().unwrap()).n2i_ge(i).deref().into()) || (ii.clone().deref() == &m.a.clone().min().unwrap() && (&m.a.clone().min().unwrap() % i.i2n_ge((ii.clone().deref() % i).deref()).deref()).deref().into()){
                                return (Some(0.0), Some(BTypes::Int(i - 1.0)))
                            }
                            return (Some(m.a.clone().min().unwrap() % i), Some(BTypes::Node(ii.clone().deref() % i)))
                        }
                    }
                }
                panic!()
            }

            NodeTypes::SumNode(s) => {
                let mut int_sum = 0.0;
                let mut node_sum: Option<Rc<NodeTypes>> = None;
            
                for x in &s.nodes {
                    match x.max().unwrap() {
                        BTypes::Int(i) => int_sum += i,
                        BTypes::Node(n) => node_sum = Some(node_sum.unwrap_or_else(|| NumNode::init(&0.0)).deref() + n.clone().deref()),
                    }
                }
            
                let min_sum: f64 = s.nodes.iter().map(|x| x.clone().min().unwrap()).sum();
                let result_node = node_sum.map(|node| BTypes::Node(node.deref() + &int_sum));
            
                (Some(min_sum), result_node)
            }

            NodeTypes::AndNode(a) => {
                (Some(a.nodes.iter().map(|x| x.clone().min().unwrap().to_isize().unwrap()).min().unwrap().to_f64().unwrap()), Some(btypes_max(a.nodes.iter().map(|x|x.clone().max().unwrap()).collect())))
            }

            _ => {
                panic!()
            }
        }
    }
}

impl Debug for BTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BTypes::Int(i) => write!(f, "{}", i),
            BTypes::Node(n) => write!(f, "{:?}", Rc::clone(n).deref()),
        }
    }
}

fn sym_render(a: &BTypes, ops: &Option<Box<dyn Any>>, ctx: &Option<Context>) -> String {
    match a {
        BTypes::Int(i) => {
            format!("{}", i)
        }
        BTypes::Node(n) => n.clone().render(&ops, &ctx),
    }
}
fn render_mulnode(m: &OpNode, ops: &Option<Box<dyn Any>>, ctx: &Option<Context>) -> String {
    match m.a.clone().deref() {
        NodeTypes::Variable(v_a) => match &m.b {
            BTypes::Node(n) => match n.clone().deref() {
                NodeTypes::Variable(v_b) => {
                    if &v_b.expr < &v_a.expr {
                        format!(
                            "{} * {}",
                            sym_render(&m.b, ops, ctx),
                            m.a.clone().deref().render(ops, ctx)
                        )
                    } else {
                        format!(
                            "{} * {}",
                            m.a.clone().deref().render(ops, ctx),
                            sym_render(&m.b, ops, ctx)
                        )
                    }
                }
                _ => {
                    format!(
                        "{} * {}",
                        m.a.clone().deref().render(ops, ctx),
                        sym_render(&m.b, ops, ctx)
                    )
                }
            },
            _ => format!(
                "{} * {}",
                m.a.clone().deref().render(ops, ctx),
                sym_render(&m.b, ops, ctx)
            ),
        },
        _ => format!(
            "{} * {}",
            m.a.clone().deref().render(ops, ctx),
            sym_render(&m.b, ops, ctx)
        ),
    }
}

fn btypes_max(b: Vec<BTypes>) -> BTypes {
    // Check if the vector is empty
    if b.is_empty() {
        // Return a default value or handle the empty case as needed
        // For now, I'll return an arbitrary default value (you may want to change this)
        return BTypes::Int(0.0);
    }

    let mut max = b[0].clone();

    for i in &b[1..] {
        match i {
            BTypes::Int(ii) => match &max {
                BTypes::Int(iii) => {
                    if ii > iii {
                        max = i.clone();
                    }
                }
                BTypes::Node(nnn) => {
                    if ii.i2n_gt(nnn.clone().deref()).deref().into() {
                        max = i.clone();
                    }
                }
            },
            BTypes::Node(nn) => match &max {
                BTypes::Int(iii) => {
                    if nn.n2i_gt(iii).deref().into() {
                        max = i.clone();
                    }
                }
                BTypes::Node(nnn) => {
                    if nn.n2n_gt(nnn.clone().deref()).deref().into() {
                        max = i.clone();
                    }
                }
            },
        }
    }

    max
}
impl NodeTypes {
    fn new_lt(a: N, b: BTypes) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::LtNode(OpNode {
            a: a.clone(),
            b: b.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::LtNode(OpNode {
            a,
            b,
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::LtNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }
    fn new_mul(a: N, b: BTypes) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::MulNode(OpNode {
            a: a.clone(),
            b: b.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::MulNode(OpNode {
            a,
            b,
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::MulNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }

    fn new_div(a: N, b: BTypes) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::DivNode(OpNode {
            a: a.clone(),
            b: b.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::DivNode(OpNode {
            a,
            b,
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::DivNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }

    fn new_mod(a: N, b: BTypes) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::ModNode(OpNode {
            a: a.clone(),
            b: b.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::ModNode(OpNode {
            a,
            b,
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::ModNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }

    fn new_sum(a: &Vec<N>) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::SumNode(RedNode {
            nodes: a.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::SumNode(RedNode {
            nodes: a.clone(),
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::SumNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }
    fn new_and(a: &Vec<N>) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::AndNode(RedNode {
            nodes: a.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::AndNode(RedNode {
            nodes: a.clone(),
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::AndNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }
    fn floordiv(&self, b: &BTypes, factoring_allowed: bool) -> Rc<Self> {
        match self {
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => match b {
                    BTypes::Int(ii) => {
                        if i % ii == 0.0 {
                            n.a.clone().deref() * &((i / ii).floor())
                        } else if ii % i == 0.0 && i > &0.0 {
                            n.a.floordiv(&BTypes::Int((ii / i).floor()), true)
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                    BTypes::Node(nn) => {
                        if (i % nn.clone().deref()).deref() == &0.0 {
                            n.a.clone().deref() * (NodeTypes::rfloordiv(i, b)).deref()
                        } else if (nn.clone().deref() % i).deref() == &0.0 && i > &0.0 {
                            n.a.clone().floordiv(
                                &BTypes::Node(nn.clone().floordiv(&n.b, factoring_allowed)),
                                true,
                            )
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                },
                BTypes::Node(i) => match b {
                    BTypes::Int(ii) => {
                        if (i.clone().deref() % ii).deref() == &0.0 {
                            n.a.clone().deref() * (i.clone().floordiv(b, true)).deref()
                        } else if (ii % i.clone().deref()).deref() == &0.0
                            && i.n2i_gt(&0.0).deref().into()
                        {
                            n.a.clone()
                                .floordiv(&BTypes::Node(NodeTypes::rfloordiv(ii, &n.b)), true)
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                    BTypes::Node(nn) => {
                        if (i.clone().deref() % nn.clone().deref()).deref() == &0.0 {
                            n.a.clone().deref() * i.floordiv(b, true).deref()
                        } else if (nn.clone().deref() % i.clone().deref()).deref() == &0.0
                            && i.n2i_gt(&0.0).deref().into()
                        {
                            n.a.clone().floordiv(
                                &BTypes::Node(nn.clone().floordiv(&n.b, factoring_allowed)),
                                true,
                            )
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                },
            },
            NodeTypes::DivNode(n) => match &n.b {
                BTypes::Int(i) => match b {
                    BTypes::Int(ii) => {
                        n.a.clone()
                            .floordiv(&BTypes::Int(ii * i), factoring_allowed)
                    }
                    BTypes::Node(ii) => {
                        n.a.clone()
                            .floordiv(&BTypes::Node(ii.clone().deref() * i), factoring_allowed)
                    }
                },
                BTypes::Node(i) => match b {
                    BTypes::Int(ii) => {
                        n.a.clone()
                            .floordiv(&BTypes::Node(i.clone().deref() * ii), factoring_allowed)
                    }
                    BTypes::Node(ii) => n.a.clone().floordiv(
                        &BTypes::Node(ii.clone().deref() * i.clone().deref()),
                        factoring_allowed,
                    ),
                },
            },
            NodeTypes::ModNode(n) => match b {
                BTypes::Int(i) => match &n.b {
                    BTypes::Int(ii) => {
                        if ii % i == 0.0 {
                            n.a.clone().floordiv(b, true).deref() % &(ii / i).floor()
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                    BTypes::Node(nn) => {
                        if (nn.clone().deref() % i).deref() == &0.0 {
                            n.a.clone().floordiv(b, factoring_allowed).deref()
                                % nn.clone().floordiv(b, true).deref()
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                },
                BTypes::Node(i) => match &n.b {
                    BTypes::Int(ii) => {
                        if (ii % i.clone().deref()).deref() == &0.0 {
                            n.a.clone().floordiv(b, factoring_allowed).deref()
                                % NodeTypes::rfloordiv(ii, b).deref()
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                    BTypes::Node(nn) => {
                        if (nn.clone().deref() % i.clone().deref()).deref() == &0.0 {
                            n.a.clone().floordiv(b, factoring_allowed).deref()
                                % nn.clone().floordiv(b, factoring_allowed).deref()
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                },
            },
            NodeTypes::SumNode(_) => match b {
                BTypes::Int(i) => {
                    if self == i {
                        return NumNode::init(&1.0).ptr();
                    }

                    let mut fully_divided: Vec<N> = vec![];

                    let mut rest: Vec<N> = vec![];
                    if i == &1.0 {
                        return self.ptr();
                    }
                    if !factoring_allowed {
                        return self.floordiv(b, factoring_allowed);
                    }
                    let mut _gcd = i.clone();
                    let mut divisior = 1.0;

                    self.flat_components()
                        .iter()
                        .for_each(|n| match n.clone().deref() {
                            NodeTypes::NumNode(x) => {
                                if &x.b % i == 0.0 {
                                    fully_divided.push(n.clone().floordiv(b, factoring_allowed));
                                } else {
                                    rest.push(n.clone());
                                    _gcd = num::integer::gcd(
                                        _gcd.floor().to_isize().unwrap(),
                                        x.b.clone().floor().to_isize().unwrap(),
                                    )
                                    .to_f64()
                                    .unwrap();
                                }
                            }
                            NodeTypes::MulNode(x) => match &x.b {
                                BTypes::Int(ii) => {
                                    if ii % i == 0.0 {
                                        fully_divided
                                            .push(n.clone().floordiv(b, factoring_allowed));
                                    } else {
                                        rest.push(n.clone());
                                        match &x.b {
                                            BTypes::Int(iii) => {
                                                _gcd = num::integer::gcd(
                                                    _gcd.clone().floor().to_isize().unwrap(),
                                                    iii.clone().floor().to_isize().unwrap(),
                                                )
                                                .to_f64()
                                                .unwrap();
                                                if divisior == 1.0 && i % ii == 0.0 {
                                                    divisior = iii.clone()
                                                }
                                            }
                                            _ => {
                                                _gcd = 1.0;
                                            }
                                        }
                                    }
                                }
                                BTypes::Node(_) => {
                                    _gcd = 1.0;
                                }
                            },
                            _ => {
                                rest.push(n.clone());
                                _gcd = 1.0;
                            }
                        });
                    if _gcd > 1.0 {
                        return (NodeTypes::sum(&fully_divided).deref()
                            + NodeTypes::sum(&rest)
                                .floordiv(&BTypes::Int(_gcd), factoring_allowed)
                                .deref())
                        .floordiv(&BTypes::Int((i / _gcd).floor()), factoring_allowed);
                    }
                    if divisior > 1.0 {
                        (NodeTypes::sum(&fully_divided).deref()
                            + NodeTypes::sum(&rest)
                                .floordiv(&BTypes::Int(divisior), factoring_allowed)
                                .deref())
                        .floordiv(&BTypes::Int((i / divisior).floor()), factoring_allowed)
                    } else {
                        NodeTypes::sum(&fully_divided).deref()
                            + NodeTypes::sum(&rest).floordiv(b, factoring_allowed).deref()
                    }
                }
                BTypes::Node(n_b1) => {
                    if self == n_b1.clone().deref() {
                        return NumNode::init(&1.0);
                    }
                    let mut fully_divided: Vec<N> = vec![];

                    let mut rest: Vec<N> = vec![];

                    self.flat_components().into_iter().for_each(|x| {
                        if (x.clone().deref() % n_b1.clone().deref()).deref() == &0.0 {
                            fully_divided.push(x.floordiv(b, factoring_allowed));
                        } else {
                            rest.push(x)
                        }
                    });
                    let sum_fully_divided = create_node(NodeTypes::new_sum(&fully_divided));
                    if sum_fully_divided.clone().deref() != &0.0 {
                        return (sum_fully_divided.deref()
                            + create_node(NodeTypes::new_sum(&rest)).deref())
                        .floordiv(b, factoring_allowed);
                    }
                    self.floordiv(b, false)
                }
            },
            _ => match b {
                BTypes::Node(n) => match n.clone().deref() {
                    NodeTypes::NumNode(num) => {
                        self.floordiv(&BTypes::Int(num.b.clone()), factoring_allowed)
                    }
                    _ => {
                        if self == n.clone().deref() {
                            NumNode::init(&1.0)
                        } else if (n.clone().deref() - self).min().unwrap() > 0.0
                            && self.min().unwrap() >= 0.0
                        {
                            NumNode::init(&0.0)
                        } else {
                            panic!("Not supported: {:?}, {:?}", self, b)
                        }
                    }
                },
                BTypes::Int(i) => {
                    assert!(i != &0.0);
                    if i < &0.0 {
                        return self.floordiv(&BTypes::Int(-i), factoring_allowed).deref() * &-1.0;
                    }
                    if i == &1.0 {
                        return self.ptr();
                    }

                    if self.min().unwrap() < 0.0 {
                        let offset: f64 = self.min().unwrap() / i.clone().floor();
                        return (self + &(-&offset * i)).floordiv(b, false).deref() + &offset;
                    }
                    create_node(NodeTypes::new_div(self.ptr(), b.clone()))
                }
            },
        }
    }
    fn rfloordiv(a: &f64, b: &BTypes) -> N {
        NumNode::init(a).floordiv(b, true)
    }

    pub fn n2n_le(&self, other: &Self) -> N {
        self.n2n_lt(&(other + &1.0))
    }
    pub fn n2n_gt(&self, other: &NodeTypes) -> N {
        -(self.n2n_lt(&((-other).deref() + &1.0))).deref()
    }
    pub fn n2n_lt(&self, other: &NodeTypes) -> N {
        match self {
            NodeTypes::MulNode(_) => self.n2n_le(other),
            NodeTypes::SumNode(_) => self.n2n_lt(other),
            _ => create_node(Self::new_lt(self.ptr(), BTypes::Node(other.ptr()))),
        }
    }
    pub fn n2n_ge(&self, other: &NodeTypes) -> N {
        (-self).n2n_lt(((-other).deref() + &1.0).deref())
    }
    pub fn n2i_lt(&self, other: &f64) -> N {
        match self {
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Node(_) => self.n2i_lt(other),
                BTypes::Int(i) => {
                    if i == &-1.0 {
                        return self.n2i_lt(other);
                    }
                    let sgn;
                    if i > &0.0 {
                        sgn = 1.0;
                    } else {
                        sgn = -1.0;
                    }
                    (n.a.clone().deref() * &sgn)
                        .n2i_lt(&((other + i.clone().abs() - 1.0) / i.clone().abs().floor()))
                }
            },
            NodeTypes::SumNode(s) => {
                let mut temp = other.clone();
                let mut new_sum: Vec<N> = vec![];
                let mut numn = other.clone();

                s.nodes.iter().for_each(|n| {
                    if let NodeTypes::NumNode(num) = n.clone().deref() {
                        numn = numn - num.b.clone();
                    } else {
                        new_sum.push(n.clone());
                    }
                });
                let mut lhs = NodeTypes::sum(&new_sum);
                let nodes;

                if let NodeTypes::SumNode(s) = lhs.clone().deref() {
                    nodes = s.nodes.clone();
                } else {
                    nodes = vec![lhs.clone()]
                }

                assert!(
                    nodes.iter().all(|nd| {
                        match nd.deref() {
                            NodeTypes::DivNode(n)
                            | NodeTypes::LtNode(n)
                            | NodeTypes::ModNode(n) => {
                                if let BTypes::Int(_) = &n.b {
                                    return true;
                                }
                                return false;
                            }
                            NodeTypes::NumNode(_) => return true,
                            NodeTypes::MulNode(n) => {
                                if let BTypes::Int(_) = &n.b {
                                    return true;
                                }
                                return false;
                            }
                            _ => return false,
                        }
                    }),
                    "Not Supported"
                );

                let (muls, others) = partition(nodes, |x| {
                    if let NodeTypes::MulNode(m) = x.clone().deref() {
                        match &m.b {
                            BTypes::Int(i) => {
                                if i > &0.0 && x.max().unwrap() >= *other {
                                    return true;
                                } else {
                                    return false;
                                }
                            }
                            BTypes::Node(n) => {
                                if n.n2i_gt(&0.0).deref().into() && x.max().unwrap() >= *other {
                                    return true;
                                } else {
                                    return false;
                                }
                            }
                        }
                    } else {
                        return false;
                    }
                });

                let mut mul_gcd = other.clone();

                muls.iter().for_each(|x| match x.clone().deref() {
                    NodeTypes::DivNode(n)
                    | NodeTypes::LtNode(n)
                    | NodeTypes::ModNode(n)
                    | NodeTypes::MulNode(n) => {
                        if let BTypes::Int(i) = &n.b {
                            mul_gcd = gcd(
                                mul_gcd.clone().floor().to_isize().unwrap(),
                                i.clone().floor().to_isize().unwrap(),
                            )
                            .to_f64()
                            .unwrap();
                        } else {
                            panic!("There is a bug here idiot");
                        }
                    }
                    NodeTypes::AndNode(_) | NodeTypes::SumNode(_) => {
                        panic!("There is a bug here idiot");
                    }
                    NodeTypes::NumNode(n) => {
                        mul_gcd = gcd(
                            mul_gcd.floor().to_isize().unwrap(),
                            n.b.clone().floor().to_isize().unwrap(),
                        )
                        .to_f64()
                        .unwrap();
                    }

                    NodeTypes::Variable(_) => {
                        panic!("There is a bug here idiot");
                    }
                });

                let all_others = NodeTypes::sum(&others);
                if all_others.min().unwrap() >= 0.0 && all_others.max().unwrap() < mul_gcd {
                    lhs = NodeTypes::sum(
                        &muls
                            .into_iter()
                            .map(|v| v.floordiv(&BTypes::Int(mul_gcd), true))
                            .collect(),
                    );
                    temp = (temp / mul_gcd).floor()
                }

                lhs.n2i_lt(&temp)
            }
            _ => create_node(Self::new_lt(self.ptr(), BTypes::Int(*other))),
        }
    }
    pub fn n2i_le(&self, other: &f64) -> N {
        self.n2i_lt(&(other + 1.0))
    }
    pub fn n2i_gt(&self, other: &f64) -> N {
        (-self).n2i_lt(&(-other))
    }

    pub fn n2i_ge(&self, other: &f64) -> N {
        (-self).n2i_lt(&(-other + 1.0))
    }

    fn recursive_max(&self) -> f64{
        match self.max().unwrap(){
            BTypes::Int(i) => {
                return i
            }
            BTypes::Node(n) => {
                NodeTypes::recursive_max(n.clone().deref())
            }
        }
    }
}
impl NumNode {
    pub fn init(a: &f64) -> N {
        let nd = Rc::new(NodeTypes::NumNode(NumNode {
            b: a.clone(),
            min: Some(a.clone()),
            max: Some(BTypes::Int(a.clone())),
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::MulNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }
}

impl Variable{
    pub fn new(expr: &String, val: &Option<f64>, min: &Option<f64>, max: &Option<BTypes>) -> N{
        let un_max = max.clone().unwrap();
        let un_min = min.clone().unwrap();
        match &un_max{
            BTypes::Int(i) => {
                assert!(i >= &0.0 && &un_min <= i);
                if &un_min == i{
                    return NumNode::init(&un_min)
                } 
            }
            BTypes::Node(n) => {
                assert!(&un_min >= &0.0 && un_min.borrow().i2n_le(n.clone().deref()).deref().into(), "{}", format!("invalid Variable {:?}, {:?}, {:?}", expr, min, max));
                if n.clone().deref()==&un_min{
                    return NumNode::init(&un_min)
                }
            }
        }
        
        let nd = Rc::new(NodeTypes::Variable(Variable { expr: expr.clone(), min: Some(un_min), max: Some(un_max), val: val.clone(), ptr:RefCell::new(None) }));

        if let NodeTypes::Variable(v) = nd.clone().deref(){
            v.ptr.borrow_mut().replace(Rc::downgrade(&nd));
        }

        nd
    }

    pub fn init(expr: String, nmin: f64, nmax: &BTypes) -> Rc<NodeTypes>{

        
        let nd = Rc::new(NodeTypes::Variable(Variable { expr: expr, min: Some(nmin), max: Some(nmax.clone()), val: None, ptr:RefCell::new(None) }));

        if let NodeTypes::Variable(v) = nd.clone().deref(){
            v.ptr.borrow_mut().replace(Rc::downgrade(&nd));
        }

        nd
    }
    pub fn val(&self) -> Option<f64>{
        assert!(self.val.is_some(), "{}", format!("Varible isn't bound can't access val of {:?}", self));
        self.val.clone()
    }
}
trait I2Ncmp {
    fn i2n_lt(&self, other: &NodeTypes) -> N;
    fn i2n_le(&self, other: &NodeTypes) -> N;
    fn i2n_gt(&self, other: &NodeTypes) -> N;
    fn i2n_ge(&self, other: &NodeTypes) -> N;
}

impl I2Ncmp for f64 {
    fn i2n_lt(&self, other: &NodeTypes) -> N {
        other.n2i_gt(self)
    }
    fn i2n_le(&self, other: &NodeTypes) -> N {
        other.n2i_gt(&(self + 1.0))
    }
    fn i2n_gt(&self, other: &NodeTypes) -> N {
        (-other).n2i_gt(&(-self))
    }
    fn i2n_ge(&self, other: &NodeTypes) -> N {
        (-other).n2i_gt(&(-self + 1.0))
    }
}

fn create_node(ret: N) -> N {
    assert!(ret.max().unwrap() >= ret.min().unwrap());
    if ret.min().unwrap() == ret.max().unwrap() {
        NumNode::init(&ret.min().unwrap())
    } else {
        ret
    }
}

impl Into<bool> for &NodeTypes {
    fn into(self) -> bool {
        match self {
            NodeTypes::DivNode(_)
            | NodeTypes::LtNode(_)
            | NodeTypes::ModNode(_)
            | NodeTypes::MulNode(_) => {
                !(self.min().unwrap() == self.max().unwrap() && self.min().unwrap() == 0.0)
            }
            _ => !(self.min().unwrap() == self.max().unwrap() && self.min().unwrap() == 0.0),
        }
    }
}

impl PartialEq<NodeTypes> for NodeTypes {
    fn eq(&self, other: &NodeTypes) -> bool {
        self.key() == other.key()
    }
}

impl PartialEq<f64> for BTypes {
    fn eq(&self, other: &f64) -> bool {
        match self {
            BTypes::Int(i) => i == other,
            BTypes::Node(n) => n.clone().deref() == other,
        }
    }
}

impl PartialEq<BTypes> for f64 {
    fn eq(&self, other: &BTypes) -> bool {
        match other {
            BTypes::Int(i) => i == self,
            BTypes::Node(n) => n.clone().deref() == self,
        }
    }
}
impl<'a> std::ops::Neg for &'a NodeTypes {
    type Output = N;

    fn neg(self) -> Self::Output {
        self * &-1.0
    }
}

impl<'a> std::ops::Add<&'a NodeTypes> for &'a NodeTypes {
    type Output = N;

    fn add(self, rhs: Self) -> Self::Output {
        NodeTypes::sum(&vec![self.ptr(), rhs.ptr()])
    }
}

impl<'a> std::ops::Add<&'a f64> for &'a NodeTypes {
    type Output = N;

    fn add(self, rhs: &'a f64) -> Self::Output {
        NodeTypes::sum(&vec![self.ptr(), NumNode::init(&rhs)])
    }
}

impl<'a> std::ops::Sub<&'a f64> for &'a NodeTypes {
    type Output = N;

    fn sub(self, rhs: &'a f64) -> Self::Output {
        (self) + &-rhs
    }
}

impl<'a> std::ops::Sub<&'a NodeTypes> for &'a NodeTypes {
    type Output = N;

    fn sub(self, rhs: &'a NodeTypes) -> Self::Output {
        self + (-rhs).deref()
    }
}

impl<'a> std::ops::Sub<&'a NodeTypes> for &'a f64 {
    type Output = N;

    fn sub(self, rhs: &'a NodeTypes) -> Self::Output {
        rhs - self
    }
}

impl PartialEq<f64> for NodeTypes {
    fn eq(&self, other: &f64) -> bool {
        NumNode::init(other).deref() == self
    }
}

impl <'a> Sub<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn sub(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i - ii)
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(i - n.clone().deref())
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().deref() - i)
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().deref() - nn.clone().deref())
            }
        }
    }
}

impl <'a> std::ops::Add<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn add(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i + ii)
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(n.clone().deref() + i)
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().deref() + i)
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().deref() + nn.clone().deref())
            }
        }
    }
}

impl <'a> std::ops::Add<&'a NodeTypes> for &'a BTypes{
    type Output = N;

    fn add(self, rhs: &'a NodeTypes) -> Self::Output {
        match &self{
            BTypes::Int(i) => rhs + i,
            BTypes::Node(n) => n.clone().deref() + rhs
        }
    }
}
impl <'a> std::ops::Mul<&'a NodeTypes> for &'a BTypes{
    type Output = N;

    fn mul(self, rhs: &'a NodeTypes) -> Self::Output {
        match &self{
            BTypes::Int(i) => rhs *i,
            BTypes::Node(n) => n.clone().deref() * rhs
        }
    }
}
impl <'a > std::ops::Add<&'a BTypes> for &'a NodeTypes{
    type Output = N;
    fn add(self, rhs: &'a BTypes) -> Self::Output {
        match &rhs{
            BTypes::Int(i) => self + i,
            BTypes::Node(n) => n.clone().deref() + self
        }
    }
}
impl <'a > std::ops::Mul<&'a BTypes> for &'a NodeTypes{
    type Output = N;
    fn mul(self, rhs: &'a BTypes) -> Self::Output {
        match &rhs{
            BTypes::Int(i) => self * i,
            BTypes::Node(n) => n.clone().deref() * self
        }
    }
}
impl <'a> std::ops::Mul<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn mul(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i * ii)
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(n.clone().deref() * i)
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().deref() * i)
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().deref() * nn.clone().deref())
            }
        }
    }
}

impl <'a> std::ops::Div<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn div(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i / ii)
            }
            _ => unreachable!()
        }
    }
}
impl Eq for BTypes{}
impl Ord for BTypes{
    fn max(self, other: Self) -> Self
        where
            Self: Sized, {
        if self > other{
            self
        } else{
            other
        }
    }

    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        unimplemented!()
    }
}
impl <'a> std::ops::Rem<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn rem(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i % ii)
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(n.clone().deref() % i)
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().deref() % i)
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().deref() % nn.clone().deref())
            }
        }
    }
}
impl PartialEq<BTypes> for BTypes {
    fn eq(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i == ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().deref() == ii,
            (BTypes::Int(i), BTypes::Node(nn)) => nn.clone().deref() == i,
            (BTypes::Node(n), BTypes::Node(nn)) => n.clone().deref() == nn.clone().deref(),
        }
    }
}
impl Eq for NodeTypes {}

impl PartialOrd<f64> for BTypes {
    fn ge(&self, other: &f64) -> bool {
        match self {
            BTypes::Int(i) => i >= other,
            BTypes::Node(n) => n.n2i_ge(other).deref().into(),
        }
    }
    fn gt(&self, other: &f64) -> bool {
        match self {
            BTypes::Int(i) => i > other,
            BTypes::Node(n) => n.n2i_gt(other).deref().into(),
        }
    }
    fn le(&self, other: &f64) -> bool {
        match self {
            BTypes::Int(i) => i <= other,
            BTypes::Node(n) => n.n2i_le(other).deref().into(),
        }
    }
    fn lt(&self, other: &f64) -> bool {
        match self {
            BTypes::Int(i) => i < other,
            BTypes::Node(n) => n.n2i_lt(other).deref().into(),
        }
    }

    fn partial_cmp(&self, _other: &f64) -> Option<std::cmp::Ordering> {
        unimplemented!()
    }
}

impl PartialEq<NodeTypes> for BTypes {
    fn eq(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => other == i,
            BTypes::Node(n) => other == n.clone().deref(),
        }
    }
}

impl PartialOrd<BTypes> for BTypes {
    fn ge(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i >= ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().n2i_ge(ii).deref().into(),
            (BTypes::Int(i), BTypes::Node(nn)) => i.i2n_ge(nn.clone().deref()).deref().into(),
            (BTypes::Node(n), BTypes::Node(nn)) => {
                n.clone().n2n_ge(nn.clone().deref()).deref().into()
            }
        }
    }
    fn gt(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i > ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().n2i_gt(ii).deref().into(),
            (BTypes::Int(i), BTypes::Node(nn)) => i.i2n_gt(nn.clone().deref()).deref().into(),
            (BTypes::Node(n), BTypes::Node(nn)) => {
                n.clone().n2n_gt(nn.clone().deref()).deref().into()
            }
        }
    }
    fn le(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i <= ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().n2i_le(ii).deref().into(),
            (BTypes::Int(i), BTypes::Node(nn)) => i.i2n_le(nn.clone().deref()).deref().into(),
            (BTypes::Node(n), BTypes::Node(nn)) => {
                n.clone().n2n_le(nn.clone().deref()).deref().into()
            }
        }
    }
    fn lt(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i < ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().n2i_lt(ii).deref().into(),
            (BTypes::Int(i), BTypes::Node(nn)) => i.i2n_lt(nn.clone().deref()).deref().into(),
            (BTypes::Node(n), BTypes::Node(nn)) => {
                n.clone().n2n_lt(nn.clone().deref()).deref().into()
            }
        }
    }

    fn partial_cmp(&self, other: &BTypes) -> Option<std::cmp::Ordering> {
        unimplemented!()
    }
}
impl PartialOrd<NodeTypes> for BTypes {
    fn gt(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => i.i2n_gt(other).deref().into(),
            BTypes::Node(n) => n.n2n_gt(other).deref().into(),
        }
    }
    fn ge(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => i.i2n_ge(other).deref().into(),
            BTypes::Node(n) => n.n2n_ge(other).deref().into(),
        }
    }
    fn le(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => i.i2n_le(other).deref().into(),
            BTypes::Node(n) => n.n2n_le(other).deref().into(),
        }
    }

    fn lt(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => i.i2n_lt(other).deref().into(),
            BTypes::Node(n) => n.n2n_lt(other).deref().into(),
        }
    }
    fn partial_cmp(&self, other: &NodeTypes) -> Option<std::cmp::Ordering> {
        unimplemented!()
    }
}
impl BTypes {
    fn sint_infer(&self) -> &Self {
        match self {
            BTypes::Node(n) => match n.clone().deref() {
                NodeTypes::AndNode(_)
                | NodeTypes::LtNode(_)
                | NodeTypes::DivNode(_)
                | NodeTypes::ModNode(_)
                | NodeTypes::NumNode(_) => {
                    panic!("Unexpected node type")
                }
                _ => return self,
            },
            _ => return self,
        }
    }

    pub fn floordiv(&self, b: &BTypes, factoring_allowed: bool) -> BTypes{
        match (self, b){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int((i /ii).floor())
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(NodeTypes::rfloordiv(i, b))
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().floordiv(b, true))
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().floordiv(b, true))
            }
        }
    }
}

impl Mul<&NodeTypes> for &NodeTypes {
    type Output = N;

    fn mul(self, rhs: &NodeTypes) -> Self::Output {
        match &self {
            NodeTypes::NumNode(n) => rhs * &n.b,
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => n.a.clone().deref() * (rhs * i).deref(),
                BTypes::Node(nn) => n.a.clone().deref() * (nn.clone().deref() * rhs).deref(),
            },
            NodeTypes::SumNode(s) => NodeTypes::sum(
                &s.nodes
                    .iter()
                    .map(|n| n.clone().deref() * rhs)
                    .collect::<Vec<N>>(),
            ),
            _ => match rhs {
                NodeTypes::NumNode(n) => {
                    create_node(NodeTypes::new_mul(self.ptr(), BTypes::Int(n.b.clone())))
                }
                _ => create_node(NodeTypes::new_mul(self.ptr(), BTypes::Node(rhs.ptr()))),
            },
        }
    }
}

impl Mul<&f64> for &NodeTypes {
    type Output = N;

    fn mul(self, rhs: &f64) -> Self::Output {
        match &self {
            NodeTypes::NumNode(n) => NumNode::init(&(n.b * rhs)),
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => n.a.clone().deref() * &(i * rhs),
                BTypes::Node(nn) => n.a.clone().deref() * (nn.clone().deref() * rhs).deref(),
            },
            NodeTypes::SumNode(s) => NodeTypes::sum(
                &s.nodes
                    .iter()
                    .map(|n| n.clone().deref() * rhs)
                    .collect::<Vec<N>>(),
            ),
            _ => {
                if rhs == &0.0 {
                    return NumNode::init(&0.0);
                } else if rhs == &1.1 {
                    return self.ptr();
                } else {
                    create_node(NodeTypes::new_mul(self.ptr(), BTypes::Int(rhs.clone())))
                }
            }
        }
    }
}
impl Rem<&NodeTypes> for &NodeTypes {
    type Output = N;

    fn rem(self, rhs: &NodeTypes) -> Self::Output {
        match rhs {
            NodeTypes::NumNode(n) => self % &n.b,
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => (n.a.clone().deref() * (i % rhs).deref()).deref() % rhs,
                BTypes::Node(i) => {
                    (n.a.clone().deref() * (i.clone().deref() % rhs).deref()).deref() % rhs
                }
            },
            NodeTypes::ModNode(_) => self % rhs,
            NodeTypes::SumNode(s) => {
                if self == rhs {
                    return NumNode::init(&0.0);
                }
                if (rhs - self).min().unwrap() > 0.0 {
                    return self.ptr();
                }
                let mut result = vec![];
                s.nodes.iter().for_each(|n| match n.clone().deref() {
                    NodeTypes::NumNode(_) => {
                        result.push(n.clone().deref() % rhs);
                    }
                    NodeTypes::MulNode(_) => {
                        result.push(n.clone().deref() % rhs);
                    }
                    _ => {
                        result.push(n.clone());
                    }
                });
                NodeTypes::sum(&result).deref() % rhs
            }
            _ => {
                if self == rhs {
                    return NumNode::init(&0.0);
                }
                if (rhs - self).min().unwrap() > 0.0 && self.min().unwrap() >= 0.0 {
                    self.ptr()
                } else {
                    panic!("Not supported: {:?} % {:?}", self, rhs);
                }
            }
        }
    }
}

impl Rem<&f64> for &NodeTypes {
    type Output = N;

    fn rem(self, rhs: &f64) -> Self::Output {
        match &self {
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => (n.a.clone().deref() * &(i % rhs)).deref() % rhs,
                BTypes::Node(i) => {
                    (n.a.clone().deref() * (i.clone().deref() % rhs).deref()).deref() % rhs
                }
            },
            NodeTypes::SumNode(s) => {
                if self == rhs {
                    return NumNode::init(&0.0);
                }
                let mut result = vec![];
                s.nodes.iter().for_each(|n| match n.clone().deref() {
                    NodeTypes::NumNode(_) => {
                        result.push(n.clone().deref() % rhs);
                    }
                    NodeTypes::MulNode(_) => {
                        result.push(n.clone().deref() % rhs);
                    }
                    _ => {
                        result.push(n.clone());
                    }
                });
                NodeTypes::sum(&result).deref() % rhs
            }

            NodeTypes::ModNode(n) => match &n.b {
                BTypes::Int(i) => {
                    if i % rhs == 0.0 {
                        n.a.clone().deref() % rhs
                    } else {
                        self % rhs
                    }
                }
                _ => self % rhs,
            },
            _ => {
                assert!(rhs > &0.0);

                if rhs == &1.0 {
                    return NumNode::init(&0.0);
                }
                if self.min().unwrap() >= 0.0 && self.max().unwrap() < rhs.clone() {
                    return self - &(rhs.clone() * ((self.min().unwrap() / rhs).floor()));
                }
                if self.min().unwrap() < 0.0 {
                    return (self - &(((self.min().unwrap() / rhs).floor()) * rhs)).deref() % rhs;
                }

                create_node(NodeTypes::new_mod(self.ptr(), BTypes::Int(rhs.clone())))
            }
        }
    }
}

impl Rem<&NodeTypes> for &f64 {
    type Output = N;

    fn rem(self, rhs: &NodeTypes) -> Self::Output {
        NumNode::init(&self).deref() % rhs
    }
}

impl Hash for NodeTypes {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl std::iter::Product<BTypes> for BTypes {
    fn product<I: Iterator<Item = BTypes>>(iter: I) -> Self {
        iter.fold(Self::Int(1.0), |acc, x| &acc * &x)
    }
}

impl std::iter::Sum<BTypes> for BTypes{
    fn sum<I: Iterator<Item = BTypes>>(iter: I) -> Self {
        iter.fold(Self::Int(0.0), |acc, s| &acc + &s)
    }
}