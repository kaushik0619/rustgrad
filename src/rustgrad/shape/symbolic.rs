use core::fmt;
use std::{any::Any, collections::{hash_map::DefaultHasher, HashMap}, fmt::write, hash::{Hash, Hasher}, ops::{Add, Mul, Neg}, rc::Rc};

use num_traits::{AsPrimitive, ToPrimitive};
use serde_json::Map;

trait NodeMethods {
    //maybe could also use dyn Any return type
    fn render(&self, ops: &Option<Rc<dyn Any>>, ctx: &Option<&str>) -> &str;

    fn vars(&self) -> Vec<&NodeTypes>;

    fn substitute(&self, var_vals: HashMap<NodeTypes, NodeTypes>) -> &NodeTypes;

    fn unbind(&self) -> (&NodeTypes, Option<f64>);

    //to be cached
    fn key(&self) -> &str;

    //to be cached
    fn hash(&self) -> u64;

    fn sum(nodes: Vec<NodeTypes>) -> NodeTypes;

    fn ands(nodes: &Vec<NodeTypes>) -> NodeTypes;
}

trait OpNodeMethods: NodeMethods {
    fn get_bounds(&self) -> (f64, f64);
}

trait RedNodeMethods: NodeMethods {

    fn get_bounds(&self) -> (f64, f64);
}
#[derive(Debug, Clone)]
struct Variable {
    expr: String,
    min: f64,
    max: f64,
    _val: Option<f64>,
}

#[derive(Debug, Clone)]
enum BTypes {
    Node(NodeTypes),
    Int(f64),
}
#[derive(Debug, Clone)]
struct NumNode {
    b: f64,
    min: f64,
    max: f64,
}
#[derive(Debug, Clone)]
struct OpNode{
    a: Rc<NodeTypes>,
    b: Rc<BTypes>,
    min: f64,
    max: f64
}
#[derive(Debug, Clone)]
struct RedNode{
    nodes: Vec<NodeTypes>,
    min: f64,
    max: f64
}
#[derive(Clone)]
enum NodeTypes {
    Variable(Variable),
    NumNode(NumNode),
    LtNode(OpNode),
    MulNode(OpNode),
    DivNode(OpNode),
    ModNode(OpNode),
    SumNode(RedNode),
    AndNode(RedNode)
}

impl NodeMethods for NodeTypes {
    fn render(&self, ops: &Option<Rc<dyn Any>>, ctx: &Option<&str>) -> &str {
        match self {
            NodeTypes::Variable(v) => {
                ops.and_then(|o|{
                    ctx.map(|c|{
                        if c == "DEBUG"{
                            format!(
                                "{}[{:?}-{:?}{}]",
                                v.expr,
                                v.min,
                                v.max,
                                if let Some(val) = v._val {
                                    format!("={}", val)
                                } else {
                                    "".to_string()
                                }
                            )
                            .as_str()
                        } else if c == "REPR"{
                            format!(
                                "Variable('{}', {:?}, {:?}){}",
                                v.expr,
                                v.min,
                                v.max,
                                if let Some(val) = v._val {
                                    format!(".bind({})", val)
                                } else {
                                    "".to_string()
                                }
                            )
                            .as_str()
                        } else {
                            v.expr.as_str()
                        }
                    })
                }).unwrap_or_else(|| v.expr.as_str())
            }
            NodeTypes::NumNode(n) => {
                ops.and_then(|_| {
                    ctx.map(|c| {
                        if c == "REPR" {
                            format!("NumNode({})", n.b).as_str()
                        } else {
                            format!("{}", n.b).as_str()
                        }
                    })
                })
                .unwrap_or_else(|| format!("{}", n.b).as_str())
            }

            NodeTypes::LtNode(l) => {
                assert!(l.min != l.max);
                format!("{} / {:?}", l.a.render(ops, ctx),l.b).as_str()
            }

            NodeTypes::MulNode(m) => {
                assert!(m.min != m.max);
                render_mulnode(m, ops, ctx)
            }

            NodeTypes::DivNode(d) => {
                assert!(d.min != d.max);
                format!("{} // {:?}", d.a.render(ops, ctx), d.b).as_str()
            }

            NodeTypes::ModNode(md) => {
                assert!(md.min != md.max);
                format!("{} % {:?}", md.a.render(ops, ctx), md.b).as_str()
            }
            NodeTypes::SumNode(s) => {
                assert!(s.min != s.max);
                let mut vec: Vec<&str> = s.nodes.iter().map(|x| x.render(ops, ctx)).collect::<Vec<&str>>();
                vec.sort();
                format!("{}", vec.join("+")).as_str()
            }
            NodeTypes::AndNode(a) => {
                assert!(a.min != a.max);
                let mut vec: Vec<&str> = a.nodes.iter().map(|x| x.render(ops, ctx)).collect::<Vec<&str>>();
                vec.sort();
                format!("{}", vec.join(" and ")).as_str()
            }
        }
    }

    fn vars(&self) -> Vec<&NodeTypes> {
        match self{
            _ => vec![],
            NodeTypes::Variable(_) => vec![self],
            NodeTypes::LtNode(n) | NodeTypes::DivNode(n) | NodeTypes::ModNode(n) | NodeTypes::MulNode(n) => {
                let mut result = n.a.vars();
                match *n.b{
                    BTypes::Node(b_n) => {result.extend(b_n.vars());},
                    _ => {result.extend::<Vec<&NodeTypes>>(vec![]);}
                }
                result
            },
            NodeTypes::SumNode(n) | NodeTypes::AndNode(n) => n.nodes.iter().flat_map(|x| x.vars()).collect()
        }
    }

    fn substitute(&self, var_vals: HashMap<NodeTypes, NodeTypes>) -> &NodeTypes {
        match self{
            NodeTypes::Variable(v) =>  return &var_vals.get(self).cloned().unwrap_or_else(|| self),
            NodeTypes::NumNode(n) => self,
            NodeTypes::LtNode(l) => {
                match *l.b{
                    BTypes::Int(i) =>{
                        return l.a.substitute(var_vals) < &i
                    },
                    BTypes::Node(n) => {
                        return l.a.substitute(var_vals) < n.substitute(var_vals);
                    }
                }
            }
            NodeTypes::MulNode(m) => {
                match *m.b{
                    BTypes::Int(i) => {
                        return &(*m.a.substitute(var_vals) * i)
                    }
                    BTypes::Node(n) => {
                        return &(*m.a.substitute(var_vals) * *n.substitute(var_vals))
                    }
                }
            }

            //may have to match the Btype enums here if ops dont support inter enum ops
            NodeTypes::DivNode(d) => {
                &d.a.substitute(var_vals).floordiv(*d.b, true)
            }
            NodeTypes::ModNode(m) => {
                match *m.b{
                    BTypes::Int(i) => {&(*m.a.substitute(var_vals) % i)},
                    BTypes::Node(n) => {&(*m.a.substitute(var_vals) % n)}
                }
            }
            NodeTypes::SumNode(s) =>{
                &Self::sum(s.nodes.iter().map(|n| *n.substitute(var_vals)).collect::<Vec<NodeTypes>>())
            }
            NodeTypes::AndNode(a) => {
                let vec = vec![];
                a.nodes.iter().for_each(|n| vec.push(*n.substitute(var_vals)));
                &Self::ands(&vec)
            }
        }
    }

    fn unbind(&self) -> (&NodeTypes, Option<f64>) {
        match self{
            _ => {

                let mut map = HashMap::new();
                self.vars().into_iter().for_each(|v|{
                    match v{
                        NodeTypes::Variable(var) => {
                            if var._val.is_some(){
                                map.insert(v.clone(), *v.unbind().0);
                            }
                        }

                        _ =>{}
                    }
                });
                (self.substitute(map), None)
            }
            NodeTypes::Variable(v) =>{
                assert!(v._val.is_some());
                (&Variable::init(v.expr.as_str(), v.min, v.max), v._val)
            }
        }
    }

    fn key(&self) -> &str {
        self.render(&None, &Some("DEBUG"))
    }
    fn hash(&self) -> u64 {
        let mut s= DefaultHasher::new();
        self.key().hash(&mut s);
        s.finish()
    }
}
impl Hash for NodeTypes{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl fmt::Debug for NodeTypes{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render(&None, &Some("DEBUG")))
    }
}

impl fmt::Display for NodeTypes{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.key())
    }
}


impl Into<bool> for NodeTypes{
fn into(self) -> bool {
    match self{
        NodeTypes::DivNode(a) | NodeTypes::LtNode(a) | NodeTypes::ModNode(a) | NodeTypes::MulNode(a) => !(a.min == a.max && a.min == 0.0),
        other @ _ =>{
            !(other.min() == other.max() && other.min() == 0.0)
        }
    }
}
}

impl PartialEq<NodeTypes> for NodeTypes{
    fn eq(&self, other: &NodeTypes) -> bool {
        self.key() == other.key()
    }
}

impl Neg for NodeTypes{
    type Output = NodeTypes;
    fn neg(self) -> Self::Output {
        self*-1.0
    }
}

impl std::ops::Add<NodeTypes> for NodeTypes {
    type Output = NodeTypes;

    fn add(self, rhs: NodeTypes) -> NodeTypes {
        NodeTypes::sum(vec![self, rhs])
    }
}

impl std::ops::Add<f64> for NodeTypes {
    type Output = NodeTypes;

    fn add(self, rhs: f64) -> NodeTypes {
        NodeTypes::sum(vec![self, NumNode::init(rhs)])
    }
}

impl std::ops::Add<NodeTypes> for f64{
    type Output = NodeTypes;

    fn add(self, rhs: NodeTypes) -> Self::Output {
        rhs + self
    }
}

impl std::ops::Sub<NodeTypes> for NodeTypes{
    type Output = NodeTypes;

    fn sub(self, rhs: NodeTypes) -> Self::Output {
        self + -rhs
    }
}

impl std::ops::Sub<f64> for NodeTypes {
    type Output = NodeTypes;

    fn sub(self, rhs: f64) -> Self::Output {
        self + -rhs
    }
}

impl std::ops::Sub<NodeTypes> for f64{
    type Output = NodeTypes;

    fn sub(self, rhs: NodeTypes) -> Self::Output {
        rhs - self
    }
}

impl PartialOrd<NodeTypes> for NodeTypes{
    fn le(&self, other: &Self) -> bool {
        self < &(other.clone() + 1.0)
    }
    fn gt(&self, other: &NodeTypes) -> bool {
        (-self.clone()) < (-other.clone())
    }
    fn ge(&self, other: &NodeTypes) -> bool {
        (-self.clone()) < (-other.clone() + 1.0)
    }
    fn lt(&self, other: &NodeTypes) -> bool {
        match self{
            NodeTypes::MulNode(n) => {
                self < other
            }
            _ =>{
                create_node(Self::new_lt(self.clone(), BTypes::Node(other.clone()))).into()
            }
        }
        
    }
    fn partial_cmp(&self, other: &NodeTypes) -> Option<std::cmp::Ordering> {
        if self < other {
            Some(std::cmp::Ordering::Less)
        } else if self > other {
            Some(std::cmp::Ordering::Greater)
        } else if self == other {
            Some(std::cmp::Ordering::Equal)
        } else {
            None
        }
    }
}
impl PartialEq<f64> for NodeTypes{
    fn eq(&self, other: &f64) -> bool {
        unimplemented!()
    }
}
impl PartialOrd<f64> for NodeTypes{
    fn lt(&self, other: &f64) -> bool {
        match self{
            _ => {create_node(Self::new_lt(self.clone(), BTypes::Int(*other))).into()}
            NodeTypes::MulNode(n) => {
                match *n.b.clone(){
                    BTypes::Node(n_b) => {
                        self < other
                    }
                    BTypes::Int(i) => {
                        if i == -1.0{
                            return self < other
                        }
                        let mut sgn = 0.0;
                        if i > 0.0 {
                            sgn = 1.0;
                        } else{
                            sgn = -1.0;
                        }
                        (*n.a.clone()*sgn) < ((other + i.abs() - 1.0)/i.abs()).floor()
                    }
                }
            }
        }
    }
    fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
        if self < other {
            Some(std::cmp::Ordering::Less)
        } else if self > other {
            Some(std::cmp::Ordering::Greater)
        } else if self == other {
            Some(std::cmp::Ordering::Equal)
        } else {
            None
        }
    }
}

impl Mul<NodeTypes> for NodeTypes{
    type Output = NodeTypes;

    fn mul(self, rhs: Self) -> Self::Output {
        match self{

            NodeTypes::NumNode(n) => {
                rhs*n.b
            }
            NodeTypes::MulNode(n) => {
                match *n.b{
                    BTypes::Int(i) => {
                        *n.a*(i*rhs)
                    }
                    BTypes::Node(nn) => {
                        *n.a*(nn*rhs)
                    }
                }
            }
            _ => {
                match rhs{
                    NodeTypes::NumNode(n) =>{
                        create_node(NodeTypes::new_mul(self, BTypes::Int(n.b)))
                    },
                    _ =>{
                        create_node(NodeTypes::new_mul(self, BTypes::Node(rhs)))
                    }
                }
            }
        }

    }
}

impl Mul<f64> for NodeTypes{
    type Output = NodeTypes;

    fn mul(self, rhs: f64) -> Self::Output {
        match self{

            NodeTypes::NumNode(n) =>{
                NumNode::init(n.b*rhs)
            }
            NodeTypes::MulNode(n) => {
                match *n.b{
                    BTypes::Int(i) => {
                        *n.a*(i*rhs)
                    }
                    BTypes::Node(nn) => {
                        *n.a*(nn*rhs)
                    }
                }
            }
            _ => {
                if rhs ==0.0 {return NumNode::init(0.0)}
                if rhs == 1.1{return self}
                create_node(NodeTypes::new_mul(self, BTypes::Int(rhs)))
            }
        }
    }
}

impl Mul<NodeTypes> for f64{
    type Output = NodeTypes;

    fn mul(self, rhs: NodeTypes) -> Self::Output {
        rhs * self
    }
}

impl std::ops::Rem<NodeTypes> for NodeTypes{
    type Output = NodeTypes;

    fn rem(self, rhs: Self) -> Self::Output {
        match rhs{
            NodeTypes::NumNode(n) => {
                self % n.b
            },
            NodeTypes::MulNode(n) =>{
                match *n.b{
                    BTypes::Int(i) => {
                        *n.a * (i % rhs) % rhs
                    }
                    BTypes::Node(i) => {
                        *n.a * (i % rhs) % rhs
                    }
                }
            },
            NodeTypes::ModNode(_) => {
                self % rhs
            }
            other @ _ =>{
                if self == rhs{return NumNode::init(0.0)}
                if (rhs - self).min() > 0.0 && self.min() >= 0.0 {self}
                else{
                    panic!("Not supported: {} % {}", self, rhs);
                }
            }
        }
    }
}

impl std::ops::Rem<f64> for NodeTypes{
    type Output = NodeTypes;

    fn rem(self, rhs: f64) -> Self::Output {

        match self{
            _=>{
                assert!(rhs > 0.0);

                if rhs == 1.0{return NumNode::init(0.0)}
        
                if self.min() >=0.0 && self.max() < rhs{return self - (rhs*((self.min()/rhs).floor()))}
                if self.min() < 0.0{return (self - (((self.min()/rhs).floor())*rhs)) % rhs}
        
                create_node(NodeTypes::new_mod(self, BTypes::Int(rhs)))
            }

            NodeTypes::MulNode(n) =>{
                match *n.b{
                    BTypes::Int(i) => {
                        *n.a * (i % rhs) % rhs
                    }
                    BTypes::Node(i) => {
                        *n.a * (i % rhs) % rhs
                    }
                }
            }
        }

    }
}

impl std::ops::Rem<NodeTypes> for f64{
    type Output = NodeTypes;

    fn rem(self, rhs: NodeTypes) -> Self::Output {
        NumNode::init(self) % rhs
    }
}

fn sym_render <'a>(a: &BTypes, ops:&Option<Rc<dyn Any>>, ctx: &'a Option<&str>) -> &'a str{
    match a{
        BTypes::Int(i) => {
            format!("{}", i).as_str()
        }
        BTypes::Node(n) => {
            n.render(ops, ctx)
        }
    }
}

fn render_mulnode<'a>(m: &OpNode, ops: &Option<Rc<dyn Any>>, ctx: &Option<&str> ) -> &'a str{
    match *m.a{
        NodeTypes::Variable(v_a) => {
            match *m.b{
                BTypes::Node(n) => {
                    match n{
                        NodeTypes::Variable(v_b) => {
                            if v_b.expr < v_a.expr{
                                return format!("{} * {}", sym_render(&*m.b, &ops, &ctx), m.a.render(&ops, ctx)).as_str();
                            } else{
                                return format!("{} * {}", m.a.render(&ops, ctx), sym_render(&*m.b, &ops, &ctx)).as_str();
                            }
                        }
                        _ => {
                            return format!("{} * {}", m.a.render(&ops, ctx), sym_render(&*m.b, &ops, &ctx)).as_str();
                        }
                    }
                }
                _ => {
                    return format!("{} * {}", m.a.render(&ops, ctx), sym_render(&*m.b, &ops, &ctx)).as_str();
                }
            }
        }
        _ =>{
            return format!("{} * {}", m.a.render(&ops, ctx), sym_render(&*m.b, &ops, &ctx)).as_str();
        }
    }
}

fn create_node(ret: NodeTypes) -> NodeTypes{
    assert!(ret.min() <= ret.max());
    if ret.min() == ret.max(){
        NumNode::init(ret.min())
    } else{
        ret
    }
}
impl Variable{
    fn init(expr: &str, nmin: f64, nmax: f64) -> NodeTypes{
        NodeTypes::Variable(Variable{
            expr: expr.to_string(),
            min: nmin,
            max: nmax,
            _val: None
        })
    }
}

impl NumNode{
    fn init(num: f64) -> NodeTypes{
        NodeTypes::NumNode(NumNode { b: num, min: num, max: num })
    }
}

impl NodeTypes{
    fn new_lt(a: NodeTypes, b: BTypes) -> NodeTypes{
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::LtNode(OpNode{a: Rc::new(a), b: Rc::new(b), min: 0.0, max: 0.0}));

        NodeTypes::LtNode(OpNode{a: a.into(), b: Rc::new(b), min, max})
    }
    fn new_mul(a: NodeTypes, b: BTypes) -> NodeTypes{
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::MulNode(OpNode{a: Rc::new(a), b: Rc::new(b), min: 0.0, max: 0.0}));

        NodeTypes::MulNode(OpNode{a: a.into(), b: Rc::new(b), min, max})
    }
    fn new_div(a: NodeTypes, b: BTypes) -> NodeTypes{
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::DivNode(OpNode{a: Rc::new(a), b: Rc::new(b), min: 0.0, max: 0.0}));

        NodeTypes::DivNode(OpNode{a: a.into(), b: Rc::new(b), min, max})
    }
    fn new_mod(a: NodeTypes, b: BTypes) -> NodeTypes{
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::ModNode(OpNode{a: Rc::new(a), b: Rc::new(b), min: 0.0, max: 0.0}));

        NodeTypes::ModNode(OpNode{a: Rc::new(a), b: Rc::new(b), min, max})
    }

    fn min(&self) -> f64{
        match self{
            NodeTypes::DivNode(n) | NodeTypes::LtNode(n) | NodeTypes::ModNode(n) | NodeTypes::MulNode(n) => {
                n.min
            },
            NodeTypes::AndNode(n) | NodeTypes::SumNode(n) => {
                n.min
            },
            NodeTypes::NumNode(n) => {
                n.min
            }
            NodeTypes::Variable(n) => {
                n.min
            }
        }
    }

    fn max(&self) -> f64{
        match self{
            NodeTypes::DivNode(n) | NodeTypes::LtNode(n) | NodeTypes::ModNode(n) | NodeTypes::MulNode(n) => {
                n.max
            },
            NodeTypes::AndNode(n) | NodeTypes::SumNode(n) => {
                n.max
            },
            NodeTypes::NumNode(n) => {
                n.max
            }
            NodeTypes::Variable(n) => {
                n.max
            }
        }
    }

    fn floordiv(&self, b: BTypes, factoring_allowed: bool) -> Self{

        match self{
            NodeTypes::MulNode(n) => {
                match *n.b{
                    BTypes::Int(i) => {
                            match b{
                                BTypes::Int(ii) => {
                                    if i % ii == 0.0{
                                        *n.a*((i/ii).floor())
                                    } else if ii % i == 0.0 && i > 0.0{
                                        n.a.floordiv(BTypes::Int((ii/i).floor()), true)
                                    } else{
                                        self.floordiv(b, factoring_allowed)
                                    }
                                }
                                BTypes::Node(nn) => {
                                    if i % nn == 0.0{
                                        *n.a*(NodeTypes::rfloordiv(i, b))
                                    } else if nn % i == 0.0 && i > 0.0{
                                        n.a.floordiv(BTypes::Node(nn.floordiv(*n.b, factoring_allowed)), true)
                                    } else{
                                        self.floordiv(b, factoring_allowed)
                                    }
                                }
                            }
                    }
                    BTypes::Node(i) => {
                        match b{
                            BTypes::Int(ii) => {
                                if i % ii == 0.0{
                                    *n.a*(i.floordiv(b, true))
                                } else if ii % i == 0.0 && i > 0.0{
                                    n.a.floordiv(BTypes::Node(NodeTypes::rfloordiv(ii, *n.b)), true)
                                } else{
                                    self.floordiv(b, factoring_allowed)
                                }
                            }
                            BTypes::Node(nn) => {
                                if i % nn == 0.0{
                                    *n.a*(i.floordiv(b, true))
                                } else if nn % i == 0.0 && i > 0.0{
                                    n.a.floordiv(BTypes::Node(nn.floordiv(*n.b, factoring_allowed)), true)
                                } else{
                                    self.floordiv(b, factoring_allowed)
                                }
                            }
                        }
                }
                }
            }
            NodeTypes::DivNode(n) => {
                match *n.b{
                    BTypes::Int(i) => {
                        match b{
                            BTypes::Int(ii) => {
                                n.a.floordiv(BTypes::Int(ii*i), factoring_allowed)
                            }
                            BTypes::Node(ii) => {
                                n.a.floordiv(BTypes::Node(ii*i), factoring_allowed)
                            }
                        }
                    }
                    BTypes::Node(i) => {
                        match b{
                            BTypes::Int(ii) => {
                                n.a.floordiv(BTypes::Node(ii*i), factoring_allowed)
                            }
                            BTypes::Node(ii) => {
                                n.a.floordiv(BTypes::Node(ii*i), factoring_allowed)
                            }
                        }
                    }
                }
            }
            NodeTypes::ModNode(n) => {
                match b{
                    BTypes::Int(i) =>{
                        match *n.b{
                            BTypes::Int(ii) => {
                                if ii % i == 0.0{
                                    n.a.floordiv(b, true) % (ii/i).floor()
                                }
                                else{
                                    self.floordiv(b, factoring_allowed)
                                }
                            }
                            BTypes::Node(nn) => {
                                if nn % i == 0.0{
                                    n.a.floordiv(b, factoring_allowed) % nn.floordiv(b, true)
                                }
                                else{
                                    self.floordiv(b, factoring_allowed)
                                }
                            }
                        }
                    }
                    BTypes::Node(i) =>{
                        match *n.b{
                            BTypes::Int(ii) => {
                                if ii % i == 0.0{
                                    n.a.floordiv(b, factoring_allowed) % NodeTypes::rfloordiv(ii, b)
                                }
                                else{
                                    self.floordiv(b, factoring_allowed)
                                }
                            }
                            BTypes::Node(nn) => {
                                if nn % i == 0.0{
                                    n.a.floordiv(b, factoring_allowed) % nn.floordiv(b, factoring_allowed)
                                }
                                else{
                                    self.floordiv(b, factoring_allowed)
                                }
                            }
                        }
                    }  
                }
            }
            _ => {
        match b{
            BTypes::Node(n) =>{
                match n{
                    NodeTypes::NumNode(num) => {
                        self.floordiv(BTypes::Int(num.b), factoring_allowed)
                    }
                    _ =>{
                        if *self == n{
                            NumNode::init(1.0)
                        } else if (n - self.clone()).min() > 0.0 && self.min() >= 0.0{
                            NumNode::init(0.0)
                        } else{
                            panic!("Not supported: {}, {:?}",self, b)
                        }
                    }
                }
            }
            BTypes::Int(i) => {
                assert!(i != 0.0);
                if i < 0.0{
                    return self.floordiv(BTypes::Int(-i), factoring_allowed)*-1.0
                }
                if i == 1.0{return self.clone()}

                if self.min() < 0.0{
                    let offset = self.min() / i.floor();
                    return (self.clone() + -offset*i).floordiv(b, false) + offset
                }
                create_node(NodeTypes::new_div(self.clone(), b))
            }
        }}
    }}
    fn rfloordiv(a: f64, b: BTypes) -> NodeTypes{
        NumNode::init(a).floordiv(b, true)
    }
}

impl OpNodeMethods for NodeTypes{
    fn get_bounds(&self) -> (f64, f64) {
        todo!()
    }
}