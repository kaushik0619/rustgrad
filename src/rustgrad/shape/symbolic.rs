use core::fmt;
use std::{any::Any, borrow::Borrow, collections::{hash_map::DefaultHasher, HashMap}, hash::{Hash, Hasher}, ops::{Deref, Mul, Neg}, rc::Rc};
use num::{integer, Float, Num, ToPrimitive};
trait NodeMethods {
    //maybe could also use dyn Any return type
    fn render(&self, ops: &Option<Rc<dyn Any>>, ctx: &Option<&str>) -> String;

    fn vars(&self) -> Vec<&NodeTypes>;

    fn substitute(&self, var_vals: &HashMap<NodeTypes, NodeTypes>) -> NodeTypes;

    fn unbind(&self) -> (NodeTypes, Option<f64>);

    //to be cached
    fn key(&self) -> String;

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
    fn render(&self, ops: &Option<Rc<dyn Any>>, ctx: &Option<&str>) -> String {
        match &self {
            NodeTypes::Variable(v) => {
                ops.as_ref().and_then(|o|{
                    ctx.map(|c|{
                        if c == "DEBUG"{
                            format!(
                                "{}[{:?}-{:?}{}]",
                                v.expr.clone(),
                                v.min.clone(),
                                v.max.clone(),
                                if let Some(val) = v._val {
                                    format!("={}", val)
                                } else {
                                    "".to_string()
                                }
                            )
                            
                        } else if c == "REPR"{
                            format!(
                                "Variable('{}', {:?}, {:?}){}",
                                v.expr.clone(),
                                v.min.clone(),
                                v.max.clone(),
                                if let Some(val) = v._val {
                                    format!(".bind({})", val)
                                } else {
                                    "".to_string()
                                }
                            )
                        } else {
                            v.expr.clone()
                        }
                    })
                }).unwrap_or_else(|| v.expr.clone())
            }
            NodeTypes::NumNode(n) => {
                ops.as_ref().and_then(|_| {
                    ctx.map(|c| {
                        if c == "REPR" {
                            format!("NumNode({})", n.b)
                        } else {
                            format!("{}", n.b)
                        }
                    })
                })
                .unwrap_or_else(|| format!("{}", n.b))
            }

            NodeTypes::LtNode(l) => {
                assert!(l.min != l.max);
                format!("{} / {:?}", l.a.render(ops, ctx),l.b.clone())
            }

            NodeTypes::MulNode(m) => {
                assert!(m.min != m.max);
                render_mulnode(m, ops, ctx)
            }

            NodeTypes::DivNode(d) => {
                assert!(d.min != d.max);
                format!("{} // {:?}", d.a.render(ops, ctx), d.b.clone())
            }

            NodeTypes::ModNode(md) => {
                assert!(md.min != md.max);
                format!("{} % {:?}", md.a.render(ops, ctx), md.b.clone())
            }
            NodeTypes::SumNode(s) => {
                assert!(s.min != s.max);
                let mut vec: Vec<String> = s.nodes.iter().map(|x| x.render(ops, ctx)).collect::<Vec<String>>();
                vec.sort();
                format!("{}", vec.join("+"))
            }
            NodeTypes::AndNode(a) => {
                assert!(a.min!= a.max);
                let mut vec: Vec<String> = a.nodes.iter().map(|x| x.render(ops, ctx)).collect::<Vec<String>>();
                vec.sort();
                format!("{}", vec.join(" and "))
            }
        }
    }

    fn vars(&self) -> Vec<&NodeTypes> {
        match &self{
            NodeTypes::Variable(_) => vec![self],
            NodeTypes::LtNode(n) | NodeTypes::DivNode(n) | NodeTypes::ModNode(n) | NodeTypes::MulNode(n) => {
                let mut result = n.a.vars();
                match n.b.as_ref(){
                    BTypes::Node(b_n) => {result.extend(b_n.vars());},
                    _ => {result.extend::<Vec<&NodeTypes>>(vec![]);}
                }
                result
            },
            NodeTypes::SumNode(n) | NodeTypes::AndNode(n) => n.nodes.iter().flat_map(|x| x.vars()).collect(),
            _ => vec![],

        }
    }

    // fn substitute(&self, var_vals: &HashMap<NodeTypes, NodeTypes>) -> NodeTypes {
    //     match &self{
    //         NodeTypes::Variable(v) =>  var_vals.get(self).cloned().unwrap_or_else(|| self.clone()),
    //         NodeTypes::NumNode(n) => self.clone(),
    //         NodeTypes::LtNode(l) => {
    //             match l.b.as_ref(){
    //                 BTypes::Int(i) =>{
    //                     return l.a.substitute(var_vals).n2i_lt(&i)
    //                 },
    //                 BTypes::Node(n) => {
    //                     return l.a.substitute(var_vals).n2n_lt(&(n.substitute(var_vals)));
    //                 }
    //             }
    //         }
    //         NodeTypes::MulNode(m) => {
    //             match m.b.as_ref(){
    //                 BTypes::Int(i) => {
    //                     return m.a.substitute(var_vals) * *i
    //                 }
    //                 BTypes::Node(n) => {
    //                     return m.a.substitute(var_vals) * n.substitute(var_vals).clone()
    //                 }
    //             }
    //         }
    //         NodeTypes::DivNode(d) => {
    //             d.a.substitute(var_vals).floordiv(d.b.as_ref().clone(), true)
    //         }
    //         NodeTypes::ModNode(m) => {
    //             match m.b.as_ref(){
    //                 BTypes::Int(i) => {m.a.substitute(var_vals) % *i},
    //                 BTypes::Node(n) => {m.a.substitute(var_vals) % n.clone()}
    //             }
    //         }
    //         NodeTypes::SumNode(s) =>{
    //             Self::sum(s.nodes.iter().map(|n| n.substitute(var_vals)).collect::<Vec<NodeTypes>>())
    //         }
    //         NodeTypes::AndNode(a) => {
    //             let mut vec = vec![];
    //             a.nodes.iter().for_each(|n| vec.push(n.substitute(var_vals)));
    //             Self::ands(&vec)
    //         }
    //     }
    // }
    fn substitute(&self, var_vals: &HashMap<NodeTypes, NodeTypes>) -> NodeTypes {
        match self {
            NodeTypes::Variable(_) => var_vals.get(self).cloned().unwrap_or_else(|| self.clone()),
            NodeTypes::NumNode(_) => self.clone(),
            NodeTypes::LtNode(l) => {
                match l.b.as_ref() {
                    BTypes::Int(i) => l.a.substitute(var_vals).n2i_lt(i),
                    BTypes::Node(n) => l.a.substitute(var_vals).n2n_lt(&n.substitute(var_vals)),
                }
            }
            NodeTypes::MulNode(m) => {
                match m.b.as_ref() {
                    BTypes::Int(i) => m.a.substitute(var_vals) * *i,
                    BTypes::Node(n) => m.a.substitute(var_vals) * n.substitute(var_vals).clone(),
                }
            }
            NodeTypes::DivNode(d) => d.a.substitute(var_vals).floordiv(d.b.as_ref().clone(), true),
            NodeTypes::ModNode(m) => {
                match m.b.as_ref() {
                    BTypes::Int(i) => m.a.substitute(var_vals) % *i,
                    BTypes::Node(n) => m.a.substitute(var_vals) % n.clone(),
                }
            }
            NodeTypes::SumNode(s) => Self::sum(s.nodes.iter().map(|n| n.substitute(var_vals)).collect()),
            NodeTypes::AndNode(a) => Self::ands(&a.nodes.iter().map(|n| n.substitute(var_vals)).collect()),
        }
    }
    fn unbind(&self) -> (NodeTypes, Option<f64>) {
        match &self{
            NodeTypes::Variable(v) =>{
                assert!(v._val.is_some());
                (Variable::init(v.expr.as_str(), v.min, v.max), v._val)
            }
            _ => {

                let mut map = HashMap::new();
                self.vars().into_iter().for_each(|v|{
                    match v{
                        NodeTypes::Variable(var) => {
                            if var._val.is_some(){
                                map.insert(v.clone(), v.unbind().0).unwrap();
                            }
                        }

                        _ =>{}
                    }
                });
                (self.substitute(&map), None)
            }

        }
    }

    fn key(&self) -> String {
        self.render(&None, &Some("DEBUG"))
    }
    fn hash(&self) -> u64 {
        let mut s= DefaultHasher::new();
        self.key().hash(&mut s);
        s.finish()
    }

    fn sum(nodes: Vec<NodeTypes>) -> NodeTypes {
        let mut nd = vec![];
        nodes.into_iter().for_each(|n|{
            if !n.max().is_nan() || !n.min().is_nan(){
                nd.push(n);
            }
        });

        if nd.is_empty(){
            return NumNode::init(0.0)
        }
        if nd.len() == 1{
            return nd[0].clone()
        }
        let mut mul_groups: HashMap<NodeTypes, BTypes> = HashMap::new();
        let mut num_node_sum = 0.0;

        for nodes in &NodeTypes::new_sum(nd).flat_components(){
            match nodes{
                NodeTypes::NumNode(n) => {
                    num_node_sum += n.b;
                }
                NodeTypes::MulNode(n) =>{
                    match n.b.as_ref(){
                        BTypes::Node(b_n)=>{
                            let getter = mul_groups.get(&n.a.clone()).unwrap_or_else(|| &BTypes::Int(0.0));
                            match getter{
                                BTypes::Int(i) => {
                                    mul_groups.insert(n.a.as_ref().clone(), BTypes::Node(i.clone() + b_n.clone()));
                                }
                                BTypes::Node(bb_N) => {
                                    mul_groups.insert(n.a.as_ref().clone(), BTypes::Node(bb_N.clone() + b_n.clone()));
                                }
                            }
                        }
                        BTypes::Int(i) => {
                            let getter = mul_groups.get(&n.a.clone()).unwrap_or_else(|| &BTypes::Int(0.0));
                            match getter{
                                BTypes::Int(ii) => {
                                    mul_groups.insert(n.a.as_ref().clone(), BTypes::Int(ii + i.clone()));
                                }
                                BTypes::Node(bb_N) => {
                                    mul_groups.insert(n.a.as_ref().clone(), BTypes::Node(bb_N.clone() + i.clone()));
                                }
                            }
                        }
                    }
                    
                }

                _ => {
                    let getter = mul_groups.get(&nodes).unwrap_or_else(||&BTypes::Int(0.0));

                    match getter{
                        BTypes::Int(i) => {
                            mul_groups.insert(nodes.clone(), BTypes::Int(i + 1.0));
                        }
                        BTypes::Node(n) => {
                            mul_groups.insert(nodes.clone(), BTypes::Node(n.clone() + 1.0));
                        }
                    }
                }
            }
        }
        let mut new_nodes= vec![];

        mul_groups.into_iter().for_each(|(a, b_sum)|{
            match &b_sum{
                BTypes::Int(i) =>{
                    if i.clone() != 0.0{
                        if i.clone() != 1.0{
                            new_nodes.push(NodeTypes::new_mul(a, b_sum.to_owned()));
                        } else{
                            new_nodes.push(a);
                        }
                    }
                }
                BTypes::Node(n) => {
                    if n.clone() != 0.0{
                        if n.clone() != 1.0{
                            new_nodes.push(NodeTypes::new_mul(a, b_sum.to_owned()));
                        } else{
                            new_nodes.push(a.to_owned());
                        }
                    }
                }
            }
        });

        if !num_node_sum.is_nan(){
            new_nodes.push(NumNode::init(num_node_sum));
        }

        if new_nodes.len() > 1{
            return create_node(NodeTypes::new_sum(new_nodes))
        } else if new_nodes.len() == 1{
            return create_node(new_nodes[0].to_owned())
        } else{
            return create_node(NumNode::init(0.0))
        }
        
    }

    fn ands(nodes: &Vec<NodeTypes>) -> NodeTypes {
        if nodes.is_empty(){
            return NumNode::init(0.0)
        }
        if nodes.len() == 0{
            return nodes[0].to_owned();
        }
        let mut tmp = vec![];
        nodes.into_iter().map(|n|{
            if n.min() != n.max(){
                tmp.push(n.to_owned());
            }
        });

        if tmp.len() > 1{
            return create_node(NodeTypes::new_and(tmp))
        } else if tmp.len() == 1{
            create_node(tmp[0].to_owned())
        } else{
            create_node(NumNode::init(1.0))
        }
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

// impl PartialOrd<NodeTypes> for NodeTypes{
//     fn le(&self, other: &Self) -> bool {
//         self < &(other.clone() + 1.0)
//     }
//     fn gt(&self, other: &NodeTypes) -> bool {
//         (-self.clone()) < (-other.clone())
//     }
//     fn ge(&self, other: &NodeTypes) -> bool {
//         (-self.clone()) < (-other.clone() + 1.0)
//     }
//     fn lt(&self, other: &NodeTypes) -> bool {
//         match self{
//             NodeTypes::MulNode(n) => {
//                 self < other
//             }
//             _ =>{
//                 create_node(Self::new_lt(self.clone(), BTypes::Node(other.clone()))).into()
//             }
//         }
        
//     }
//     fn partial_cmp(&self, other: &NodeTypes) -> Option<std::cmp::Ordering> {
//         if self < other {
//             Some(std::cmp::Ordering::Less)
//         } else if self > other {
//             Some(std::cmp::Ordering::Greater)
//         } else if self == other {
//             Some(std::cmp::Ordering::Equal)
//         } else {
//             None
//         }
//     }
// }
impl PartialEq<f64> for NodeTypes{
    fn eq(&self, other: &f64) -> bool {
        NumNode::init(other.clone()) == self.clone()
    }
}
impl Eq for NodeTypes{}
// impl PartialOrd<f64> for NodeTypes{
//     fn lt(&self, other: &f64) -> bool {
//         match self{
//             _ => {create_node(Self::new_lt(self.clone(), BTypes::Int(*other))).into()}
//             NodeTypes::MulNode(n) => {
//                 match *n.b.clone(){
//                     BTypes::Node(n_b) => {
//                         self < other
//                     }
//                     BTypes::Int(i) => {
//                         if i == -1.0{
//                             return self < other
//                         }
//                         let mut sgn = 0.0;
//                         if i > 0.0 {
//                             sgn = 1.0;
//                         } else{
//                             sgn = -1.0;
//                         }
//                         (*n.a.clone()*sgn) < ((other + i.abs() - 1.0)/i.abs()).floor()
//                     }
//                 }
//             }
//         }
//     }
//     fn partial_cmp(&self, other: &f64) -> Option<std::cmp::Ordering> {
//         if self < other {
//             Some(std::cmp::Ordering::Less)
//         } else if self > other {
//             Some(std::cmp::Ordering::Greater)
//         } else if self == other {
//             Some(std::cmp::Ordering::Equal)
//         } else {
//             None
//         }
//     }
// }

impl Mul<NodeTypes> for NodeTypes{
    type Output = NodeTypes;

    fn mul(self, rhs: Self) -> Self::Output {
        match &self{

            NodeTypes::NumNode(n) => {
                rhs*n.b
            }
            NodeTypes::MulNode(n) => {
                match n.b.as_ref(){
                    BTypes::Int(i) => {
                        n.a.as_ref().to_owned()*(i.to_owned()*rhs)
                    }
                    BTypes::Node(nn) => {
                        n.a.as_ref().to_owned()*(nn.to_owned()*rhs)
                    }
                }
            }
            NodeTypes::SumNode(s) => {
                NodeTypes::sum(s.nodes.iter().map(|n| n.clone()*rhs.borrow().to_owned()).collect::<Vec<NodeTypes>>())
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
                match n.b.as_ref(){
                    BTypes::Int(i) => {
                        n.a.as_ref().to_owned()*(i*rhs)
                    }
                    BTypes::Node(nn) => {
                        n.a.as_ref().to_owned()*(nn.to_owned()*rhs)
                    }
                }
            }
            NodeTypes::SumNode(s) => {
                NodeTypes::sum(s.nodes.iter().map(|n| n.clone()*rhs).collect::<Vec<NodeTypes>>())
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
        match &rhs{
            NodeTypes::NumNode(n) => {
                self % n.b
            },
            NodeTypes::MulNode(n) =>{
                match n.b.as_ref(){
                    BTypes::Int(i) => {
                        n.a.as_ref().to_owned() * (i.to_owned() % rhs.clone()) % rhs.to_owned()
                    }
                    BTypes::Node(i) => {
                        n.a.as_ref().to_owned() * (i.to_owned() % rhs.clone()) % rhs.to_owned()
                    }
                }
            },
            NodeTypes::ModNode(_) => {
                self % rhs.to_owned()
            }
            NodeTypes::SumNode(s) => {
                if self.clone() ==rhs.clone() {return NumNode::init(0.0)}
                if (rhs.clone()-self.clone()).min() > 0.0{return self.to_owned()}
                let mut result = vec![];
                s.nodes.iter().for_each(|n|{
                    match n{
                        NodeTypes::NumNode(_) => {
                            result.push(n.clone()%rhs.to_owned());
                        }
                        NodeTypes::MulNode(_) => {
                            result.push(n.clone()%rhs.to_owned());
                        }
                        _ => {
                            result.push(n.clone());
                        }
                    }
                });
                NodeTypes::sum(result) % rhs
            }
             _ =>{
                if self == rhs{return NumNode::init(0.0)}
                if (rhs.clone() - self.clone()).min() > 0.0 && self.min() >= 0.0 {self.to_owned()}
                else{
                    panic!("Not supported: {} % {}", self.borrow(), rhs.borrow());
                }
            }
        }
    }
}

impl std::ops::Rem<f64> for NodeTypes{
    type Output = NodeTypes;

    fn rem(self, rhs: f64) -> Self::Output {

        match &self{
            NodeTypes::MulNode(n) =>{
                match n.b.as_ref(){
                    BTypes::Int(i) => {
                        n.a.as_ref().to_owned() * (i % rhs) % rhs
                    }
                    BTypes::Node(i) => {
                        n.a.as_ref().to_owned() * (i.to_owned() % rhs.clone()) % rhs
                    }
                }
            }
            NodeTypes::SumNode(s) => {
                if self ==rhs {return NumNode::init(0.0)}
                let mut result = vec![];
                s.nodes.iter().for_each(|n|{
                    match n{
                        NodeTypes::NumNode(_) => {
                            result.push(n.clone()%rhs);
                        }
                        NodeTypes::MulNode(_) => {
                            result.push(n.clone()%rhs);
                        }
                        _ => {
                            result.push(n.clone());
                        }
                    }
                });
                NodeTypes::sum(result) % rhs
            }

            NodeTypes::ModNode(n) => {
                match n.b.as_ref(){
                    BTypes::Int(i) => {
                        if i%rhs == 0.0{
                            n.a.as_ref().to_owned()%rhs
                        } else{
                            self % rhs
                        }
                    }
                    _ => {
                        self % rhs
                    }
                }
            },
            _=>{
                assert!(rhs > 0.0);

                if rhs.clone() == 1.0{return NumNode::init(0.0)}
        
                if self.min() >=0.0 && self.max() < rhs.clone(){return self.clone() - (rhs.clone()*((self.min()/rhs.to_owned()).floor()))}
                if self.min() < 0.0{return (self.clone() - (((self.min()/rhs.clone()).floor())*rhs.clone())) % rhs.to_owned()}
        
                create_node(NodeTypes::new_mod(self, BTypes::Int(rhs)))
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

fn sym_render (a: &BTypes, ops:&Option<Rc<dyn Any>>, ctx: Option<&str>) -> String{
    match &a{
        BTypes::Int(i) => {
            format!("{}", i)
        }
        BTypes::Node(n) => {
            n.render(ops, &ctx)
        }
    }
}

fn render_mulnode(m: &OpNode, ops: &Option<Rc<dyn Any>>, ctx: &Option<&str> ) -> String{
    match m.a.as_ref(){
        NodeTypes::Variable(v_a) => {
            match m.b.as_ref(){
                BTypes::Node(n) => {
                    match n{
                        NodeTypes::Variable(v_b) => {
                            if v_b.expr < v_a.expr{
                                return format!("{} * {}", sym_render(&*m.b, &ops, *ctx), m.a.render(&ops, ctx));
                            } else{
                                return format!("{} * {}", m.a.render(&ops, ctx), sym_render(&*m.b, &ops, *ctx));
                            }
                        }
                        _ => {
                            return format!("{} * {}", m.a.render(&ops, ctx), sym_render(&*m.b, &ops, *ctx));
                        }
                    }
                }
                _ => {
                    return format!("{} * {}", m.a.render(&ops, ctx), sym_render(&*m.b, &ops, *ctx));
                }
            }
        }
        _ =>{
            return format!("{} * {}", m.a.render(&ops, ctx), sym_render(&*m.b, &ops, *ctx));
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
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::LtNode(OpNode{a: Rc::new(a.clone()), b: Rc::new(b.clone()), min: 0.0, max: 0.0}));

        NodeTypes::LtNode(OpNode{a: a.into(), b: Rc::new(b), min, max})
    }
    fn new_mul(a: NodeTypes, b: BTypes) -> NodeTypes{
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::MulNode(OpNode{a: Rc::new(a.clone()), b: Rc::new(b.clone()), min: 0.0, max: 0.0}));

        NodeTypes::MulNode(OpNode{a: a.into(), b: Rc::new(b), min, max})
    }
    fn new_div(a: NodeTypes, b: BTypes) -> NodeTypes{
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::DivNode(OpNode{a: Rc::new(a.clone()), b: Rc::new(b.clone()), min: 0.0, max: 0.0}));

        NodeTypes::DivNode(OpNode{a: a.into(), b: Rc::new(b), min, max})
    }
    fn new_mod(a: NodeTypes, b: BTypes) -> NodeTypes{
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::ModNode(OpNode{a: Rc::new(a.clone()), b: Rc::new(b.clone()), min: 0.0, max: 0.0}));

        NodeTypes::ModNode(OpNode{a: Rc::new(a), b: Rc::new(b), min, max})
    }

    fn new_sum(a: Vec<NodeTypes>) -> NodeTypes{
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::SumNode(RedNode{nodes: a.clone(), min: 0.0, max: 0.0}));

        NodeTypes::SumNode(RedNode{nodes: a, min, max})
    }
    fn new_and(a: Vec<NodeTypes>) -> NodeTypes{
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::AndNode(RedNode{nodes: a.clone(), min: 0.0, max: 0.0}));

        NodeTypes::AndNode(RedNode{nodes: a, min, max})
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
                match n.b.as_ref(){
                    BTypes::Int(i) => {
                            match &b{
                                BTypes::Int(ii) => {
                                    if i % ii == 0.0{
                                        n.a.as_ref().to_owned()*((i/ii).floor())
                                    } else if ii % i == 0.0 && i > &0.0{
                                        n.a.floordiv(BTypes::Int((ii.to_owned()/i.to_owned()).floor()), true)
                                    } else{
                                        self.floordiv(b, factoring_allowed)
                                    }
                                }
                                BTypes::Node(nn) => {
                                    if i.clone() % nn.clone() == 0.0{
                                        n.a.as_ref().to_owned()*(NodeTypes::rfloordiv(i.to_owned(), b))
                                    } else if nn.clone() % i.clone() == 0.0 && i > &0.0{
                                        n.a.floordiv(BTypes::Node(nn.floordiv(n.b.as_ref().to_owned(), factoring_allowed)), true)
                                    } else{
                                        self.floordiv(b.to_owned(), factoring_allowed)
                                    }
                                }
                            }
                    }
                    BTypes::Node(i) => {
                        match &b{
                            BTypes::Int(ii) => {
                                if i.clone() % ii.to_owned() == 0.0{
                                    n.a.as_ref().to_owned()*(i.floordiv(b, true))
                                } else if ii.clone() % i.clone() == 0.0 && i.n2i_gt(&0.0).into(){
                                    n.a.floordiv(BTypes::Node(NodeTypes::rfloordiv(ii.to_owned(), n.b.as_ref().to_owned())), true)
                                } else{
                                    self.floordiv(b.to_owned(), factoring_allowed)
                                }
                            }
                            BTypes::Node(nn) => {
                                if i.to_owned() % nn.clone() == 0.0{
                                    n.a.as_ref().to_owned()*(i.floordiv(b, true))
                                } else if nn.clone() % i.clone() == 0.0 && i.n2i_gt(&0.0).into(){
                                    n.a.floordiv(BTypes::Node(nn.floordiv(n.b.as_ref().to_owned(), factoring_allowed)), true)
                                } else{
                                    self.floordiv(b.to_owned(), factoring_allowed)
                                }
                            }
                        }
                }
                }
            }
            NodeTypes::DivNode(n) => {
                match n.b.as_ref(){
                    BTypes::Int(i) => {
                        match &b{
                            BTypes::Int(ii) => {
                                n.a.floordiv(BTypes::Int(ii*i), factoring_allowed)
                            }
                            BTypes::Node(ii) => {
                                n.a.floordiv(BTypes::Node(ii.to_owned()*i.to_owned()), factoring_allowed)
                            }
                        }
                    }
                    BTypes::Node(i) => {
                        match &b{
                            BTypes::Int(ii) => {
                                n.a.floordiv(BTypes::Node(ii.to_owned()*i.to_owned()), factoring_allowed)
                            }
                            BTypes::Node(ii) => {
                                n.a.floordiv(BTypes::Node(ii.to_owned()*i.to_owned()), factoring_allowed)
                            }
                        }
                    }
                }
            }
            NodeTypes::ModNode(n) => {
                match &b{
                    BTypes::Int(i) =>{
                        match n.b.as_ref(){
                            BTypes::Int(ii) => {
                                if ii % i == 0.0{
                                    n.a.floordiv(b.to_owned(), true) % (ii/i).floor()
                                }
                                else{
                                    self.floordiv(b, factoring_allowed)
                                }
                            }
                            BTypes::Node(nn) => {
                                if nn.to_owned() % i.to_owned() == 0.0{
                                    n.a.floordiv(b.to_owned(), factoring_allowed) % nn.floordiv(b, true)
                                }
                                else{
                                    self.floordiv(b, factoring_allowed)
                                }
                            }
                        }
                    }
                    BTypes::Node(i) =>{
                        match n.b.as_ref(){
                            BTypes::Int(ii) => {
                                if ii.clone() % i.to_owned() == 0.0{
                                    n.a.floordiv(b.to_owned(), factoring_allowed) % NodeTypes::rfloordiv(ii.to_owned(), b)
                                }
                                else{
                                    self.floordiv(b.to_owned(), factoring_allowed)
                                }
                            }
                            BTypes::Node(nn) => {
                                if nn.to_owned() % i.to_owned() == 0.0{
                                    n.a.floordiv(b.clone(), factoring_allowed) % nn.floordiv(b, factoring_allowed)
                                }
                                else{
                                    self.floordiv(b, factoring_allowed)
                                }
                            }
                        }
                    }  
                }
            }
            NodeTypes::SumNode(_) => {
                match &b{
                    BTypes::Int(i) => {
                        if self.clone() == i.clone() {return NumNode::init(1.0);}
                        let mut fully_divided:Vec<NodeTypes> = vec![];

                        let mut rest:Vec<NodeTypes> = vec![];
                        if i.clone() == 1.0{
                            return self.to_owned()
                        } if !factoring_allowed{
                            return self.deref().floordiv(b.to_owned(), factoring_allowed)
                        }
                        let mut _gcd = i.clone();
                        let mut divisor = 1.0;
                        self.flat_components().iter().map(|n| {
                            match n{
                                NodeTypes::NumNode(x) => {
                                    if x.b%i.clone() == 0.0{
                                        fully_divided.push(n.floordiv(b.to_owned(), factoring_allowed));
                                    } else{
                                        rest.push(n.to_owned());
                                            _gcd = num::integer::gcd(_gcd.clone().floor().to_isize().unwrap(), x.b.clone().floor().to_isize().unwrap()).to_f64().unwrap();
                                    }
                                }
                                NodeTypes::MulNode(x) => {
                                    match x.b.as_ref(){
                                        BTypes::Int(ii)=>{
                                            if ii%i == 0.0{
                                                fully_divided.push(n.floordiv(b.to_owned(), factoring_allowed));
                                            } else{
                                                rest.push(n.to_owned());
                                                match *x.b{
                                                    BTypes::Int(iii) => {
                                                        _gcd = num::integer::gcd(_gcd.clone().floor().to_isize().unwrap(), iii.clone().floor().to_isize().unwrap()).to_f64().unwrap();
                                                        if divisor == 1.0 && i%ii == 0.0{
                                                            divisor = iii;
                                                        }
                                                    }
                                                    _ =>{
                                                        _gcd = 1.0;
                                                    }
                                                }
                                            }
                                        }
                                        BTypes::Node(nnn) => {
                                            _gcd = 1.0;
                                        }
                                    }
                                }
                                _ => {
                                    rest.push(n.to_owned());
                                    _gcd = 1.0;
                                }
                            }
                        });
                        if _gcd > 1.0{
                            return (NodeTypes::sum(fully_divided) + NodeTypes::sum(rest).floordiv(BTypes::Int(_gcd), factoring_allowed)).floordiv(BTypes::Int((i/_gcd).floor()), factoring_allowed)
                        }
                        if divisor > 1.0{
                            return (NodeTypes::sum(fully_divided) + NodeTypes::sum(rest).floordiv(BTypes::Int(divisor), factoring_allowed)).floordiv(BTypes::Int((i/divisor).floor()), factoring_allowed)
                        } else{
                            return NodeTypes::sum(fully_divided) + NodeTypes::sum(rest).floordiv(b, factoring_allowed)
                        }
                    }
                    BTypes::Node(n_b1) => {
                        if self.clone() == n_b1.clone() {return NumNode::init(1.0);}
                        let mut fully_divided:Vec<NodeTypes> = vec![];

                        let mut rest:Vec<NodeTypes> = vec![];
                        self.flat_components().iter().map(|x|{
                            if x.clone()%n_b1.to_owned() ==0.0{
                                fully_divided.push(x.floordiv(b.clone(), factoring_allowed).to_owned());
                            }
                            else{
                                rest.push(x.to_owned());
                            }
                        });
                        let sum_fully_divided = create_node(NodeTypes::new_sum(fully_divided));
                        if sum_fully_divided != 0.0{
                            return (sum_fully_divided + create_node(NodeTypes::new_sum(rest))).floordiv(b.to_owned(), factoring_allowed)
                        }
                        return self.floordiv(b, false)
                    }
                }
            }
            _ => {
        match &b{
            BTypes::Node(n) =>{
                match n{
                    NodeTypes::NumNode(num) => {
                        self.floordiv(BTypes::Int(num.b), factoring_allowed)
                    }
                    _ =>{
                        if self == n{
                            NumNode::init(1.0)
                        } else if (n.to_owned() - self.clone()).min() > 0.0 && self.min() >= 0.0{
                            NumNode::init(0.0)
                        } else{
                            panic!("Not supported: {}, {:?}",self, b)
                        }
                    }
                }
            }
            BTypes::Int(i) => {
                assert!(i.clone() != 0.0);
                if i.clone() < 0.0{
                    return self.floordiv(BTypes::Int(-i.to_owned()), factoring_allowed)*-1.0
                }
                if i.clone() == 1.0{return self.to_owned()}

                if self.min() < 0.0{
                    let offset = self.min() / i.borrow().floor();
                    return (self.clone() + -offset.borrow()*i.to_owned()).floordiv(b, false) + offset.to_owned()
                }
                create_node(NodeTypes::new_div(self.to_owned(), b.to_owned()))
            }
        }}
    }}
    fn rfloordiv(a: f64, b: BTypes) -> NodeTypes{
        NumNode::init(a).floordiv(b, true)
    }

    fn n2n_le(&self, other: &Self) -> NodeTypes {
        self.n2n_lt(&(other.clone() + 1.0))
    }
    fn n2n_gt(&self, other: &NodeTypes) -> NodeTypes {
        (-self.clone()).n2n_lt(&(-other.clone()))
    }
    fn n2n_ge(&self, other: &NodeTypes) -> NodeTypes {
        (-self.clone()).n2n_lt(&(-other.clone() + 1.0))
    }
    fn n2n_lt(&self, other: &NodeTypes) -> NodeTypes {
        match self{
            NodeTypes::MulNode(n) => {
                self.n2n_le(&(other.clone()))
            }
            _ =>{
                create_node(Self::new_lt(self.clone(), BTypes::Node(other.clone())))
            }
        }
        
    }
    fn n2i_lt(&self, other: &f64) -> NodeTypes {
        match self{
            NodeTypes::MulNode(n) => {
                match n.b.as_ref(){
                    BTypes::Node(n_b) => {
                        self.n2i_lt(&other.clone())
                    }
                    BTypes::Int(i) => {
                        if *i == -1.0{
                            return self.n2i_lt(&(other.clone()))
                        }
                        let mut sgn = 0.0;
                        if i.to_owned() > 0.0 {
                            sgn = 1.0;
                        } else{
                            sgn = -1.0;
                        }
                        (n.a.as_ref().to_owned()*sgn).n2i_lt(&((other + i.abs() - 1.0)/i.abs()).floor())
                    }
                }
            },
            _ => {create_node(Self::new_lt(self.clone(), BTypes::Int(*other)))}

        }
    }
    fn n2i_le(&self, other: &f64) -> NodeTypes {
        self.n2i_lt(&(other.clone() + 1.0))
    }
    fn n2i_gt(&self, other: &f64) -> NodeTypes {
        (-self.clone()).n2i_lt(&(-other.clone()))
    }
    fn n2i_ge(&self, other: &f64) -> NodeTypes {
        (-self.clone()).n2i_lt(&(-other.clone() + 1.0))
    }
    fn flat_components(&self) -> Vec<NodeTypes>{
        match self{
            
            NodeTypes::SumNode(s) => {
                let mut result = vec![];
                s.nodes.iter().for_each(|x| {
                    match x{
                        NodeTypes::SumNode(s)=>{
                            result.extend(x.flat_components());
                        }
                        _ => {
                            result.push(x.clone());
                        }
                    }
                });
                result
            }
            _ => {
                panic!("Not for {}", self.clone())
            }
        }
    }
}

impl OpNodeMethods for NodeTypes{
    fn get_bounds(&self) -> (f64, f64) {
        todo!()
    }
}
trait I2Ncmp{
    fn i2n_lt(&self, other: NodeTypes) -> NodeTypes;
    fn i2n_le(&self, other: NodeTypes) -> NodeTypes;
    fn i2n_gt(&self, other: NodeTypes) -> NodeTypes;
    fn i2n_ge(&self, other: NodeTypes) -> NodeTypes;
}

impl I2Ncmp for f64{
    fn i2n_lt(&self, other: NodeTypes) -> NodeTypes {
        other.n2i_gt(&self.clone())
    }
    fn i2n_le(&self, other: NodeTypes) -> NodeTypes {
        other.n2i_gt(&(self.clone() + 1.0))
    }
    fn i2n_gt(&self, other: NodeTypes) -> NodeTypes {
        (-other.clone()).n2i_gt(&(-self.clone()))
    }
    fn i2n_ge(&self, other: NodeTypes) -> NodeTypes {
        (-other.clone()).n2i_gt(&(-self.clone() + 1.0))
    }
}