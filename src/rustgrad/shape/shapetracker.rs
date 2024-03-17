use std::{cell::{Ref, RefCell}, collections::HashMap, ops::{Add, Deref}, option::Iter, rc::{Rc, Weak}};

use itertools::Itertools;

use crate::rustgrad::{helpers::{merge_maps, MERGE_VIEW}, shape::sym::{BTypes, NodeMethods, NumNode}};

use super::{sym::{NodeTypes, Variable}, view::View};

fn _expr_view(view: &View, idxs: Vec<Rc<NodeTypes>>, valid: Option<Rc<NodeTypes>>) -> (Rc<NodeTypes>, Rc<NodeTypes>){
    assert!(idxs.len() == view.shape.len(), "{}", format!("need an idx for all dimentions {:?} vs {:?}", idxs, view.shape));
    let mut iexpr = {
        match &view.offset{
            BTypes::Int(i) => vec![NumNode::init(i)],
            BTypes::Node(n) => vec![n.clone()]
        }
    };

    let mut vexpr = {
        if let Some(v) = valid{
            vec![v]
        }else{
            vec![]
        }
    };

    let mm = {
        if let Some(v) = &view.mask{
            v.clone()
        }else{
            vec![]
        }
    };
    idxs.iter().zip(view.shape.iter()).zip(view.strides.iter()).zip(mm.iter()).for_each(|(((idx, sh), st), m)|{
        if sh != &BTypes::Int(1.0) && st != &BTypes::Int(0.0){
            iexpr.push(idx.clone().deref() * st)
        } else if !mm.is_empty(){
            vexpr.push(NodeTypes::new_ge(idx.clone(), m.0.clone()));
            vexpr.push(NodeTypes::new_ge(idx.clone(), m.1.clone()))
        }
    });
    return (NodeTypes::sum(&iexpr), NodeTypes::ands(&vexpr))
}

struct ShapeTracker{
    views: Vec<Rc<View>>,
    ptr: RefCell<Option<Weak<Self>>>,
}

impl Add<&ShapeTracker> for &ShapeTracker{
    type Output = Rc<ShapeTracker>;

    fn add(self, rhs: &ShapeTracker) -> Self::Output {
        let mut views = self.views.clone();
        rhs.views.iter().for_each(|v|{
            views.push(v.clone())
        });

        ShapeTracker::new(views)
    }
}

impl ShapeTracker{
    fn new(views: Vec<Rc<View>>) -> Rc<ShapeTracker>{
        let ret = Rc::new(ShapeTracker{views, ptr: RefCell::new(None)});
        ret.ptr.borrow_mut().replace(Rc::downgrade(&ret));
        ret 
    }   
    fn invert(&self, out_shape: Vec<BTypes>) -> Option<Rc<ShapeTracker>>{
        let ret = self.views.iter().rev().zip(self.views.iter().rev().skip(1).map(|x| x.shape).chain(std::iter::once(out_shape)))
        .map(|(v, s)| v.invert(&s)).collect::<Vec<_>>();
    
    // Check if all inversions were successful
    if ret.iter().all(|x| x.is_some()) {
        Some(ShapeTracker::new(ret.into_iter().map(|x| x.unwrap()).collect()).reshape(out_shape))
    } else {
        None
    }
    }

    fn contiguous(&self) -> bool{
        return self.views.len() == 1 && self.views[0].clone().deref().contiguous
    }

    fn shape(&self) -> Vec<BTypes>{
        return self.views[self.views.len() - 1].clone().deref().shape
    }

    fn size(&self) -> usize{
        return self.views[self.views.len() - 1].clone().deref().size() as usize;
    }

    fn from_shape(shape: &Vec<BTypes>) -> Rc<ShapeTracker>{
        ShapeTracker::new(vec![Rc::new(View::create(shape, None, BTypes::Int(0.0), None))])
    }

    fn real_size(&self) -> usize{
        if self.shape().contains(&BTypes::Int(0.0)){
            return 0
        }
        let (idx, valid) = self.expr_idxs(None);

        // valid can never be None tinygrad bug
        // if valid{
        //     return 0
        // }
        let mut ret = idx.max();
        if let BTypes::Node(n) = &ret.unwrap(){
            ret = n.max();
        }

        match &ret.unwrap(){
            BTypes::Int(i) => (i + 1.0) as usize,
            BTypes::Node(n) => panic!("{}", format!("ret must be integer, {:?} isnt", ret)),
        }
    }

    fn vars(&self) -> Vec<Rc<NodeTypes>>{
        let vars = self.views.iter().map(|v| v.vars()).flatten().collect_vec();

        vars
    }

    fn var_vals(&self) -> HashMap<Rc<NodeTypes>, f64>{
        merge_maps(self.vars().iter().map(|v|{
            let (x, y) = v.clone().unbind();
            let mut h = HashMap::new();
            h.insert(x, y.unwrap());
            h
        }))
    }

    fn unbind(&self) -> (Rc<ShapeTracker>, HashMap<Rc<NodeTypes>, f64>){
        let mut unbound_views = vec![];
        let mut var_vals = vec![];

        self.views.iter().for_each(|v|{
            let (x, y) = v.clone().unbind();
            unbound_views.push(Rc::new(x));
            y.into_iter().map(|(z,w)| {
                let mut h = HashMap::new();
                h.insert(z, w.unwrap());
                var_vals.push(h)
            });
        });

        return (ShapeTracker::new(unbound_views), merge_maps(var_vals.into_iter()))
    }

    fn real_strides(&self, ignore_valid: bool) -> Vec<Option<BTypes>>{
        if self.views.len() == 1 && self.views[0].mask.is_none(){
            return self.views[0].strides.clone().into_iter().map(|x| Some(x)).collect();
        }
        let idxs: Vec<Rc<NodeTypes>> = self.shape().iter().enumerate().map(|(i, s)|{
            Variable::init(format!("idx{}", i), 0.0, &(s - &BTypes::Int(1.0)))
        }).collect();

        let (idx, valid) = self.expr_idxs(Some(idxs));
        let mut ret: Vec<Option<BTypes>> = vec![None; self.views[self.views.len() - 1].shape.len()];
        let mut bad_idx_vars: Vec<Rc<NodeTypes>> = vec![];
        if let NodeTypes::SumNode(s) = &idx.clone().deref(){
            s.nodes.iter().for_each(|this_dim|{
                let (idx_maybe, stride_maybe) = {
                    if let NodeTypes::MulNode(m) = this_dim.clone().deref(){
                        (m.a.clone(), m.b.clone())
                    }else{
                        (this_dim.clone(), BTypes::Int(1.0))
                    }
                };

                if let Some(i_idxs) = {
                    if idxs.iter().any(|x|{
                        // idxs.get(idx_maybe.clone()
                        if idx_maybe.clone().deref() == x.clone().deref(){
                            return true;
                        }
                        return false;
                    }){
                        Some(idx_maybe.clone())
                    }else{
                        None
                    }
                }{
                    if let Some(i_ret) = i_idxs.checked_sub(BTypes::Int(1.0)){
                        ret[i_ret] = Some(stride_maybe)
                    }
                }else{
                    idx_maybe.vars().into_iter().for_each(|x| bad_idx_vars.push(x))
                }
            });
        }
        let idx_vars: Vec<Rc<NodeTypes>> = idx.vars();
        let valid_vars = valid.vars();

        idxs.into_iter().enumerate().for_each(|(i , tidx)|{
            if bad_idx_vars.contains(&tidx) || (valid_vars.contains(&tidx) && !ignore_valid){
                ret[i] = None
            }else if !idx_vars.contains(&tidx){
                ret[i] = Some(BTypes::Int(0.0))
            }
        });

        return ret
    }

    fn unit_stride_axes(&self, ignore_valid: bool) -> Vec<usize>{
        self.real_strides(ignore_valid).iter().enumerate().filter_map(|(i ,st)|{
            if st.unwrap() == BTypes::Int(1.0){
                return Some(i)
            }
            None
        }).collect_vec()
    }
    fn expr_idxs(&self, idxs: Option<Vec<Rc<NodeTypes>>>) -> (Rc<NodeTypes>, Rc<NodeTypes>){
        let idxs_n = {
            if let Some(idx) = idxs{
                idx
            }else{
                self.shape().iter().enumerate().map(|(i, s)|{
                    Variable::init(format!("idx{}", i), 0.0, &(s - &BTypes::Int(1.0)))
                }).collect_vec()
            }
        };

        let (mut idx, mut valid) = _expr_view(self.views[self.views.len() - 1].clone().deref(), idxs_n.clone(), None);

        for view in self.views[0.. self.views.len() - 1].iter().rev(){
            if valid.max().unwrap() == BTypes::Int(0.0){
                return (NumNode::init(&-1.0), valid)
            }
            let view_n = view.minify();

            let mut acc = BTypes::Int(1.0);
            let mut idxs = vec![];

            for d in view.shape.iter().rev(){
                match d{
                    BTypes::Int(i) => {
                        idxs.push(idx.floordiv(&acc, true).deref() % i);
                    },
                    BTypes::Node(n) => {
                        idxs.push(idx.floordiv(&acc, true).deref() % n.clone().deref());
                    }
                }
                acc = &acc * d 
            }

            (idx, valid) = _expr_view(view.clone().deref(), idxs_n[..idxs_n.len() - 1].to_vec(), Some(valid));
        }

        return (idx, valid)
    }

    fn axis_is_masked(&self, axis: isize) -> bool{
        let (_, valid) = self.expr_idxs(None);
        valid.vars().into_iter().any(|v|{
            match v.deref(){
                NodeTypes::Variable(vv) => {
                    if vv.expr.clone() == format!("idx{}", axis){
                        return true
                    }
                    false
                },
                _ => false
            }
        })
    }

    fn simplify(&self) -> Rc<ShapeTracker>{
        let new_view = self.views[self.views.len() - 2].clone().deref() + self.views[self.views.len() - 1].clone().deref();
        if self.views.len() >= 2 && new_view.is_some(){
            return ShapeTracker::new({
                let mut var = self.views[..self.views.len() - 2].to_vec();
                var.push(Rc::new(new_view.unwrap()));
                var
            }).simplify()
        }
        return self.ptr.borrow().as_ref().unwrap().upgrade().unwrap()
    }


    //tinygrad bug view pad takes int while this function gives it as sint...
    fn pad(&self, arg: &Vec<(BTypes, BTypes)>) -> Rc<ShapeTracker>{
        ShapeTracker::new({
            let mut var = self.views[0..self.views.len() - 1].to_vec();
            var.push(self.views[self.views.len() - 1].pad(arg).into());
            var
        })
    }
    fn shrink(&self, arg: &Vec<(BTypes, BTypes)>) -> Rc<ShapeTracker>{
        ShapeTracker::new({
            let mut var = self.views[0..self.views.len() - 1].to_vec();
            var.push(self.views[self.views.len() - 1].shrink(arg).into());
            var
        })
    }
    fn expand(&self, new_shape: &Vec<BTypes>) -> Rc<ShapeTracker>{
        ShapeTracker::new({
            let mut var = self.views[0..self.views.len() - 1].to_vec();
            var.push(self.views[self.views.len() - 1].expand(new_shape).into());
            var
        })
    }
    fn permute(&self, axis: &Vec<isize>) -> Rc<ShapeTracker>{
        ShapeTracker::new({
            let mut var = self.views[0..self.views.len() - 1].to_vec();
            var.push(self.views[self.views.len() - 1].permute(axis).into());
            var
        })
    }

    fn stride(&self, mul: &Vec<isize>) -> Rc<ShapeTracker>{
        ShapeTracker::new({
            let mut var = self.views[0..self.views.len() - 1].to_vec();
            var.push(self.views[self.views.len() - 1].stride(mul).into());
            var
        })
    }

    fn reshape(&self, new_shape: &Vec<BTypes>) -> Rc<ShapeTracker>{
        let new_view = self.views[self.views.len() - 1].reshape(new_shape);
        if MERGE_VIEW.clone().deref().lock().unwrap().deref().value == 1 && new_view.is_some(){
            return ShapeTracker::new({
                let mut var = self.views[0..self.views.len() - 1].to_vec();
                var.push(Rc::new(new_view.unwrap()));
                var
            });
        }
        return ShapeTracker::new({
            let mut var = self.views.clone();
            var.push(Rc::new(new_view.unwrap()));
            var
        })
    }
}