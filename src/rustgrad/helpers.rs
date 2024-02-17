use num_traits::{Num, NumCast, One};
use std::iter::{ Product};
use std::num::NonZeroUsize;
use std::ops::Mul;
use std::str::FromStr;
use std::usize::MAX;
use std::{default, env, result, vec};
use lazy_static::lazy_static;
use std::collections::{HashSet, HashMap};
use colored::*;
use regex::Regex;
use std::iter::Flatten;
use libloading::{Library, Symbol};
use lru::LruCache;
use std::sync::{Mutex, Arc};
lazy_static! {
    static ref OSX:bool = cfg!(target_os = "macos");
    static ref CI: bool = env::var("CI").is_ok();
    //adjust the CACHE size dynamically
    static ref CACHE: Arc<Mutex<LruCache<Arc<String>, Arc<String>>>> = Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(MAX).unwrap())));
}

pub fn prod<T>(x: impl IntoIterator<Item = T>) -> T
where
    T: Mul<Output = T> + Product + One,
{
    x.into_iter().fold(T::one(), |acc, val| acc * val)
}

pub fn dedup<T>(x: Vec<T>) -> Vec<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    let mut set = HashSet::with_capacity(x.len());
    let mut result = Vec::with_capacity(x.len());

    for item in x.into_iter() {
        if set.insert(item.clone()) {
            result.push(item);
        }
    }

    result
}

// pub fn argfix<T>(x: Option<&Vec<T,>>) -> Option<Vec<T>> 
// where 
//     T: Clone
// {
//     match x {
//         Some(values) if values.is_empty() => None,
//         Some(values) => Some(values.to_vec()),
//         None => None,
//     }
// }

// pub fn argfix<T, C>(x:(T, C)) -> Vec<_>
// where
//     C: AsRef<[T]>,
// {
//     let mut temp = Vec::new();
//     temp.push(x);
//     temp
// }


pub fn argsort<T, F>(data: &Vec<T>, key: F) -> Vec<usize>
where
    T: PartialOrd,
    F: Fn(&T) -> usize,
{
    let mut indices: Vec<usize> = (0..data.len()).collect();

    indices.sort_by_key(|&a| key(&data[a]));

    indices
}

pub fn all_same<T: PartialEq>(items: &Vec<T>) -> bool {
    if let Some(first) = items.first() {
        items.iter().all(|x| x == first)
    } else {
        true
    }
}

pub fn all_int<T: Num>(t: &Vec<T>) -> bool {
    t.iter().all(|_| true)
}

pub fn colored(st: &str, color: Option<&str>, background: bool) -> String {
    if let Some(color_str) = color {
        let color_enum = match color_str.to_lowercase().as_str() {
            "black" => Color::Black,
            "red" => Color::Red,
            "green" => Color::Green,
            "yellow" => Color::Yellow,
            "blue" => Color::Blue,
            "magenta" => Color::Magenta,
            "cyan" => Color::Cyan,
            "white" => Color::White,
            _ => Color::White, // Default to white if the color is not recognized
        };

        let colored_st = if background {
            st.on_color(color_enum)
        } else {
            st.color(color_enum)
        };

        return colored_st.to_string();
    }

    st.to_string()
}

pub fn ansistrip(s: &str) -> String {
    let re = Regex::new(r"\x1b\[([Km]|.*?m)").unwrap();
    re.replace_all(s, "").to_string()
}

pub fn ansilen(s: &str) -> usize{
    ansistrip(s).len()
}

pub fn make_pair(x: isize, cnt: usize) -> Vec<isize> {
    vec![x; cnt]
}

pub(crate) trait DeepFlattenIteratorOf<Depth,T> {
    type DeepFlattenStruct: Iterator<Item = T>;
    fn deep_flatten(this: Self) -> Self::DeepFlattenStruct;
}


impl<I: Iterator> DeepFlattenIteratorOf<(),I::Item> for I {
    type DeepFlattenStruct = Self;
    fn deep_flatten(this: Self) -> Self::DeepFlattenStruct {
        this
    }
}

impl<Depth, I: Iterator, T> DeepFlattenIteratorOf<(Depth,),T> for I
where
    Flatten<I>: DeepFlattenIteratorOf<Depth,T>,
    I: Iterator,
    <I as Iterator>::Item: IntoIterator, 
{
    type DeepFlattenStruct = <Flatten<I> as DeepFlattenIteratorOf<Depth,T>>::DeepFlattenStruct;
    fn deep_flatten(this: Self) -> Self::DeepFlattenStruct {
        DeepFlattenIteratorOf::deep_flatten(this.flatten())
    }
}

// wrapper type to help out type inference
pub(crate) struct DeepFlattenStruct<Depth, I, T>
where
    I: DeepFlattenIteratorOf<Depth, T>,
{
    inner: I::DeepFlattenStruct,
}

pub trait DeepFlatten: Iterator + Sized {
    fn deep_flatten<Depth, T>(self) -> DeepFlattenStruct<Depth, Self, T>
    where
        Self: DeepFlattenIteratorOf<Depth, T>,
    {
        DeepFlattenStruct {
            inner: DeepFlattenIteratorOf::deep_flatten(self)
        }
    }
}
impl<I: Iterator> DeepFlatten for I {}
impl<Depth, I, T> Iterator for DeepFlattenStruct<Depth, I, T>
where
    I: DeepFlattenIteratorOf<Depth, T>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}


pub fn flatten<T, I>(l: impl IntoIterator<Item = I>) -> Vec<T>
where
    I: IntoIterator<Item = T>,
{
    l.into_iter().flat_map(|item| item.into_iter()).collect()
}

// pub fn flatten<T, F>(l: Vec<Vec<T>>) -> Vec<T>{
//     l.into_iter().flatten().collect()
// }

pub fn from_import(mod_name: &str, frm_name: &str) -> Result<extern "C" fn(), String> {
    let lib = Library::new(mod_name).map_err(|e| format!("Failed to load module: {:?}", e))?;

    unsafe {
        let symbol: Result<Symbol<extern "C" fn()>, _> = lib.get(frm_name.as_bytes());
        match symbol {
            Ok(func) => Ok(*func.into_raw()),
            Err(_) => Err(format!("Failed to find function in module: {}", frm_name)),
        }
    }
}

pub fn strip_parens(fst: &str) -> &str {
    if fst.starts_with('(') && fst.ends_with(')') {
        let inner_str = &fst[1..fst.len() - 1];
        if inner_str.find('(') <= inner_str.find(')') {
            return inner_str;
        }
    }
    fst
}

pub fn round_up(num: f64, amt: isize) -> f64 {
    (((num + amt as f64 - 1.0) / amt as f64).floor() as isize * amt) as f64
}

pub fn merge_maps<T, U>(maps: impl IntoIterator<Item = HashMap<T, U>>) -> HashMap<T, U>
where
    T: Eq + std::hash::Hash + Clone,
{
    let mut merged_map = HashMap::new();
    let mut seen_keys = HashSet::new();

    for map in maps {
        for (key, value) in map.into_iter() {
            // Check if the key is already seen
            if !seen_keys.insert(key.clone()) {
                panic!("Cannot merge, multiple values for the same key");
            }

            // Insert the key-value pair into the merged map
            merged_map.insert(key, value);
        }
    }

    merged_map
}

pub fn partition<T, F>(lst: Vec<T>, fxn: F) -> (Vec<T>, Vec<T>)
where
    T: Copy,
    F: Fn(&T) -> Option<T>,
{
    let (a, b): (Vec<T>, Vec<T>) = lst.into_iter().partition(|s| fxn(s).is_some());
    (a, b)
}


fn get_child<'a>(obj: &'a Box<dyn std::any::Any>, key: &'a str) -> Option<&'a Box<dyn std::any::Any>> {
    let mut current_obj = obj;

    for k in key.split('.') {
        if k.parse::<usize>().is_ok() {
            if let Some(inner_obj) = current_obj.downcast_ref::<HashMap<String, Box<dyn std::any::Any>>>() {
                if let Some(inner_box) = inner_obj.get(&k.to_string()) {
                    current_obj = inner_box;
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else if let Some(inner_obj) = current_obj.downcast_ref::<HashMap<String, Box<dyn std::any::Any>>>() {
            if let Some(inner_box) = inner_obj.get(k) {
                current_obj = inner_box;
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    Some(current_obj)
}

pub fn get_contraction(old_shape: &[usize], new_shape: &[usize]) -> Option<Vec<Vec<f64>>> {
    let acc_old: Vec<f64> = old_shape.iter().scan(1.0, |acc, &x| {*acc *= x as f64; Some(*acc)}).collect();
    let acc_new: Vec<f64> = new_shape.iter().scan(1.0, |acc, &x| {*acc *= x as f64; Some(*acc)}).collect();

    let split: Vec<usize> = acc_new.iter().map(|&acc| acc_old.iter().position(|&x| (x as usize) == acc as usize).unwrap_or(0) + 1).collect();

    let contraction: Vec<Vec<f64>> = (0..split.len()).map(|i| (if i == 0 { 0 } else { split[i - 1] })..split[i])
                                                        .map(|range| range.map(|idx| idx as f64).collect())
                                                        .collect();

    Some(contraction)
}

pub fn to_function_name(s: &str) -> String {
    let cache_clone = Arc::clone(&CACHE);
    let mut cache = cache_clone.lock().unwrap();

    if let Some(result) = cache.get(&Arc::new(s.to_string())) {
        return Arc::clone(result).as_str().to_string();
    }

    let result: String = ansistrip(s)
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c.to_string()
            } else {
                format!("{:02X}", c as u32)
            }
        })
        .collect();

    cache.put(Arc::new(s.to_string()), Arc::new(result.clone()));

    result
}

pub fn getenv(key: &str, default: String) -> String {
    let cache_clone = Arc::clone(&CACHE);
    let mut cache = cache_clone.lock().unwrap();

    if let Some(result) = cache.get(&Arc::new(key.to_string())) {
        return Arc::clone(result).as_str().to_string();
    }

    match env::var(key) {
        Ok(value) => {
            let result = value.parse().unwrap_or_else(|_| default.clone());
            cache.put(Arc::new(key.to_string()), Arc::new(result.clone()));
            result
        }
        Err(_) => {
            cache.put(Arc::new(key.to_string()), Arc::new(default.clone()));
            default
        }
    }
}