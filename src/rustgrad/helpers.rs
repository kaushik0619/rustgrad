use num_traits::{ Num, One};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::env::{temp_dir};
use std::iter::{ Product};
use std::num::NonZeroUsize;
use std::ops::Mul;
use std::usize::MAX;
use std::{env, vec};
use lazy_static::lazy_static;
use std::collections::{HashSet, HashMap};
use colored::*;
use regex::Regex;
use std::iter::Flatten;
use libloading::{Library, Symbol};
use lru::LruCache;
use std::sync::{Mutex, Arc};
use std::path::{Path};
use std::error::Error;
use std::fmt::{self};
use std::time::{Instant};
use dirs::home_dir;
use rusqlite::{Connection, params};
use std::fs;
lazy_static! {
    static ref OSX: Arc<bool> = Arc::new(cfg!(target_os = "macos"));
    static ref CI: Arc<bool> = Arc::new(env::var("CI").is_ok());
    //adjust the CACHE_LRU size dynamically
    static ref CACHE_LRU: Arc<Mutex<LruCache<Arc<String>, Arc<String>>>> = Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(MAX).unwrap())));
    
    pub static ref DEBUG: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("DEBUG", 0)));
    pub static ref IMAGE: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("IMAGE", 0)));
    pub static ref WINO: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("WINO", 0)));
    pub static ref BEAM: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("BEAM", 0)));
    pub static ref NOOPT: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("NOOPT", 0)));
    pub static ref GRAPH: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("GRAPH", 0)));

    
    pub static ref GRAPHPATH: Arc<String> =  match getenv("GRAPHPATH", "/temp/net".to_string()){
        Ok(s) => Arc::new(s),
        Err(s) => Arc::new(s)
    };


    pub static ref CACHE_DIR: Arc<String> = match getenv("XDG_CACHE_HOME", home_dir().map_or("".to_string(), |home| home.join(if *Arc::clone(&OSX) { "Library/Caches" } else { ".cache" }).to_string_lossy().into_owned())){
        Ok(s) => Arc::new(s),
        Err(s) => Arc::new(s)
    };
    pub static ref CACHEDB: Arc<String> = match getenv("CACHEDB", Path::new(Arc::clone(&CACHE_DIR).as_ref()).join("rustgrad").join("cache_db").canonicalize().expect("Failed to get abs path").to_string_lossy().to_owned().to_string()){
        Ok(s) => Arc::new(s),
        Err(s) => Arc::new(s)
    };
    pub static ref CACHELEVEL: Arc<String> = match getenv("CACHELEVEL", 2.to_string()){
        Ok(s) => Arc::new(s),
        Err(s) => Arc::new(s)
    };


    pub static ref VERSION:Arc<usize> = Arc::new(1);
    pub static ref DB_CONNECTION: Arc<Mutex<Option<Connection>>> = Arc::new(Mutex::new(None));
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
    let cache_clone = Arc::clone(&CACHE_LRU);
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

pub fn getenv(key: &str, default: String) -> Result<String, String> {
    let cache_clone = Arc::clone(&CACHE_LRU);
    let mut cache = cache_clone.lock().unwrap();

    if let Some(result) = cache.get(&Arc::new(key.to_string())) {
        return Ok(Arc::clone(result).as_str().to_string());
    }

    match env::var(key) {
        Ok(value) => {
            let result = value.parse().unwrap_or_else(|_| default.clone());
            cache.put(Arc::new(key.to_string()), Arc::new(result.clone()));
            Ok(result)
        }
        Err(_) => {
            cache.put(Arc::new(key.to_string()), Arc::new(default.clone()));
            Err(default)
        }
    }
}

pub fn temp(x: String) -> String {
    let temp_dir = temp_dir();
    let temp_path = Path::new(&temp_dir).join(x);
    temp_path.to_string_lossy().into_owned()
}

//xxxxxxxxxxxxxxxx TO BE DEPRECATED AND REPLACED BY panic! OR Result enum xxxxxxxxxxxxxxxxxxxx
#[derive(Debug)]
pub struct GraphError {
    message: Option<String>,
}

impl GraphError {
    fn new(message: Option<String>) -> Self {
        GraphError { message }
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref message) = self.message {
            write!(f, "Graph error: {}", message)
        } else {
            write!(f, "Graph error")
        }
    }
}
impl Error for GraphError {}

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

pub struct Context {
    stack: Arc<RefCell<Vec<HashMap<String, isize>>>>,
}

impl Context {
    fn new() -> Self {
        Context {
            stack: Arc::new(RefCell::new(vec![HashMap::new()])),
        }
    }

    fn enter(&self, kwargs: HashMap<String, isize>){
        let mut stack = self.stack.borrow_mut();
        stack.last_mut().unwrap().extend(kwargs.clone());

        for (k,v) in kwargs.iter(){
            stack.last_mut().unwrap().insert(k.clone(), *v);
        }

        stack.push(kwargs);
    }

    fn exit(&self){
        let mut stack = self.stack.borrow_mut();
        if let Some(undo_state) = stack.pop(){
            for (k, v) in undo_state.iter(){
                stack.last_mut().unwrap().insert(k.clone(), *v);
            }
        }
    }
}

//automatic exit when dropped can be overriden
impl <'a> Drop for Context {
    fn drop(&mut self) {
            let mut stack = self.stack.borrow_mut();
            if let Some(undo_state) = stack.pop(){
                for (k, v) in undo_state.iter(){
                    stack.last_mut().unwrap().insert(k.clone(), *v);
                }
            }
    }
}
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Context Usage xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// let context = Context::new();

// // Example of using the context
// {
//     let mut stack = context.stack.lock().unwrap();
//     stack.last_mut().unwrap().insert(String::from("example"), 42);
// }

// context.enter(HashMap::new()); // Enter the context with empty temporary state
// context.enter(HashMap::from([("key1", 10), ("key2", 20)])); // Enter the context with new temporary state

// // ... Do some work with the modified state ...

// context.exit(); 

#[derive(Debug)]
pub struct ContextVar {
    key: Arc<String>,
    value: isize,
}

impl ContextVar {
    pub fn new(key: &str, default_value: isize) -> ContextVar {
        let cache_clone = Arc::clone(&CACHE_LRU);
        let mut cache = cache_clone.lock().unwrap();

        if let Some(result) = cache.get(&Arc::new(key.to_string())){
            return ContextVar{
                key: Arc::new(key.to_string()),
                value: result.parse().unwrap_or(default_value),
            };
        }

        let result= match getenv(key, default_value.to_string()){
            Ok(value) => value.parse().unwrap_or(default_value),
            Err(_) => default_value,
        };

        cache.put(Arc::new(key.to_string()), Arc::new(result.to_string()));

        ContextVar{
            key: Arc::new(key.to_string()),
            value: result,
        }
    }
}

impl std::cmp::PartialEq for ContextVar{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl std::cmp::PartialOrd for ContextVar{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.value.cmp(&other.value))
    }
}

pub struct Timing<'a> {
    prefix: &'a str,
    on_exit: Option<Box<dyn Fn(u128) -> String>>,
    enabled: bool,
    st: Option<Instant>,
}

impl<'a> Timing<'a> {
    pub fn new(prefix: &'a str, on_exit: Option<Box<dyn Fn(u128) -> String>>, enabled: bool) -> Self {
        Timing { prefix, on_exit, enabled, st: None }
    }

    pub fn enter(&mut self){
        self.st = Some(Instant::now());
    }

    pub fn exit(&mut self){
        if let Some(start_time) = self.st {
            let elapsed_time = start_time.elapsed().as_nanos();
            if self.enabled {
                let message = format!("{}{:.2} ms", self.prefix, elapsed_time as f64 * 1e-6);
                if let Some(ref on_exit_fn) = self.on_exit {
                    print!("{}{}", message, on_exit_fn(elapsed_time));
                } else {
                    print!("{}", message);
                }
            }
        }
    }
}

// automatic when dropped can be overridden later
impl<'a> Drop for Timing<'a> {
    fn drop(&mut self) {
        if let Some(start_time) = self.st {
            let elapsed_time = start_time.elapsed().as_nanos();
            if self.enabled {
                let message = format!("{}{:.2} ms", self.prefix, elapsed_time as f64 * 1e-6);
                if let Some(ref on_exit_fn) = self.on_exit {
                    print!("{}{}", message, on_exit_fn(elapsed_time));
                } else {
                    print!("{}", message);
                }
            }
        }
    }
}

fn format_fcn(fcn: (u32, u32, u32)) -> String {
    format!("{}:{}:{}", fcn.0, fcn.1, fcn.2)
}

pub fn db_connection() -> Result<String, String>{
    {   
        if Arc::clone(&DB_CONNECTION).lock().unwrap().as_ref().is_some(){
            return Err("DB already running".to_string());
        }
    }
 
    let cache_db = Arc::clone(&CACHEDB);
    fs::create_dir_all(Path::new(cache_db.as_str()).parent().unwrap()).ok();
    let mut connection = Connection::open(cache_db.as_str()).expect("Failed to connect to db");

    if Arc::clone(&DEBUG).lock().unwrap().key.parse::<i32>().unwrap() >= 7 {
        connection.trace(Some(|i: &str| print!("{}", i)));
    }

    // let db = Arc::clone(&DB_CONNECTION);
    // let mut db_guard = db.lock().unwrap();
    // *db_guard = Some(connection);

    *(Arc::clone(&DB_CONNECTION).lock().unwrap()) = Some(connection);
    
    Ok("DB started".to_string())
}

// pub fn discache_get<T>(tabel: String, key: HashMap<String, String>, val: T) -> Option<serde_pickle>{
//     if Arc::clone(&CACHELEVEL).parse::<isize>().unwrap() == 0{
//         return None;
//     }
// }

fn diskcache_get<T>(table: &str, key: &impl serde::Serialize) -> Option<T>
where
    T: serde::de::DeserializeOwned,
{
    if Arc::clone(&CACHELEVEL).parse().unwrap() == 0 {
        return None;
    }
    let key_str = serde_pickle::to_vec(key, Default::default()).expect("Failed to serialize key");
    let key_value: serde_pickle::Value =
        serde_pickle::from_slice(&key_str, Default::default()).expect("Failed to deserialize key");

    let conn = match db_connection() {
        Ok(s) => *Arc::clone(&DB_CONNECTION).lock().unwrap(),
        Err(s) => panic!("DB CONNECTION ERR"),
    };
    let mut stmt = conn?
        .prepare(&format!(
            "SELECT val FROM {}_{} WHERE {}",
            table,
            Arc::clone(&VERSION).as_ref(),
            match &key_value {
                serde_pickle::Value::Dict(dict) => dict
                    .iter()
                    .map(|(k, _)| format!("{}=?", k))
                    .collect::<Vec<_>>()
                    .join(" AND "),
                _ => panic!("Key must be a dictionary"),
            }
        ))
        .expect("Failed to prepare SQL statement");

    let mut rows = stmt
        .query(params!(
            match &key_value {
                serde_pickle::Value::Dict(dict) => dict.iter().map(|(_, v)| v),
                _ => panic!("Key must be a dictionary"),
            }
        ))
        .expect("Failed to execute SQL query");

    if let Some(val) = rows
        .next().unwrap()?
        .map(|row| row.unwrap().get::<_, Vec<u8>>(0).map(|v| &v[..]))
    {
        return Some(serde_pickle::from_slice(val?, Default::default()).expect("Failed to deserialize value"));
    }

    None
}