pub mod metric_computer;
pub mod normalizer;
pub mod types;
pub mod views;

use polars::lazy::dsl::{lit, when, Expr};
use polars::prelude::*;
use std::collections::VecDeque;
use types::MetricDefinition;

const OPS: [u8; 12] = [
    b'+', b'-', b'*', b'/', b'&', b'|', b'^', b'>', b'<', b'=', b'?', b':',
];
const BRACKETS: [u8; 3] = [b'(', b'[', b'{'];
const SS: usize = std::mem::size_of::<usize>();
const ONES: usize = usize::from_be_bytes([1_u8; SS]);
const HIGHS: usize = usize::from_be_bytes([128_u8; SS]);

#[macro_export]
macro_rules! map {
    ($($key:expr => $value:expr),+) => {
        {
            let mut m = std::collections::HashMap::new();
            $(
                m.insert($key.to_owned(), $value);
            )+
            m
        }
    };
}

#[macro_export]
macro_rules! define {
    (| $($var:ident),* | $f:expr) => {
        {
            let mut definitions = std::vec::Vec::new();
            $(
                let definition = $crate::redp::parsers::metrics::Definition::Event {
                    alias: stringify!($var).to_owned(),
                    name: stringify!($var).to_owned(),
                };
                definitions.push(definition);
            )*
            let formula = $crate::redp::parsers::metrics::Definition::Formula {
                socket: None,
                form: stringify!($f).replace("\n", "").into(),
            };
            definitions.push(formula);
            let metric = $crate::redp::parsers::metrics::Metric {
                name: "metric".into(),
                definition: definitions,
            };
            $crate::redp::core::types::MetricDefinition::from(metric)
        }
    };
}

#[macro_export]
macro_rules! prior {
    ( $byte:ident ) => {
        match $byte[0] {
            b'?' | b':' => (0x01u8, 1u8),
            b'>' | b'<' | b'=' => (0x02u8, 1u8),
            b'+' | b'-' | b'&' | b'|' | b'^' => (0x04u8, 1u8),
            b'*' | b'/' => match $byte[1] {
                b'*' => (0x10u8, 2u8),
                b'/' => (0x08u8, 2u8),
                _ => (0x08u8, 1u8),
            },
            _ => (0xffu8, 0u8),
        }
    };
}

pub trait SliceExt {
    fn trim(&self) -> &Self;
    fn contains_aligned(&self, x: &Self) -> bool;
}

impl SliceExt for [u8] {
    #[inline]
    fn trim(&self) -> &[u8] {
        fn is_whitespace(c: &u8) -> bool {
            *c == b'\t' || *c == b' ' || *c == b'\n'
        }

        fn is_not_whitespace(c: &u8) -> bool {
            !is_whitespace(c)
        }

        if let Some(first) = self.iter().position(is_not_whitespace) {
            if let Some(last) = self.iter().rposition(is_not_whitespace) {
                &self[first..last + 1]
            } else {
                unreachable!();
            }
        } else {
            &[]
        }
    }

    #[inline]
    fn contains_aligned(&self, x: &[u8]) -> bool {
        let (prefix, aligned, suffix) = unsafe { self.align_to::<usize>() };
        x.iter().any(|c| prefix.contains(c))
            || x.iter().any(|c| {
                aligned.iter().any(|a| {
                    let x = usize::from_be_bytes([*c; SS]) ^ a;
                    ((x as isize - ONES as isize) as usize & !x & HIGHS) != 0
                })
            })
            || x.iter().any(|c| suffix.contains(c))
    }
}

pub trait Vec2Ext {
    fn transpose(self) -> Self;
}

impl<T: Sized> Vec2Ext for Vec<Vec<T>> {
    #[inline]
    fn transpose(self) -> Self {
        assert!(!self.is_empty());
        let len = self[0].len();
        let mut iters: Vec<_> = self.into_iter().map(|x| x.into_iter()).collect();
        (0..len)
            .map(|_| {
                iters
                    .iter_mut()
                    .map(|x| x.next().unwrap())
                    .collect::<Vec<T>>()
            })
            .collect()
    }
}

pub trait Arithmetic<T, S>
where
    T: std::ops::Add + std::ops::Mul + std::ops::Sub + std::ops::Div,
    S: Clone,
{
    fn arithmetic(&self, data: S) -> T;
    fn do_arithmetic(&self, e: &[u8], data: S) -> T;
    fn ternary(&self, condition: &Token, first: &Token, second: &Token, data: S) -> T;
}

impl Arithmetic<Expr, &DataFrame> for MetricDefinition {
    fn arithmetic(&self, data: &DataFrame) -> Expr {
        let metric_name = self.name.as_str();
        let _description = self.description.as_str();

        // Check if brackets are balanced in the formula
        assert!(
            brackets_are_balanced(self.formula.as_bytes()).0,
            "{:?}: Brackets are not balanced",
            self.formula.as_str()
        );

        // Perform arithmetic operations and alias the result with the metric name
        self.do_arithmetic(self.formula.as_bytes(), data)
            .alias(metric_name)
    }

    #[inline]
    fn do_arithmetic(&self, e: &[u8], data: &DataFrame) -> Expr {
        let bc = Lexer::from(e);
        let mut collection = bc.into_iter().collect::<VecDeque<Token>>();
        let head = collection.pop_front().unwrap();
        let mut curr: Expr;

        // Match the head element to determine the type of operation
        match head {
            Token::Expr(v) => {
                if !collection.is_empty()
                    && v.contains_aligned(b"><=")
                    && String::from(collection.pop_front().unwrap()) == "?"
                {
                    // Handle ternary operation
                    let first = collection.pop_front().unwrap();
                    collection.pop_front();
                    let second = collection.pop_front().unwrap();
                    curr = self.ternary(&head, &first, &second, data);
                } else {
                    // Recursively handle nested arithmetic expressions
                    curr = self.do_arithmetic(v, data);
                }
            }
            Token::Ident(v, _) => {
                // Handle constants and column references
                let key = std::str::from_utf8(v).unwrap();
                let try_extract = data.column(
                    self.event_aliases
                        .get(key)
                        .unwrap_or_else(|| self.constants.get(key).unwrap())
                        .as_str(),
                );
                curr = match try_extract {
                    Ok(v) => lit(v.to_owned()).alias("values"),
                    Err(_) => lit(self.constants.get(key).unwrap().parse::<f32>().unwrap())
                        .alias("values"),
                };
            }
            Token::Literal(v) => {
                let literal = std::str::from_utf8(v).unwrap();
                curr = lit(literal.parse::<f32>().unwrap()).alias("values");
            }
            Token::Max(v) => {
                // Handle max operation
                curr = v.reduce(self.do_arithmetic(v.a, data), self.do_arithmetic(v.b, data));
            }
            Token::Min(v) => {
                // Handle min operation
                curr = v.reduce(self.do_arithmetic(v.a, data), self.do_arithmetic(v.b, data));
            }
            Token::Op(v) => {
                // Handle negative head
                if v == [b'-'] {
                    curr = lit(0).alias("values");
                    collection.push_front(head);
                } else {
                    unreachable!();
                }
            }
        }

        if collection.is_empty() {
            return curr;
        }

        let mut next: Expr;
        let mut i = 0;
        while i + 1 < collection.len() {
            let op = &collection[i];
            let next_raw = &collection[i + 1];

            // Match the next element to determine the type of operation
            match next_raw {
                Token::Expr(v) => {
                    next = self.do_arithmetic(v, data);
                }
                Token::Ident(v, _) => {
                    let key = std::str::from_utf8(v).unwrap();
                    let try_extract = data.column(
                        self.event_aliases
                            .get(key)
                            .unwrap_or_else(|| self.constants.get(key).unwrap())
                            .as_str(),
                    );
                    next = match try_extract {
                        Ok(v) => lit(v.to_owned()).alias("values"),
                        Err(_) => lit(self.constants.get(key).unwrap().parse::<f32>().unwrap())
                            .alias("values"),
                    };
                }
                Token::Literal(v) => {
                    let literal = std::str::from_utf8(v).unwrap();
                    next = lit(literal.parse::<f32>().unwrap()).alias("values");
                }
                Token::Max(v) => {
                    // Handle max operation
                    next = v.reduce(self.do_arithmetic(v.a, data), self.do_arithmetic(v.b, data));
                }
                Token::Min(v) => {
                    // Handle min operation
                    next = v.reduce(self.do_arithmetic(v.a, data), self.do_arithmetic(v.b, data));
                }
                Token::Op(_v) => {
                    unimplemented!();
                }
            }

            // Perform arithmetic operation based on the operator
            match op {
                Token::Op(v) => match v[..] {
                    [b'+'] => {
                        curr = curr + next;
                    }
                    [b'-'] => {
                        curr = curr - next;
                    }
                    [b'*'] => {
                        curr = curr * next;
                    }
                    [b'/'] => {
                        curr = curr / next;
                    }
                    [b'&'] => {
                        curr = curr
                            .fill_nan(0)
                            .strict_cast(DataType::UInt64)
                            .and(next.fill_nan(0).strict_cast(DataType::UInt64))
                            .strict_cast(DataType::Float64);
                    }
                    [b'|'] => {
                        curr = curr
                            .fill_nan(0)
                            .strict_cast(DataType::UInt64)
                            .or(next.fill_nan(0).strict_cast(DataType::UInt64))
                            .strict_cast(DataType::Float64);
                    }
                    [b'^'] => {
                        curr = curr
                            .fill_nan(0)
                            .strict_cast(DataType::UInt64)
                            .xor(next.fill_nan(0).strict_cast(DataType::UInt64))
                            .strict_cast(DataType::Float64);
                    }
                    [b'*', b'*'] => {
                        curr = curr.pow(next);
                    }
                    [b'/', b'/'] => {
                        curr = curr.floor_div(next);
                    }
                    _ => {
                        unreachable!();
                    }
                },
                _ => unreachable!(),
            }
            i += 2;
        }
        curr
    }

    #[inline]
    fn ternary(&self, condition: &Token, first: &Token, second: &Token, data: &DataFrame) -> Expr {
        // Evaluate the first expression in the ternary operation
        let first = match first {
            Token::Expr(v) => self.do_arithmetic(v, data),
            Token::Ident(v, _) => {
                let key = std::str::from_utf8(v).unwrap();
                let try_extract = data.column(
                    self.event_aliases
                        .get(key)
                        .unwrap_or_else(|| self.constants.get(key).unwrap())
                        .as_str(),
                );
                match try_extract {
                    Ok(v) => lit(v.to_owned()).alias("values"),
                    Err(_) => lit(self.constants.get(key).unwrap().parse::<f32>().unwrap())
                        .alias("values"),
                }
            }
            Token::Literal(v) => {
                let literal = std::str::from_utf8(v).unwrap();
                lit(literal.parse::<f32>().unwrap()).alias("values")
            }
            Token::Max(v) => v.reduce(self.do_arithmetic(v.a, data), self.do_arithmetic(v.b, data)),
            Token::Min(v) => v.reduce(self.do_arithmetic(v.a, data), self.do_arithmetic(v.b, data)),
            Token::Op(_v) => {
                unreachable!()
            }
        };

        // Evaluate the second expression in the ternary operation
        let second = match second {
            Token::Expr(v) => self.do_arithmetic(v, data),
            Token::Ident(v, _) => {
                let key = std::str::from_utf8(v).unwrap();
                let try_extract = data.column(
                    self.event_aliases
                        .get(key)
                        .unwrap_or_else(|| self.constants.get(key).unwrap())
                        .as_str(),
                );
                match try_extract {
                    Ok(v) => lit(v.to_owned()).alias("values"),
                    Err(_) => lit(self.constants.get(key).unwrap().parse::<f32>().unwrap())
                        .alias("values"),
                }
            }
            Token::Literal(v) => {
                let literal = std::str::from_utf8(v).unwrap();
                lit(literal.parse::<f32>().unwrap()).alias("values")
            }
            Token::Max(v) => v.reduce(self.do_arithmetic(v.a, data), self.do_arithmetic(v.b, data)),
            Token::Min(v) => v.reduce(self.do_arithmetic(v.a, data), self.do_arithmetic(v.b, data)),
            Token::Op(_v) => {
                unreachable!()
            }
        };

        // Evaluate the condition and return the appropriate expression
        match condition {
            Token::Expr(v) => {
                let c: Vec<&[u8]> = v
                    .split_inclusive(|x| [b'>', b'<', b'='].contains(x))
                    .collect();
                assert_eq!(c.len(), 2, "Ternary expression error...");

                let (mut _left, op) = c[0].split_at(c[0].len() - 1);
                let mut _right = c[1];

                let (_, mut bracket_overflow) = brackets_are_balanced(_left);
                if bracket_overflow > 0 {
                    (_, _left) = _left.split_at(bracket_overflow);
                }
                let left = self.do_arithmetic(_left, data);

                (_, bracket_overflow) = brackets_are_balanced(c[1]);
                if bracket_overflow > 0 {
                    (_right, _) = _right.split_at(_right.len() - bracket_overflow);
                }
                let right = self.do_arithmetic(_right, data);
                match op[..] {
                    [b'>'] => when(left.gt(right)).then(first).otherwise(second),
                    [b'<'] => when(left.lt(right)).then(first).otherwise(second),
                    [b'='] => when(left.eq(right)).then(first).otherwise(second),
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }
}

enum Bracket {
    Open(u8),
    Close(u8),
}

impl Bracket {
    #[inline]
    pub fn from_u8(c: u8) -> Option<Bracket> {
        match c {
            b'{' | b'[' | b'(' => Some(Bracket::Open(c)),
            b'}' => Some(Bracket::Close(b'{')),
            b']' => Some(Bracket::Close(b'[')),
            b')' => Some(Bracket::Close(b'(')),
            _ => None,
        }
    }
}

#[inline]
pub fn brackets_are_balanced(string: &[u8]) -> (bool, usize) {
    let mut brackets: Vec<u8> = vec![];
    for c in string {
        match Bracket::from_u8(*c) {
            Some(Bracket::Open(char_bracket)) => {
                brackets.push(char_bracket);
            }
            Some(Bracket::Close(char_close_bracket)) => {
                if brackets.pop() != Some(char_close_bracket) {
                    return (false, 0);
                }
            }
            _ => {}
        };
    }
    (brackets.is_empty(), brackets.len())
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Lexer<'a> {
    full: &'a [u8],
    buf: (usize, usize),
    punct: usize,
    need_join: bool,
    potential: u8,
    next_op: u8,
}

impl<'a> Lexer<'a> {
    #[inline]
    fn new(input: &'a [u8]) -> Self {
        Lexer {
            full: input,
            buf: (0, 0),
            punct: 0,
            need_join: false,
            potential: 0xff,
            next_op: 1,
        }
    }
}

impl<'a, 'b> From<&'a [u8]> for Lexer<'b>
where
    'a: 'b,
{
    #[inline]
    fn from(value: &'a [u8]) -> Self {
        Lexer::new(value)
    }
}

#[derive(Debug, Clone)]
pub enum Token<'a> {
    Expr(&'a [u8]),
    Ident(&'a [u8], Option<usize>),
    Literal(&'a [u8]),
    Op(&'a [u8]),
    Max(Max<&'a [u8], &'a [u8]>),
    Min(Min<&'a [u8], &'a [u8]>),
}

impl PartialEq for Token<'_> {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Token::Expr(p) => match other {
                Token::Expr(q) => p[..] == q[..],
                _ => false,
            },
            Token::Max(p) => match other {
                Token::Max(q) => (p.a[..] == q.a[..]) & (p.b[..] == q.b[..]),
                _ => false,
            },
            Token::Min(p) => match other {
                Token::Min(q) => (p.a[..] == q.a[..]) & (p.b[..] == q.b[..]),
                _ => false,
            },
            Token::Ident(p, i) => match other {
                Token::Ident(q, j) => (p[..] == q[..]) & (i == j),
                _ => false,
            },
            Token::Op(p) => match other {
                Token::Op(q) => p[..] == q[..],
                _ => false,
            },
            Token::Literal(p) => match other {
                Token::Literal(q) => p[..] == q[..],
                _ => false,
            },
        }
    }
}

impl From<&Token<'_>> for String {
    #[inline]
    fn from(value: &Token) -> Self {
        match value {
            Token::Expr(v) => std::str::from_utf8(v).unwrap().to_owned(),
            Token::Op(v) => std::str::from_utf8(v).unwrap().to_owned(),
            Token::Literal(v) => std::str::from_utf8(v).unwrap().to_owned(),
            Token::Max(v) => {
                String::from("max(")
                    + std::str::from_utf8(v.a).unwrap()
                    + ", "
                    + std::str::from_utf8(v.b).unwrap()
                    + ")"
            }
            Token::Min(v) => {
                String::from("min(")
                    + std::str::from_utf8(v.a).unwrap()
                    + ", "
                    + std::str::from_utf8(v.b).unwrap()
                    + ")"
            }
            Token::Ident(v, i) => match i {
                Some(idx) => std::str::from_utf8(v).unwrap().to_owned() + idx.to_string().as_str(),
                None => std::str::from_utf8(v).unwrap().to_owned(),
            },
        }
    }
}

impl From<Token<'_>> for String {
    #[inline]
    fn from(value: Token) -> Self {
        match value {
            Token::Expr(v) => String::from_utf8(v.to_vec()).unwrap(),
            Token::Op(v) => String::from_utf8(v.to_vec()).unwrap(),
            Token::Literal(v) => std::str::from_utf8(v).unwrap().to_owned(),
            Token::Max(v) => {
                String::from("max(")
                    + std::str::from_utf8(v.a).unwrap()
                    + ", "
                    + std::str::from_utf8(v.b).unwrap()
                    + ")"
            }
            Token::Min(v) => {
                String::from("min(")
                    + std::str::from_utf8(v.a).unwrap()
                    + ", "
                    + std::str::from_utf8(v.b).unwrap()
                    + ")"
            }
            Token::Ident(v, i) => match i {
                Some(idx) => std::str::from_utf8(v).unwrap().to_owned() + idx.to_string().as_str(),
                None => std::str::from_utf8(v).unwrap().to_owned(),
            },
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut in_bracket = false;
        let mut in_square = false;
        let mut bracket_degree = 0;
        let mut square_degree = 0;
        self.buf.0 = self.buf.0.max(self.buf.1);
        self.buf.1 = self.buf.0;
        let mut i = self.buf.0;
        let mut buf_ret: &'a [u8] = &[];
        'outer: while i < self.full.len() {
            match self.full[i] {
                b'{' | b'(' => {
                    if !in_square {
                        bracket_degree += 1;
                        if !in_bracket {
                            in_bracket = true;
                        } else {
                            self.buf.1 = i + 1;
                        }
                    } else {
                        self.buf.1 = i + 1;
                    }
                }
                b'[' => {
                    if !in_bracket {
                        square_degree += 1;
                        if !in_square {
                            in_square = true;
                        } else {
                            self.buf.1 = i + 1;
                        }
                    } else {
                        self.buf.1 = i + 1;
                    }
                }
                b']' => 'state: {
                    if !in_bracket {
                        square_degree -= 1;
                        if square_degree <= 0 {
                            in_square = false;
                            if self.need_join {
                                if self.full[self.buf.0 - 1] == b'[' {
                                    self.buf.0 -= 1;
                                }
                                self.buf.1 = i + 1;

                                if i + 1 == self.full.len() {
                                    buf_ret = &self.full[self.buf.0..self.buf.1];
                                    break 'state;
                                }

                                if i + 4 < self.full.len()
                                    && (self.full[i + 2..i + 5] == [b'm', b'a', b'x']
                                        || self.full[i + 2..i + 5] == [b'm', b'i', b'n'])
                                {
                                    self.buf.1 = i + 5;
                                    if i + 5 == self.full.len() {
                                        buf_ret = &self.full[self.buf.0..self.buf.1];
                                    }
                                }
                                break 'state;
                            } else {
                                if i + 1 < self.full.len() {
                                    match self.full[i + 1] {
                                        b'.' => {
                                            if i + 5 < self.full.len()
                                                && (self.full[i + 2..i + 5] == [b'm', b'a', b'x']
                                                    || self.full[i + 2..i + 5]
                                                        == [b'm', b'i', b'n'])
                                            {
                                                let next = &self.full[i + 5..];
                                                let (next_prior, _) = prior!(next);
                                                if next_prior > self.potential {
                                                    if self.full[self.buf.0 - 1] == b'[' {
                                                        self.buf.0 -= 1;
                                                    }
                                                    self.buf.1 = i + 5;
                                                    break 'state;
                                                }
                                            }
                                        }
                                        _ => {
                                            let next = &self.full[i + 1..];
                                            let (next_prior, _) = prior!(next);
                                            if next_prior > self.potential {
                                                if self.full[self.buf.0 - 1] == b'[' {
                                                    self.buf.0 -= 1;
                                                }
                                                self.buf.1 = i + 1;
                                                break 'state;
                                            }
                                        }
                                    }
                                }
                                self.buf.1 = i;
                            }

                            // case: ident with idx
                            if self.full.len() < i + 5 {
                                let start = self.buf.0;
                                for (j, byte) in
                                    self.full.iter().enumerate().take(i).skip(self.buf.0)
                                {
                                    if byte == &b'[' {
                                        self.buf.1 = j;
                                    }
                                }
                                self.buf.0 = i + 1;
                                return Some(Token::Ident(&self.full[start..self.buf.1], unsafe {
                                    std::str::from_utf8_unchecked(&self.full[self.buf.1 + 1..i])
                                        .parse()
                                        .ok()
                                }));
                            }

                            return match self.full[i + 2..i + 5] {
                                [b'm', b'a', b'x'] => {
                                    assert!(self.buf.0 < self.punct);
                                    assert!(self.buf.1 > self.punct);
                                    let a = &self.full[self.buf.0..self.punct];
                                    let b = &self.full[self.punct + 1..self.buf.1];
                                    self.buf.0 = i + 5;
                                    Some(Token::Max(Max { a, b }))
                                }
                                [b'm', b'i', b'n'] => {
                                    assert!(self.buf.0 < self.punct);
                                    assert!(self.buf.1 > self.punct);
                                    let a = &self.full[self.buf.0..self.punct];
                                    let b = &self.full[self.punct + 1..self.buf.1];
                                    self.buf.0 = i + 5;
                                    Some(Token::Min(Min { a, b }))
                                }
                                // case: ident with idx
                                _ => {
                                    let start = self.buf.0;
                                    for (j, byte) in
                                        self.full.iter().enumerate().take(i).skip(self.buf.0)
                                    {
                                        if byte == &b'[' {
                                            self.buf.1 = j;
                                        }
                                    }
                                    self.buf.0 = i + 1;
                                    Some(Token::Ident(
                                        &self.full[start..self.buf.1],
                                        unsafe {
                                            std::str::from_utf8_unchecked(
                                                &self.full[self.buf.1 + 1..i],
                                            )
                                                .parse()
                                                .ok()
                                        },
                                    ))
                                }
                            }
                        } else {
                            self.buf.1 = i + 1;
                        }
                    } else {
                        self.buf.1 = i + 1;
                    }
                }
                b'}' | b')' => {
                    if !in_square {
                        bracket_degree -= 1;
                        if bracket_degree <= 0 {
                            in_bracket = false;
                            if self.need_join {
                                if self.full[self.buf.0 - 1] == b'(' {
                                    self.buf.0 -= 1;
                                }
                                self.buf.1 = i + 1;
                            } else {
                                if i + 1 < self.full.len() {
                                    let next = &self.full[i + 1..];
                                    let (next_prior, _) = prior!(next);
                                    if next_prior > self.potential
                                        && self.full[self.buf.0 - 1] == b'('
                                    {
                                        self.buf.0 -= 1;
                                    }
                                }
                                self.buf.1 = i;
                            }
                            if i == self.full.len() - 1 {
                                buf_ret = &self.full[self.buf.0..self.buf.1];
                            }
                        } else {
                            self.buf.1 = i + 1;
                        }
                    } else {
                        self.buf.1 = i + 1;
                    }
                }
                _ => {
                    if !in_bracket && !in_square {
                        match self.full[i] {
                            b'?' | b':' | b'>' | b'<' | b'=' | b'+' | b'-' | b'&' | b'|' | b'^'
                            | b'*' | b'/' => {
                                let op = &self.full[i..];
                                let (priority, n_byte) = prior!(op);
                                self.next_op = n_byte;
                                self.need_join = priority > self.potential;
                                if self.buf.0 == self.buf.1 {
                                    self.potential = self.potential.min(priority);
                                    self.buf.1 = i + n_byte as usize;
                                    buf_ret = &self.full[self.buf.0..self.buf.1];
                                    self.buf.0 = self.buf.1;
                                    break 'outer;
                                }
                                if !self.need_join {
                                    buf_ret = &self.full[self.buf.0..self.buf.1];
                                    self.buf.0 = i;
                                    break 'outer;
                                } else {
                                    self.buf.1 = i + self.next_op as usize;
                                    if self.next_op as usize > 1 {
                                        i += self.next_op as usize - 1;
                                    }
                                }
                            }
                            _ => {
                                self.buf.1 = i + 1;
                                if i == self.full.len() - 1 {
                                    buf_ret = &self.full[self.buf.0..self.buf.1];
                                }
                            }
                        }
                    } else if self.full[i] == b',' && square_degree == 1 {
                        self.punct = i;
                        self.buf.1 = i + 1;
                    } else {
                        self.buf.1 = i + 1;
                    }
                }
            }
            i += 1;
            if self.buf.0 >= self.buf.1 {
                self.buf.0 += 1;
            }
        }
        match buf_ret {
            &[] => None,
            &[b'+']
            | &[b'-']
            | &[b'*']
            | &[b'/']
            | &[b'&']
            | &[b'|']
            | &[b'^']
            | &[b'?']
            | &[b':']
            | &[b'>']
            | &[b'<']
            | &[b'=']
            | &[b'/', b'/']
            | &[b'*', b'*'] => Some(Token::Op(buf_ret)),
            _ => {
                if buf_ret.contains_aligned(&OPS) || buf_ret.contains_aligned(&BRACKETS) {
                    Some(Token::Expr(buf_ret))
                } else {
                    match buf_ret[0] {
                        b'0' | b'1' | b'2' | b'3' | b'4' | b'5' | b'6' | b'7' | b'8' | b'9' => {
                            Some(Token::Literal(buf_ret))
                        }
                        _ => Some(Token::Ident(buf_ret, None)),
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Max<T, S> {
    a: T,
    b: S,
}

#[derive(Debug, Clone)]
pub struct Min<T, S> {
    a: T,
    b: S,
}

trait Reducer<T> {
    fn reduce(&self, candidate_a: T, candidate_b: T) -> T;
}

impl Reducer<Expr> for Max<&[u8], &[u8]> {
    #[inline]
    fn reduce(&self, candidate_a: Expr, candidate_b: Expr) -> Expr {
        when(
            candidate_a
                .clone()
                .is_null()
                .or(candidate_a.clone().gt(candidate_b.clone())),
        )
        .then(candidate_a)
        .otherwise(candidate_b)
    }
}

impl Reducer<Expr> for Min<&[u8], &[u8]> {
    #[inline]
    fn reduce(&self, candidate_a: Expr, candidate_b: Expr) -> Expr {
        when(
            candidate_a
                .clone()
                .is_null()
                .or(candidate_a.clone().lt(candidate_b.clone())),
        )
        .then(candidate_a)
        .otherwise(candidate_b)
    }
}
