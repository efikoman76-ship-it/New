//! TOML subset parser and writer for configuration files.
//!
//! Supports: comments, bare/dotted/quoted keys, `[table]` and
//! `[[array-of-tables]]` headers, inline tables, basic and literal strings
//! (single- and multi-line), integers with underscores, floats with
//! exponents/inf/nan, booleans, and (possibly multiline, trailing-comma)
//! arrays. Not supported: datetimes (config has no use for them; a datetime
//! value is a parse error, not a silent misread).

use crate::RhizomeError;
use std::collections::BTreeMap;

/// A parsed TOML value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// String value.
    Str(String),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// Array value.
    Array(Vec<Value>),
    /// Table value (preserves document order; keys unique by construction).
    Table(Table),
}

/// Ordered table: key → value, insertion-ordered.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Table {
    entries: Vec<(String, Value)>,
}

impl Table {
    /// Empty table.
    pub fn new() -> Self {
        Table {
            entries: Vec::new(),
        }
    }

    /// Insert (or replace) a key.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        let key = key.into();
        for e in self.entries.iter_mut() {
            if e.0 == key {
                e.1 = value;
                return;
            }
        }
        self.entries.push((key, value));
    }

    /// Look up a key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|e| e.0 == key).map(|e| &e.1)
    }

    /// Iterate entries in document order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|e| (e.0.as_str(), &e.1))
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Deep lookup by dotted path.
    pub fn get_path(&self, path: &str) -> Option<&Value> {
        let mut cur = self;
        let parts: Vec<&str> = path.split('.').collect();
        for (i, p) in parts.iter().enumerate() {
            let v = cur.get(p)?;
            if i + 1 == parts.len() {
                return Some(v);
            }
            match v {
                Value::Table(t) => cur = t,
                _ => return None,
            }
        }
        None
    }
}

impl Value {
    /// Borrow as str; error otherwise.
    pub fn as_str(&self) -> Result<&str, RhizomeError> {
        match self {
            Value::Str(s) => Ok(s),
            other => Err(RhizomeError::Serde(format!(
                "expected string, got {other:?}"
            ))),
        }
    }

    /// Borrow as integer; error otherwise.
    pub fn as_int(&self) -> Result<i64, RhizomeError> {
        match self {
            Value::Int(i) => Ok(*i),
            other => Err(RhizomeError::Serde(format!(
                "expected integer, got {other:?}"
            ))),
        }
    }

    /// Borrow as float (accepts integers too).
    pub fn as_float(&self) -> Result<f64, RhizomeError> {
        match self {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            other => Err(RhizomeError::Serde(format!(
                "expected float, got {other:?}"
            ))),
        }
    }

    /// Borrow as bool; error otherwise.
    pub fn as_bool(&self) -> Result<bool, RhizomeError> {
        match self {
            Value::Bool(b) => Ok(*b),
            other => Err(RhizomeError::Serde(format!("expected bool, got {other:?}"))),
        }
    }

    /// Borrow as array; error otherwise.
    pub fn as_array(&self) -> Result<&[Value], RhizomeError> {
        match self {
            Value::Array(a) => Ok(a),
            other => Err(RhizomeError::Serde(format!(
                "expected array, got {other:?}"
            ))),
        }
    }

    /// Borrow as table; error otherwise.
    pub fn as_table(&self) -> Result<&Table, RhizomeError> {
        match self {
            Value::Table(t) => Ok(t),
            other => Err(RhizomeError::Serde(format!(
                "expected table, got {other:?}"
            ))),
        }
    }

    /// Human-readable type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Str(_) => "string",
            Value::Int(_) => "integer",
            Value::Float(_) => "float",
            Value::Bool(_) => "bool",
            Value::Array(_) => "array",
            Value::Table(_) => "table",
        }
    }
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

type PResult<T> = Result<T, RhizomeError>;

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Parser {
            src: src.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ') | Some(b'\t')) {
            self.pos += 1;
        }
    }

    /// Skip whitespace, comments and newlines.
    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') => {
                    self.pos += 1;
                }
                Some(b'#') => {
                    while !matches!(self.peek(), None | Some(b'\n')) {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn expect_line_end(&mut self) -> PResult<()> {
        self.skip_ws();
        if self.peek() == Some(b'#') {
            while !matches!(self.peek(), None | Some(b'\n')) {
                self.pos += 1;
            }
        }
        match self.peek() {
            None => Ok(()),
            Some(b'\n') => {
                self.pos += 1;
                Ok(())
            }
            Some(b'\r') if self.src.get(self.pos + 1) == Some(&b'\n') => {
                self.pos += 2;
                Ok(())
            }
            Some(c) => Err(RhizomeError::Serde(format!(
                "expected end of line, got {:?} at {}",
                c as char, self.pos
            ))),
        }
    }

    fn parse_key_part(&mut self) -> PResult<String> {
        match self.peek() {
            Some(b'"') => self.parse_basic_string(),
            Some(b'\'') => self.parse_literal_string(),
            _ => {
                let start = self.pos;
                while matches!(
                    self.peek(),
                    Some(b'a'..=b'z')
                        | Some(b'A'..=b'Z')
                        | Some(b'0'..=b'9')
                        | Some(b'_')
                        | Some(b'-')
                ) {
                    self.pos += 1;
                }
                if start == self.pos {
                    return Err(RhizomeError::Serde(format!("empty key at {}", self.pos)));
                }
                Ok(String::from_utf8_lossy(&self.src[start..self.pos]).into_owned())
            }
        }
    }

    /// Dotted key path.
    fn parse_key(&mut self) -> PResult<Vec<String>> {
        let mut parts = vec![self.parse_key_part()?];
        loop {
            self.skip_ws();
            if self.peek() == Some(b'.') {
                self.pos += 1;
                self.skip_ws();
                parts.push(self.parse_key_part()?);
            } else {
                return Ok(parts);
            }
        }
    }

    fn parse_basic_string(&mut self) -> PResult<String> {
        // Multi-line?
        if self.src[self.pos..].starts_with(b"\"\"\"") {
            self.pos += 3;
            // Trim first immediate newline.
            if self.peek() == Some(b'\n') {
                self.pos += 1;
            } else if self.src[self.pos..].starts_with(b"\r\n") {
                self.pos += 2;
            }
            let mut out = String::new();
            loop {
                if self.src[self.pos..].starts_with(b"\"\"\"") {
                    self.pos += 3;
                    // TOML allows up to two extra quotes to be part of content.
                    let mut extra = 0;
                    while extra < 2 && self.peek() == Some(b'"') {
                        out.push('"');
                        self.pos += 1;
                        extra += 1;
                    }
                    return Ok(out);
                }
                match self.bump() {
                    None => return Err(RhizomeError::Serde("unterminated string".into())),
                    Some(b'\\') => {
                        // Line-ending backslash: trim whitespace.
                        if matches!(
                            self.peek(),
                            Some(b'\n') | Some(b'\r') | Some(b' ') | Some(b'\t')
                        ) {
                            let save = self.pos;
                            let mut p = self.pos;
                            while matches!(self.src.get(p), Some(b' ') | Some(b'\t') | Some(b'\r'))
                            {
                                p += 1;
                            }
                            if matches!(self.src.get(p), Some(b'\n')) {
                                self.pos = p + 1;
                                while matches!(
                                    self.peek(),
                                    Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
                                ) {
                                    self.pos += 1;
                                }
                            } else {
                                self.pos = save;
                                out.push(self.parse_escape()?);
                            }
                        } else {
                            out.push(self.parse_escape()?);
                        }
                    }
                    Some(c) => push_utf8(&mut out, c, self)?,
                }
            }
        }
        // Single-line.
        if self.bump() != Some(b'"') {
            return Err(RhizomeError::Serde("expected quote".into()));
        }
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err(RhizomeError::Serde("unterminated string".into())),
                Some(b'"') => return Ok(out),
                Some(b'\n') => {
                    return Err(RhizomeError::Serde("newline in single-line string".into()))
                }
                Some(b'\\') => out.push(self.parse_escape()?),
                Some(c) => push_utf8(&mut out, c, self)?,
            }
        }
    }

    fn parse_escape(&mut self) -> PResult<char> {
        let c = self
            .bump()
            .ok_or_else(|| RhizomeError::Serde("dangling escape".into()))?;
        Ok(match c {
            b'n' => '\n',
            b't' => '\t',
            b'r' => '\r',
            b'"' => '"',
            b'\\' => '\\',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'u' | b'U' => {
                let n = if c == b'u' { 4 } else { 8 };
                let mut v: u32 = 0;
                for _ in 0..n {
                    let d = self
                        .bump()
                        .ok_or_else(|| RhizomeError::Serde("short unicode escape".into()))?;
                    let hv = (d as char)
                        .to_digit(16)
                        .ok_or_else(|| RhizomeError::Serde("bad hex escape".into()))?;
                    v = v * 16 + hv;
                }
                char::from_u32(v).ok_or_else(|| RhizomeError::Serde("invalid codepoint".into()))?
            }
            other => {
                return Err(RhizomeError::Serde(format!(
                    "bad escape \\{}",
                    other as char
                )));
            }
        })
    }

    fn parse_literal_string(&mut self) -> PResult<String> {
        if self.src[self.pos..].starts_with(b"'''") {
            self.pos += 3;
            if self.peek() == Some(b'\n') {
                self.pos += 1;
            }
            let mut out = String::new();
            loop {
                if self.src[self.pos..].starts_with(b"'''") {
                    self.pos += 3;
                    let mut extra = 0;
                    while extra < 2 && self.peek() == Some(b'\'') {
                        out.push('\'');
                        self.pos += 1;
                        extra += 1;
                    }
                    return Ok(out);
                }
                match self.bump() {
                    None => return Err(RhizomeError::Serde("unterminated literal".into())),
                    Some(c) => push_utf8(&mut out, c, self)?,
                }
            }
        }
        if self.bump() != Some(b'\'') {
            return Err(RhizomeError::Serde("expected quote".into()));
        }
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err(RhizomeError::Serde("unterminated literal".into())),
                Some(b'\'') => return Ok(out),
                Some(b'\n') => return Err(RhizomeError::Serde("newline in literal".into())),
                Some(c) => push_utf8(&mut out, c, self)?,
            }
        }
    }

    fn parse_value(&mut self) -> PResult<Value> {
        match self
            .peek()
            .ok_or_else(|| RhizomeError::Serde("unexpected EOF".into()))?
        {
            b'"' => Ok(Value::Str(self.parse_basic_string()?)),
            b'\'' => Ok(Value::Str(self.parse_literal_string()?)),
            b'[' => self.parse_array(),
            b'{' => self.parse_inline_table(),
            b't' | b'f' => self.parse_bool(),
            _ => self.parse_number(),
        }
    }

    fn parse_bool(&mut self) -> PResult<Value> {
        if self.src[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Value::Bool(true))
        } else if self.src[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Value::Bool(false))
        } else {
            Err(RhizomeError::Serde(format!("bad bool at {}", self.pos)))
        }
    }

    fn parse_number(&mut self) -> PResult<Value> {
        let start = self.pos;
        while matches!(
            self.peek(),
            Some(b'0'..=b'9')
                | Some(b'a'..=b'z')
                | Some(b'A'..=b'Z')
                | Some(b'+')
                | Some(b'-')
                | Some(b'.')
                | Some(b'_')
                | Some(b':')
        ) {
            // Datetime colon: reject later.
            self.pos += 1;
        }
        let raw = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| RhizomeError::Serde("bad utf8".into()))?;
        if raw.is_empty() {
            return Err(RhizomeError::Serde(format!("empty value at {start}")));
        }
        if raw.contains(':') {
            return Err(RhizomeError::Serde(format!(
                "datetimes not supported: {raw}"
            )));
        }
        let cleaned = raw.replace('_', "");
        if let Some(hex) = cleaned.strip_prefix("0x") {
            return i64::from_str_radix(hex, 16)
                .map(Value::Int)
                .map_err(|_| RhizomeError::Serde(format!("bad hex int {raw}")));
        }
        if cleaned.contains(['e', 'E', '.'])
            || cleaned == "inf"
            || cleaned == "+inf"
            || cleaned == "-inf"
            || cleaned == "nan"
            || cleaned == "+nan"
            || cleaned == "-nan"
        {
            let f = cleaned
                .parse::<f64>()
                .map_err(|_| RhizomeError::Serde(format!("bad float {raw}")))?;
            return Ok(Value::Float(f));
        }
        cleaned
            .parse::<i64>()
            .map(Value::Int)
            .map_err(|_| RhizomeError::Serde(format!("bad int {raw}")))
    }

    fn parse_array(&mut self) -> PResult<Value> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        loop {
            self.skip_trivia();
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Ok(Value::Array(items));
            }
            items.push(self.parse_value()?);
            self.skip_trivia();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => {
                    return Err(RhizomeError::Serde(format!(
                        "expected , or ] at {}",
                        self.pos
                    )))
                }
            }
        }
    }

    fn parse_inline_table(&mut self) -> PResult<Value> {
        self.pos += 1; // '{'
        let mut table = Table::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Table(table));
        }
        loop {
            self.skip_ws();
            let key = self.parse_key()?;
            self.skip_ws();
            if self.bump() != Some(b'=') {
                return Err(RhizomeError::Serde(format!(
                    "expected = in inline table at {}",
                    self.pos
                )));
            }
            self.skip_ws();
            let value = self.parse_value()?;
            insert_path(&mut table, &key, value)?;
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => return Ok(Value::Table(table)),
                _ => {
                    return Err(RhizomeError::Serde(
                        "expected , or } in inline table".into(),
                    ))
                }
            }
        }
    }
}

fn push_utf8(out: &mut String, first: u8, p: &mut Parser<'_>) -> PResult<()> {
    if first < 0x80 {
        out.push(first as char);
        return Ok(());
    }
    // Multibyte: collect continuation bytes validated by from_utf8.
    let start = p.pos - 1;
    let len = if first >= 0xF0 {
        4
    } else if first >= 0xE0 {
        3
    } else {
        2
    };
    for _ in 1..len {
        if p.bump().is_none() {
            return Err(RhizomeError::Serde("truncated utf8".into()));
        }
    }
    let slice = &p.src[start..p.pos];
    let s = std::str::from_utf8(slice).map_err(|_| RhizomeError::Serde("invalid utf8".into()))?;
    out.push_str(s);
    Ok(())
}

fn insert_path(root: &mut Table, path: &[String], value: Value) -> PResult<()> {
    if path.is_empty() {
        return Err(RhizomeError::Serde("empty key path".into()));
    }
    if path.len() == 1 {
        if root.get(&path[0]).is_some() {
            return Err(RhizomeError::Serde(format!("duplicate key {}", path[0])));
        }
        root.insert(path[0].clone(), value);
        return Ok(());
    }
    let head = &path[0];
    let needs_new = root.get(head).is_none();
    if needs_new {
        root.insert(head.clone(), Value::Table(Table::new()));
    }
    match root.get_mut(head).unwrap() {
        Value::Table(t) => insert_path(t, &path[1..], value),
        other => Err(RhizomeError::Serde(format!(
            "key {head} is a {}, want table",
            other.type_name()
        ))),
    }
}

impl Table {
    fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.entries
            .iter_mut()
            .find(|e| e.0 == key)
            .map(|e| &mut e.1)
    }
}

/// Parse a TOML document into a root table.
pub fn parse(src: &str) -> PResult<Table> {
    let mut p = Parser::new(src);
    let mut root = Table::new();
    // Current table path for [headers]; arrays track their open index.
    let mut cur: Vec<String> = Vec::new();
    // Exact paths opened by [table] headers; re-opening is an error.
    let mut defined: Vec<Vec<String>> = Vec::new();
    loop {
        p.skip_trivia();
        if p.peek().is_none() {
            return Ok(root);
        }
        if p.peek() == Some(b'[') {
            let is_array = p.src.get(p.pos + 1) == Some(&b'[');
            p.pos += if is_array { 2 } else { 1 };
            p.skip_ws();
            let key = p.parse_key()?;
            p.skip_ws();
            match (p.bump(), is_array) {
                (Some(b']'), false) => {}
                (Some(b']'), true) => {
                    if p.bump() != Some(b']') {
                        return Err(RhizomeError::Serde("expected ]]".into()));
                    }
                }
                _ => return Err(RhizomeError::Serde("expected ]".into())),
            }
            if is_array {
                push_array_table(&mut root, &key)?;
                cur = key;
            } else {
                if defined.contains(&key) {
                    return Err(RhizomeError::Serde(format!(
                        "table [{}] defined twice",
                        key.join(".")
                    )));
                }
                defined.push(key.clone());
                ensure_table(&mut root, &key)?;
                cur = key;
            }
            p.expect_line_end()?;
            continue;
        }
        let key = p.parse_key()?;
        p.skip_ws();
        if p.bump() != Some(b'=') {
            return Err(RhizomeError::Serde(format!("expected = at {}", p.pos)));
        }
        p.skip_ws();
        let value = p.parse_value()?;
        // Resolve current header path.
        let mut path: Vec<String> = cur.clone();
        path.extend(key);
        // For arrays-of-tables the last path element refers to the open entry.
        insert_header_path(&mut root, &path, value)?;
        p.expect_line_end()?;
    }
}

fn ensure_table(root: &mut Table, path: &[String]) -> PResult<()> {
    let mut node = root;
    for part in path {
        if node.get(part).is_none() {
            node.insert(part.clone(), Value::Table(Table::new()));
        }
        node = match node.get_mut(part) {
            Some(Value::Table(t)) => t,
            Some(Value::Array(a)) => match a.last_mut() {
                Some(Value::Table(t)) => t,
                _ => return Err(RhizomeError::Serde(format!("array at {part} has no table"))),
            },
            Some(other) => {
                return Err(RhizomeError::Serde(format!(
                    "{part} is a {} not a table",
                    other.type_name()
                )))
            }
            None => return Err(RhizomeError::Serde("unreachable".into())),
        };
    }
    Ok(())
}

fn push_array_table(root: &mut Table, path: &[String]) -> PResult<()> {
    let (last, parents) = path
        .split_last()
        .ok_or_else(|| RhizomeError::Serde("empty path".into()))?;
    ensure_table(root, parents)?;
    let mut node = root;
    for part in parents {
        node = match node.get_mut(part) {
            Some(Value::Table(t)) => t,
            Some(Value::Array(a)) => match a.last_mut() {
                Some(Value::Table(t)) => t,
                _ => return Err(RhizomeError::Serde("bad array entry".into())),
            },
            _ => return Err(RhizomeError::Serde("bad header".into())),
        };
    }
    match node.get_mut(last) {
        None => {
            node.insert(last.clone(), Value::Array(vec![Value::Table(Table::new())]));
        }
        Some(Value::Array(a)) => a.push(Value::Table(Table::new())),
        Some(other) => {
            return Err(RhizomeError::Serde(format!(
                "{last} is a {} not an array",
                other.type_name()
            )))
        }
    }
    Ok(())
}

fn insert_header_path(root: &mut Table, path: &[String], value: Value) -> PResult<()> {
    let (last, parents) = path
        .split_last()
        .ok_or_else(|| RhizomeError::Serde("empty key".into()))?;
    ensure_table(root, parents)?;
    let mut node = root;
    for part in parents {
        node = match node.get_mut(part) {
            Some(Value::Table(t)) => t,
            Some(Value::Array(a)) => match a.last_mut() {
                Some(Value::Table(t)) => t,
                _ => return Err(RhizomeError::Serde("bad array entry".into())),
            },
            _ => return Err(RhizomeError::Serde("bad header".into())),
        };
    }
    if node.get(last).is_some() {
        return Err(RhizomeError::Serde(format!("duplicate key {last}")));
    }
    node.insert(last.clone(), value);
    Ok(())
}

/// Render a value back to TOML (used for canonical re-emission in tests and
/// tooling; formatting is stable).
pub fn write_value(v: &Value) -> String {
    let mut s = String::new();
    write_value_into(v, &mut s);
    s
}

fn write_value_into(v: &Value, out: &mut String) {
    match v {
        Value::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\t' => out.push_str("\\t"),
                    '\r' => out.push_str("\\r"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::Float(f) => {
            if f.is_nan() {
                out.push_str("nan");
            } else if f.is_infinite() {
                out.push_str(if *f > 0.0 { "inf" } else { "-inf" });
            } else {
                let r = format!("{f}");
                out.push_str(&r);
                if !r.contains('.') && !r.contains('e') && !r.contains('E') {
                    out.push_str(".0");
                }
            }
        }
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value_into(item, out);
            }
            out.push(']');
        }
        Value::Table(t) => {
            out.push('{');
            for (i, (k, val)) in t.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("{k} = "));
                write_value_into(val, out);
            }
            out.push('}');
        }
    }
}

/// Convenience: sorted map view of a table (for canonical hashing).
pub fn sorted_pairs(t: &Table) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for (k, v) in t.iter() {
        m.insert(k.to_string(), write_value(v));
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = r#"
# RHIZOME test document
name = "test-s"
d_model = 64
gravity = 9.80665
exp = 1e-3
neg_inf = -inf
not_a_number = nan
flag = true
hex = 0x10
big = 1_000_000
list = [1, 2, 3,]
nested = [[1, 2], [3]]
inline = { a = 1, b = "two" }

[model]
n_heads = 2
[model.moe]
experts = 4
topk = 2

[[phases]]
name = "p0"
steps = 100

[[phases]]
name = "p1"
steps = 200

[escape]
basic = "a\"b\\c\nd\tŕ"
literal = 'C:\path'
multi = """
line1
line2"""
"#;

    #[test]
    fn parse_document() {
        let t = parse(DOC).unwrap();
        assert_eq!(t.get("name").unwrap().as_str().unwrap(), "test-s");
        assert_eq!(t.get("d_model").unwrap().as_int().unwrap(), 64);
        assert!((t.get("gravity").unwrap().as_float().unwrap() - 9.80665).abs() < 1e-12);
        assert_eq!(t.get("exp").unwrap().as_float().unwrap(), 1e-3);
        assert_eq!(
            t.get("neg_inf").unwrap().as_float().unwrap(),
            f64::NEG_INFINITY
        );
        assert!(t.get("not_a_number").unwrap().as_float().unwrap().is_nan());
        assert!(t.get("flag").unwrap().as_bool().unwrap());
        assert_eq!(t.get("hex").unwrap().as_int().unwrap(), 16);
        assert_eq!(t.get("big").unwrap().as_int().unwrap(), 1_000_000);
        assert_eq!(t.get("list").unwrap().as_array().unwrap().len(), 3);
        assert_eq!(
            t.get("nested").unwrap().as_array().unwrap()[1]
                .as_array()
                .unwrap()[0]
                .as_int()
                .unwrap(),
            3
        );
        assert_eq!(
            t.get_path("model.moe.experts").unwrap().as_int().unwrap(),
            4
        );
        assert_eq!(t.get_path("model.moe.topk").unwrap().as_int().unwrap(), 2);
        let phases = t.get("phases").unwrap().as_array().unwrap();
        assert_eq!(phases.len(), 2);
        assert_eq!(
            phases[1]
                .as_table()
                .unwrap()
                .get("steps")
                .unwrap()
                .as_int()
                .unwrap(),
            200
        );
        assert_eq!(
            t.get_path("escape.basic").unwrap().as_str().unwrap(),
            "a\"b\\c\nd\tŕ"
        );
        assert_eq!(
            t.get_path("escape.literal").unwrap().as_str().unwrap(),
            "C:\\path"
        );
        assert_eq!(
            t.get_path("escape.multi").unwrap().as_str().unwrap(),
            "line1\nline2"
        );
    }

    #[test]
    fn parse_errors() {
        assert!(parse("a = ").is_err());
        assert!(parse("a = tru").is_err());
        assert!(parse("= 1").is_err());
        assert!(parse("a = 1\na = 2").is_err());
        assert!(parse("[t]\n[t]").is_err());
        assert!(parse("a = [1, 2").is_err());
        assert!(parse("a = \"unterminated").is_err());
        assert!(
            parse("d = 1979-05-27T07:32:00Z").is_err(),
            "datetimes rejected"
        );
        assert!(parse("a = 1 b = 2").is_err());
    }

    #[test]
    fn writer_roundtrip() {
        let t = parse(DOC).unwrap();
        // Spot-check writer output.
        assert_eq!(write_value(t.get("name").unwrap()), "\"test-s\"");
        assert_eq!(write_value(t.get("gravity").unwrap()), "9.80665");
        assert_eq!(write_value(t.get("flag").unwrap()), "true");
        assert_eq!(
            write_value(t.get("inline").unwrap()),
            "{a = 1, b = \"two\"}"
        );
        assert_eq!(write_value(t.get("big").unwrap()), "1000000");
        // Integer-valued float keeps ".0".
        assert_eq!(write_value(&Value::Float(2.0)), "2.0");
    }

    #[test]
    fn dotted_keys() {
        let t = parse("a.b.c = 1\na.b.d = 2\n").unwrap();
        assert_eq!(t.get_path("a.b.c").unwrap().as_int().unwrap(), 1);
        assert_eq!(t.get_path("a.b.d").unwrap().as_int().unwrap(), 2);
    }
}
