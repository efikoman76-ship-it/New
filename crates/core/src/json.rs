//! Minimal RFC 8259 JSON: value type, parser (depth-limited, duplicate keys
/// last-wins, surrogate-pair aware) and deterministic writer (insertion
/// order). Used for metrics, API payloads, eval reports, plan dumps.
use crate::{RhizomeError, RhizomeResult};

/// Maximum nesting depth accepted by the parser (fuzz safety).
pub const MAX_DEPTH: u32 = 128;

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// null
    Null,
    /// true / false
    Bool(bool),
    /// IEEE double (integers are kept exact when ≤ 2^53 via f64).
    Number(f64),
    /// UTF-8 string.
    String(String),
    /// Array of values.
    Array(Vec<Json>),
    /// Object; insertion order preserved, duplicate keys overwrite.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Build an object from key/value pairs.
    pub fn object(pairs: Vec<(&str, Json)>) -> Json {
        Json::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// Convenience constructors.
    pub fn str(s: impl Into<String>) -> Json {
        Json::String(s.into())
    }

    /// Integer-valued number.
    pub fn int(i: i64) -> Json {
        Json::Number(i as f64)
    }

    /// Borrow as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Borrow as number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Borrow as str.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow as array.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Look up an object key.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

struct JParser<'a> {
    s: &'a [u8],
    i: usize,
}

type JResult<T> = Result<T, RhizomeError>;

impl<'a> JParser<'a> {
    fn ws(&mut self) {
        while matches!(
            self.s.get(self.i),
            Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r')
        ) {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn value(&mut self, depth: u32) -> JResult<Json> {
        if depth > MAX_DEPTH {
            return Err(RhizomeError::Serde("json nesting too deep".into()));
        }
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.lit(b"true", Json::Bool(true)),
            Some(b'f') => self.lit(b"false", Json::Bool(false)),
            Some(b'n') => self.lit(b"null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(RhizomeError::Serde(format!(
                "unexpected byte {:?} at {}",
                c as char, self.i
            ))),
            None => Err(RhizomeError::Serde("unexpected eof".into())),
        }
    }

    fn lit(&mut self, word: &[u8], v: Json) -> JResult<Json> {
        if self.s.len() >= self.i + word.len() && &self.s[self.i..self.i + word.len()] == word {
            self.i += word.len();
            Ok(v)
        } else {
            Err(RhizomeError::Serde(format!("bad literal at {}", self.i)))
        }
    }

    fn object(&mut self, depth: u32) -> JResult<Json> {
        self.i += 1; // {
        let mut out: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Object(out));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return Err(RhizomeError::Serde(format!("expected : at {}", self.i)));
            }
            self.i += 1;
            let val = self.value(depth + 1)?;
            // Duplicate keys: last wins.
            if let Some(slot) = out.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = val;
            } else {
                out.push((key, val));
            }
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(out));
                }
                _ => {
                    return Err(RhizomeError::Serde(format!(
                        "expected , or }} at {}",
                        self.i
                    )))
                }
            }
        }
    }

    fn array(&mut self, depth: u32) -> JResult<Json> {
        self.i += 1; // [
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Array(out));
        }
        loop {
            out.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(out));
                }
                _ => {
                    return Err(RhizomeError::Serde(format!(
                        "expected , or ] at {}",
                        self.i
                    )))
                }
            }
        }
    }

    fn string(&mut self) -> JResult<String> {
        if self.peek() != Some(b'"') {
            return Err(RhizomeError::Serde(format!(
                "expected string at {}",
                self.i
            )));
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| RhizomeError::Serde("unterminated string".into()))?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self
                        .peek()
                        .ok_or_else(|| RhizomeError::Serde("dangling escape".into()))?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) {
                                // High surrogate: require a following \uXXXX low.
                                if self.peek() == Some(b'\\') {
                                    self.i += 1;
                                    if self.peek() == Some(b'u') {
                                        self.i += 1;
                                        let lo = self.hex4()?;
                                        if (0xDC00..0xE000).contains(&lo) {
                                            let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                            out.push(char::from_u32(c).ok_or_else(|| {
                                                RhizomeError::Serde("bad pair".into())
                                            })?);
                                            continue;
                                        }
                                    }
                                }
                                return Err(RhizomeError::Serde("lone surrogate".into()));
                            } else if (0xDC00..0xE000).contains(&cp) {
                                return Err(RhizomeError::Serde("lone low surrogate".into()));
                            } else {
                                out.push(
                                    char::from_u32(cp)
                                        .ok_or_else(|| RhizomeError::Serde("bad cp".into()))?,
                                );
                            }
                        }
                        other => {
                            return Err(RhizomeError::Serde(format!(
                                "bad escape \\{}",
                                other as char
                            )))
                        }
                    }
                }
                c if c < 0x20 => return Err(RhizomeError::Serde("control char in string".into())),
                c if c < 0x80 => out.push(c as char),
                _ => {
                    // Multibyte utf8.
                    let start = self.i - 1;
                    let len = if c >= 0xF0 {
                        4
                    } else if c >= 0xE0 {
                        3
                    } else {
                        2
                    };
                    if self.s.len() < start + len {
                        return Err(RhizomeError::Serde("truncated utf8".into()));
                    }
                    let s = std::str::from_utf8(&self.s[start..start + len])
                        .map_err(|_| RhizomeError::Serde("invalid utf8".into()))?;
                    self.i = start + len;
                    out.push_str(s);
                }
            }
        }
    }

    fn hex4(&mut self) -> JResult<u32> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self
                .peek()
                .ok_or_else(|| RhizomeError::Serde("short \\u escape".into()))?;
            let d = (c as char)
                .to_digit(16)
                .ok_or_else(|| RhizomeError::Serde("bad \\u hex".into()))?;
            v = v * 16 + d;
            self.i += 1;
        }
        Ok(v)
    }

    fn number(&mut self) -> JResult<Json> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        // int part
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(c) if c.is_ascii_digit() => {
                while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    self.i += 1;
                }
            }
            _ => return Err(RhizomeError::Serde(format!("bad number at {start}"))),
        }
        // frac
        if self.peek() == Some(b'.') {
            self.i += 1;
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(RhizomeError::Serde("bad fraction".into()));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        // exp
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.i += 1;
            }
            if !matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                return Err(RhizomeError::Serde("bad exponent".into()));
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        let raw = std::str::from_utf8(&self.s[start..self.i])
            .map_err(|_| RhizomeError::Serde("bad utf8".into()))?;
        raw.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| RhizomeError::Serde(format!("unparsable number {raw}")))
    }
}

/// Parse a JSON document.
pub fn parse(src: &[u8]) -> RhizomeResult<Json> {
    let mut p = JParser { s: src, i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i != src.len() {
        return Err(RhizomeError::Serde(format!("trailing bytes at {}", p.i)));
    }
    Ok(v)
}

/// Parse a JSON document from a string.
pub fn parse_str(src: &str) -> RhizomeResult<Json> {
    parse(src.as_bytes())
}

/// Serialize to a compact string.
pub fn to_string(v: &Json) -> String {
    let mut s = String::new();
    write_json(v, &mut s);
    s
}

/// Serialize to a pretty (2-space) string.
pub fn to_string_pretty(v: &Json) -> String {
    let mut s = String::new();
    write_pretty(v, &mut s, 0);
    s
}

fn write_json(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Number(n) => {
            if n.is_finite() {
                if *n == n.trunc() && n.abs() < 9.007199254740992e15 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{n}"));
                }
            } else {
                out.push_str("null"); // non-finite not representable in JSON
            }
        }
        Json::String(s) => write_string(s, out),
        Json::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Json::Object(o) => {
            out.push('{');
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(k, out);
                out.push(':');
                write_json(val, out);
            }
            out.push('}');
        }
    }
}

fn write_pretty(v: &Json, out: &mut String, indent: usize) {
    match v {
        Json::Array(a) if !a.is_empty() => {
            out.push_str("[\n");
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                out.push_str(&"  ".repeat(indent + 1));
                write_pretty(item, out, indent + 1);
            }
            out.push('\n');
            out.push_str(&"  ".repeat(indent));
            out.push(']');
        }
        Json::Object(o) if !o.is_empty() => {
            out.push_str("{\n");
            for (i, (k, val)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                out.push_str(&"  ".repeat(indent + 1));
                write_string(k, out);
                out.push_str(": ");
                write_pretty(val, out, indent + 1);
            }
            out.push('\n');
            out.push_str(&"  ".repeat(indent));
            out.push('}');
        }
        other => write_json(other, out),
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Helper to build objects with a small builder API.
pub struct ObjBuilder {
    pairs: Vec<(String, Json)>,
}

impl ObjBuilder {
    /// New builder.
    pub fn new() -> Self {
        ObjBuilder { pairs: Vec::new() }
    }

    /// Add a field.
    pub fn field(mut self, k: &str, v: Json) -> Self {
        self.pairs.push((k.to_string(), v));
        self
    }

    /// Add a string field.
    pub fn str_field(self, k: &str, v: impl Into<String>) -> Self {
        self.field(k, Json::String(v.into()))
    }

    /// Add an integer field.
    pub fn int_field(self, k: &str, v: i64) -> Self {
        self.field(k, Json::int(v))
    }

    /// Add a float field.
    pub fn float_field(self, k: &str, v: f64) -> Self {
        self.field(k, Json::Number(v))
    }

    /// Add a boolean field.
    pub fn bool_field(self, k: &str, v: bool) -> Self {
        self.field(k, Json::Bool(v))
    }

    /// Finish.
    pub fn build(self) -> Json {
        Json::Object(self.pairs)
    }
}

impl Default for ObjBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let src = r#"{"a":1,"b":[true,null,"x"],"c":{"d":1.5e3},"е":-0.25}"#;
        let v = parse_str(src).unwrap();
        assert_eq!(v.get("a").unwrap().as_f64().unwrap(), 1.0);
        assert_eq!(
            v.get("c").unwrap().get("d").unwrap().as_f64().unwrap(),
            1500.0
        );
        let out = to_string(&v);
        let v2 = parse_str(&out).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn strings_and_escapes() {
        let v = parse_str(r#""aA\n\"\\é😀""#).unwrap();
        assert_eq!(v.as_str().unwrap(), "aA\n\"\\é😀");
        let out = to_string(&Json::String("tab\there\u{1}".into()));
        assert_eq!(out, "\"tab\\there\\u0001\"");
    }

    #[test]
    fn surrogate_pairs() {
        let v = parse_str("\"\\ud83d\\ude00\"").unwrap();
        assert_eq!(v.as_str().unwrap(), "😀");
        assert!(parse_str("\"\\ud83d\"").is_err(), "lone high surrogate");
        assert!(parse_str("\"\\ude00\"").is_err(), "lone low surrogate");
    }

    #[test]
    fn parse_errors() {
        assert!(parse(b"").is_err());
        assert!(parse(b"{").is_err());
        assert!(parse(b"[1,]").is_err());
        assert!(parse(b"{\"a\":}").is_err());
        assert!(parse(b"01").is_err(), "leading zero");
        assert!(parse(b"1.").is_err());
        assert!(parse(b"nul").is_err());
        assert!(parse(b"\"abc").is_err());
        assert!(parse(b"1 2").is_err(), "trailing bytes");
        // Deep nesting rejected before stack exhaustion.
        let deep = "[".repeat(500) + &"]".repeat(500);
        assert!(parse(deep.as_bytes()).is_err());
    }

    #[test]
    fn numbers_kept_exact_for_ints() {
        let v = Json::int(123456789012345);
        assert_eq!(to_string(&v), "123456789012345");
        assert_eq!(to_string(&Json::Number(0.5)), "0.5");
        assert_eq!(to_string(&Json::Number(f64::NAN)), "null");
    }

    #[test]
    fn pretty_printer() {
        let v = Json::object(vec![
            ("a", Json::int(1)),
            ("b", Json::Array(vec![Json::int(2)])),
        ]);
        let s = to_string_pretty(&v);
        assert!(s.contains("\n"));
        assert_eq!(parse_str(&s).unwrap(), v);
    }
}
