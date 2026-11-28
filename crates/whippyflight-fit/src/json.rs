//! JSON with its keys in the order written, printed as Python's
//! `json.dump(indent=2)` does (what the fit's files have always looked like).

use std::fmt::Write;

#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Int(i64),
    /// Non-finite numbers are written as null.
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl From<f64> for J {
    fn from(v: f64) -> J {
        J::Num(v)
    }
}

impl From<i64> for J {
    fn from(v: i64) -> J {
        J::Int(v)
    }
}

impl From<usize> for J {
    fn from(v: usize) -> J {
        J::Int(v as i64)
    }
}

impl From<u32> for J {
    fn from(v: u32) -> J {
        J::Int(v as i64)
    }
}

impl From<bool> for J {
    fn from(v: bool) -> J {
        J::Bool(v)
    }
}

impl From<&str> for J {
    fn from(v: &str) -> J {
        J::Str(v.to_string())
    }
}

impl From<String> for J {
    fn from(v: String) -> J {
        J::Str(v)
    }
}

impl<T: Into<J>> From<Option<T>> for J {
    fn from(v: Option<T>) -> J {
        v.map_or(J::Null, Into::into)
    }
}

impl<T: Into<J>> From<Vec<T>> for J {
    fn from(v: Vec<T>) -> J {
        J::Arr(v.into_iter().map(Into::into).collect())
    }
}

/// An object from (key, value) pairs.
#[macro_export]
macro_rules! obj {
    ($($k:expr => $v:expr),* $(,)?) => {
        $crate::json::J::Obj(vec![$(($k.to_string(), $crate::json::J::from($v))),*])
    };
}

impl J {
    pub fn obj() -> J {
        J::Obj(vec![])
    }

    /// Set `key` (in place if present, else appended).
    pub fn set(&mut self, key: &str, value: impl Into<J>) {
        let J::Obj(pairs) = self else { panic!("set on a non-object") };
        let value = value.into();
        match pairs.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => pairs.push((key.to_string(), value)),
        }
    }

    pub fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut J> {
        match self {
            J::Obj(pairs) => pairs.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            J::Num(v) => Some(v),
            J::Int(v) => Some(v as f64),
            _ => None,
        }
    }

    /// From parsed JSON (key order as serde_json keeps it).
    pub fn from_value(v: &serde_json::Value) -> J {
        use serde_json::Value;
        match v {
            Value::Null => J::Null,
            Value::Bool(b) => J::Bool(*b),
            Value::Number(n) => n.as_i64().map_or_else(|| J::Num(n.as_f64().unwrap_or(f64::NAN)), J::Int),
            Value::String(s) => J::Str(s.clone()),
            Value::Array(a) => J::Arr(a.iter().map(J::from_value).collect()),
            Value::Object(o) => J::Obj(o.iter().map(|(k, v)| (k.clone(), J::from_value(v))).collect()),
        }
    }

    pub fn to_value(&self) -> serde_json::Value {
        serde_json::from_str(&self.pretty()).expect("written JSON parses")
    }

    /// Indented two spaces, with a trailing newline.
    pub fn pretty(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out.push('\n');
        out
    }

    fn write(&self, out: &mut String, depth: usize) {
        let pad = |out: &mut String, d: usize| out.extend(std::iter::repeat_n(' ', 2 * d));
        match self {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Int(v) => write!(out, "{v}").unwrap(),
            J::Num(v) => out.push_str(&float(*v)),
            J::Str(s) => string(out, s),
            J::Arr(items) if items.is_empty() => out.push_str("[]"),
            J::Arr(items) => {
                out.push_str("[\n");
                for (i, item) in items.iter().enumerate() {
                    pad(out, depth + 1);
                    item.write(out, depth + 1);
                    out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
                }
                pad(out, depth);
                out.push(']');
            }
            J::Obj(pairs) if pairs.is_empty() => out.push_str("{}"),
            J::Obj(pairs) => {
                out.push_str("{\n");
                for (i, (k, v)) in pairs.iter().enumerate() {
                    pad(out, depth + 1);
                    string(out, k);
                    out.push_str(": ");
                    v.write(out, depth + 1);
                    out.push_str(if i + 1 < pairs.len() { ",\n" } else { "\n" });
                }
                pad(out, depth);
                out.push('}');
            }
        }
    }
}

fn string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || !c.is_ascii() => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    write!(out, "\\u{unit:04x}").unwrap();
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's float repr: the shortest round-trip digits, in exponent form
/// below 1e-4 and from 1e16.
pub fn float(v: f64) -> String {
    if !v.is_finite() {
        return "null".into();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sci = format!("{v:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if !(-4..16).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exp.abs());
    }
    let s = format!("{v}");
    if s.contains('.') { s } else { format!("{s}.0") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_does() {
        assert_eq!(float(1.0), "1.0");
        assert_eq!(float(1.7525621493441606e-05), "1.7525621493441606e-05");
        assert_eq!(float(0.0001), "0.0001");
        assert_eq!(float(445.5), "445.5");
        assert_eq!(float(-1.512245366558469), "-1.512245366558469");
        assert_eq!(float(1e16), "1e+16");
        assert_eq!(float(f64::NAN), "null");
    }

    #[test]
    fn objects_keep_their_order() {
        let mut o = obj! {"b" => 1i64, "a" => vec![1.5, 2.0]};
        o.set("b", "x");
        o.set("c", J::obj());
        assert_eq!(o.pretty(), "{\n  \"b\": \"x\",\n  \"a\": [\n    1.5,\n    2.0\n  ],\n  \"c\": {}\n}\n");
    }
}
