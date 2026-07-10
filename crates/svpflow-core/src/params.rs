pub fn parse(input: &[u8]) -> Result<Value, String> {
    let mut parser = Parser::new(input);
    parser.skip_ws();
    let value = parser.value()?;
    parser.skip_ws();
    if parser.finished() || matches!(value, Value::Object(_)) {
        Ok(value)
    } else {
        Err(parser.error("extra data after JSON value"))
    }
}

pub enum Value {
    Object(Vec<(String, Self)>),
    Array(Vec<Self>),
    String(String),
    Number(f64),
    Bool(bool),
    Null,
}

impl Value {
    pub fn bool_at(&self, path: &[&str]) -> Option<bool> {
        match self.at(path) {
            Some(Self::Number(value)) => Some(*value != 0.0),
            Some(Self::String(value)) => Some(!value.is_empty()),
            Some(Self::Object(value)) => Some(!value.is_empty()),
            Some(Self::Array(value)) => Some(!value.is_empty()),
            Some(Self::Bool(value)) => Some(*value),
            Some(Self::Null) => Some(false),
            _ => None,
        }
    }

    pub fn int_at(&self, path: &[&str]) -> Option<i64> {
        match self.at(path) {
            Some(Self::Number(value)) if value.is_finite() => {
                value.trunc().to_string().parse().ok()
            }
            Some(Self::Bool(value)) => Some(i64::from(*value)),
            Some(Self::Null) => Some(0),
            _ => None,
        }
    }

    pub fn float_at(&self, path: &[&str]) -> Option<f64> {
        match self.at(path) {
            Some(Self::Number(value)) => Some(*value),
            Some(Self::Bool(value)) => Some(i32::from(*value).into()),
            Some(Self::Null) => Some(0.0),
            _ => None,
        }
    }

    pub fn string_at(&self, path: &[&str]) -> Option<&str> {
        match self.at(path) {
            Some(Self::String(value)) => Some(value),
            _ => None,
        }
    }

    pub fn array_at(&self, path: &[&str]) -> Option<&[Self]> {
        match self.at(path) {
            Some(Self::Array(value)) => Some(value),
            _ => None,
        }
    }

    fn at(&self, path: &[&str]) -> Option<&Self> {
        let mut value = self;
        for key in path {
            value = value.get(key)?;
        }
        Some(value)
    }

    fn get(&self, key: &str) -> Option<&Self> {
        let Self::Object(entries) = self else {
            return None;
        };
        entries
            .iter()
            .rev()
            .find_map(|(entry_key, value)| (entry_key == key).then_some(value))
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn finished(&self) -> bool {
        self.pos == self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    fn skip_ws(&mut self) {
        loop {
            self.skip_space();
            if self.bytes.get(self.pos..self.pos + 2) == Some(b"//") {
                self.pos += 2;
                while !matches!(self.peek(), None | Some(b'\n' | b'\r')) {
                    self.pos += 1;
                }
            } else if self.bytes.get(self.pos..self.pos + 2) == Some(b"/*") {
                self.pos += 2;
                while self.bytes.get(self.pos..self.pos + 2) != Some(b"*/") {
                    if self.next().is_none() {
                        return;
                    }
                }
                self.pos += 2;
            } else {
                return;
            }
        }
    }

    fn skip_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn skip_member_space(&mut self) -> Result<(), String> {
        self.skip_space();
        if self.bytes.get(self.pos..self.pos + 2) == Some(b"//") {
            self.pos += 2;
            while !matches!(self.peek(), None | Some(b'\n' | b'\r')) {
                self.pos += 1;
            }
            self.skip_space();
            return Err(self.error("Missing '}' or object member name"));
        }
        if self.bytes.get(self.pos..self.pos + 2) == Some(b"/*") {
            self.pos += 2;
            while self.bytes.get(self.pos..self.pos + 2) != Some(b"*/") {
                if self.next().is_none() {
                    return Err(self.error("Missing '}' or object member name"));
                }
            }
            self.pos += 2;
            self.skip_space();
            return Err(self.error("Missing '}' or object member name"));
        }
        Ok(())
    }

    fn value(&mut self) -> Result<Value, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Value::String),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(b't') => self.literal(b"true", Value::Bool(true)),
            Some(b'f') => self.literal(b"false", Value::Bool(false)),
            Some(b'n') => self.literal(b"null", Value::Null),
            _ => Err(self.error("Syntax error: value, object or array expected.")),
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        self.expect(b'{')?;
        self.skip_ws();
        if self.consume(b'}') {
            return Ok(Value::Object(Vec::new()));
        }
        let mut entries = Vec::new();
        loop {
            self.skip_member_space()?;
            let key = if self.peek() == Some(b'"') {
                self.string()?
            } else {
                self.identifier()?
            };
            self.skip_ws();
            if !self.consume(b':') {
                return Err(self.error("Missing ':' after object member name"));
            }
            let value = self.value()?;
            entries.push((key, value));
            self.skip_ws();
            if self.consume(b'}') {
                return Ok(Value::Object(entries));
            }
            if !self.consume(b',') {
                return Err(self.error("Missing ',' or '}' in object declaration"));
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.expect(b'[')?;
        self.skip_ws();
        if self.consume(b']') {
            return Ok(Value::Array(Vec::new()));
        }
        let mut values = Vec::new();
        loop {
            values.push(self.value()?);
            self.skip_ws();
            if self.consume(b']') {
                return Ok(Value::Array(values));
            }
            self.expect(b',')?;
        }
    }

    fn identifier(&mut self) -> Result<String, String> {
        let start = self.pos;
        match self.peek() {
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'_') => self.pos += 1,
            _ => return Err(self.error("Missing '}' or object member name")),
        }
        while matches!(
            self.peek(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
        ) {
            self.pos += 1;
        }
        Ok(String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
    }

    fn string(&mut self) -> Result<String, String> {
        let start = self.pos;
        self.expect(b'"')?;
        let mut bytes = Vec::new();
        while let Some(byte) = self.next() {
            match byte {
                b'"' => return Ok(String::from_utf8_lossy(&bytes).into_owned()),
                b'\\' => {
                    let escaped = self.escape(start)?;
                    bytes.extend_from_slice(escaped.as_bytes());
                }
                _ => bytes.push(byte),
            }
        }
        Err(self.error_at(start, "Syntax error: value, object or array expected."))
    }

    fn escape(&mut self, string_start: usize) -> Result<String, String> {
        match self.next() {
            Some(b'"') => Ok("\"".to_owned()),
            Some(b'\\') => Ok("\\".to_owned()),
            Some(b'/') => Ok("/".to_owned()),
            Some(b'b') => Ok('\u{0008}'.to_string()),
            Some(b'f') => Ok('\u{000c}'.to_string()),
            Some(b'n') => Ok("\n".to_owned()),
            Some(b'r') => Ok("\r".to_owned()),
            Some(b't') => Ok("\t".to_owned()),
            Some(b'u') => {
                let code = self.unicode_escape(string_start)?;
                self.escaped_unicode(string_start, code)
            }
            _ => {
                Err(self.error_with_detail(string_start, "Bad escape sequence in string", self.pos))
            }
        }
    }

    fn unicode_escape(&mut self, string_start: usize) -> Result<u32, String> {
        let mut code = 0;
        for _ in 0..4 {
            code = code * 16
                + match self.next() {
                    Some(byte @ b'0'..=b'9') => u32::from(byte - b'0'),
                    Some(byte @ b'a'..=b'f') => u32::from(byte - b'a' + 10),
                    Some(byte @ b'A'..=b'F') => u32::from(byte - b'A' + 10),
                    _ => {
                        return Err(self.error_with_detail(
                            string_start,
                            "Bad unicode escape sequence in string: hexadecimal digit expected.",
                            self.pos,
                        ));
                    }
                };
        }
        Ok(code)
    }

    fn escaped_unicode(&mut self, string_start: usize, code: u32) -> Result<String, String> {
        match code {
            0xd800..=0xdbff => self.surrogate_pair(string_start, code),
            0xdc00..=0xdfff => Ok('\u{fffd}'.to_string()),
            _ => Ok(char::from_u32(code).unwrap_or('\u{fffd}').to_string()),
        }
    }

    fn surrogate_pair(&mut self, string_start: usize, high: u32) -> Result<String, String> {
        let detail_pos = self.pos;
        if self.next() != Some(b'\\') || self.next() != Some(b'u') {
            return Err(self.error_with_detail(
                string_start,
                "additional six characters expected to parse unicode surrogate pair.",
                detail_pos,
            ));
        }
        let low = self.unicode_escape(string_start)?;
        if !(0xdc00..=0xdfff).contains(&low) {
            return Err(self.error_with_detail(
                string_start,
                "additional six characters expected to parse unicode surrogate pair.",
                detail_pos,
            ));
        }
        let code = 0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00);
        Ok(char::from_u32(code).unwrap_or('\u{fffd}').to_string())
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        let negative = self.consume(b'-');
        let leading_dot = if matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
            self.consume_digits();
            false
        } else if negative && self.consume(b'.') {
            if !self.consume_digits() {
                return Err(self.bad_number(start));
            }
            true
        } else if negative {
            return Ok(Value::Number(0.0));
        } else {
            return Err(self.error("invalid number"));
        };
        if !leading_dot && self.consume(b'.') {
            self.consume_digits();
        }
        if self.consume(b'e') || self.consume(b'E') {
            if !self.consume(b'+') {
                self.consume(b'-');
            }
            if !self.consume_digits() {
                return Err(self.bad_number(start));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.error("invalid number"))?;
        let normalized;
        let text = if leading_dot {
            normalized = format!("-0{}", &text[1..]);
            normalized.as_str()
        } else {
            text
        };
        text.parse::<f64>()
            .map(Value::Number)
            .map_err(|_| self.error("invalid number"))
    }

    fn bad_number(&self, start: usize) -> String {
        let text = String::from_utf8_lossy(&self.bytes[start..self.pos]);
        self.error_at(start, &format!("'{text}' is not a number."))
    }

    fn consume_digits(&mut self) -> bool {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        self.pos != start
    }

    fn literal(&mut self, literal: &[u8], value: Value) -> Result<Value, String> {
        if self.bytes.get(self.pos..self.pos + literal.len()) == Some(literal) {
            self.pos += literal.len();
            Ok(value)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), String> {
        if self.consume(expected) {
            Ok(())
        } else {
            Err(self.error("unexpected token"))
        }
    }

    fn error(&self, message: &str) -> String {
        self.error_at(self.pos, message)
    }

    fn error_at(&self, pos: usize, message: &str) -> String {
        let (line, column) = self.line_column_at(pos);
        format!("* Line {line}, Column {column}\n  {message}\n")
    }

    fn error_with_detail(&self, pos: usize, message: &str, detail_pos: usize) -> String {
        let (line, column) = self.line_column_at(pos);
        let (detail_line, detail_column) = self.line_column_at(detail_pos);
        format!(
            "* Line {line}, Column {column}\n  {message}\nSee Line {detail_line}, Column {detail_column} for detail.\n"
        )
    }

    fn line_column_at(&self, pos: usize) -> (usize, usize) {
        let mut line = 1;
        let mut line_start = 0;
        let stop = pos.min(self.bytes.len());
        let mut index = 0;
        while index < stop {
            match self.bytes[index] {
                b'\n' => {
                    line += 1;
                    line_start = index + 1;
                }
                b'\r' => {
                    line += 1;
                    line_start = if self.bytes.get(index + 1) == Some(&b'\n') {
                        index += 1;
                        index + 1
                    } else {
                        index + 1
                    };
                }
                _ => {}
            }
            index += 1;
        }
        (line, stop - line_start + 1)
    }
}
