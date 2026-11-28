//! Betaflight blackbox (.bbl / .bfl) decoder.
//!
//! Decodes data version 2 logs as written by Betaflight 4.x. Values are the
//! raw logged integers (deg/s gyro, centivolt vbat, DShot motor command,
//! eRPM/100), as a Blackbox Explorer CSV export has them, except that a
//! `blackbox_high_resolution` log's gyro, rcCommand and rate setpoints, stored
//! at 10×, are divided back. A file may hold several logs (one per arm).
//!
//! Corrupt frames are handled the way blackbox-tools does: a frame is only
//! accepted if it is followed by a valid frame marker and its iteration/time
//! advance sanely; after a bad frame, P-frames are dropped until the next
//! I-frame.

use std::collections::HashMap;

const LOG_START: &[u8] = b"H Product:Blackbox flight data recorder by Nicholas Sherlock\n";
const LOG_END_MSG: &[u8] = b"End of log\x00";

const ENC_SIGNED_VB: i64 = 0;
const ENC_UNSIGNED_VB: i64 = 1;
const ENC_NEG_14BIT: i64 = 3;
const ENC_TAG8_8SVB: i64 = 6;
const ENC_TAG2_3S32: i64 = 7;
const ENC_TAG8_4S16: i64 = 8;
const ENC_NULL: i64 = 9;

const PRED_ZERO: i64 = 0;
const PRED_PREVIOUS: i64 = 1;
const PRED_STRAIGHT_LINE: i64 = 2;
const PRED_AVERAGE_2: i64 = 3;
const PRED_MINTHROTTLE: i64 = 4;
const PRED_MOTOR_0: i64 = 5;
const PRED_INCREMENT: i64 = 6;
const PRED_1500: i64 = 8;
const PRED_VBATREF: i64 = 9;
const PRED_MINMOTOR: i64 = 11;

const EVENT_SYNC_BEEP: u8 = 0;
const EVENT_INFLIGHT_ADJUSTMENT: u8 = 13;
const EVENT_LOGGING_RESUME: u8 = 14;
const EVENT_DISARM: u8 = 15;
const EVENT_FLIGHTMODE: u8 = 30;
const EVENT_LOG_END: u8 = 255;

const MAX_TIME_JUMP_US: i64 = 10 * 1_000_000;
const MAX_ITERATION_JUMP: i64 = 500 * 10;

const FRAME_MARKERS: &[u8] = b"IPSEGH";

const HIGH_RESOLUTION_SCALE: f64 = 10.0;
const HIGH_RESOLUTION_FIELDS: [&str; 4] = ["gyroADC", "gyroUnfilt", "rcCommand", "setpoint"];
/// The mixer throttle, logged at 1000× regardless.
const HIGH_RESOLUTION_UNSCALED: [&str; 1] = ["setpoint[3]"];

/// Frames of each kind decoded, and what was thrown away.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub i: usize,
    pub p: usize,
    pub s: usize,
    pub e: usize,
    pub corrupt: usize,
    pub dropped_p: usize,
}

/// One decoded log: its header and main-frame columns.
#[derive(Clone, Debug)]
pub struct Log {
    pub header: HashMap<String, String>,
    pub names: Vec<String>,
    /// Row-major, `names.len()` raw values per frame.
    data: Vec<i64>,
    scaled: Vec<bool>,
    index: HashMap<String, usize>,
    pub stats: Stats,
}

impl Log {
    fn new(header: HashMap<String, String>, names: Vec<String>, data: Vec<i64>, stats: Stats) -> Log {
        let high_res = header.get("blackbox_high_resolution").and_then(|v| v.trim().parse::<i64>().ok()).unwrap_or(0) != 0;
        let scaled = names
            .iter()
            .map(|n| {
                high_res
                    && HIGH_RESOLUTION_FIELDS.contains(&n.split('[').next().unwrap_or(""))
                    && !HIGH_RESOLUTION_UNSCALED.contains(&n.as_str())
            })
            .collect();
        let index = names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        Log { header, names, data, scaled, index, stats }
    }

    pub fn len(&self) -> usize {
        if self.names.is_empty() { 0 } else { self.data.len() / self.names.len() }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// A column in the usual units (see the module docs), or None if not logged.
    pub fn column(&self, name: &str) -> Option<Vec<f64>> {
        let &c = self.index.get(name)?;
        let width = self.names.len();
        let scale = if self.scaled[c] { 1.0 / HIGH_RESOLUTION_SCALE } else { 1.0 };
        Some(self.data.iter().skip(c).step_by(width).map(|&v| v as f64 * scale).collect())
    }

    /// A column that must be logged.
    pub fn require(&self, name: &str) -> Result<Vec<f64>, String> {
        self.column(name).ok_or_else(|| format!("the log has no {name} field"))
    }

    pub fn header_str(&self, key: &str) -> Option<&str> {
        self.header.get(key).map(String::as_str)
    }

    /// A header value as comma-separated integers.
    pub fn header_ints(&self, key: &str) -> Option<Vec<i64>> {
        let v = self.header.get(key)?;
        v.split(',').map(|x| x.trim().parse().ok()).collect()
    }

    pub fn header_int(&self, key: &str) -> Option<i64> {
        self.header_ints(key)?.first().copied()
    }
}

#[derive(Debug)]
struct Corrupt;

struct Defs {
    names: Vec<String>,
    signed: Vec<bool>,
    predictor: Vec<i64>,
    encoding: Vec<i64>,
}

fn wrap(value: i64, signed: bool) -> i64 {
    let v = value & 0xFFFF_FFFF;
    if signed && v & 0x8000_0000 != 0 { v - (1 << 32) } else { v }
}

fn sign_extend(value: i64, bits: u32) -> i64 {
    let sign = 1i64 << (bits - 1);
    (value & (sign - 1)) - (value & sign)
}

struct Stream<'a> {
    buf: &'a [u8],
    pos: usize,
    end: usize,
}

impl Stream<'_> {
    fn byte(&mut self) -> Result<i64, Corrupt> {
        if self.pos >= self.end {
            return Err(Corrupt);
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b as i64)
    }

    fn uvb(&mut self) -> Result<i64, Corrupt> {
        let mut result = 0i64;
        for shift in (0..35).step_by(7) {
            let b = self.byte()?;
            result |= (b & 0x7F) << shift;
            if b < 0x80 {
                return Ok(result);
            }
        }
        Err(Corrupt)
    }

    fn svb(&mut self) -> Result<i64, Corrupt> {
        let u = self.uvb()?;
        Ok((u >> 1) ^ -(u & 1))
    }

    fn tag8_8svb(&mut self, count: usize, out: &mut Vec<i64>) -> Result<(), Corrupt> {
        if count == 1 {
            out.push(self.svb()?);
            return Ok(());
        }
        let mut header = self.byte()?;
        for _ in 0..count {
            out.push(if header & 1 != 0 { self.svb()? } else { 0 });
            header >>= 1;
        }
        Ok(())
    }

    fn tag2_3s32(&mut self, out: &mut Vec<i64>) -> Result<(), Corrupt> {
        let lead = self.byte()?;
        match lead >> 6 {
            0 => out.extend([sign_extend(lead >> 4, 2), sign_extend(lead >> 2, 2), sign_extend(lead, 2)]),
            1 => {
                let b = self.byte()?;
                out.extend([sign_extend(lead, 4), sign_extend(b >> 4, 4), sign_extend(b, 4)]);
            }
            2 => {
                let b1 = self.byte()?;
                let b2 = self.byte()?;
                out.extend([sign_extend(lead, 6), sign_extend(b1, 6), sign_extend(b2, 6)]);
            }
            _ => {
                let mut selector = lead;
                for _ in 0..3 {
                    let v = match selector & 3 {
                        0 => sign_extend(self.byte()?, 8),
                        1 => {
                            let b1 = self.byte()?;
                            sign_extend(b1 | (self.byte()? << 8), 16)
                        }
                        2 => {
                            let b1 = self.byte()?;
                            let b2 = self.byte()?;
                            sign_extend(b1 | (b2 << 8) | (self.byte()? << 16), 24)
                        }
                        _ => {
                            let b1 = self.byte()?;
                            let b2 = self.byte()?;
                            let b3 = self.byte()?;
                            sign_extend(b1 | (b2 << 8) | (b3 << 16) | (self.byte()? << 24), 32)
                        }
                    };
                    out.push(v);
                    selector >>= 2;
                }
            }
        }
        Ok(())
    }

    fn tag8_4s16(&mut self, out: &mut Vec<i64>) -> Result<(), Corrupt> {
        let mut selector = self.byte()?;
        let mut nibble = false;
        let mut buffer = 0i64;
        for _ in 0..4 {
            let v = match selector & 3 {
                0 => 0,
                1 => {
                    let v = if !nibble {
                        buffer = self.byte()?;
                        sign_extend(buffer >> 4, 4)
                    } else {
                        sign_extend(buffer, 4)
                    };
                    nibble = !nibble;
                    v
                }
                2 => {
                    if !nibble {
                        sign_extend(self.byte()?, 8)
                    } else {
                        let hi = (buffer << 4) & 0xFF;
                        buffer = self.byte()?;
                        sign_extend(hi | (buffer >> 4), 8)
                    }
                }
                _ => {
                    if !nibble {
                        let b1 = self.byte()?;
                        sign_extend((b1 << 8) | self.byte()?, 16)
                    } else {
                        let b1 = self.byte()?;
                        let b2 = self.byte()?;
                        let v = sign_extend(((buffer << 12) | (b1 << 4) | (b2 >> 4)) & 0xFFFF, 16);
                        buffer = b2;
                        v
                    }
                }
            };
            out.push(v);
            selector >>= 2;
        }
        Ok(())
    }

    /// One frame's worth of encoded values (before prediction).
    fn raw_values(&mut self, encodings: &[i64]) -> Result<Vec<i64>, Corrupt> {
        let n = encodings.len();
        let mut out = Vec::with_capacity(n + 4);
        let mut i = 0;
        while i < n {
            match encodings[i] {
                ENC_SIGNED_VB => {
                    out.push(self.svb()?);
                    i += 1;
                }
                ENC_UNSIGNED_VB => {
                    out.push(self.uvb()?);
                    i += 1;
                }
                ENC_NEG_14BIT => {
                    out.push(-sign_extend(self.uvb()?, 14));
                    i += 1;
                }
                ENC_NULL => {
                    out.push(0);
                    i += 1;
                }
                ENC_TAG8_8SVB => {
                    let mut j = i + 1;
                    while j < i + 8 && j < n && encodings[j] == ENC_TAG8_8SVB {
                        j += 1;
                    }
                    self.tag8_8svb(j - i, &mut out)?;
                    i = j;
                }
                ENC_TAG2_3S32 => {
                    self.tag2_3s32(&mut out)?;
                    i += 3;
                }
                ENC_TAG8_4S16 => {
                    self.tag8_4s16(&mut out)?;
                    i += 4;
                }
                _ => return Err(Corrupt),
            }
        }
        out.truncate(n);
        Ok(out)
    }
}

fn find(buf: &[u8], needle: &[u8], from: usize, to: usize) -> Option<usize> {
    if to < from + needle.len() {
        return None;
    }
    buf[from..to].windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}

fn parse_header(buf: &[u8], mut pos: usize, end: usize) -> (HashMap<String, String>, usize) {
    let mut header = HashMap::new();
    while pos + 2 <= end && &buf[pos..pos + 2] == b"H " {
        let nl = buf[pos..end].iter().position(|&b| b == b'\n').map_or(end, |p| p + pos);
        let line: String = buf[pos + 2..nl].iter().map(|&b| b as char).collect();
        let (key, value) = line.split_once(':').unwrap_or((line.as_str(), ""));
        header.insert(key.to_string(), value.to_string());
        pos = (nl + 1).min(end);
        if nl == end {
            break;
        }
    }
    (header, pos)
}

fn ints(header: &HashMap<String, String>, key: &str) -> Result<Option<Vec<i64>>, String> {
    match header.get(key) {
        None => Ok(None),
        Some(v) if v.is_empty() => Ok(None),
        Some(v) => v
            .split(',')
            .map(|x| x.trim().parse::<i64>().map_err(|e| format!("header {key}: {e}")))
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
    }
}

fn defs(header: &HashMap<String, String>, kind: char, names: Option<&[String]>, signed: Option<&[bool]>) -> Result<Defs, String> {
    let names: Vec<String> = match names {
        Some(n) => n.to_vec(),
        None => header
            .get(&format!("Field {kind} name"))
            .ok_or_else(|| format!("the header has no Field {kind} name"))?
            .split(',')
            .map(str::to_string)
            .collect(),
    };
    let signed = match signed {
        Some(s) => s.to_vec(),
        None => ints(header, &format!("Field {kind} signed"))?
            .map_or_else(|| vec![false; names.len()], |v| v.iter().map(|&x| x != 0).collect()),
    };
    let predictor = ints(header, &format!("Field {kind} predictor"))?
        .ok_or_else(|| format!("the header has no Field {kind} predictor"))?;
    let encoding = ints(header, &format!("Field {kind} encoding"))?
        .ok_or_else(|| format!("the header has no Field {kind} encoding"))?;
    Ok(Defs { names, signed, predictor, encoding })
}

fn decode_log(buf: &[u8], pos: usize, end: usize) -> Result<Log, String> {
    let (header, pos) = parse_header(buf, pos, end);
    if header.get("Data version").map(String::as_str) != Some("2") {
        return Err(format!("unsupported blackbox data version {:?}", header.get("Data version")));
    }
    let idef = defs(&header, 'I', None, None)?;
    let pdef = defs(&header, 'P', Some(&idef.names), Some(&idef.signed))?;
    let sdef = if header.contains_key("Field S name") { Some(defs(&header, 'S', None, None)?) } else { None };
    let nfields = idef.names.len();
    for d in [&idef, &pdef].into_iter().chain(sdef.as_ref()) {
        if d.predictor.len() < d.names.len() || d.signed.len() < d.names.len() {
            return Err("the header's field definitions disagree in length".into());
        }
    }

    let int = |key: &str, default: i64| header.get(key).and_then(|v| v.split(',').next()?.trim().parse().ok()).unwrap_or(default);
    let i_interval = int("I interval", 1).max(1);
    let (p_num, p_denom) = match header.get("P interval").map(String::as_str) {
        Some(v) if v.contains('/') => {
            let (a, b) = v.split_once('/').unwrap();
            (a.trim().parse::<i64>().unwrap_or(1), b.trim().parse::<i64>().unwrap_or(1).max(1))
        }
        Some(v) => (1, v.trim().parse::<i64>().unwrap_or(1).max(1)),
        None => (1, 1),
    };
    let minthrottle = int("minthrottle", 0);
    let vbatref = int("vbatref", 0);
    let minmotor = int("motorOutput", 0);
    let position = |name: &str| idef.names.iter().position(|n| n == name);
    let idx_iter = position("loopIteration").ok_or("the log has no loopIteration field")?;
    let idx_time = position("time").ok_or("the log has no time field")?;
    let idx_motor0 = position("motor[0]");

    let should_have_frame = |index: i64| (index % i_interval + p_num - 1).rem_euclid(p_denom) < p_num;

    let predict = |defs: &Defs, raw: &[i64], prev: Option<&[i64]>, prev2: Option<&[i64]>, skipped: i64| -> Result<Vec<i64>, Corrupt> {
        let mut cur = vec![0i64; nfields];
        for i in 0..nfields {
            let mut v = raw[i];
            match defs.predictor[i] {
                PRED_ZERO => {}
                PRED_PREVIOUS => {
                    if let Some(p) = prev {
                        v += p[i];
                    }
                }
                PRED_STRAIGHT_LINE => {
                    if let (Some(p), Some(p2)) = (prev, prev2) {
                        v += 2 * p[i] - p2[i];
                    }
                }
                PRED_AVERAGE_2 => {
                    if let (Some(p), Some(p2)) = (prev, prev2) {
                        if defs.signed[i] {
                            v += wrap(p[i] + p2[i], true) / 2;
                        } else {
                            v += ((p[i] & 0xFFFF_FFFF) + (p2[i] & 0xFFFF_FFFF)) / 2;
                        }
                    }
                }
                PRED_MINTHROTTLE => v += minthrottle,
                PRED_MOTOR_0 => v += cur[idx_motor0.ok_or(Corrupt)?],
                PRED_INCREMENT => {
                    if let Some(p) = prev {
                        v += skipped + 1 + p[i];
                    }
                }
                PRED_1500 => v += 1500,
                PRED_VBATREF => v += vbatref,
                PRED_MINMOTOR => v += minmotor,
                _ => return Err(Corrupt),
            }
            cur[i] = wrap(v, defs.signed[i]);
        }
        Ok(cur)
    };

    let mut stream = Stream { buf, pos, end };
    let slow_width = sdef.as_ref().map_or(0, |d| d.names.len());
    let mut rows: Vec<i64> = vec![];
    let mut slow_rows: Vec<i64> = vec![];
    let mut slow_cur = vec![0i64; slow_width];
    let mut prev: Option<Vec<i64>> = None;
    let mut prev2: Option<Vec<i64>> = None;
    let mut valid = false;
    let mut last_iter = -1i64;
    let mut last_time = -1i64;
    let mut stats = Stats::default();

    while stream.pos < end {
        let start = stream.pos;
        let kind = buf[start];
        stream.pos += 1;
        let next_is_marker = |s: &Stream| s.pos >= end || FRAME_MARKERS.contains(&buf[s.pos]);
        let sane = |frame: &[i64], last_iter: i64, last_time: i64| {
            if last_iter < 0 {
                return true;
            }
            let (it, t) = (frame[idx_iter], frame[idx_time]);
            last_iter <= it && it <= last_iter + MAX_ITERATION_JUMP && last_time <= t && t <= last_time + MAX_TIME_JUMP_US
        };
        let result: Result<bool, Corrupt> = (|| {
            match kind {
                b'I' => {
                    let raw = stream.raw_values(&idef.encoding)?;
                    let frame = predict(&idef, &raw, prev.as_deref(), prev.as_deref(), 0)?;
                    if !next_is_marker(&stream) || !sane(&frame, last_iter, last_time) {
                        return Err(Corrupt);
                    }
                    last_iter = frame[idx_iter];
                    last_time = frame[idx_time];
                    rows.extend_from_slice(&frame);
                    slow_rows.extend_from_slice(&slow_cur);
                    prev2 = Some(frame.clone());
                    prev = Some(frame);
                    valid = true;
                    stats.i += 1;
                }
                b'P' => {
                    let raw = stream.raw_values(&pdef.encoding)?;
                    if !valid {
                        if !next_is_marker(&stream) {
                            return Err(Corrupt);
                        }
                        stats.dropped_p += 1;
                        return Ok(false);
                    }
                    let mut skipped = 0;
                    let mut index = last_iter + 1;
                    while !should_have_frame(index) {
                        skipped += 1;
                        index += 1;
                    }
                    let frame = predict(&pdef, &raw, prev.as_deref(), prev2.as_deref(), skipped)?;
                    if !next_is_marker(&stream) || !sane(&frame, last_iter, last_time) {
                        return Err(Corrupt);
                    }
                    last_iter = frame[idx_iter];
                    last_time = frame[idx_time];
                    rows.extend_from_slice(&frame);
                    slow_rows.extend_from_slice(&slow_cur);
                    prev2 = prev.take();
                    prev = Some(frame);
                    stats.p += 1;
                }
                b'S' if sdef.is_some() => {
                    let sd = sdef.as_ref().unwrap();
                    let raw = stream.raw_values(&sd.encoding)?;
                    if !next_is_marker(&stream) {
                        return Err(Corrupt);
                    }
                    slow_cur = raw.iter().zip(&sd.signed).map(|(&v, &s)| wrap(v, s)).collect();
                    if stats.s == 0 {
                        for row in slow_rows.chunks_mut(slow_width) {
                            row.copy_from_slice(&slow_cur);
                        }
                    }
                    stats.s += 1;
                }
                b'E' => {
                    let etype = stream.byte()? as u8;
                    match etype {
                        EVENT_SYNC_BEEP => {
                            stream.uvb()?;
                        }
                        EVENT_INFLIGHT_ADJUSTMENT => {
                            let func = stream.byte()?;
                            if func & 0x80 != 0 {
                                for _ in 0..4 {
                                    stream.byte()?;
                                }
                            } else {
                                stream.svb()?;
                            }
                        }
                        EVENT_LOGGING_RESUME => {
                            let it = stream.uvb()?;
                            let t = stream.uvb()?;
                            last_iter = it;
                            last_time = t;
                            valid = false;
                        }
                        EVENT_DISARM => {
                            stream.uvb()?;
                        }
                        EVENT_FLIGHTMODE => {
                            stream.uvb()?;
                            stream.uvb()?;
                        }
                        EVENT_LOG_END => {
                            if buf.get(stream.pos..stream.pos + LOG_END_MSG.len()) != Some(LOG_END_MSG) {
                                return Err(Corrupt);
                            }
                            stats.e += 1;
                            return Ok(true);
                        }
                        _ => return Err(Corrupt),
                    }
                    if !next_is_marker(&stream) {
                        return Err(Corrupt);
                    }
                    stats.e += 1;
                }
                _ => return Err(Corrupt),
            }
            Ok(false)
        })();
        match result {
            Ok(true) => break,
            Ok(false) => {}
            Err(Corrupt) => {
                stats.corrupt += 1;
                valid = false;
                stream.pos = start + 1;
                while stream.pos < end && !FRAME_MARKERS.contains(&buf[stream.pos]) {
                    stream.pos += 1;
                }
            }
        }
    }

    let mut names = idef.names.clone();
    if let Some(sd) = &sdef {
        names.extend(sd.names.iter().cloned());
    }
    let n = rows.len() / nfields.max(1);
    let mut data = Vec::with_capacity(n * names.len());
    for k in 0..n {
        data.extend_from_slice(&rows[k * nfields..(k + 1) * nfields]);
        data.extend_from_slice(&slow_rows[k * slow_width..(k + 1) * slow_width]);
    }
    Ok(Log::new(header, names, data, stats))
}

/// Every non-empty log in a blackbox file's bytes.
pub fn decode(buf: &[u8]) -> Result<Vec<Log>, String> {
    let mut starts = vec![];
    let mut pos = 0;
    while let Some(p) = find(buf, LOG_START, pos, buf.len()) {
        starts.push(p);
        pos = p + 1;
    }
    let mut logs = vec![];
    for (i, &start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(buf.len());
        let log = decode_log(buf, start, end)?;
        if !log.is_empty() {
            logs.push(log);
        }
    }
    Ok(logs)
}

/// Every non-empty log in a blackbox file.
pub fn read_logs(path: &std::path::Path) -> Result<Vec<Log>, String> {
    let buf = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode(&buf).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_and_sign_extension() {
        let buf = [0x96, 0x01, 0x03, 0xFF];
        let mut s = Stream { buf: &buf, pos: 0, end: 4 };
        assert_eq!(s.uvb().unwrap(), 150);
        assert_eq!(s.svb().unwrap(), -2);
        assert!(s.uvb().is_err());
        assert_eq!(sign_extend(0b11, 2), -1);
        assert_eq!(wrap(0xFFFF_FFFF, true), -1);
        assert_eq!(wrap(-1, false), 0xFFFF_FFFF);
    }

    fn log_with(header: &[(&str, &str)], names: &[&str], data: Vec<i64>) -> Log {
        let header = header.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        Log::new(header, names.iter().map(|s| s.to_string()).collect(), data, Stats::default())
    }

    #[test]
    fn high_resolution_divides_rates_but_not_the_mixer_throttle() {
        let names = ["gyroADC[0]", "setpoint[0]", "setpoint[3]", "motor[0]"];
        let hi = log_with(&[("blackbox_high_resolution", "1")], &names, vec![5400, 3600, 500, 1200]);
        assert_eq!(hi.column("gyroADC[0]").unwrap(), vec![540.0]);
        assert_eq!(hi.column("setpoint[0]").unwrap(), vec![360.0]);
        assert_eq!(hi.column("setpoint[3]").unwrap(), vec![500.0]);
        assert_eq!(hi.column("motor[0]").unwrap(), vec![1200.0]);
        let lo = log_with(&[("blackbox_high_resolution", "0")], &names, vec![5400, 3600, 500, 1200]);
        assert_eq!(lo.column("gyroADC[0]").unwrap(), vec![5400.0]);
    }
}
