// strlib.cy — pure string helpers, zero capabilities.
// Shared by every .cy module; det end to end.

host det fn trim(s: str) -> str {
    var lo = 0;
    var hi = len(s);
    var done = false;
    while lo < hi {
        if done { break; }
        let c = slice_str(s, lo, lo + 1);
        if c == " " || c == "\t" || c == "\r" || c == "\n" { lo = lo + 1; } else { done = true; }
    }
    done = false;
    while hi > lo {
        if done { break; }
        let c = slice_str(s, hi - 1, hi);
        if c == " " || c == "\t" || c == "\r" || c == "\n" { hi = hi - 1; } else { done = true; }
    }
    return slice_str(s, lo, hi);
}

host det fn is_hex(s: str) -> bool {
    if len(s) == 0 { return false; }
    let hexd = "0123456789abcdefABCDEF";
    for i in 0..len(s) {
        if !contains(hexd, slice_str(s, i, i + 1)) { return false; }
    }
    return true;
}

host det fn dir_of(path: str) -> str {
    var last = -1;
    for i in 0..len(path) {
        if slice_str(path, i, i + 1) == "/" { last = i; }
    }
    if last < 0 { return "."; }
    return slice_str(path, 0, last);
}

host det fn starts_with(s: str, pre: str) -> bool {
    if len(pre) > len(s) { return false; }
    return slice_str(s, 0, len(pre)) == pre;
}

host det fn ends_with(s: str, suf: str) -> bool {
    if len(suf) > len(s) { return false; }
    return slice_str(s, len(s) - len(suf), len(s)) == suf;
}

// first index of `needle` in `hay`, -1 when absent
host det fn find_sub(hay: str, needle: str) -> i64 {
    if len(needle) == 0 { return 0; }
    for i in 0..(len(hay) - len(needle) + 1) {
        if slice_str(hay, i, i + len(needle)) == needle { return i; }
    }
    return -1;
}

host det fn lines(s: str) -> [str] {
    return split(s, "\n");
}

host det fn hex_val(c: str) -> i64 {
    let d = "0123456789abcdef";
    var v = -1;
    for j in 0..16 {
        if slice_str(d, j, j + 1) == c { v = j; }
    }
    if v < 0 {
        let D = "0123456789ABCDEF";
        for j in 10..16 {
            if slice_str(D, j, j + 1) == c { v = j; }
        }
    }
    return v;
}

host det fn hex_to_i64(h: str) -> i64 {
    var n = 0;
    for i in 0..len(h) {
        let v = hex_val(slice_str(h, i, i + 1));
        if v < 0 { return -1; }
        n = n * 16 + v;
    }
    return n;
}

host det fn u32be_hex(n: i64) -> str {
    let d = "0123456789abcdef";
    var acc = "";
    var div = 268435456; // 16^7
    for i in 0..8 {
        let nib = (n / div) % 16;
        acc = acc + slice_str(d, nib, nib + 1);
        div = div / 16;
    }
    return acc;
}

host det fn byte_hex(b: i64) -> str {
    let d = "0123456789abcdef";
    let hi = (b / 16) % 16;
    let lo = b % 16;
    return slice_str(d, hi, hi + 1) + slice_str(d, lo, lo + 1);
}

// little-endian ints in hex form — safetensors headers, i32 payloads
host det fn hex_le_i64(h: str) -> i64 {
    var n = 0;
    var mul = 1;
    for i in 0..(len(h) / 2) {
        let byte = hex_to_i64(slice_str(h, i * 2, i * 2 + 2));
        if byte < 0 { return -1; }
        n = n + byte * mul;
        mul = mul * 256;
    }
    return n;
}

host det fn i64_le_hex(n: i64, bytes: i64) -> str {
    var acc = "";
    var v = n;
    for i in 0..bytes {
        acc = acc + byte_hex(v % 256);
        v = v / 256;
    }
    return acc;
}

// signed i32 from 4 LE hex bytes
host det fn hex_le_i32(h: str) -> i64 {
    var n = hex_le_i64(slice_str(h, 0, 8));
    if n >= 2147483648 { n = n - 4294967296; }
    return n;
}

// replace every occurrence of `old` with `new`
host det fn replace_all(s: str, old: str, new: str) -> str {
    if len(old) == 0 { return s; }
    var r = "";
    var i = 0;
    let n = len(s);
    let m = len(old);
    while i < n {
        if i + m <= n {
            if slice_str(s, i, i + m) == old {
                r = r + new;
                i = i + m;
            } else {
                r = r + slice_str(s, i, i + 1);
                i = i + 1;
            }
        } else {
            r = r + slice_str(s, i, i + 1);
            i = i + 1;
        }
    }
    return r;
}

// last index of `needle` in `hay`, -1 when absent
host det fn rfind_sub(hay: str, needle: str) -> i64 {
    var last = -1;
    if len(needle) == 0 { return len(hay); }
    for i in 0..(len(hay) - len(needle) + 1) {
        if slice_str(hay, i, i + len(needle)) == needle { last = i; }
    }
    return last;
}

// split on runs of whitespace (space, tab, newline), skipping empties
host det fn split_ws(s: str) -> [str] {
    var words: [str] = [];
    var cur = "";
    for i in 0..len(s) {
        let c = slice_str(s, i, i + 1);
        if c == " " || c == "\t" || c == "\n" || c == "\r" {
            if len(cur) > 0 { push(words, cur); cur = ""; }
        } else {
            cur = cur + c;
        }
    }
    if len(cur) > 0 { push(words, cur); }
    return words;
}

host det fn trim_end(s: str) -> str {
    var hi = len(s);
    var done = false;
    while hi > 0 {
        if done { break; }
        let c = slice_str(s, hi - 1, hi);
        if c == " " || c == "\t" || c == "\r" || c == "\n" { hi = hi - 1; } else { done = true; }
    }
    return slice_str(s, 0, hi);
}

// count occurrences of a single-char needle
host det fn count_char(s: str, ch: str) -> i64 {
    var n = 0;
    for i in 0..len(s) {
        if slice_str(s, i, i + 1) == ch { n += 1; }
    }
    return n;
}
