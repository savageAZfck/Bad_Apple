// vault.cy — the encrypted secrets vault, in Coyote.
//
// Port of src/vault.rs: AES-256-GCM entries at rest, master key derived
// from the SLICKS secret so the vault is tied to the install. The wire
// format is preserved — [nonce||ct||tag] over a JSON payload
// {"entries":{k:v},"version":1} — so vaults written by either
// implementation open in the other.

cap fs, crypto

import "sys.cy";
import "strlib.cy";

host det fn vault_file() -> str { return "vault.enc"; }

host det fn default_vault_path() -> str {
    let d = env_get("BADAPPLE_DATA_DIR");
    if d != "" { return d + "/" + vault_file(); }
    return "/var/lib/bad_apple/" + vault_file();
}

// master key = sha256(secret bytes) → hex. Sources in precedence order:
// BADAPPLE_VAULT_KEY → SLICKS key file → BADAPPLE_SLICKS_SECRET.
host fn master_key_hex() -> str cap fs {
    let vk = env_get("BADAPPLE_VAULT_KEY");
    if vk != "" { return sha256(vk); }
    let kp = env_get("BADAPPLE_SLICKS_KEY_PATH");
    if kp != "" {
        if !fs_exists(kp) { return ""; }
        let raw = trim(read_file(kp));
        if raw == "" { return ""; }
        return sha256(raw);
    }
    let ss = env_get("BADAPPLE_SLICKS_SECRET");
    if ss != "" { return sha256(ss); }
    return "";
}

// decrypt vault.enc → Json Obj {entries: Obj, version: Num} (empty vault
// when absent; Null on decrypt failure — wrong key or corruption)
host fn vault_load(path: str, key_hex: str) -> Json cap fs, crypto {
    var empty_entries: {str: Json} = {};
    if !fs_exists(path) {
        var fresh: {str: Json} = {};
        fresh["entries"] = Json::Obj(empty_entries);
        fresh["version"] = Json::Num(1.0);
        return Json::Obj(fresh);
    }
    let ct = fs_read_hex(path);
    if ct == "" { return Json::Null; }
    let pt_hex = aead_open(key_hex, ct);
    if pt_hex == "" { return Json::Null; }
    let v = json_parse(hex_decode(pt_hex));
    match v {
        Json::Obj(_) => { return v; }
        _ => { return Json::Null; }
    }
}

// encrypt + atomic write (tmp + rename)
host fn vault_save(path: str, key_hex: str, payload: Json) -> bool cap fs, crypto {
    fs_mkdir(dir_of(path));
    let ct = aead_seal(key_hex, hex_encode(json_emit(payload)));
    if ct == "" { return false; }
    let tmp = path + ".tmp";
    if !fs_write_hex(tmp, ct) { return false; }
    return fs_rename(tmp, path);
}

host det fn key_valid(key: str) -> bool {
    if len(key) == 0 { return false; }
    if contains(key, "/") { return false; }
    if contains(key, "..") { return false; }
    return true;
}

host det fn jobj(o: {str: Json}, k: str) -> {str: Json} {
    if has(o, k) {
        match o[k] { Json::Obj(m) => { return m; } _ => { } }
    }
    var empty: {str: Json} = {};
    return empty;
}

// vault_set(path, key, value) → true on success
host fn vault_set(path: str, key: str, value: str) -> bool cap fs, crypto {
    if !key_valid(key) { return false; }
    let khex = master_key_hex();
    if khex == "" { return false; }
    let payload = vault_load(path, khex);
    match payload {
        Json::Obj(m) => {
            var entries = jobj(m, "entries");
            entries[key] = Json::Str(value);
            m["entries"] = Json::Obj(entries);
            return vault_save(path, khex, Json::Obj(m));
        }
        _ => { return false; }
    }
}

// vault_get(path, key) → value or ""
host fn vault_get(path: str, key: str) -> str cap fs, crypto {
    let khex = master_key_hex();
    if khex == "" { return ""; }
    let payload = vault_load(path, khex);
    match payload {
        Json::Obj(m) => {
            let entries = jobj(m, "entries");
            if has(entries, key) {
                match entries[key] {
                    Json::Str(v) => { return v; }
                    _ => { }
                }
            }
            return "";
        }
        _ => { return ""; }
    }
}

// → true when the key existed
host fn vault_remove(path: str, key: str) -> bool cap fs, crypto {
    let khex = master_key_hex();
    if khex == "" { return false; }
    let payload = vault_load(path, khex);
    match payload {
        Json::Obj(m) => {
            var entries = jobj(m, "entries");
            let had = has(entries, key);
            // "remove" — store entries minus the key by rebuilding
            var kept: {str: Json} = {};
            let ks = keys(entries);
            for i in 0..len(ks) {
                if ks[i] != key { kept[ks[i]] = entries[ks[i]]; }
            }
            m["entries"] = Json::Obj(kept);
            let ok = vault_save(path, khex, Json::Obj(m));
            return had && ok;
        }
        _ => { return false; }
    }
}

// list keys (not values), sorted — matches Rust's keys.sort()
host fn vault_list(path: str) -> [str] cap fs, crypto {
    var out_v: [str] = [];
    let khex = master_key_hex();
    if khex == "" { return out_v; }
    let payload = vault_load(path, khex);
    match payload {
        Json::Obj(m) => {
            let entries = jobj(m, "entries");
            let ks = keys(entries);
            // insertion sort — key counts are small
            for i in 0..len(ks) {
                var ins = len(out_v);
                var j = 0;
                while j < len(out_v) {
                    if ks[i] < out_v[j] {
                        ins = j;
                        j = len(out_v);
                    }
                    j += 1;
                }
                var tmp: [str] = [];
                var a = 0;
                while a < ins { push(tmp, out_v[a]); a += 1; }
                push(tmp, ks[i]);
                var b = ins;
                while b < len(out_v) { push(tmp, out_v[b]); b += 1; }
                out_v = tmp;
            }
            return out_v;
        }
        _ => { return out_v; }
    }
}
