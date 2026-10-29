// watcher.cy — workspace change detection, in Coyote.
//
// Port of the policy half of src/workspace_watcher.rs. The FSEvents
// callback loop stays in the shell (the VM is synchronous); what lives
// here is the file filter and a polling scan that answers the same
// question the supervisor actually asks — "what changed since T" —
// without a resident notify thread.

cap fs

import "sys.cy";
import "strlib.cy";

// Default workspace filter: text and code documents safe to index.
// Same extension list as workspace_watcher::default_file_filter.
host det fn watch_extensions() -> [str] {
    return ["txt","md","markdown","rs","swift","py","js","ts","tsx","jsx",
        "c","cpp","h","hpp","m","mm","sh","zsh","bash","json","yaml","yml",
        "toml","css","html","xml","plist","sql"];
}

host det fn default_file_filter(path: str) -> bool {
    let dot = rfind_sub(path, ".");
    let slash = rfind_sub(path, "/");
    if dot >= 0 && dot > slash {
        let ext = str_lower(slice_str(path, dot + 1, len(path)));
        let exts = watch_extensions();
        for i in 0..len(exts) {
            if exts[i] == ext { return true; }
        }
        return false;
    }
    // no extension — accept anything with a file name component
    return len(path) > 0 && slice_str(path, len(path) - 1, len(path)) != "/";
}

// changed since since_ns — polling alternative to the FSEvents loop for
// supervisor-cadence rescans. → [relative-ish paths under root]
host fn watch_scan(root: str, since_ns: i64, depth: i64) -> [str] cap fs {
    var out_v: [str] = [];
    var canon = fs_realpath(root);
    if canon == "" { canon = root; }
    let files = fs_walk(canon, depth);
    match files {
        Json::Arr(fa) => {
            for i in 0..len(fa) {
                match fa[i] {
                    Json::Obj(f) => {
                        if jnumi_(f, "mtime") > since_ns {
                            let p = jstr_(f, "path");
                            if default_file_filter(p) {
                                // strip the canonical root prefix when present
                                var rel = p;
                                if starts_with(rel, canon + "/") {
                                    rel = slice_str(rel, len(canon) + 1, len(rel));
                                }
                                push(out_v, rel);
                            }
                        }
                    }
                    _ => { }
                }
            }
        }
        _ => { }
    }
    return out_v;
}

host det fn jstr_(o: {str: Json}, k: str) -> str {
    if has(o, k) {
        match o[k] { Json::Str(s) => { return s; } _ => { } }
    }
    return "";
}

host det fn jnumi_(o: {str: Json}, k: str) -> i64 {
    if has(o, k) {
        match o[k] { Json::Num(n) => { return i64(n); } _ => { } }
    }
    return 0;
}
