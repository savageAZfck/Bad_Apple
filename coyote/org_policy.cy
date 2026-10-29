// org_policy.cy — org-signed policy, ported from src/org_policy.rs.
//
// Ed25519 signatures over policy.yaml. Personal installs are sovereign —
// no trust root means owner-edited policy. Once org_policy.pub exists the
// engine must verify policy.yaml.sig against the pinned key; tampered or
// unsigned fails closed. Keys/sigs are hex text files, matching the Swift
// policy engine's CryptoKit format.

import "sys.cy";
import "strlib.cy";

cap fs, crypto

host det fn org_pub_path() -> str { return "/var/lib/bad_apple/org_policy.pub"; }
host det fn org_sig_path() -> str { return "/var/lib/bad_apple/policy.yaml.sig"; }
host det fn org_policy_path() -> str { return "/var/lib/bad_apple/policy.yaml"; }

host fn org_mode_active() -> bool cap fs {
    return fs_exists(org_pub_path());
}

// returns {key: str, pub: str} hex paths, or {error: str}
host fn keygen(dir: str) -> Json cap fs, crypto {
    fs_mkdir(dir);
    let kp = dir + "/org_policy.key";
    let pp = dir + "/org_policy.pub";
    if fs_exists(kp) || fs_exists(pp) {
        return Json::Obj({ "error": Json::Str("refusing to overwrite existing keys") });
    }
    let sec = ed25519_keygen();
    if sec == "" {
        return Json::Obj({ "error": Json::Str("keygen failed") });
    }
    write_file(kp, sec);
    fs_chmod(kp, 384); // 0600
    write_file(pp, ed25519_pub(sec));
    return Json::Obj({ "key": Json::Str(kp), "pub": Json::Str(pp) });
}

host fn sign_file(policy: str, key_path: str, sig_path: str) -> bool cap fs, crypto {
    if !fs_exists(policy) || !fs_exists(key_path) { return false; }
    let policy_hex = fs_read_hex(policy);
    let sec = trim(read_file(key_path));
    let sig = ed25519_sign(sec, policy_hex);
    if sig == "" { return false; }
    write_file(sig_path, sig);
    return true;
}

// tri-state — matches the Rust Result<bool>: 1 valid, 0 invalid/missing
// signature, -1 unreadable inputs (fails closed upstream either way).
host fn verify_file(policy: str, sig_path: str, pub_path: str) -> i64 cap fs, crypto {
    if !fs_exists(policy) || !fs_exists(pub_path) { return -1; }
    let policy_hex = fs_read_hex(policy);
    let pub_hex = trim(read_file(pub_path));
    if pub_hex == "" || !is_hex(pub_hex) { return -1; }
    if !fs_exists(sig_path) { return 0; }
    let sig = trim(read_file(sig_path));
    if sig == "" || !is_hex(sig) { return 0; }
    if ed25519_verify(pub_hex, policy_hex, sig) { return 1; }
    return 0;
}
