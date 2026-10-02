// coreml_crash_guard.h — C shim for SIGSEGV/SIGBUS crash isolation.
//
// Swift cannot call sigsetjmp/siglongjmp directly (returns_twice attribute
// is rejected). The jump target must also stay on the stack while the
// guarded code runs, so the C side owns the frame and calls back into Swift.

#ifndef COREML_CRASH_GUARD_H
#define COREML_CRASH_GUARD_H

/// Run `body(ctx)` with SIGSEGV/SIGBUS protection on the calling thread.
/// Returns 0 if `body` completed, 1 if a fault was caught. After a caught
/// fault, whatever `body` was touching is in an undefined state — callers
/// must discard it rather than reuse it.
int coreml_guard_run(void (*body)(void *ctx), void *ctx);

#endif // COREML_CRASH_GUARD_H
