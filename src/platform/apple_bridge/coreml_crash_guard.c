// coreml_crash_guard.c — C shim for SIGSEGV/SIGBUS crash isolation.
//
// The handler is installed once per process and chains to whatever handler
// was there before. Only threads currently inside coreml_guard_run (armed,
// with a live jump buffer on their own stack) are rescued; every other fault
// goes to the previous handler exactly as if this shim did not exist.

#include "coreml_crash_guard.h"
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <string.h>

static _Thread_local sigjmp_buf *_guard_jmpbuf = NULL;
static struct sigaction _prev_segv;
static struct sigaction _prev_bus;
static pthread_once_t _install_once = PTHREAD_ONCE_INIT;

static void _chain(int sig, siginfo_t *info, void *uctx) {
    struct sigaction *prev = (sig == SIGSEGV) ? &_prev_segv : &_prev_bus;
    if (prev->sa_flags & SA_SIGINFO) {
        if (prev->sa_sigaction) {
            prev->sa_sigaction(sig, info, uctx);
            return;
        }
    } else if (prev->sa_handler != SIG_DFL && prev->sa_handler != SIG_IGN) {
        prev->sa_handler(sig);
        return;
    }
    // Default disposition: restore it and re-raise so the process crashes
    // (and reports) normally.
    signal(sig, SIG_DFL);
    raise(sig);
}

static void _crash_handler(int sig, siginfo_t *info, void *uctx) {
    sigjmp_buf *target = _guard_jmpbuf;
    if (target) {
        _guard_jmpbuf = NULL;
        siglongjmp(*target, 1);
    }
    _chain(sig, info, uctx);
}

static void _install(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = _crash_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &_prev_segv);
    sigaction(SIGBUS, &sa, &_prev_bus);
}

int coreml_guard_run(void (*body)(void *ctx), void *ctx) {
    pthread_once(&_install_once, _install);

    sigjmp_buf env;
    sigjmp_buf *outer = _guard_jmpbuf; // support nesting
    if (sigsetjmp(env, 1) != 0) {
        _guard_jmpbuf = outer;
        return 1;
    }
    _guard_jmpbuf = &env;
    body(ctx);
    _guard_jmpbuf = outer;
    return 0;
}
