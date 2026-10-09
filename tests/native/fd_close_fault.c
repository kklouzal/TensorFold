/* Isolated LD_PRELOAD qualification only. One live ordinary owned descriptor,
 * one arming thread, no outside reassignment, no foreign pthread cancellation.
 * The injected failure follows an actual successful libc close and actual
 * replacement open. Unarmed descriptors use the real libc close unchanged.
 */
#define _GNU_SOURCE
#include "fd_close_fault.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef __linux__
#error "This qualification shim targets Linux only"
#endif

_Static_assert(sizeof(struct tf_fd_fault_stats) == 14 * sizeof(int64_t), "test protocol size");
static pthread_mutex_t guard = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t resolver = PTHREAD_ONCE_INIT;
static int (*real_close)(int);
static pthread_t arming_thread;
static struct stat target_identity;
static char replacement_path[PATH_MAX];
static struct tf_fd_fault_stats stats;

static void resolve_close(void)
{
    /* POSIX dlsym supports this function-pointer conversion. No native FD is
     * armed during loader initialization; a missing supplier aborts the test.
     */
    dlerror();
    real_close = (int (*)(int))dlsym(RTLD_NEXT, "close");
    if (dlerror() != NULL || real_close == NULL) {
        _exit(127);
    }
}

static void lock_state(void)
{
    if (pthread_mutex_lock(&guard) != 0) {
        _exit(126);
    }
}

static void unlock_state(void)
{
    if (pthread_mutex_unlock(&guard) != 0) {
        _exit(126);
    }
}

static void initialize(void)
{
    if (pthread_once(&resolver, resolve_close) != 0) {
        _exit(126);
    }
}

TF_FD_FAULT_EXPORT unsigned int tf_fd_fault_stats_size(void)
{
    return (unsigned int)sizeof(struct tf_fd_fault_stats);
}

TF_FD_FAULT_EXPORT int tf_fd_fault_arm(int descriptor, int code, const char *replacement)
{
    struct stat identity;
    size_t length;
    initialize();
    if (descriptor < 0 || (code != 0 && code != EIO && code != EINTR) || replacement == NULL) {
        return EINVAL;
    }
    length = strnlen(replacement, sizeof(replacement_path));
    if (length == 0 || length == sizeof(replacement_path)) {
        return ENAMETOOLONG;
    }
    if (fstat(descriptor, &identity) < 0) {
        return errno;
    }
    if (!S_ISREG(identity.st_mode)) {
        return EINVAL;
    }
    lock_state();
    if (stats.phase != 0) {
        unlock_state();
        return EBUSY;
    }
    memset(&stats, 0, sizeof(stats));
    stats.version = 1;
    stats.phase = 1;
    stats.target_fd = descriptor;
    stats.injected_errno = code;
    stats.underlying_result = -2; /* not called */
    stats.replacement_fd = -1;
    target_identity = identity;
    arming_thread = pthread_self();
    memcpy(replacement_path, replacement, length + 1);
    unlock_state();
    return 0;
}

TF_FD_FAULT_EXPORT int tf_fd_fault_snapshot(struct tf_fd_fault_stats *out)
{
    if (out == NULL) {
        return EINVAL;
    }
    lock_state();
    *out = stats;
    unlock_state();
    return 0;
}

TF_FD_FAULT_EXPORT int tf_fd_fault_reset(void)
{
    /* Acquiring this mutex awaits any underlying close/open sequence. Reset
     * disarms but never consumes the replacement: the test explicitly owns it.
     */
    lock_state();
    if (stats.phase != 0 && !pthread_equal(arming_thread, pthread_self())) {
        unlock_state();
        return EPERM;
    }
    memset(&stats, 0, sizeof(stats));
    stats.version = 1;
    stats.target_fd = -1;
    stats.replacement_fd = -1;
    replacement_path[0] = '\0';
    unlock_state();
    return 0;
}

TF_FD_FAULT_EXPORT int close(int descriptor);
TF_FD_FAULT_EXPORT int close(int descriptor)
{
    int entry_errno = errno;
    int result, result_errno;
    struct stat observed;
    initialize();
    lock_state();
    if (stats.phase == 1 && descriptor == stats.target_fd && pthread_equal(arming_thread, pthread_self())) {
        if (fstat(descriptor, &observed) < 0
            || observed.st_dev != target_identity.st_dev || observed.st_ino != target_identity.st_ino) {
            ++stats.identity_mismatches;
            stats.setup_error = ESTALE;
            unlock_state();
            errno = ESTALE;
            return -1; /* refuse to close a foreign replacement */
        }
        errno = entry_errno;
        result = real_close(descriptor);
        result_errno = errno;
        ++stats.underlying_calls;
        stats.underlying_result = result;
        stats.underlying_errno = result < 0 ? result_errno : 0;
        if (result < 0) {
            stats.setup_error = result_errno;
            unlock_state();
            errno = result_errno;
            return result; /* no synthetic evidence after a real close error */
        }
        stats.phase = 2;
        if (fcntl(descriptor, F_GETFD) == -1 && errno == EBADF) {
            stats.consumed_witness = 1;
        } else {
            stats.setup_error = EBUSY;
        }
        result = openat(AT_FDCWD, replacement_path, O_RDONLY | O_CLOEXEC);
        result_errno = errno;
        stats.replacement_fd = result;
        stats.replacement_errno = result < 0 ? result_errno : 0;
        if (result < 0) {
            stats.setup_error = result_errno;
        } else if (result != descriptor) {
            /* Do not dup2 over any unrelated descriptor to manufacture reuse. */
            if (real_close(result) < 0) {
                stats.setup_error = errno;
            } else {
                stats.setup_error = EBUSY;
            }
            stats.replacement_fd = -1;
        }
        result_errno = stats.setup_error != 0 ? (int)stats.setup_error : (int)stats.injected_errno;
        unlock_state();
        errno = result_errno != 0 ? result_errno : entry_errno;
        return result_errno != 0 ? -1 : 0;
    }
    if (stats.phase == 2 && descriptor == stats.target_fd) {
        ++stats.retry_calls; /* observe a bug on any thread; never inject twice */
    } else {
        ++stats.passthrough_calls;
    }
    unlock_state();
    errno = entry_errno;
    return real_close(descriptor);
}
