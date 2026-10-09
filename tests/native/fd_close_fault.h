/* Project-owned Linux qualification shim; never link into a production binary. */
#ifndef TENSORFOLD_FD_CLOSE_FAULT_H
#define TENSORFOLD_FD_CLOSE_FAULT_H

#include <stdint.h>

/* Fixed-width native test protocol, checked against the ctypes declaration. */
struct tf_fd_fault_stats {
    int64_t version;
    int64_t phase; /* 0: reset, 1: armed, 2: original descriptor consumed */
    int64_t target_fd;
    int64_t injected_errno;
    int64_t underlying_calls;
    int64_t underlying_result;
    int64_t underlying_errno;
    int64_t consumed_witness;
    int64_t replacement_fd;
    int64_t replacement_errno;
    int64_t retry_calls;
    int64_t passthrough_calls;
    int64_t identity_mismatches;
    int64_t setup_error;
};

#define TF_FD_FAULT_EXPORT __attribute__((visibility("default")))
TF_FD_FAULT_EXPORT int tf_fd_fault_arm(int descriptor, int code, const char *replacement);
TF_FD_FAULT_EXPORT int tf_fd_fault_snapshot(struct tf_fd_fault_stats *out);
TF_FD_FAULT_EXPORT int tf_fd_fault_reset(void);
TF_FD_FAULT_EXPORT unsigned int tf_fd_fault_stats_size(void);

#endif
