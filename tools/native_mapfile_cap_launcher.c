/* Qualification-only Linux exec bridge. No UID/GID transitions or libcap.
 * Build/install ONLY as the root-owned, read-only file-capability executable
 * in the isolated verification image. Protected Python is never modified.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/capability.h>
#include <limits.h>
#include <stdlib.h>
#include <linux/prctl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <unistd.h>

_Static_assert(CAP_CHECKPOINT_RESTORE == 40, "review capability ABI");
_Static_assert(_LINUX_CAPABILITY_U32S_3 == 2, "review capability ABI");
static const char python_path[] = "/opt/venv/bin/python3";
static const uint32_t bit = UINT32_C(1) << (CAP_CHECKPOINT_RESTORE - 32);
extern char **environ;

static int refuse(const char *phase, int error)
{
    /* Fixed labels/integers only: no argv/environment/secret rendering. */
    char record[256];
    int length = snprintf(record, sizeof record,
        "MAPFILE_CAP_LAUNCH_REFUSED phase=%s errno=%d\n", phase, error);
    if (length > 0 && (size_t)length < sizeof record) {
        size_t written = 0;
        while (written < (size_t)length) {
            ssize_t count = write(STDERR_FILENO, record + written, (size_t)length - written);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) break;
            written += (size_t)count;
        }
    }
    return 125;
}

static int ids(uid_t *uid, gid_t *gid)
{
    uid_t real, effective, saved;
    gid_t gr, ge, gs;
    if (getresuid(&real, &effective, &saved) || getresgid(&gr, &ge, &gs)) return -1;
    if (!real || real != effective || real != saved || !gr || gr != ge || gr != gs) {
        errno = EPERM;
        return -1;
    }
    *uid = real; *gid = gr;
    return 0;
}

static int caps(int prepared)
{
    struct __user_cap_header_struct header = {_LINUX_CAPABILITY_VERSION_3, 0};
    struct __user_cap_data_struct values[_LINUX_CAPABILITY_U32S_3] = {{0}};
    if (syscall(SYS_capget, &header, values)) return -1;
    if (values[0].effective || values[0].permitted || values[0].inheritable ||
        values[1].effective != bit || values[1].permitted != bit ||
        values[1].inheritable != (prepared ? bit : 0)) {
        errno = EPERM;
        return -1;
    }
    for (unsigned cap = 0; cap < 64; ++cap) {
        errno = 0;
        int bounded = prctl(PR_CAPBSET_READ, (unsigned long)cap, 0L, 0L, 0L);
        if (bounded < 0) {
            if (errno == EINVAL && cap > CAP_CHECKPOINT_RESTORE) break;
            return -1;
        }
        int ambient = prctl(PR_CAP_AMBIENT, (unsigned long)PR_CAP_AMBIENT_IS_SET, (unsigned long)cap, 0L, 0L);
        if (ambient < 0) return -1;
        if (bounded != (cap == CAP_CHECKPOINT_RESTORE) ||
            ambient != (prepared && cap == CAP_CHECKPOINT_RESTORE)) {
            errno = EPERM;
            return -1;
        }
    }
    errno = 0;
    if (prctl(PR_CAPBSET_READ, 64L, 0L, 0L, 0L) != -1 || errno != EINVAL) {
        errno = ENOTSUP;
        return -1;
    }
    int secure = prctl(PR_GET_SECUREBITS, 0L, 0L, 0L, 0L);
    if (secure < 0) return -1;
    int no_new = prctl(PR_GET_NO_NEW_PRIVS, 0L, 0L, 0L, 0L);
    if (no_new < 0) return -1;
    if (secure || no_new) { errno = EPERM; return -1; }
    return 0;
}

static int close_status(int descriptor, int primary)
{
    if (close(descriptor)) {
        int cleanup = errno;
        if (!primary) return cleanup;
        (void)refuse("descriptor-close-secondary", cleanup);
    }
    return primary; /* No retry: close failure never authorizes exec. */
}

static int protected_parents(const char *path)
{
    char copy[PATH_MAX];
    size_t length = strlen(path);
    if (length >= sizeof copy || !length || path[0] != '/') { errno = EINVAL; return -1; }
    memcpy(copy, path, length + 1);
    for (size_t index = 0; index < length; ++index) {
        if (copy[index] != '/' || (index && copy[index - 1] == '/')) continue;
        char saved = copy[index];
        if (index) copy[index] = 0;
        int descriptor = open(index ? copy : "/", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        copy[index] = saved;
        if (descriptor < 0) return -1;
        struct stat info;
        struct statvfs filesystem;
        int error = 0;
        if (fstat(descriptor, &info) || fstatvfs(descriptor, &filesystem)) error = errno;
        else if (!S_ISDIR(info.st_mode) || info.st_uid != 0 ||
                 (info.st_mode & (S_IWGRP | S_IWOTH)) || !(filesystem.f_flag & ST_RDONLY)) error = EPERM;
        error = close_status(descriptor, error);
        if (error) { errno = error; return -1; }
    }
    return 0;
}

static int executable(const char *path, int launcher)
{
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return -1;
    int error = 0;
    struct stat info;
    struct statvfs filesystem;
    unsigned char actual[24];
    const unsigned char expected[20] = {
        1, 0, 0, 2, /* revision2, effective; P(low)=I(low)=0 */
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, /* P(high)=1<<(40-32) */
        0, 0, 0, 0
    };
    if (fstat(descriptor, &info) || fstatvfs(descriptor, &filesystem)) error = errno;
    else if (!S_ISREG(info.st_mode) || info.st_uid != 0 ||
             (info.st_mode & (S_ISUID | S_ISGID | S_IWGRP | S_IWOTH)) ||
             (launcher && (info.st_mode & S_IWUSR)) || !(info.st_mode & S_IXUSR) ||
             !(filesystem.f_flag & ST_RDONLY)) error = EPERM;
    if (!error) {
        errno = 0;
        ssize_t length = fgetxattr(descriptor, "security.capability", actual, sizeof actual);
        if (launcher) {
            if (length != (ssize_t)sizeof expected || memcmp(actual, expected, sizeof expected))
                error = length < 0 ? errno : EPERM;
        } else if (length >= 0 || errno != ENODATA) error = length < 0 ? errno : EPERM;
    }
    error = close_status(descriptor, error);
    if (error) { errno = error; return -1; }
    return 0;
}

int main(int argc, char **argv)
{
    uid_t uid, after_uid;
    gid_t gid, after_gid;
    if (argc < 1 || !argv || !argv[0]) return refuse("argv", EINVAL);
    if (ids(&uid, &gid)) return refuse("initial-ids", errno);
    if (caps(0)) return refuse("initial-caps", errno);
    if (executable("/proc/self/exe", 1)) return refuse("launcher-file", errno);
    char resolved[PATH_MAX];
    if (protected_parents(python_path) || !realpath(python_path, resolved) || protected_parents(resolved) ||
        executable(python_path, 0)) return refuse("protected-python", errno);
    struct __user_cap_header_struct header = {_LINUX_CAPABILITY_VERSION_3, 0};
    struct __user_cap_data_struct values[_LINUX_CAPABILITY_U32S_3] = {{0}};
    values[1].effective = values[1].permitted = values[1].inheritable = bit;
    if (syscall(SYS_capset, &header, values)) return refuse("inheritable", errno);
    if (prctl(PR_CAP_AMBIENT, (unsigned long)PR_CAP_AMBIENT_RAISE, (unsigned long)CAP_CHECKPOINT_RESTORE, 0L, 0L))
        return refuse("ambient", errno);
    if (ids(&after_uid, &after_gid) || after_uid != uid || after_gid != gid)
        return refuse("final-ids", EPERM);
    if (caps(1)) return refuse("final-caps", errno);
    char record[256];
    int length = snprintf(record, sizeof record,
        "MAPFILE_CAP_LAUNCH_READY uid=%lu gid=%lu P=0000010000000000 E=0000010000000000 "
        "I=0000010000000000 A=0000010000000000 B=0000010000000000\n",
        (unsigned long)uid, (unsigned long)gid);
    if (length <= 0 || (size_t)length >= sizeof record) return refuse("receipt-format", EOVERFLOW);
    size_t written = 0;
    while (written < (size_t)length) {
        ssize_t count = write(STDERR_FILENO, record + written, (size_t)length - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return refuse("receipt-write", count < 0 ? errno : EIO);
        written += (size_t)count;
    }
    argv[0] = (char *)python_path; /* Same argument vector; no shell or PATH lookup. */
    execve(python_path, argv, environ);
    return refuse("exec", errno);
}
