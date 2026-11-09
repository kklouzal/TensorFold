"""Compile the canonical launcher; stdlib-only controls never grant capability."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/native_mapfile_cap_launcher.c'
FLAGS = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-Wconversion', '-Wshadow', '-pedantic',
         '-fPIE', '-pie', '-fstack-protector-strong', '-D_FORTIFY_SOURCE=3',
         '-Wl,-z,relro,-z,now,-z,noexecstack']

# Include the actual implementation after system headers. Interpose only the
# OS boundary, so branch/state tests exercise canonical C without privileges.
HARNESS = r'''
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/capability.h>
#include <linux/prctl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <unistd.h>
static const char *scenario;
static int changed, mock_ambient, descriptors, execs;
static int kinds[64];
static int is(const char *value) { return strcmp(scenario, value) == 0; }
static int ids_u(uid_t *r, uid_t *e, uid_t *s) {
    *r = *e = *s = is("root") ? 0U : 1000U;
    if (is("saved-uid")) *s = 1001;
    if (is("changed-uid") && changed) *r = *e = *s = 1001;
    return 0;
}
static int ids_g(gid_t *r, gid_t *e, gid_t *s) {
    *r = *e = *s = is("root-gid") ? 0U : 1000U;
    if (is("saved-gid")) *s = 1001;
    return 0;
}
static long caps_call(long number, ...) {
    va_list args;
    va_start(args, number);
    struct __user_cap_header_struct *header = va_arg(args, struct __user_cap_header_struct *);
    struct __user_cap_data_struct *data = va_arg(args, struct __user_cap_data_struct *);
    va_end(args);
    if (header->version != _LINUX_CAPABILITY_VERSION_3 || header->pid) abort();
    if (number == SYS_capget) {
        if (is("capget-fail")) { errno = EIO; return -1; }
        memset(data, 0, 2 * sizeof *data);
        data[1].effective = data[1].permitted = is("missing-cap") ? 0U : 256U;
        data[1].inheritable = changed && !is("bad-final-inherit") ? 256U : 0U;
        if (is("extra-cap")) data[0].permitted |= 1U << CAP_SYS_ADMIN;
        if (is("initial-inherit")) data[1].inheritable = 256;
        return 0;
    }
    if (number != SYS_capset) abort();
    if (is("capset-fail")) { errno = EPERM; return -1; }
    if (data[0].effective || data[0].permitted || data[0].inheritable ||
        data[1].effective != 256 || data[1].permitted != 256 || data[1].inheritable != 256) abort();
    changed = 1;
    return 0;
}
static int caps_prctl(int option, ...) {
    va_list args;
    va_start(args, option);
    unsigned long a = va_arg(args, unsigned long);
    unsigned long b = va_arg(args, unsigned long);
    (void)va_arg(args, unsigned long); (void)va_arg(args, unsigned long);
    va_end(args);
    if (option == PR_CAPBSET_READ) {
        if (a == 64 && is("unknown-abi")) return 0;
        if (a > 40) { errno = EINVAL; return -1; }
        return a == 40 || (is("extra-bounding") && a == CAP_SYS_ADMIN);
    }
    if (option == PR_CAP_AMBIENT) {
        if (a == PR_CAP_AMBIENT_IS_SET)
            return (mock_ambient && b == 40) || (is("initial-ambient") && b == CAP_SYS_ADMIN);
        if (a != PR_CAP_AMBIENT_RAISE || b != 40 || !changed) abort();
        if (is("ambient-fail")) { errno = EPERM; return -1; }
        mock_ambient = 1;
        return 0;
    }
    if (option == PR_GET_SECUREBITS) return is("securebits") ? 16 : 0;
    if (option == PR_GET_NO_NEW_PRIVS) return is("no-new-privs");
    abort();
}
static int file_open(const char *path, int flags, ...) {
    if (is("open-fail")) { errno = EACCES; return -1; }
    int descriptor = ++descriptors;
    if (descriptor >= 64) abort();
    kinds[descriptor] = (flags & O_DIRECTORY) ? 1 : (!strcmp(path, "/proc/self/exe") ? 2 : 3);
    if (kinds[descriptor] == 1 && !(flags & O_NOFOLLOW)) abort();
    return descriptor;
}
static int file_stat(int descriptor, struct stat *info) {
    if (is("stat-fail") || is("dual-failure")) { errno = EIO; return -1; }
    memset(info, 0, sizeof *info);
    info->st_mode = (mode_t)(kinds[descriptor] == 1 ? S_IFDIR | 0755 : S_IFREG | 0555);
    if (is("foreign-owner")) info->st_uid = 1000;
    if (is("writable-launcher") && kinds[descriptor] == 2) info->st_mode |= S_IWUSR;
    if (is("writable-parent") && kinds[descriptor] == 1) info->st_mode |= S_IWGRP;
    if (is("setuid-file")) info->st_mode |= S_ISUID;
    return 0;
}
static int file_fs(int descriptor, struct statvfs *info) {
    (void)descriptor; memset(info, 0, sizeof *info);
    info->f_flag = is("rw-filesystem") ? 0U : ST_RDONLY;
    return 0;
}
static ssize_t file_caps(int descriptor, const char *name, void *out, size_t extent) {
    if (strcmp(name, "security.capability") || extent < 20) abort();
    if (kinds[descriptor] != 2) {
        if (is("python-cap")) { memset(out, 0, 20); return 20; }
        errno = ENODATA; return -1;
    }
    if (is("missing-filecap")) { errno = ENODATA; return -1; }
    unsigned char value[20] = {1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    if (is("extra-filecap")) value[6] = 32; /* Extra CAP_SYS_ADMIN permitted. */
    memcpy(out, value, sizeof value); return 20;
}
static int file_close(int descriptor) {
    (void)descriptor;
    if (is("close-fail") || is("dual-failure")) { errno = EBADF; return -1; }
    return 0;
}
static char *path_real(const char *path, char *resolved) {
    if (strcmp(path, "/opt/venv/bin/python3")) abort();
    if (is("realpath-fail")) { errno = ENOENT; return NULL; }
    strcpy(resolved, "/usr/bin/python3.12"); return resolved;
}
static ssize_t receipt_write(int descriptor, const void *buffer, size_t length) {
    if (is("write-fail")) { errno = EIO; return -1; }
    return write(descriptor, buffer, length > 7 ? 7 : length); /* Also exercises partial writes. */
}
static int python_exec(const char *path, char *const args[], char *const env[]) {
    ++execs;
    if (strcmp(path, "/opt/venv/bin/python3") || strcmp(args[0], path) ||
        strcmp(args[1], "-c") || strcmp(args[2], "owned-code") || args[3] || !env || !changed || !mock_ambient) abort();
    if (is("exec-fail")) { errno = ENOENT; return -1; }
    puts("MOCK_PROTECTED_EXEC uid=1000 gid=1000 capability=40 args=preserved");
    exit(0);
}
#define getresuid ids_u
#define getresgid ids_g
#define syscall caps_call
#define prctl caps_prctl
#define open file_open
#define fstat file_stat
#define fstatvfs file_fs
#define fgetxattr file_caps
#define close file_close
#define realpath path_real
#define write receipt_write
#define execve python_exec
#define main launcher_main
#include "native_mapfile_cap_launcher.c"
#undef main
int main(int argc, char **argv) {
    if (argc != 2) return 99;
    scenario = argv[1];
    char *arguments[] = {"owned-launcher", "-c", "owned-code", NULL};
    int result = launcher_main(3, arguments);
    if (result != 125 || (execs && !is("exec-fail"))) return 98;
    return result;
}
'''


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.production = cls.root / 'launcher'
        cls.harness = cls.root / 'harness'
        # Compile production first; no shim/capability ever enters this output.
        cls.compile(SOURCE, cls.production)
        harness_source = cls.root / 'harness.c'
        harness_source.write_text(HARNESS)
        cls.compile(harness_source, cls.harness)

    @classmethod
    def compile(cls, source, output):
        result = subprocess.run(['cc', *FLAGS, '-I', str(SOURCE.parent), str(source), '-o', str(output)],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise AssertionError('canonical launcher compilation failed:\n' + result.stderr)

    def case(self, scenario):
        return subprocess.run([str(self.harness), scenario], capture_output=True, text=True, check=False)

    def test_positive_transition_execs_only_fixed_python_and_preserves_arguments(self):
        result = self.case('valid')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('uid=1000 gid=1000 capability=40 args=preserved', result.stdout)
        self.assertIn('P=0000010000000000 E=0000010000000000', result.stderr)

    def test_invalid_identity_capability_and_ownership_never_exec(self):
        cases = ['root', 'root-gid', 'saved-uid', 'saved-gid', 'changed-uid', 'missing-cap', 'extra-cap',
                 'initial-inherit', 'initial-ambient', 'extra-bounding', 'unknown-abi', 'securebits',
                 'no-new-privs', 'foreign-owner', 'writable-launcher', 'writable-parent', 'setuid-file',
                 'rw-filesystem', 'python-cap', 'missing-filecap', 'extra-filecap', 'bad-final-inherit']
        for scenario in cases:
            with self.subTest(scenario=scenario):
                result = self.case(scenario)
                self.assertEqual(result.returncode, 125, result.stderr)
                self.assertNotIn('MOCK_PROTECTED_EXEC', result.stdout)
                self.assertIn('MAPFILE_CAP_LAUNCH_REFUSED', result.stderr)

    def test_syscall_receipt_and_exec_failures_are_nonpassing(self):
        for scenario in ['capget-fail', 'capset-fail', 'ambient-fail', 'open-fail', 'stat-fail',
                         'close-fail', 'realpath-fail', 'write-fail', 'exec-fail']:
            with self.subTest(scenario=scenario):
                result = self.case(scenario)
                self.assertEqual(result.returncode, 125, result.stderr)
                self.assertNotIn('MOCK_PROTECTED_EXEC', result.stdout)

    def test_close_failure_reports_secondary_without_losing_primary(self):
        result = self.case('dual-failure')
        self.assertEqual(result.returncode, 125)
        self.assertIn('descriptor-close-secondary errno=9', result.stderr)
        self.assertIn('launcher-file errno=5', result.stderr)

    def test_real_binary_refuses_root_or_missing_cap_without_file_capability(self):
        # No file capability assignment/activation; only a negative boundary.
        result = subprocess.run([str(self.production), '-c', 'raise SystemExit(99)'],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 125)
        expected = 'initial-ids' if os.getuid() == 0 else 'initial-caps'
        self.assertIn('phase=' + expected, result.stderr)


if __name__ == '__main__':
    unittest.main()
