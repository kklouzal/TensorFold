"""CPU-only faults at the short diagnostic's actual ownership boundaries."""
import json
import subprocess
import unittest

from tools.v100_short_container import OwnedContainer, retire_client

CID = "a" * 64
LIVE = {"Status": "running", "Running": True, "Pid": 123, "ExitCode": 0, "OOMKilled": False}
DONE = {"Status": "exited", "Running": False, "Pid": 0, "ExitCode": 0, "OOMKilled": False}


class BadRepr(MemoryError):
    def __repr__(self):
        raise AssertionError("retirement must not format failures")


class Docker:
    def __init__(self, *, inspect=LIVE, terminal=DONE, fail=None):
        self.calls = []
        self.initial, self.terminal, self.fail = inspect, terminal, fail or {}
        self.inspections = 0

    def __call__(self, argv, **kwargs):
        key = " ".join(argv[1:3])
        self.calls.append(tuple(argv))
        if key in self.fail:
            raise self.fail[key]
        if argv[1] == "create":
            raw = (CID + "\n").encode()
        elif argv[1] == "inspect":
            self.inspections += 1
            raw = json.dumps(self.initial if self.inspections == 1 else self.terminal).encode()
        else:
            raw = b"0\n"
        return subprocess.CompletedProcess(argv, 0, raw, b"")


def bind(owner):
    owner.create(["docker", "create", "image"], lambda *_: None)


class Tests(unittest.TestCase):
    def test_explicit_remaining_wait_never_renews_an_expired_deadline(self):
        for remaining in (0,2.5):
            with self.subTest(remaining=remaining):
                docker=Docker();observed=[]
                def run(argv,**kwargs):
                    if argv[1]=='wait':observed.append(kwargs['timeout'])
                    return docker(argv,**kwargs)
                owner=OwnedContainer(run);bind(owner);owner.retire(wait_timeout_s=remaining)
                self.assertEqual(len(observed),1);self.assertTrue(0<=observed[0]<=remaining)
                self.assertEqual(owner.exit_code(None),0)

    def test_remaining_wait_pays_inspect_and_stop_command_latency(self):
        from unittest.mock import patch
        docker=Docker();observed=[]
        def run(argv,**kwargs):
            if argv[1]=='wait':observed.append(kwargs['timeout'])
            return docker(argv,**kwargs)
        owner=OwnedContainer(run);bind(owner)
        with patch('tools.v100_short_container.time.monotonic',side_effect=[100,105]):
            owner.retire(wait_timeout_s=9)
        self.assertEqual(observed,[4]);self.assertEqual(owner.exit_code(None),0)

    def test_invalid_retirement_wait_refuses_before_docker_effects(self):
        for remaining in (-1,True,float('nan'),float('inf')):
            with self.subTest(remaining=remaining):
                docker=Docker();owner=OwnedContainer(docker)
                with self.assertRaises(ValueError):owner.retire(wait_timeout_s=remaining)
                self.assertEqual(docker.calls,[])

    def test_client_failed_kill_still_gets_final_wait(self):
        from unittest.mock import Mock
        process = Mock()
        timeout, kill_error = subprocess.TimeoutExpired("docker exec", 30), BadRepr()
        process.wait.side_effect = [timeout, 0]
        process.kill.side_effect = kill_error
        errors = [None] * 3
        retire_client(process, errors)
        self.assertEqual(process.wait.call_count, 2)
        self.assertIs(errors[0], timeout)
        self.assertIs(errors[1], kill_error)
        self.assertIsNone(errors[2])

    def test_client_final_wait_failure_is_retained_independently(self):
        from unittest.mock import Mock
        process = Mock()
        timeout, last = subprocess.TimeoutExpired("docker exec", 30), BadRepr()
        process.wait.side_effect = [timeout, last]
        errors = [None] * 3
        retire_client(process, errors)
        self.assertIs(errors[0], timeout)
        self.assertIsNone(errors[1])
        self.assertIs(errors[2], last)

    def test_already_exited_client_needs_no_kill(self):
        from unittest.mock import Mock
        process = Mock()
        process.wait.return_value = 0
        errors = [None] * 3
        retire_client(process, errors)
        process.kill.assert_not_called()
        self.assertEqual(errors, [None] * 3)

    def test_create_publication_failure_retains_cid_and_retires(self):
        docker = Docker()
        owner = OwnedContainer(docker)
        primary = None
        def publish(*_):
            raise BadRepr()
        try:
            owner.create(["docker", "create", "image"], publish)
        except BaseException as error:
            primary = error
        finally:
            owner.retire()
        self.assertIsInstance(primary, BadRepr)
        self.assertEqual(owner.cid, CID)
        self.assertIn(("docker", "kill", "--signal=SIGINT", CID), docker.calls)
        self.assertEqual(owner.exit_code(primary), 1)

    def test_clean_retirement_is_zero(self):
        docker = Docker()
        owner = OwnedContainer(docker)
        bind(owner)
        slots = owner.errors
        owner.retire()
        self.assertIs(owner.errors, slots)
        self.assertEqual(owner.exit_code(None), 0)
        self.assertNotIn(("docker", "kill", "--signal=SIGKILL", CID), docker.calls)

    def test_stop_failure_still_waits_and_fails(self):
        failure = BadRepr()
        docker = Docker(fail={"kill --signal=SIGINT": failure})
        owner = OwnedContainer(docker)
        bind(owner)
        owner.retire()
        self.assertIs(owner.errors[1], failure)
        self.assertIn(("docker", "wait", CID), docker.calls)
        self.assertEqual(owner.exit_code(None), 1)

    def test_wait_timeout_attempts_exact_cid_containment_and_remains_failure(self):
        failure = subprocess.TimeoutExpired("docker wait", 90)
        docker = Docker(fail={"wait " + CID: failure})
        owner = OwnedContainer(docker)
        bind(owner)
        owner.retire()
        self.assertIs(owner.errors[2], failure)
        self.assertIs(owner.errors[4], failure)
        self.assertIn(("docker", "kill", "--signal=SIGKILL", CID), docker.calls)
        self.assertEqual(owner.exit_code(None), 1)

    def test_bad_first_inspect_still_attempts_retirement(self):
        docker = Docker(fail={"inspect --format": BadRepr()})
        owner = OwnedContainer(docker)
        bind(owner)
        owner.retire()
        self.assertIn(("docker", "kill", "--signal=SIGINT", CID), docker.calls)
        self.assertEqual(owner.exit_code(None), 1)

    def test_container_nonzero_oom_or_unretired_is_nonzero(self):
        for changes in ({"ExitCode": 15}, {"OOMKilled": True}, {"Running": True, "Pid": 123}):
            with self.subTest(changes=changes):
                docker = Docker(terminal={**DONE, **changes})
                owner = OwnedContainer(docker)
                bind(owner)
                owner.retire()
                self.assertEqual(owner.exit_code(None), 1)

    def test_unacquired_or_missing_final_state_cannot_pass(self):
        owner = OwnedContainer(Docker())
        owner.retire()
        self.assertEqual(owner.exit_code(None), 1)

    def test_already_terminal_container_needs_no_signal(self):
        docker = Docker(inspect=DONE)
        owner = OwnedContainer(docker)
        bind(owner)
        owner.retire()
        self.assertEqual(owner.exit_code(None), 0)
        self.assertFalse(any(argv[1] == "kill" for argv in docker.calls))


if __name__ == "__main__":
    unittest.main()
