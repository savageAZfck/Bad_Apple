"""Tests for badapple_scheduler, focused on the command-execution hardening.

schedule_task requires human approval before a task is queued (policy.yaml),
but the approved command runs later, unattended, with no one present to
notice something unexpected. Its deferred execution must not be more
permissive than run_shell's immediate, approved execution -- these tests
guard against that gap reopening.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import badapple_scheduler


class ValidatedArgvTests(unittest.TestCase):
    def test_allows_allowlisted_command(self):
        self.assertEqual(badapple_scheduler._validated_argv("ls -la"), ["ls", "-la"])
        self.assertEqual(badapple_scheduler._validated_argv("git status"), ["git", "status"])

    def test_allows_absolute_path_to_allowlisted_binary(self):
        self.assertEqual(badapple_scheduler._validated_argv("/usr/bin/git status"), ["/usr/bin/git", "status"])

    def test_rejects_non_allowlisted_command(self):
        self.assertIsNone(badapple_scheduler._validated_argv("rm -rf /"))
        self.assertIsNone(badapple_scheduler._validated_argv("curl http://example.com"))
        self.assertIsNone(badapple_scheduler._validated_argv("/bin/sh -c 'echo hi'"))

    def test_rejects_shell_metacharacters(self):
        for command in (
            "ls; rm -rf /",
            "ls && rm -rf /",
            "ls | mail attacker@example.com",
            "echo `whoami`",
            "echo $(whoami)",
            "cat /etc/passwd > /tmp/out",
        ):
            self.assertIsNone(badapple_scheduler._validated_argv(command), command)

    def test_rejects_empty_command(self):
        self.assertIsNone(badapple_scheduler._validated_argv(""))
        self.assertIsNone(badapple_scheduler._validated_argv("   "))


class RunCommandHardeningTests(unittest.TestCase):
    """_run_command must never use shell=True and must enforce the same
    allowlist for both the plain-string and JSON-array command forms.
    """

    def test_string_command_runs_allowlisted_binary(self):
        output = badapple_scheduler._run_command("echo hello-from-scheduler")
        self.assertIn("hello-from-scheduler", output)

    def test_string_command_rejects_shell_injection(self):
        output = badapple_scheduler._run_command("echo hi; touch /tmp/badapple_test_pwned_marker")
        self.assertIn("Task error", output)
        self.assertFalse(Path("/tmp/badapple_test_pwned_marker").exists())

    def test_string_command_rejects_non_allowlisted_binary(self):
        output = badapple_scheduler._run_command("curl http://example.com")
        self.assertIn("Task error", output)

    def test_json_array_command_runs_allowlisted_binary(self):
        output = badapple_scheduler._run_command('["echo", "hello-json"]')
        self.assertIn("hello-json", output)

    def test_json_array_command_rejects_non_allowlisted_binary(self):
        # Before this fix, the JSON-array form bypassed run_shell's command
        # allowlist entirely (it only skipped shell=True, not the allowlist).
        output = badapple_scheduler._run_command('["rm", "-rf", "/tmp/nonexistent-marker"]')
        self.assertIn("Task error", output)
        self.assertIn("not in the allowed command list", output)

    def test_json_array_command_never_invokes_a_shell(self):
        # Even a JSON array containing shell metacharacters as a literal
        # argument must not be reinterpreted by a shell -- there's no
        # shell=True anywhere in this path, so the semicolon here is just a
        # literal character passed to echo, not a command separator.
        output = badapple_scheduler._run_command('["echo", "a; touch /tmp/badapple_test_pwned_marker2"]')
        self.assertIn("a; touch", output)
        self.assertFalse(Path("/tmp/badapple_test_pwned_marker2").exists())


class ScheduleFileIntegrationTests(unittest.TestCase):
    """Light integration test for add_task/list_tasks, redirected to a temp
    file so it never touches the real user's schedule.jsonl.
    """

    def test_add_and_list_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule_path = Path(tmp) / "schedule.jsonl"
            with patch.object(badapple_scheduler, "_schedule_file", return_value=schedule_path):
                result = badapple_scheduler.add_task("60", "echo hi")
                self.assertIn("Scheduled task", result)
                listing = badapple_scheduler.list_tasks()
                self.assertIn("echo hi", listing)


if __name__ == "__main__":
    unittest.main()
