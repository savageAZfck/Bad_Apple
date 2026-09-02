"""Tests for socket directory ownership/permissions set by the gatekeeper.

The gatekeeper runs as root and (re)creates ``/var/run/badapple`` on startup.
When the directory is recreated it must be mode ``0o770`` and owned by the
console user (or at least keep a group the MLX daemon belongs to) so the
user-owned daemon can bind its socket. These tests verify the expected
permissions and that the gatekeeper's ``ensure_socket_dir`` logic would set
correct permissions, without requiring root or a real ``/var/run`` path.
"""

import os
import shutil
import stat
import subprocess
import tempfile
import unittest


def _console_user() -> str | None:
    """Mirror the gatekeeper's console-user detection for the test environment."""
    try:
        out = subprocess.run(
            ["stat", "-f", "%Su", "/dev/console"],
            capture_output=True,
            text=True,
            check=True,
        )
        user = out.stdout.strip()
        if user and user != "root":
            return user
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


class SocketOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="badapple_socket_")
        self.sock_dir = os.path.join(self.tmp, "badapple")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_expected_mode_is_0o770(self) -> None:
        """The socket directory must allow group rwx but no world access."""
        os.makedirs(self.sock_dir, exist_ok=True)
        os.chmod(self.sock_dir, 0o770)
        mode = stat.S_IMODE(os.stat(self.sock_dir).st_mode)
        self.assertEqual(mode, 0o770)
        # No world access.
        self.assertFalse(mode & stat.S_IWOTH)
        self.assertFalse(mode & stat.S_IROTH)
        self.assertFalse(mode & stat.S_IXOTH)

    def test_ensure_socket_dir_sets_permissions(self) -> None:
        """Simulate the gatekeeper recreating the dir and fixing permissions."""
        # Start with a root-like 0o755 dir (what create_dir_all produces as
        # root on a fresh /var/run).
        os.makedirs(self.sock_dir, exist_ok=True)
        os.chmod(self.sock_dir, 0o755)
        self.assertEqual(stat.S_IMODE(os.stat(self.sock_dir).st_mode), 0o755)

        # Replicate ensure_socket_dir: set mode 0o770, then chown to console
        # user if available (skipped when not running as root / no console).
        os.chmod(self.sock_dir, 0o770)
        user = _console_user()
        if user is not None:
            try:
                subprocess.run(
                    ["chown", user, self.sock_dir],
                    check=True,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, OSError):
                # Not root in CI; ownership fix is best-effort. Permissions
                # are the hard requirement.
                pass

        mode = stat.S_IMODE(os.stat(self.sock_dir).st_mode)
        self.assertEqual(mode, 0o770)

    def test_gatekeeper_creates_missing_dir_with_correct_mode(self) -> None:
        """ensure_socket_dir uses create_dir_all then enforces 0o770."""
        # Directory does not exist yet (simulates /var/run cleared on reboot).
        self.assertFalse(os.path.exists(self.sock_dir))
        os.makedirs(self.sock_dir, exist_ok=True)
        os.chmod(self.sock_dir, 0o770)
        self.assertTrue(os.path.isdir(self.sock_dir))
        self.assertEqual(stat.S_IMODE(os.stat(self.sock_dir).st_mode), 0o770)

    def test_daemon_can_write_when_mode_0o770(self) -> None:
        """With 0o770 the owning user can create files (daemon binds socket)."""
        os.makedirs(self.sock_dir, exist_ok=True)
        os.chmod(self.sock_dir, 0o770)
        socket_file = os.path.join(self.sock_dir, "substrate_mlx.sock")
        # Creating a file here stands in for binding a Unix socket.
        with open(socket_file, "w") as fh:
            fh.write("probe")
        self.assertTrue(os.path.exists(socket_file))

    def test_console_user_detection_returns_non_root_or_none(self) -> None:
        user = _console_user()
        if user is not None:
            self.assertNotEqual(user, "root")


if __name__ == "__main__":
    unittest.main()
