"""Exercise launcher path resolution without requiring a robot policy server."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

LAUNCHER = Path(__file__).resolve().parents[1] / 'scripts/run_all_libero_10.sh'


class LiberoLauncherTests(unittest.TestCase):
    def test_runs_all_tasks_from_another_directory(self):
        with tempfile.TemporaryDirectory(prefix='libero launcher ') as tmp:
            root = Path(tmp)
            checkout = root / 'checkout with spaces'
            rollout = checkout / 'gr00t/eval/rollout_policy.py'
            rollout.parent.mkdir(parents=True)
            rollout.write_text(
                'import json, os, sys\n'
                'with open(os.environ["TEST_CALL_LOG"], "a") as f:\n'
                '    f.write(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}) + "\\n")\n'
            )
            log = root / 'calls.jsonl'
            env = dict(os.environ, GR00T_ROOT='checkout with spaces',
                       PYTHON_BIN=sys.executable, TEST_CALL_LOG=str(log),
                       POLICY_CLIENT_HOST='test-server', POLICY_CLIENT_PORT='6000')
            result = subprocess.run(['bash', str(LAUNCHER)], cwd=root, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 10)
            tasks = set()
            for call in calls:
                self.assertEqual(call['cwd'], str(checkout))
                args = call['args']
                self.assertEqual(args[args.index('--policy_client_host') + 1], 'test-server')
                self.assertEqual(args[args.index('--policy_client_port') + 1], '6000')
                tasks.add(args[args.index('--env_name') + 1])
            self.assertEqual(len(tasks), 10)

    def test_reports_missing_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            rollout = checkout / 'gr00t/eval/rollout_policy.py'
            rollout.parent.mkdir(parents=True)
            rollout.touch()
            env = dict(os.environ, GR00T_ROOT=tmp, PYTHON_BIN=str(checkout / 'missing-python'))
            result = subprocess.run(['bash', str(LAUNCHER)], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing LIBERO Python environment', result.stderr)
            self.assertIn('PYTHON_BIN', result.stderr)


if __name__ == '__main__':
    unittest.main()
