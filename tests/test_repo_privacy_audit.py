import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
from scripts.repo_privacy_audit import emit_report, scan_repo  # noqa: E402


class RepoPrivacyAuditTests(unittest.TestCase):
    def make_repo(self, files: dict[str, str], tracked: list[str] | None = None) -> tuple[Path, list[str]]:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        repo = Path(tempdir.name) / 'repo'
        repo.mkdir()
        subprocess.run(['git', 'init'], cwd=repo, check=True, capture_output=True)

        tracked = tracked or list(files)
        for rel, content in files.items():
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')

        if tracked:
            subprocess.run(['git', 'add', *tracked], cwd=repo, check=True, capture_output=True)

        return repo, tracked

    def test_scan_repo_flags_absolute_workspace_path(self):
        leaked_path = '/'.join(['', 'home', 'tiresias']) + '/' + '.openclaw' + '/workspace/projects/CiteWeave'
        repo, tracked = self.make_repo(
            {'README.md': f'local path: {leaked_path}'},
        )

        violations = scan_repo(repo, tracked)

        self.assertEqual(violations, [('README.md', 'absolute local workspace path')])

    def test_scan_repo_flags_tracked_local_override_file(self):
        repo, tracked = self.make_repo(
            {'config/neo4j_config.local.json': '{"password": "CHANGE_ME_LOCAL_ONLY"}'},
        )

        violations = scan_repo(repo, tracked)

        self.assertEqual(violations, [('config/neo4j_config.local.json', 'tracked local override config')])

    def test_scan_repo_ignores_untracked_local_override_file(self):
        repo, tracked = self.make_repo(
            {
                'config/neo4j_config.json': '{"password": "CHANGE_ME_LOCAL_ONLY"}',
                'config/neo4j_config.local.json': '{"password": "' + '1234' + '5678' + '"}',
            },
            tracked=['config/neo4j_config.json'],
        )

        violations = scan_repo(repo, tracked)

        self.assertEqual(violations, [])

    def test_emit_report_returns_success_for_clean_repo(self):
        self.assertEqual(emit_report([]), 0)


if __name__ == '__main__':
    unittest.main()
