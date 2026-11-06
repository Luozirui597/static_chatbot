"""Execute the injected B test-page script against a tiny fake DOM."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.option_b_web.launcher import _build_test_html, load_frozen_spec


class TestPageScriptState(unittest.TestCase):
    """The real injected script must handle terminal changes, not just source text."""

    def test_terminal_and_branch_state_sequences(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is required for the page-script state test")

        frozen = load_frozen_spec()
        index = (
            '<!doctype html><html><body>'
            '<div id="app" class="app-layout"></div>'
            '</body></html>'
        )
        page = _build_test_html(index, frozen, "page-state-harness")

        with tempfile.TemporaryDirectory(prefix="option_b_page_state_") as tmp:
            page_path = Path(tmp) / "page.html"
            page_path.write_text(page, encoding="utf-8")
            harness = Path(__file__).with_name("page_state_harness.js")
            result = subprocess.run(
                [node, str(harness), str(page_path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                self.fail(
                    "page script state harness failed:\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            self.assertIn("page script state sequences ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
