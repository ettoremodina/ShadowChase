import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SKILL_ROOT / "scripts" / "render_html.py"
TEMPLATE_PATH = SKILL_ROOT / "assets" / "reader-template.html"
SPEC = importlib.util.spec_from_file_location("render_html", SCRIPT_PATH)
assert SPEC and SPEC.loader
RENDER_HTML = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER_HTML)


class RenderHtmlTests(unittest.TestCase):
    def render(self, source: str) -> str:
        return RENDER_HTML.build_document(source, "sample.md", None, None, TEMPLATE_PATH)

    def test_renders_longform_structures_and_navigation(self) -> None:
        source = """# Delivery Plan

## Milestones

- [x] Discovery
- [ ] Build

| Owner | Work |
| --- | --- |
| Sam | API |

```python
print("ready")
```
"""
        document = self.render(source)
        self.assertIn('href="#milestones"', document)
        self.assertIn('type="checkbox" disabled checked', document)
        self.assertIn("<table>", document)
        self.assertIn('class="language-python"', document)

    def test_escapes_raw_html_and_unsafe_links(self) -> None:
        document = self.render("# Safe\n\n<script>alert(1)</script> [bad](javascript:alert(1))")
        self.assertNotIn("<script>alert(1)</script>", document)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)
        self.assertNotIn('href="javascript:', document)

    def test_output_is_self_contained_and_resolves_placeholders(self) -> None:
        document = self.render("# Offline Reader\n\nContent")
        self.assertNotRegex(document, r'<(?:script|link)[^>]+(?:src|href)=["\']https?://')
        self.assertNotIn("{{CONTENT}}", document)
        self.assertNotIn("{{TITLE}}", document)

    def test_cli_writes_default_html_beside_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "release-plan.md"
            input_path.write_text("# Release Plan\n\nShip it.", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), str(input_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            output_path = input_path.with_suffix(".html")
            self.assertTrue(output_path.exists())
            self.assertEqual(Path(result.stdout.strip()), output_path)
            self.assertIn("Release Plan", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
