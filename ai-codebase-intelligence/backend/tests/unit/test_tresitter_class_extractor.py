"""
Unit tests for graph/treesitter_class_extractor.py.

These tests run only if tree-sitter-javascript is installed.
If it's not installed, the tests are skipped.
"""
import pytest

try:
    import tree_sitter_javascript  # noqa: F401
    TREESITTER_AVAILABLE = True
except ImportError:
    TREESITTER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TREESITTER_AVAILABLE,
    reason="tree-sitter-javascript not installed",
)

from graph.treesitter_class_extractor import extract_js_class_hierarchy

REPO_ID = "test-repo-uuid"
FILE_PATH = "src/models.js"
CHUNK_ID = "chunk-js-001"


class TestJSClassExtraction:

    def test_simple_extends(self):
        source = "class Dog extends Animal {\n  bark() {}\n}"
        result = extract_js_class_hierarchy(source, CHUNK_ID, "Dog", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.class_name == "Dog"
        assert len(result.bases) == 1
        assert result.bases[0].name == "Animal"
        assert result.bases[0].is_abstract is False

    def test_no_extends(self):
        source = "class Animal {\n  breathe() {}\n}"
        result = extract_js_class_hierarchy(source, CHUNK_ID, "Animal", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.bases == []
        assert result.has_bases is False

    def test_mixin_base_detected(self):
        source = "class Dog extends FetchMixin {\n  bark() {}\n}"
        result = extract_js_class_hierarchy(source, CHUNK_ID, "Dog", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.bases[0].is_mixin is True

    def test_no_class_returns_none(self):
        source = "function add(a, b) { return a + b; }"
        result = extract_js_class_hierarchy(source, CHUNK_ID, "Unknown", FILE_PATH, REPO_ID)
        assert result is None


class TestTSClassExtraction:

    def test_implements_interface(self):
        if not TREESITTER_AVAILABLE:
            pytest.skip("tree-sitter-typescript not installed")

        source = "class Circle extends Shape implements Drawable {\n  draw() {}\n}"
        result = extract_js_class_hierarchy(
            source, CHUNK_ID, "Circle", FILE_PATH, REPO_ID, language="typescript"
        )
        assert result is not None
        parent_names = [b.name for b in result.bases]
        assert "Shape" in parent_names
        assert "Drawable" in parent_names

    def test_abstract_class_ts(self):
        if not TREESITTER_AVAILABLE:
            pytest.skip("tree-sitter-typescript not installed")

        source = "abstract class Vehicle {\n  abstract start(): void;\n}"
        result = extract_js_class_hierarchy(
            source, CHUNK_ID, "Vehicle", FILE_PATH, REPO_ID, language="typescript"
        )
        assert result is not None
        assert result.is_abstract is True