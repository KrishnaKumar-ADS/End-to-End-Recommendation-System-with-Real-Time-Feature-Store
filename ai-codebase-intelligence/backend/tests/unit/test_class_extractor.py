"""
Unit tests for graph/class_extractor.py.
No Neo4j, no PostgreSQL — pure AST parsing.
"""
import pytest
from graph.class_extractor import (
    extract_class_hierarchy,
    extract_class_hierarchy_from_file,
    compute_mro,
    ClassHierarchyData,
)

REPO_ID = "test-repo-uuid"
FILE_PATH = "myapp/models.py"
CHUNK_ID = "chunk-uuid-001"


class TestExtractClassHierarchy:

    def test_simple_inheritance(self):
        source = "class Dog(Animal):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Dog", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.class_name == "Dog"
        assert len(result.bases) == 1
        assert result.bases[0].name == "Animal"
        assert result.bases[0].is_mixin is False
        assert result.is_abstract is False

    def test_no_bases_returns_empty_bases(self):
        source = "class Animal:\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Animal", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.bases == []
        assert result.has_bases is False

    def test_multiple_inheritance(self):
        source = "class C(A, B):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "C", FILE_PATH, REPO_ID)
        assert result is not None
        assert len(result.bases) == 2
        assert result.bases[0].name == "A"
        assert result.bases[0].position == 0
        assert result.bases[1].name == "B"
        assert result.bases[1].position == 1

    def test_mixin_detected(self):
        source = "class FetchMixin(object):\n    def fetch(self): pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "FetchMixin", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.is_mixin is True

    def test_base_mixin_detected_in_bases(self):
        source = "class Dog(Animal, FetchMixin):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Dog", FILE_PATH, REPO_ID)
        assert result is not None
        fetch_base = next(b for b in result.bases if b.name == "FetchMixin")
        assert fetch_base.is_mixin is True

    def test_abc_abstract_detected(self):
        source = "from abc import ABC\nclass Shape(ABC):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Shape", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.is_abstract is True

    def test_abstractmethod_decorator_marks_abstract(self):
        source = (
            "from abc import abstractmethod\n"
            "class Shape:\n"
            "    @abstractmethod\n"
            "    def area(self): pass\n"
        )
        result = extract_class_hierarchy(source, CHUNK_ID, "Shape", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.is_abstract is True

    def test_metaclass_abcmeta_detected(self):
        source = "class Shape(metaclass=ABCMeta):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Shape", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.is_abstract is True

    def test_module_qualified_base(self):
        # class Foo(models.Model) → base name "Model"
        source = "class UserProfile(models.Model):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "UserProfile", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.bases[0].name == "Model"

    def test_generic_subscript_base(self):
        # class Foo(Generic[T]) → base name "Generic"
        source = "from typing import Generic, TypeVar\nT = TypeVar('T')\nclass Box(Generic[T]):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Box", FILE_PATH, REPO_ID)
        assert result is not None
        assert result.bases[0].name == "Generic"

    def test_syntax_error_returns_none(self):
        source = "class Broken(\n    # no closing paren\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Broken", FILE_PATH, REPO_ID)
        assert result is None

    def test_no_class_def_returns_none(self):
        source = "def some_function():\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "SomeClass", FILE_PATH, REPO_ID)
        assert result is None

    def test_direct_parent_names_property(self):
        source = "class Dog(Animal, FetchMixin):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Dog", FILE_PATH, REPO_ID)
        # FetchMixin is a mixin, so direct_parent_names should only contain Animal
        assert "Animal" in result.direct_parent_names
        assert "FetchMixin" not in result.direct_parent_names

    def test_mixin_names_property(self):
        source = "class Dog(Animal, FetchMixin):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Dog", FILE_PATH, REPO_ID)
        assert "FetchMixin" in result.mixin_names

    def test_repr(self):
        source = "class Dog(Animal):\n    pass\n"
        result = extract_class_hierarchy(source, CHUNK_ID, "Dog", FILE_PATH, REPO_ID)
        assert "Dog" in repr(result)
        assert "Animal" in repr(result)


class TestExtractFromFile:

    def test_extracts_multiple_classes(self):
        source = (
            "class Animal:\n    pass\n\n"
            "class Dog(Animal):\n    pass\n\n"
            "class Cat(Animal):\n    pass\n"
        )
        chunk_id_map = {"Animal": "id-1", "Dog": "id-2", "Cat": "id-3"}
        results = extract_class_hierarchy_from_file(source, FILE_PATH, REPO_ID, chunk_id_map)
        assert len(results) == 3
        names = {r.class_name for r in results}
        assert names == {"Animal", "Dog", "Cat"}

    def test_skips_classes_not_in_map(self):
        source = "class Dog(Animal):\n    pass\nclass Cat(Animal):\n    pass\n"
        # Only Dog is in the map
        chunk_id_map = {"Dog": "id-2"}
        results = extract_class_hierarchy_from_file(source, FILE_PATH, REPO_ID, chunk_id_map)
        assert len(results) == 1
        assert results[0].class_name == "Dog"

    def test_syntax_error_returns_empty(self):
        source = "class Broken(\n"
        results = extract_class_hierarchy_from_file(source, FILE_PATH, REPO_ID, {"Broken": "id-1"})
        assert results == []


class TestComputeMRO:

    def _make_hierarchy(self, class_name: str, bases: list[str]) -> ClassHierarchyData:
        from graph.class_extractor import BaseClassRef
        return ClassHierarchyData(
            chunk_id=f"chunk-{class_name}",
            class_name=class_name,
            file_path=FILE_PATH,
            repo_id=REPO_ID,
            bases=[BaseClassRef(name=b, position=i) for i, b in enumerate(bases)],
        )

    def test_single_inheritance_mro(self):
        all_h = [
            self._make_hierarchy("Animal", []),
            self._make_hierarchy("Dog", ["Animal"]),
        ]
        mro = compute_mro("Dog", all_h)
        assert mro[0] == "Dog"
        assert "Animal" in mro
        assert mro[-1] == "object"

    def test_no_bases_mro(self):
        all_h = [self._make_hierarchy("Animal", [])]
        mro = compute_mro("Animal", all_h)
        assert mro == ["Animal", "object"]

    def test_unknown_class_returns_just_name(self):
        mro = compute_mro("UnknownClass", [])
        assert mro == ["UnknownClass"]

    def test_multi_level_mro(self):
        all_h = [
            self._make_hierarchy("Animal", []),
            self._make_hierarchy("Mammal", ["Animal"]),
            self._make_hierarchy("Dog", ["Mammal"]),
            self._make_hierarchy("GoldenRetriever", ["Dog"]),
        ]
        mro = compute_mro("GoldenRetriever", all_h)
        assert mro.index("GoldenRetriever") < mro.index("Dog")
        assert mro.index("Dog") < mro.index("Mammal")
        assert mro.index("Mammal") < mro.index("Animal")