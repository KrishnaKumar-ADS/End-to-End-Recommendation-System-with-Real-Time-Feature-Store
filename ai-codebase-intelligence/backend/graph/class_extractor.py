"""
Class Hierarchy Extractor — Python AST.

Scans every Python CodeChunk of type "class" in the PostgreSQL database
and extracts:
  - Direct parent class names (base classes)
  - Whether the class is abstract (ABCMeta / abc.ABC)
  - Whether the class looks like a mixin (name ends with Mixin or Base)
  - C3 MRO list (best-effort, using only names available in the repo)

Output is a list of ClassHierarchyData objects — plain dataclasses, no DB
or Neo4j dependency. The hierarchy_builder.py (Day 2) turns these into
Neo4j edges.

Why not use importlib / inspect?
  We never install the repo — we work only on source text. inspect requires
  a running Python interpreter with all dependencies installed. AST parsing
  is dependency-free and works on any codebase.
"""

import ast
from dataclasses import dataclass, field
from typing import Optional
from core.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Class names that signal abstract base class behaviour
ABSTRACT_BASE_NAMES = frozenset({
    "ABC",
    "ABCMeta",
    "Protocol",
    "TypedDict",
    "NamedTuple",
})

# Suffixes that conventionally signal a mixin class
MIXIN_SUFFIXES = ("Mixin", "mixin", "Base", "base", "Interface", "interface")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BaseClassRef:
    """
    A single base class reference found in a class definition.

    For `class Dog(Mammal, FetchMixin)`:
      BaseClassRef(name="Mammal",    is_abstract=False, is_mixin=False, position=0)
      BaseClassRef(name="FetchMixin",is_abstract=False, is_mixin=True,  position=1)
    """
    name: str                   # simple name of the parent class
    is_abstract: bool = False   # True if this base looks like an ABC
    is_mixin: bool = False      # True if this base looks like a Mixin
    position: int = 0           # 0 = leftmost base in the tuple
    module_path: str = ""       # e.g. "abc" if we see `from abc import ABC`


@dataclass
class ClassHierarchyData:
    """
    Everything we know about one class's position in the hierarchy.

    This is the output of extract_class_hierarchy() for one class chunk.
    It feeds directly into hierarchy_builder.py which turns it into Neo4j edges.
    """
    chunk_id: str                          # UUID — same as code_chunks.id in PostgreSQL
    class_name: str                        # e.g. "GoldenRetriever"
    file_path: str                         # relative path, e.g. "animals/dog.py"
    repo_id: str                           # UUID — same as repositories.id
    bases: list[BaseClassRef] = field(default_factory=list)
    is_abstract: bool = False              # True if class uses ABC / ABCMeta
    is_mixin: bool = False                 # True if name ends with Mixin/Base
    mro_list: list[str] = field(default_factory=list)  # best-effort C3 MRO

    @property
    def has_bases(self) -> bool:
        """True if this class inherits from anything other than `object`."""
        real_bases = [b for b in self.bases if b.name not in ("object", "type")]
        return len(real_bases) > 0

    @property
    def direct_parent_names(self) -> list[str]:
        """Return just the names of direct parents, in order."""
        return [b.name for b in self.bases if not b.is_mixin and not b.is_abstract]

    @property
    def mixin_names(self) -> list[str]:
        """Return names of mixin bases only."""
        return [b.name for b in self.bases if b.is_mixin]

    @property
    def abstract_base_names(self) -> list[str]:
        """Return names of abstract/interface bases only."""
        return [b.name for b in self.bases if b.is_abstract]

    def __repr__(self) -> str:
        base_str = ", ".join(b.name for b in self.bases) or "object"
        return f"<ClassHierarchy {self.class_name}({base_str}) abstract={self.is_abstract}>"


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_class_hierarchy(
    source_code: str,
    chunk_id: str,
    class_name: str,
    file_path: str,
    repo_id: str,
) -> Optional[ClassHierarchyData]:
    """
    Parse `source_code` (a single class chunk's content) and extract its
    inheritance information.

    Returns None if:
      - The source has a syntax error
      - No class definition is found in the chunk

    Called once per CodeChunk of type "class".

    Example:
      source_code = '''
      class GoldenRetriever(Dog, FetchMixin):
                    '''A friendly dog breed.'''
          pass
      '''
      → ClassHierarchyData(
            class_name="GoldenRetriever",
            bases=[
                BaseClassRef("Dog",       is_mixin=False, position=0),
                BaseClassRef("FetchMixin", is_mixin=True,  position=1),
            ],
            is_abstract=False,
            is_mixin=False,
        )
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        logger.warning(
            "class_extractor_syntax_error",
            chunk_id=chunk_id,
            class_name=class_name,
            error=str(exc),
        )
        return None

    # Walk the AST and find the first ClassDef node whose name matches
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name == class_name:
                return _extract_from_classdef(node, chunk_id, file_path, repo_id)

    # If name doesn't match exactly, take the first ClassDef we find
    # (handles cases where the chunk has a slightly different name in source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return _extract_from_classdef(node, chunk_id, file_path, repo_id)

    logger.warning(
        "class_extractor_no_classdef",
        chunk_id=chunk_id,
        class_name=class_name,
        file_path=file_path,
    )
    return None


def extract_class_hierarchy_from_file(
    source_code: str,
    file_path: str,
    repo_id: str,
    chunk_id_map: dict[str, str],
) -> list[ClassHierarchyData]:
    """
    Parse an entire file and extract hierarchy data for ALL classes defined in it.

    This is used when we want to process a whole file at once rather than
    chunk by chunk — useful for resolving references between sibling classes
    in the same file.

    Args:
        source_code:  Full source text of the file
        file_path:    Relative path of the file within the repo
        repo_id:      UUID of the repository
        chunk_id_map: Dict mapping class_name → chunk_id from PostgreSQL
                      e.g. {"GoldenRetriever": "uuid-1", "Dog": "uuid-2"}
                      Classes not in this map are skipped.

    Returns:
        List of ClassHierarchyData — one per class found in the file.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        logger.warning(
            "file_class_extractor_syntax_error",
            file_path=file_path,
            error=str(exc),
        )
        return []

    results: list[ClassHierarchyData] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Only process classes we have a chunk_id for
        chunk_id = chunk_id_map.get(node.name)
        if chunk_id is None:
            logger.debug(
                "class_extractor_no_chunk_id",
                class_name=node.name,
                file_path=file_path,
            )
            continue

        data = _extract_from_classdef(node, chunk_id, file_path, repo_id)
        results.append(data)

    return results


def compute_mro(
    class_name: str,
    all_hierarchy: list[ClassHierarchyData],
) -> list[str]:
    """
    Compute a best-effort C3 linearization (MRO) for a class.

    This is a simplified version of Python's real MRO algorithm.
    It works on ClassHierarchyData objects — no live Python objects needed.

    Args:
        class_name:    Name of the class to compute MRO for
        all_hierarchy: All ClassHierarchyData objects in the repo

    Returns:
        List of class names in MRO order, starting with the class itself.
        Always ends with "object".

    Example:
        class A: pass
        class B(A): pass
        class C(A): pass
        class D(B, C): pass

        compute_mro("D", [...]) → ["D", "B", "C", "A", "object"]
    """
    # Build a lookup by class name
    by_name: dict[str, ClassHierarchyData] = {h.class_name: h for h in all_hierarchy}

    def _mro(name: str, visited: set[str]) -> list[str]:
        """Recursive helper — returns linearized list starting with `name`."""
        if name in visited:
            return []  # cycle guard
        if name not in by_name:
            return [name]  # external class (e.g. from stdlib) — include but don't expand

        visited = visited | {name}
        data = by_name[name]

        # Get direct parents (not mixins, not abc bases — for the main chain)
        parents = [b.name for b in data.bases if b.name not in ("object", "type")]

        if not parents:
            return [name, "object"]

        # Simplified C3: [class] + merge of all parent linearizations
        result = [name]
        seen: set[str] = {name}

        for parent in parents:
            parent_mro = _mro(parent, visited)
            for p in parent_mro:
                if p not in seen:
                    result.append(p)
                    seen.add(p)

        if "object" not in seen:
            result.append("object")

        return result

    return _mro(class_name, set())


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_from_classdef(
    node: ast.ClassDef,
    chunk_id: str,
    file_path: str,
    repo_id: str,
) -> ClassHierarchyData:
    """
    Extract hierarchy data from a single ast.ClassDef node.

    This is the core logic — shared between chunk-level and file-level extractors.
    """
    bases: list[BaseClassRef] = []

    for position, base in enumerate(node.bases):
        base_name = _resolve_base_name(base)
        if base_name is None:
            continue

        is_abstract = base_name in ABSTRACT_BASE_NAMES
        is_mixin = base_name.endswith(MIXIN_SUFFIXES)

        bases.append(BaseClassRef(
            name=base_name,
            is_abstract=is_abstract,
            is_mixin=is_mixin,
            position=position,
        ))

    # Determine if THIS class is abstract
    this_is_abstract = _is_abstract_class(node)

    # Determine if THIS class is a mixin
    this_is_mixin = (
        node.name.endswith(MIXIN_SUFFIXES)
        or any(b.is_abstract for b in bases)
    )

    data = ClassHierarchyData(
        chunk_id=chunk_id,
        class_name=node.name,
        file_path=file_path,
        repo_id=repo_id,
        bases=bases,
        is_abstract=this_is_abstract,
        is_mixin=this_is_mixin,
    )

    logger.debug(
        "class_extracted",
        class_name=node.name,
        bases=[b.name for b in bases],
        is_abstract=this_is_abstract,
    )

    return data


def _resolve_base_name(base_node: ast.expr) -> Optional[str]:
    """
    Extract a base class name from an AST expression node.

    Handles three common patterns:
      class Foo(Bar)          → ast.Name      → "Bar"
      class Foo(module.Bar)   → ast.Attribute → "Bar"
      class Foo(Generic[T])   → ast.Subscript → "Generic"
    """
    if isinstance(base_node, ast.Name):
        return base_node.id

    if isinstance(base_node, ast.Attribute):
        # module.ClassName — we just want the class name
        return base_node.attr

    if isinstance(base_node, ast.Subscript):
        # Generic[T], List[str], etc. — extract the outer name
        return _resolve_base_name(base_node.value)

    # Call nodes like `register.adapter()` — skip
    return None


def _is_abstract_class(node: ast.ClassDef) -> bool:
    """
    Return True if the class is abstract.

    Detection rules:
      1. One of the bases is ABC, ABCMeta, or Protocol
      2. The class has any method decorated with @abstractmethod
      3. The metaclass keyword is ABCMeta:  class Foo(metaclass=ABCMeta)
    """
    # Rule 1 — base class check
    for base in node.bases:
        name = _resolve_base_name(base)
        if name in ABSTRACT_BASE_NAMES:
            return True

    # Rule 2 — metaclass=ABCMeta keyword argument
    for keyword in node.keywords:
        if keyword.arg == "metaclass":
            meta_name = _resolve_base_name(keyword.value)
            if meta_name == "ABCMeta":
                return True

    # Rule 3 — has @abstractmethod decorator on any method
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in child.decorator_list:
                dec_name = _resolve_base_name(decorator)
                if dec_name == "abstractmethod":
                    return True

    return False