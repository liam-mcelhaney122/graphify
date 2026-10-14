"""Cross-file resolution of Rust `use` paths.

extractors/rust.py emits a `use` as `file --imports_from--> _make_id(<last path
segment>)` and a type reference to an imported name as an edge onto a sourceless
per-file stub (`ensure_named_node`). On a multi-crate workspace neither endpoint
names a node: `use gt_core::ItemStack;` points at the bare id `itemstack` while
the struct's node is `gt_core/src/item.rs::ItemStack`, so build mints an external
stub for it, and a `Direction` defined in several modules leaves every file's
`<file>_rs_direction` placeholder standing because the unique-label rewire
(`_rewire_unique_stub_nodes`) cannot pick one. The path written in the `use`
decides both, and the extractor now keeps it (`metadata.use_path`), so this pass
walks it against the module tree the corpus's own files form.

The module tree is read off file paths plus the nearest `Cargo.toml`'s
`[package] name` (`-` -> `_`): `src/lib.rs`/`src/main.rs` are the crate root,
`src/a.rs` and `src/a/mod.rs` are `crate::a`, `src/a/b.rs` is `crate::a::b`;
`src/bin/x.rs`, `tests/x.rs`, `examples/x.rs` and `benches/x.rs` are roots of
their own target (a `tests/x/main.rs` layout counts as one too). A workspace
crate is importable by name when its `src/lib.rs` is in the corpus. A path is
followed segment by segment through module files, then through the module's own
`use` bindings (re-exports, `pub use a::B as C`) and its glob imports, so both
`gt_core::ItemStack` (via `pub use item::ItemStack;` in lib.rs) and
`use super::*` are landed on real nodes. Something the module tree cannot see
(an inline `mod`, `#[path]`, `include!`, a macro-generated item) falls back to a
name that is defined exactly once in the target crate, and that edge is marked
INFERRED with a score.

Honesty rules: a path resolved through the module tree and the written
re-export chain is exact (confidence unchanged, EXTRACTED); the name-in-crate
fallback is INFERRED at 0.85; a path into a crate the corpus does not hold
(`std`, `serde`, `bevy`), or one that is ambiguous, is left exactly as the
extractor emitted it. No edge is invented and none is removed: a resolved stub
simply loses its referrers and is dropped once nothing points at it.

Not modelled (documented limits): `[lib] path` overrides, `#[path = ".."]`
module attributes, dependency renames (`package = ".."` in Cargo.toml), inline
`mod a { .. }` as a path segment (its items are attributed to the file), the
2015-edition `extern crate` namespace, macro paths (`use crate::my_macro;`),
and a type written with its full path at the use site (`fn f(x: a::b::Foo)`),
which the extractor still reduces to its last segment.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from graphify.extractors.resolution import _js_source_path, _resolve_cached

_LOG = logging.getLogger(__name__)

_ROOT_STEMS = frozenset({"lib", "main", "mod"})
_TARGET_DIRS = frozenset({"tests", "examples", "benches"})
_LIB_UNIT = "lib"
_IMPORT_RELATIONS = frozenset({"imports", "imports_from"})
_CHILD_RELATIONS = frozenset({"case_of", "contains", "method"})
_NAME_IN_CRATE_SCORE = 0.85
_MAX_DEPTH = 24

# How a target was found. `exact` and `reexport`/`glob` follow written paths and
# keep the edge's confidence; `name_in_crate` is the INFERRED fallback.
_EXACT_MODES = frozenset({"exact", "reexport", "glob"})


def _package_name(manifest: Path) -> tuple[str, bool] | None:
    """(`[package] name` with `-` as `_`, whether it declares `[lib]`) or None."""
    from graphify.cargo_introspect import _load_toml

    try:
        data = _load_toml(manifest)
    except Exception:
        return None
    package = data.get("package") if isinstance(data, dict) else None
    name = package.get("name") if isinstance(package, dict) else None
    if not isinstance(name, str) or not name:
        return None
    lib = data.get("lib")
    lib_name = lib.get("name") if isinstance(lib, dict) else None
    if isinstance(lib_name, str) and lib_name:
        name = lib_name
    return name.replace("-", "_"), isinstance(lib, dict)


def _unit_and_mods(rel_parts: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """(target unit, module path) of a `.rs` file from its package-relative parts."""
    stem = rel_parts[-1][:-3]
    dirs = list(rel_parts[:-1])

    def mods(tail: list[str]) -> tuple[str, ...]:
        return tuple(tail) + (() if stem in _ROOT_STEMS else (stem,))

    if dirs[:1] == ["src"]:
        if dirs[1:2] == ["bin"]:
            if len(dirs) == 2:
                return f"bin/{stem}", ()
            return f"bin/{dirs[2]}", mods(dirs[3:])
        return _LIB_UNIT, mods(dirs[1:])
    if dirs and dirs[0] in _TARGET_DIRS:
        if len(dirs) == 1:
            return f"{dirs[0]}/{stem}", ()
        return f"{dirs[0]}/{dirs[1]}", mods(dirs[2:])
    if not dirs and stem == "build":
        return "build", ()
    return "/".join(dirs) or ".", mods([])


class _Workspace:
    """The module tree of the Rust files in the corpus."""

    def __init__(self, root: Path) -> None:
        self.root = root
        try:
            self._home = Path.home()
        except (RuntimeError, KeyError):
            self._home = Path(root.anchor)
        self._manifest_cache: dict[str, tuple[str, Path, bool] | None] = {}
        # file -> (package, unit, mods); (package, unit, mods) -> file
        self.file_module: dict[str, tuple[str, str, tuple[str, ...]]] = {}
        self.module_file: dict[tuple[str, str, tuple[str, ...]], str] = {}
        self.lib_packages: set[str] = set()
        self.unit_files: dict[tuple[str, str], list[str]] = {}

    def _package_of(self, directory: Path) -> tuple[str, Path, bool] | None:
        """The nearest enclosing `[package]`: (name, package dir, has [lib])."""
        key = str(directory)
        if key in self._manifest_cache:
            return self._manifest_cache[key]
        result: tuple[str, Path, bool] | None = None
        manifest = directory / "Cargo.toml"
        try:
            is_file = manifest.is_file()
        except OSError:
            is_file = False
        if is_file:
            named = _package_name(manifest)
            if named is not None:
                result = (named[0], directory, named[1])
        # Keep climbing past the scan root (a scan of `crates/x/src` still has
        # its manifest one level up), stopping at the home directory or the
        # filesystem root like `_find_js_project_anchor` does.
        if result is None and directory != self._home and directory.parent != directory:
            result = self._package_of(directory.parent)
        self._manifest_cache[key] = result
        return result

    def add_file(self, path: str) -> None:
        if path in self.file_module or not path.endswith(".rs"):
            return
        abs_path = Path(path)
        package = self._package_of(abs_path.parent)
        if package is None:
            return
        name, package_dir, _has_lib = package
        try:
            rel_parts = abs_path.relative_to(package_dir).parts
        except ValueError:
            return
        unit, mods = _unit_and_mods(tuple(rel_parts))
        key = (name, unit, mods)
        self.file_module[path] = key
        self.module_file.setdefault(key, path)
        self.unit_files.setdefault((name, unit), []).append(path)
        if unit == _LIB_UNIT and not mods and rel_parts[-1] == "lib.rs":
            self.lib_packages.add(name)


class RustUseResolver:
    """Resolve `use` paths and imported-name stubs against the corpus's nodes."""

    def __init__(
        self,
        root: Path,
        nodes: list[dict],
        edges: list[dict],
        context_nodes: list[dict] | None = None,
        context_edges: list[dict] | None = None,
    ) -> None:
        self.root = root
        self.workspace = _Workspace(_resolve_cached(root))
        self.node_by_id: dict[str, dict] = {}
        self.file_node: dict[str, str] = {}  # file path -> file node id
        self.file_of_node: dict[str, str] = {}  # file node id -> file path
        self.defs: dict[tuple[str, str], list[str]] = {}  # (file, name) -> node ids
        self.children: dict[tuple[str, str], list[str]] = {}  # (node, name) -> node ids
        self.uses: dict[str, dict[str, list[str]]] = {}  # file -> local name -> paths
        self.globs: dict[str, list[str]] = {}  # file -> glob paths (`a::b::*`)
        self._path_cache: dict[str, str | None] = {}

        for node in nodes:
            self._index_node(node)
        for node in context_nodes or ():
            if node.get("id") not in self.node_by_id:
                self._index_node(node)
        for edge in edges:
            self._index_edge(edge)
        for edge in context_edges or ():
            self._index_edge(edge)

    # -- indexing -----------------------------------------------------------

    def _path_key(self, source_file: object) -> str | None:
        """Absolute resolved path of a node's/edge's `source_file`, memoised."""
        text = str(source_file or "")
        if not text:
            return None
        if text not in self._path_cache:
            resolved = _js_source_path(text, self.root)
            self._path_cache[text] = str(resolved) if resolved is not None else None
        return self._path_cache[text]

    def _index_node(self, node: dict) -> None:
        nid = node.get("id")
        if not isinstance(nid, str) or not nid:
            return
        self.node_by_id[nid] = node
        path = self._path_key(node.get("source_file"))
        if path is None or not path.endswith(".rs"):
            return
        self.workspace.add_file(path)
        if str(node.get("label", "")) == Path(path).name and path not in self.file_node:
            self.file_node[path] = nid
            self.file_of_node[nid] = path

    def _index_edge(self, edge: dict) -> None:
        relation = edge.get("relation")
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            return
        if relation in _CHILD_RELATIONS:
            target_node = self.node_by_id.get(target)
            if target_node is None:
                return
            name = str(target_node.get("label", "")).strip()
            if name.startswith("."):
                name = name[1:]
            if name.endswith("()"):
                name = name[:-2]
            if not name:
                return
            self.children.setdefault((source, name), []).append(target)
            file_path = self.file_of_node.get(source)
            if relation == "contains" and file_path is not None:
                self.defs.setdefault((file_path, name), []).append(target)
            return
        if relation in _IMPORT_RELATIONS and not edge.get("_rust_use_group"):
            metadata = edge.get("metadata")
            use_path = metadata.get("use_path") if isinstance(metadata, dict) else None
            if not isinstance(use_path, str) or not use_path:
                return
            file_path = self._path_key(edge.get("source_file"))
            if file_path is None:
                return
            path, local = _split_alias(use_path)
            if path.endswith("::*"):
                self.globs.setdefault(file_path, []).append(path[:-3])
            elif local:
                self.uses.setdefault(file_path, {}).setdefault(local, []).append(path)

    # -- resolution ---------------------------------------------------------

    def resolve_path(self, from_file: str, path: str) -> tuple[str, str, str] | None:
        """Resolve a written `use` path from `from_file`.

        Returns (kind, id, mode): kind is `node` (a definition's node id), `file`
        (a module's file node id) or `external` (a crate the corpus does not
        hold; id is ""), and mode is how it was found (see `_EXACT_MODES`).
        None means the path is internal but could not be resolved.
        """
        segments = [s for s in path.split("::") if s]
        return self._resolve(from_file, segments, set(), 0)

    def _resolve(
        self, from_file: str, segments: list[str], seen: set, depth: int
    ) -> tuple[str, str, str] | None:
        if depth > _MAX_DEPTH or not segments:
            return None
        located = self.workspace.file_module.get(from_file)
        if located is None:
            return None
        package, unit, base_mods = located
        first = segments[0]
        rest = segments[1:]
        if first == "crate":
            mods: tuple[str, ...] = ()
        elif first == "self":
            mods = base_mods
        elif first == "super":
            mods = base_mods
            rest = segments
            while rest and rest[0] == "super":
                if not mods:
                    return None
                mods = mods[:-1]
                rest = rest[1:]
        elif first in self.workspace.lib_packages:
            package, unit, mods = first, _LIB_UNIT, ()
        elif (package, unit, base_mods + (first,)) in self.workspace.module_file:
            # 2018 uniform path to a child module of this one (`use util::parse;`).
            mods, rest = base_mods, segments
        elif "/" in unit and (package, f"{unit.split('/')[0]}/{first}", ()) in self.workspace.module_file:
            # `mod common;` shared by integration tests: `tests/common.rs` or
            # `tests/common/mod.rs` next to `tests/x.rs` (same for examples/benches).
            package, unit, mods, rest = package, f"{unit.split('/')[0]}/{first}", (), rest
        elif len(segments) == 1 and (
            (from_file, first) in self.defs or first in self.uses.get(from_file, {})
        ):
            # `use Name as Alias;`: a single segment names an item in scope here.
            mods, rest = base_mods, segments
        else:
            return "external", "", "exact"

        module_file = self.workspace.module_file
        while rest and (package, unit, mods + (rest[0],)) in module_file:
            mods += (rest[0],)
            rest = rest[1:]
        module_path = module_file.get((package, unit, mods))
        if not rest or rest == ["*"]:
            if module_path is None:
                return None
            file_nid = self.file_node.get(module_path)
            return ("file", file_nid, "exact") if file_nid else None
        if "*" in rest or module_path is None:
            return self._name_in_crate(package, unit, rest)
        found = self._lookup(module_path, rest[0], seen, depth + 1)
        if found is None:
            return self._name_in_crate(package, unit, rest)
        kind, nid, mode = found
        for segment in rest[1:]:
            if kind == "file":
                # A re-exported module (`pub use gt_core;` then `api::gt_core::X`).
                owner = self.file_of_node.get(nid)
                found = self._lookup(owner, segment, seen, depth + 1) if owner else None
                if found is None:
                    return None
                kind, nid, mode = found
                continue
            if kind != "node":
                return None
            owned = sorted(set(self.children.get((nid, segment), [])))
            if len(owned) != 1:
                return None
            nid = owned[0]
        return kind, nid, mode

    def _lookup(
        self, module_path: str, name: str, seen: set, depth: int
    ) -> tuple[str, str, str] | None:
        """`name` as the module in `module_path` sees it: its own definition, the
        target of its `use` binding of that name, or a glob import's definition."""
        key = (module_path, name)
        if key in seen or depth > _MAX_DEPTH:
            return None
        seen = seen | {key}
        defined = sorted(set(self.defs.get(key, [])))
        if len(defined) == 1:
            return "node", defined[0], "exact"
        if defined:
            return None  # two definitions in one module (cfg twins kept apart): ambiguous
        bound = self.uses.get(module_path, {}).get(name)
        if bound:
            return _unique(
                [self._resolve(module_path, p.split("::"), seen, depth + 1) for p in sorted(set(bound))],
                "reexport",
            )
        found: list[tuple[str, str, str] | None] = []
        for glob in sorted(set(self.globs.get(module_path, []))):
            target = self._resolve(module_path, glob.split("::"), seen, depth + 1)
            if target is None:
                found.append(None)
            elif target[0] == "file":
                source_path = self.file_of_node.get(target[1])
                found.append(self._lookup(source_path, name, seen, depth + 1) if source_path else None)
            elif target[0] == "external":
                found.append(target)
        return _unique(found, "glob") if found else None

    def _name_in_crate(
        self, package: str, unit: str, rest: list[str]
    ) -> tuple[str, str, str] | None:
        """INFERRED fallback: the one definition of the path's last name in the crate."""
        name = rest[-1]
        if name == "*":
            return None
        candidates: set[str] = set()
        for file_path in self.workspace.unit_files.get((package, unit), []):
            candidates.update(self.defs.get((file_path, name), []))
        if len(candidates) != 1:
            return None
        return "node", next(iter(candidates)), "name_in_crate"

    def resolve_name(self, from_file: str, name: str) -> tuple[str, str, str] | None:
        """A bare name referenced in `from_file`, through its `use` declarations."""
        bound = self.uses.get(from_file, {}).get(name)
        if bound:
            return _unique(
                [self._resolve(from_file, p.split("::"), {(from_file, name)}, 1) for p in sorted(set(bound))],
                "exact",
            )
        globs = sorted(set(self.globs.get(from_file, [])))
        if not globs:
            return None
        found: list[tuple[str, str, str] | None] = []
        for glob in globs:
            target = self._resolve(from_file, glob.split("::"), {(from_file, name)}, 1)
            if target is None:
                found.append(None)
            elif target[0] == "file":
                source_path = self.file_of_node.get(target[1])
                found.append(self._lookup(source_path, name, {(from_file, name)}, 1) if source_path else None)
            else:
                found.append(target)
        return _unique(found, "glob")


def _split_alias(use_path: str) -> tuple[str, str]:
    """`a::b as c` -> (`a::b`, `c`); `a::b` -> (`a::b`, `b`); `a::*` -> (`a::*`, "")."""
    path, _, alias = use_path.partition(" as ")
    path = path.strip()
    if alias.strip():
        return path, alias.strip()
    last = path.rsplit("::", 1)[-1]
    return path, "" if last == "*" else last


def _unique(
    found: list[tuple[str, str, str] | None], mode: str
) -> tuple[str, str, str] | None:
    """One resolved target among `found`, or None when the candidates disagree.

    An external candidate is only reported when nothing internal was found: a
    name a known module supplies cannot also come from an external glob in
    compiling code (that is an ambiguity error), so the known one wins.
    """
    resolved = [item for item in found if item is not None]
    internal = {(kind, nid) for kind, nid, _ in resolved if kind != "external"}
    if len(internal) == 1:
        kind, nid = next(iter(internal))
        inferred = any(m == "name_in_crate" for k, n, m in resolved if (k, n) == (kind, nid))
        return kind, nid, "name_in_crate" if inferred else mode
    if internal:
        return None
    if resolved:
        return "external", "", "exact"
    return None


def resolve_rust_use_paths(
    paths: list[Path],
    all_nodes: list[dict],
    all_edges: list[dict],
    root: Path,
    resolution_context_nodes: list[dict] | None = None,
    resolution_context_edges: list[dict] | None = None,
) -> None:
    """Retarget Rust `use` edges and imported-name stubs onto the nodes they name.

    Runs before `_rewire_unique_stub_nodes` so an exact path beats the unique-label
    heuristic, and before the shared call pass indexes import evidence. Mutates
    `all_edges` in place (target, relation, confidence) and drops from `all_nodes`
    the sourceless stubs that no edge references any more.
    """
    if not any(str(p).endswith(".rs") for p in paths):
        return
    resolver = RustUseResolver(root, all_nodes, all_edges, resolution_context_nodes, resolution_context_edges)
    if not resolver.workspace.file_module:
        return

    def retarget(edge: dict, found: tuple[str, str, str]) -> bool:
        kind, nid, mode = found
        if kind == "external" or not nid or nid == edge.get("source"):
            return False
        edge["target"] = nid
        if kind == "node" and edge.get("relation") == "imports_from":
            edge["relation"] = "imports"
        if mode not in _EXACT_MODES:
            edge["confidence"] = "INFERRED"
            edge["confidence_score"] = _NAME_IN_CRATE_SCORE
        return True

    # Imports: the written path.
    for edge in all_edges:
        if edge.get("relation") not in _IMPORT_RELATIONS or edge.get("_rust_use_group"):
            continue
        metadata = edge.get("metadata")
        use_path = metadata.get("use_path") if isinstance(metadata, dict) else None
        if not isinstance(use_path, str) or not use_path:
            continue
        from_file = resolver._path_key(edge.get("source_file"))
        if from_file is None:
            continue
        path, _local = _split_alias(use_path)
        found = resolver.resolve_path(from_file, path)
        if found is not None and found[0] == "file" and found[1] == edge.get("source"):
            # `use self::*` from an inline `mod tests`: the file importing itself.
            edge["target"] = found[1]
            continue
        if found is not None:
            retarget(edge, found)

    # Imported-name stubs: the sourceless node a Rust file minted for a type it
    # references but does not define, keyed by the file that references it.
    stub_ids = {
        str(n.get("id"))
        for n in all_nodes
        if not n.get("source_file") and str(n.get("origin_file", "")).endswith(".rs") and n.get("id")
    }
    if not stub_ids:
        return
    memo: dict[tuple[str, str], tuple[str, str, str] | None] = {}
    touched = False
    for edge in all_edges:
        target = edge.get("target")
        if target not in stub_ids:
            continue
        from_file = resolver._path_key(edge.get("source_file"))
        if from_file is None or not from_file.endswith(".rs"):
            continue
        name = str(resolver.node_by_id.get(str(target), {}).get("label", "")).strip()
        if not name:
            continue
        key = (from_file, name)
        if key not in memo:
            memo[key] = resolver.resolve_name(from_file, name)
        found = memo[key]
        if found is not None and found[0] == "node" and retarget(edge, found):
            touched = True
    if touched:
        referenced = {x for e in all_edges for x in (e.get("source"), e.get("target"))}
        all_nodes[:] = [n for n in all_nodes if n.get("id") not in stub_ids or n.get("id") in referenced]
