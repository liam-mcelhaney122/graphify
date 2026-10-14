"""A Rust `use` lands on the node its written path names, across files and crates.

extract_rust emitted `use gt_core::ItemStack;` as an `imports_from` edge to the
bare id `itemstack`, and typed a parameter `s: ItemStack` against a sourceless
per-file stub (`<file>_rs_itemstack`). Neither was ever matched to the struct's
node `gt_core/src/item.rs::ItemStack`: on a multi-crate workspace thousands of
import edges dangled (build minted an external stub for each) and a type defined
in more than one module kept a placeholder in every file that used it. The
extractor now keeps the written path on the edge (`metadata.use_path`) and
graphify/rust_resolution.py walks it through the module tree the corpus's files
form, the modules' re-exports and their glob imports.

Honesty: an exact path keeps EXTRACTED; a name found only by being defined once in
the target crate is INFERRED at 0.85; an ambiguous name is left alone and an
external crate keeps the bare target the extractor emitted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import graphify.extractors.rust as rust_module
from graphify.extract import extract
from graphify.extractors.rust import _rust_use_path


def _cargo(name: str) -> str:
    return f'[package]\nname = "{name}"\nversion = "0.1.0"\nedition = "2021"\n'


# Two crates. `demo-core` (imported as `demo_core`) re-exports an item, an aliased
# item and an inline module; `app` imports from it in every shape the resolver
# must handle, and defines `Direction` twice so that name is ambiguous.
_WORKSPACE = {
    "core/Cargo.toml": _cargo("demo-core"),
    "core/src/lib.rs": (
        "pub mod item;\npub mod geom;\n"
        "pub use item::ItemStack;\n"
        "pub use geom::Direction as Dir;\n"
        "pub mod outer { pub struct Deep; }\n"
    ),
    "core/src/item.rs": "pub struct ItemStack { pub count: u32 }\npub fn parse() {}\n",
    "core/src/geom.rs": "pub enum Direction { North, South }\n",
    "app/Cargo.toml": _cargo("app"),
    "app/src/main.rs": (
        "mod util;\nmod dup;\nmod other;\nmod sub;\n"
        "use demo_core::ItemStack;\n"                 # re-export chain
        "use demo_core::item::parse as parse_item;\n"  # alias of a plain path
        "use demo_core::Dir;\n"                        # re-export with `as`
        "use demo_core::outer::Deep;\n"                # inline module: INFERRED
        "use crate::util::{make_helper as mh, *};\n"   # group, alias, glob
        "use crate::missing::Direction;\n"             # ambiguous in crate: untouched
        "use serde::Deserialize;\n"                    # external: untouched
        "use std::fmt::Formatter;\n"
        "fn run(s: ItemStack, d: Dir, h: Helper, deep: Deep, f: Formatter) -> Direction { todo!() }\n"
        "fn main() {}\n"
    ),
    "app/src/util.rs": (
        "pub struct Helper;\npub fn make_helper() {}\n"
        "#[cfg(test)]\nmod tests {\n    use super::*;\n    use crate::dup::Direction;\n    fn t(h: Helper) {}\n}\n"
    ),
    "app/src/dup.rs": "pub struct Direction;\n",
    "app/src/other.rs": "pub struct Direction;\n",
    "app/src/sub/mod.rs": "use super::util::Helper;\nuse super::super::nope;\npub fn f(h: Helper) {}\n",
}


def _write(root: Path, files: dict[str, str]) -> None:
    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body, encoding="utf-8")


def _extract(root: Path, files: dict[str, str] = _WORKSPACE) -> dict:
    _write(root, files)
    paths = sorted(root / name for name in files if name.endswith(".rs"))
    return extract(paths, root=root, cache_root=root / "graphify-out", parallel=False)


def _nid(result: dict, file_suffix: str, label: str) -> str:
    matches = [n["id"] for n in result["nodes"]
               if str(n.get("source_file", "")).endswith(file_suffix) and n.get("label") == label]
    assert len(matches) == 1, (file_suffix, label, matches)
    return matches[0]


def _imports(result: dict, file_suffix: str) -> dict[str, tuple[str, str, str, float | None]]:
    """use_path -> (relation, target, confidence, score) of the file's import edges."""
    out = {}
    for e in result["edges"]:
        if e.get("context") == "import" and e["source_file"].endswith(file_suffix) and "metadata" in e:
            out[e["metadata"]["use_path"]] = (e["relation"], e["target"], e["confidence"], e.get("confidence_score"))
    return out


def _refs(result: dict, file_suffix: str, label: str) -> set[str]:
    """Targets of the references edges from `label`'s node in the file."""
    src = _nid(result, file_suffix, label)
    return {e["target"] for e in result["edges"] if e["source"] == src and e["relation"] == "references"}


def test_use_paths_land_on_the_nodes_they_name(tmp_path: Path):
    result = _extract(tmp_path)
    item_stack = _nid(result, "core/src/item.rs", "ItemStack")
    parse = _nid(result, "core/src/item.rs", "parse()")
    direction = _nid(result, "core/src/geom.rs", "Direction")
    deep = _nid(result, "core/src/lib.rs", "Deep")
    util_file = _nid(result, "app/src/util.rs", "util.rs")
    helper_fn = _nid(result, "app/src/util.rs", "make_helper()")

    assert _imports(result, "app/src/main.rs") == {
        "demo_core::ItemStack": ("imports", item_stack, "EXTRACTED", None),
        "demo_core::item::parse as parse_item": ("imports", parse, "EXTRACTED", None),
        "demo_core::Dir": ("imports", direction, "EXTRACTED", None),
        # `outer` is an inline module the module tree cannot see: `Deep` is the
        # one definition of that name in demo_core, so the edge is inferred.
        "demo_core::outer::Deep": ("imports", deep, "INFERRED", 0.85),
        "crate::util::make_helper as mh": ("imports", helper_fn, "EXTRACTED", None),
        "crate::util::*": ("imports_from", util_file, "EXTRACTED", None),
        # Two `Direction`s in `app`, neither at `crate::missing`: left as emitted.
        "crate::missing::Direction": ("imports_from", "direction", "EXTRACTED", None),
        # External crates keep v8's bare last segment; build mints their stubs.
        "serde::Deserialize": ("imports_from", "deserialize", "EXTRACTED", None),
        "std::fmt::Formatter": ("imports_from", "formatter", "EXTRACTED", None),
    }


def test_imported_type_references_are_retargeted_and_the_placeholder_dropped(tmp_path: Path):
    result = _extract(tmp_path)
    refs = _refs(result, "app/src/main.rs", "run()")
    assert _nid(result, "core/src/item.rs", "ItemStack") in refs          # via re-export
    assert _nid(result, "core/src/geom.rs", "Direction") in refs          # via `as Dir`
    assert _nid(result, "app/src/util.rs", "Helper") in refs              # via `crate::util::*`
    assert _nid(result, "core/src/lib.rs", "Deep") in refs                # INFERRED
    deep_edges = [e for e in result["edges"] if e["target"] == _nid(result, "core/src/lib.rs", "Deep")
                  and e["source_file"].endswith("main.rs") and e["relation"] == "references"]
    assert deep_edges and all((e["confidence"], e.get("confidence_score")) == ("INFERRED", 0.85) for e in deep_edges)
    # The ambiguous `Direction` return type and the external `Formatter` keep
    # their sourceless placeholders; nothing is guessed. Every resolved
    # placeholder is gone with its last referrer.
    placeholders = {n["label"] for n in result["nodes"]
                    if not n.get("source_file") and n.get("file_type") == "code"}
    assert placeholders == {"Direction", "Formatter"}
    run = _nid(result, "app/src/main.rs", "run()")
    assert {next(n["label"] for n in result["nodes"] if n["id"] == e["target"])
            for e in result["edges"] if e["source"] == run and e["relation"] == "references"} == {
        "ItemStack", "Direction", "Helper", "Deep", "Formatter"}


def test_super_and_glob_from_an_inline_test_module(tmp_path: Path):
    result = _extract(tmp_path)
    util_file = _nid(result, "app/src/util.rs", "util.rs")
    helper = _nid(result, "app/src/util.rs", "Helper")
    dup_direction = _nid(result, "app/src/dup.rs", "Direction")
    imports = _imports(result, "app/src/util.rs")
    # `use super::*;` inside `mod tests` is the file importing itself; the path is
    # recorded relative to the file's module and the edge targets the file node.
    assert imports["self::*"] == ("imports_from", util_file, "EXTRACTED", None)
    assert imports["crate::dup::Direction"] == ("imports", dup_direction, "EXTRACTED", None)
    assert helper in _refs(result, "app/src/util.rs", "t()")
    # `super::util::Helper` from app/src/sub/mod.rs; `super::super::nope` climbs
    # past the crate root and is left as emitted.
    sub = _imports(result, "app/src/sub/mod.rs")
    assert sub["super::util::Helper"] == ("imports", helper, "EXTRACTED", None)
    assert sub["super::super::nope"] == ("imports_from", "nope", "EXTRACTED", None)
    assert helper in _refs(result, "app/src/sub/mod.rs", "f()")


def test_an_ambiguous_name_is_never_resolved(tmp_path: Path):
    files = {**_WORKSPACE, "app/src/user.rs": "pub fn pick(d: Direction) {}\n"}
    files["app/src/main.rs"] = files["app/src/main.rs"].replace("mod util;", "mod util;\nmod user;")
    result = _extract(tmp_path, files)
    targets = _refs(result, "app/src/user.rs", "pick()")
    assert len(targets) == 1
    (target,) = targets
    node = next(n for n in result["nodes"] if n["id"] == target)
    assert node["label"] == "Direction" and not node.get("source_file")


@pytest.mark.parametrize("binding, inline, expected", [
    ("crate::util :: {parse /* note */ as p}".split("{")[0] + "parse /* note */ as p", (), "crate::util::parse as p"),
    ("::std::io::Read", (), "std::io::Read"),
    ("crate::{r#match}".split("{")[0] + "r#match", (), "crate::match"),
    ("super::*", ("tests",), "self::*"),
    ("super::super::top::Thing", ("tests",), "super::top::Thing"),
    ("self::local::X", ("inner",), "self::inner::local::X"),
    ("super::Y", ("a", "b"), "self::a::Y"),
    ("crate::x", ("tests",), "crate::x"),
])
def test_use_path_normalisation(binding: str, inline: tuple[str, ...], expected: str):
    assert _rust_use_path(binding, inline) == expected


def _build(root: Path, changed: list[Path] | None = None) -> dict:
    from graphify.watch import _rebuild_code

    assert _rebuild_code(root, changed_paths=changed, no_cluster=True, acquire_lock=False) is True
    return json.loads((root / "graphify-out" / "graph.json").read_text(encoding="utf-8"))


def _view(graph: dict) -> tuple:
    return (sorted(json.dumps(e, sort_keys=True) for e in graph.get("links", graph.get("edges", []))),
            sorted(json.dumps(n, sort_keys=True) for n in graph["nodes"] if not n.get("external")))


def test_incremental_build_resolves_through_unchanged_files_like_a_full_build(tmp_path: Path):
    # The re-export chain `demo_core::ItemStack` -> `pub use item::ItemStack`
    # runs through core/src/lib.rs, which the incremental build does not
    # re-extract: its `use` edges must reach the resolver as context.
    root = tmp_path / "inc"
    _write(root, _WORKSPACE)
    _build(root)
    edited = _WORKSPACE["app/src/main.rs"] + "fn extra(s: ItemStack) {}\n"
    (root / "app/src/main.rs").write_text(edited, encoding="utf-8")
    incremental = _build(root, [root / "app/src/main.rs"])

    fresh = tmp_path / "fresh"
    _write(fresh, {**_WORKSPACE, "app/src/main.rs": edited})
    assert _view(incremental) == _view(_build(fresh))
    links = incremental["links"]
    item_stack = next(n["id"] for n in incremental["nodes"]
                      if n.get("label") == "ItemStack" and str(n.get("source_file", "")).endswith("item.rs"))
    assert any(e["target"] == item_stack and e["relation"] == "imports"
               and e.get("metadata", {}).get("use_path") == "demo_core::ItemStack" for e in links)


def test_a_cache_entry_without_the_use_path_is_not_served(tmp_path: Path, monkeypatch):
    import graphify.cache as cache_mod
    from graphify.cache import save_cached

    _write(tmp_path, _WORKSPACE)
    path = tmp_path / "app" / "src" / "main.rs"
    current = cache_mod._AST_CACHE_SCHEMA
    assert current >= 16
    monkeypatch.setattr(cache_mod, "_AST_CACHE_SCHEMA", 15)  # 0.9.90's
    with monkeypatch.context() as patch:
        patch.setattr(rust_module, "_rust_use_path", lambda *args: "")  # what 0.9.90 cached: no path
        stale = rust_module.extract_rust(path)
    assert not any("metadata" in e for e in stale["edges"])
    save_cached(path, stale, tmp_path, cache_root=tmp_path / "graphify-out")
    monkeypatch.setattr(cache_mod, "_AST_CACHE_SCHEMA", current)
    result = extract(sorted(tmp_path / n for n in _WORKSPACE if n.endswith(".rs")), root=tmp_path,
                     cache_root=tmp_path / "graphify-out", parallel=False)
    assert _imports(result, "app/src/main.rs")["demo_core::ItemStack"][1] == _nid(result, "core/src/item.rs", "ItemStack")
