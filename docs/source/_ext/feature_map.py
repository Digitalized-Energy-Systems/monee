"""Feature overview page: the interactive feature map and the task table.

Both are generated from ``_data/features.yaml``. Two directives place them on a
page, ``feature-map`` and ``feature-tasks``. Documentation references in the
YAML (``doc`` plus a heading ``section``) are validated after reading and
resolved with the builder at ``doctree-resolved``, so the links work for the
html and dirhtml builders alike. HTML builders get the interactive markup plus
``_static/feature_map``; every other builder (LaTeX for the PDF) gets a plain
feature list and task tables.
"""

from __future__ import annotations

import html
import json
import os
import re
import unicodedata

import yaml
from docutils import nodes
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode

logger = logging.getLogger(__name__)

STATIC_DIR = "feature_map"
PALETTE_SLOTS = 10
OTHER_GROUP = {
    "id": "other",
    "title": "More features",
    "tagline": "",
    "style": None,
    "slot": 0,
}
KIND_LABELS = {
    "quickstart": "Quickstart",
    "install": "Install",
    "concept": "Concept",
    "how-to": "How-to",
    "tutorial": "Tutorial",
    "problem": "Problem",
    "component": "Component",
    "benchmark": "Benchmark",
    "api": "API",
}
LEVELS = ("core", "advanced")
CARRIERS = ("electricity", "gas", "heat", "all")
GROUP_STYLES = ("open",)
FEATURE_KEYS = (
    "id", "group", "title", "short", "tagline", "level", "carriers", "details",
    "highlights", "snippet", "api", "keywords", "related",
)  # fmt: skip
STRING_FIELDS = ("title", "short", "tagline", "level", "details", "snippet")
LIST_FIELDS = ("carriers", "highlights", "api", "keywords", "related")
TOP_KEYS = ("groups", "features", "tasks")
GROUP_KEYS = ("id", "title", "order", "tagline", "style")
TASK_KEYS = ("id", "category", "want", "how", "features", "docs")
DOC_KEYS = ("doc", "section", "label", "kind")
TASK_PREFIX = "task-"


class feature_map_node(nodes.comment):
    """Placeholder that ``doctree-resolved`` replaces with the rendered output."""


class _Map(dict):
    line = 0


class _Loader(yaml.SafeLoader):
    pass


def _construct_map(loader, node):
    data = _Map()
    data.line = node.start_mark.line + 1
    seen = set()
    for key_node, _ in node.value:
        key = key_node.value
        if isinstance(key, str) and key in seen:
            logger.warning(
                "feature map data: duplicate key '%s'; the last value wins",
                key,
                location=f"{loader.name}:{key_node.start_mark.line + 1}",
                type="feature_map",
            )
        seen.add(key)
    yield data
    data.update(loader.construct_mapping(node))


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_map)


def _norm(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _anchors(env):
    if not hasattr(env, "feature_map_anchors"):
        env.feature_map_anchors = {}
        env.feature_map_users = set()
    return env.feature_map_anchors


def _users(env):
    _anchors(env)
    return env.feature_map_users


class _HeadingCollector(EnvironmentCollector):
    """Index the section titles and rubrics of every document by normalized text.

    Rubrics count because MyST turns headings inside sphinx-design tab items into
    rubrics, and several catalog links point at those.
    """

    def clear_doc(self, app, env, docname):
        _anchors(env).pop(docname, None)
        _users(env).discard(docname)

    def merge_other(self, app, env, docnames, other):
        for docname in docnames:
            if docname in _anchors(other):
                _anchors(env)[docname] = _anchors(other)[docname]
            if docname in _users(other):
                _users(env).add(docname)

    def process_doc(self, app, doctree):
        index = {}
        for node in doctree.findall(
            lambda n: isinstance(n, (nodes.section, nodes.rubric))
        ):
            if not node["ids"]:
                continue
            is_section = isinstance(node, nodes.section)
            title = node.next_node(nodes.title) if is_section else node
            if title is None:
                continue
            index.setdefault(_norm(title.astext()), []).append(
                (node["ids"][0], is_section)
            )
        _anchors(app.env)[app.env.docname] = index
        if any(doctree.findall(feature_map_node)):
            _users(app.env).add(app.env.docname)


class _FeatureDirective(SphinxDirective):
    kind = ""

    def run(self):
        self.env.note_dependency(_data_path(self.env.srcdir, self.config))
        node = feature_map_node()
        node["feature_map"] = self.kind
        self.set_source_info(node)
        return [node]


class FeatureMapDirective(_FeatureDirective):
    kind = "map"


class FeatureTasksDirective(_FeatureDirective):
    kind = "tasks"


def _data_path(srcdir, config):
    return os.path.normpath(os.path.join(srcdir, config.feature_map_data))


class _Catalog:
    """The validated catalog. Invalid entries are reported once and then skipped."""

    def __init__(self, env, path):
        self.env = env
        self.path = path
        self.groups = []
        self.by_group = {}
        self.features = []
        self.tasks = []
        raw = self._read()
        if raw is not None:
            self._build(raw)

    def warn(self, message, obj=None):
        line = getattr(obj, "line", 0)
        location = f"{self.path}:{line}" if line else self.path
        logger.warning(message, location=location, type="feature_map")

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = yaml.load(f, Loader=_Loader)
        except OSError as exc:
            logger.warning("feature map data not readable: %s", exc, type="feature_map")
            return None
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            location = f"{self.path}:{mark.line + 1}" if mark else self.path
            logger.warning(
                "feature map data is not valid YAML: %s",
                getattr(exc, "problem", None) or exc,
                location=location,
                type="feature_map",
            )
            return None
        if not isinstance(raw, dict):
            self.warn("feature map data must be a mapping with groups, features, tasks")
            return None
        self._known_keys(raw, TOP_KEYS, "feature map data")
        return raw

    def _known_keys(self, entry, allowed, what):
        for key in entry:
            if key not in allowed:
                self.warn(
                    f"{what}: unknown key '{key}' (expected one of {', '.join(allowed)})",
                    entry,
                )

    def _entries(self, raw, key):
        value = raw.get(key)
        if value is None:
            self.warn(f"feature map data has no '{key}' list", raw)
            return []
        if not isinstance(value, list):
            self.warn(f"'{key}' must be a list", raw)
            return []
        out = []
        for entry in value:
            if isinstance(entry, dict):
                out.append(entry)
            else:
                self.warn(
                    f"every entry of '{key}' must be a mapping, not {entry!r}", raw
                )
        return out

    def _string(self, entry, key, what, required=False):
        value = entry.get(key)
        if value is None or value == "":
            if required:
                self.warn(f"{what} has no '{key}'", entry)
            return ""
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            self.warn(f"{what}: '{key}' must be text", entry)
            return ""
        return str(value)

    def _list(self, entry, key, what, required=False):
        value = entry.get(key)
        if value is None or value == []:
            if required:
                self.warn(f"{what} has no '{key}'", entry)
            return []
        if not isinstance(value, list):
            self.warn(f"{what}: '{key}' must be a list", entry)
            return []
        out = []
        for item in value:
            if isinstance(item, str) and item:
                out.append(item)
            else:
                self.warn(f"{what}: '{key}' entries must be text, not {item!r}", entry)
        return out

    def _build(self, raw):
        self._build_groups(self._entries(raw, "groups"))
        self._build_features(self._entries(raw, "features"))
        self._build_tasks(self._entries(raw, "tasks"))

    def _build_groups(self, entries):
        seen = set()
        groups = []
        for entry in entries:
            gid = self._string(entry, "id", "group", required=True)
            if not gid:
                continue
            what = f"group '{gid}'"
            self._known_keys(entry, GROUP_KEYS, what)
            if gid == OTHER_GROUP["id"]:
                self.warn(f"{what}: the id '{gid}' is reserved", entry)
                continue
            if gid in seen:
                self.warn(f"duplicate {what}", entry)
                continue
            seen.add(gid)
            order = entry.get("order", 0)
            if not isinstance(order, (int, float)) or isinstance(order, bool):
                self.warn(f"{what}: 'order' must be a number", entry)
                order = 0
            style = self._string(entry, "style", what) or None
            if style and style not in GROUP_STYLES:
                self.warn(f"{what}: unknown style '{style}'", entry)
                style = None
            groups.append(
                {
                    "id": gid,
                    "title": self._string(entry, "title", what, required=True) or gid,
                    "tagline": self._string(entry, "tagline", what),
                    "style": style,
                    "order": order,
                    "entry": entry,
                }
            )
        groups.sort(key=lambda g: g["order"])
        self.groups = groups

    def _build_features(self, entries):
        known_groups = {g["id"] for g in self.groups}
        features = []
        seen = set()
        for entry in entries:
            fid = self._string(entry, "id", "feature", required=True)
            if not fid:
                continue
            what = f"feature '{fid}'"
            self._known_keys(entry, (*FEATURE_KEYS, "docs"), what)
            if fid in seen:
                self.warn(f"duplicate {what}", entry)
                continue
            seen.add(fid)
            feature = {"id": fid, "entry": entry}
            for key in STRING_FIELDS:
                required = key in ("title", "tagline", "level")
                feature[key] = self._string(entry, key, what, required=required)
            for key in LIST_FIELDS:
                feature[key] = self._list(entry, key, what, required=key == "carriers")
            feature["title"] = feature["title"] or fid
            group = self._string(entry, "group", what, required=True)
            if group and group not in known_groups:
                self.warn(f"{what}: unknown group '{group}'", entry)
            feature["group"] = group if group in known_groups else OTHER_GROUP["id"]
            if feature["level"] and feature["level"] not in LEVELS:
                self.warn(
                    f"{what}: level must be one of {', '.join(LEVELS)}, "
                    f"not '{feature['level']}'",
                    entry,
                )
            for carrier in feature["carriers"]:
                if carrier not in CARRIERS:
                    self.warn(f"{what}: unknown carrier '{carrier}'", entry)
            feature["docs"] = self._docs(entry, what)
            features.append(feature)
        ids = {f["id"] for f in features}
        for feature in features:
            related = []
            for rid in feature["related"]:
                if rid not in ids or rid == feature["id"]:
                    self.warn(
                        f"feature '{feature['id']}': related id '{rid}' is not "
                        "another feature",
                        feature["entry"],
                    )
                elif rid not in related:
                    related.append(rid)
            feature["related"] = related
        used = {f["group"] for f in features}
        for group in self.groups:
            if group["id"] not in used:
                self.warn(f"group '{group['id']}' has no features", group["entry"])
        self.groups = [g for g in self.groups if g["id"] in used]
        for n, group in enumerate(self.groups):
            group["slot"] = n % PALETTE_SLOTS + 1
        if OTHER_GROUP["id"] in used:
            self.groups.append(dict(OTHER_GROUP))
        self.by_group = {g["id"]: [] for g in self.groups}
        for feature in features:
            self.by_group[feature["group"]].append(feature)
        self.features = features

    def _build_tasks(self, entries):
        ids = {f["id"] for f in self.features}
        seen = set()
        for entry in entries:
            tid = self._string(entry, "id", "task", required=True)
            if not tid:
                continue
            what = f"task '{tid}'"
            self._known_keys(entry, TASK_KEYS, what)
            if not tid.startswith(TASK_PREFIX):
                self.warn(f"{what}: task ids must start with '{TASK_PREFIX}'", entry)
                continue
            if tid in seen:
                self.warn(f"duplicate {what}", entry)
                continue
            seen.add(tid)
            want = self._string(entry, "want", what, required=True)
            if not want:
                continue
            fids = []
            for fid in self._list(entry, "features", what, required=True):
                if fid in ids:
                    fids.append(fid)
                else:
                    self.warn(f"{what}: unknown feature id '{fid}'", entry)
            self.tasks.append(
                {
                    "id": tid,
                    "category": self._string(entry, "category", what, required=True)
                    or "Other",
                    "want": want,
                    "how": self._string(entry, "how", what, required=True),
                    "features": fids,
                    "docs": self._docs(entry, what),
                }
            )

    def _docs(self, entry, what):
        value = entry.get("docs")
        if not isinstance(value, list) or not value:
            self.warn(f"{what}: 'docs' must be a non-empty list", entry)
            return []
        docs = []
        for ref in value:
            if not isinstance(ref, dict):
                self.warn(f"{what}: every docs entry must be a mapping", entry)
                continue
            resolved = self._resolve(ref, what)
            if resolved:
                docs.append(resolved)
        return docs

    def _resolve(self, ref, what):
        self._known_keys(ref, DOC_KEYS, f"{what} docs entry")
        doc = self._string(ref, "doc", f"{what} docs entry", required=True)
        label = self._string(ref, "label", f"{what} docs entry", required=True)
        kind = self._string(ref, "kind", f"{what} docs entry", required=True)
        section = self._string(ref, "section", what)
        if not doc or not label:
            return None
        if kind and kind not in KIND_LABELS:
            self.warn(
                f"{what}: unknown docs kind '{kind}' "
                f"(expected one of {', '.join(KIND_LABELS)})",
                ref,
            )
        if doc not in self.env.found_docs:
            self.warn(f"{what}: unknown document '{doc}'", ref)
            return None
        anchor, is_section = None, False
        if section:
            matches = _anchors(self.env).get(doc, {}).get(_norm(section), [])
            if not matches:
                self.warn(
                    f"{what}: no heading or rubric '{section}' in '{doc}', "
                    "linking the page instead",
                    ref,
                )
            else:
                anchor, is_section = matches[0]
        return {
            "doc": doc,
            "anchor": anchor,
            "is_section": is_section,
            "label": label,
            "kind": kind,
        }


def _catalog(app):
    path = _data_path(app.srcdir, app.config)
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        stamp = None
    cached = getattr(app, "_feature_map_catalog", None)
    if cached and cached[0] == (path, stamp, id(app.env)):
        return cached[1]
    catalog = _Catalog(app.env, path)
    app._feature_map_catalog = ((path, stamp, id(app.env)), catalog)
    return catalog


def _check_consistency(app, env):
    if _users(env):
        _catalog(app)


def _outdated(app, env, added, changed, removed):
    # A renamed or removed heading anywhere must re-validate the page links.
    if added or changed or removed:
        return sorted(_users(env) - set(removed))
    return []


def _kind_label(kind):
    return KIND_LABELS.get(kind, kind.title())


def _kind_rank(ref):
    order = list(KIND_LABELS)
    return order.index(ref["kind"]) if ref["kind"] in KIND_LABELS else len(order)


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _url(builder, fromdoc, ref):
    if builder.name == "singlehtml":
        return "#" + (ref["anchor"] or "document-" + ref["doc"])
    uri = builder.get_relative_uri(fromdoc, ref["doc"])
    return f"{uri}#{ref['anchor']}" if ref["anchor"] else uri


esc = html.escape


def _code_html(match):
    return "<code>" + re.sub(r"([._(])", r"\1<wbr>", match.group(1)) + "</code>"


def _inline(text):
    return re.sub(r"`([^`]+)`", _code_html, esc(text or ""))


def _badge(n, cls="mfm-num"):
    return f'<span class="{cls}" aria-hidden="true">{n or ""}</span>'


def _payload(catalog, builder, fromdoc):
    features = []
    for gid, items in catalog.by_group.items():
        for feature in items:
            slim = {k: feature[k] for k in FEATURE_KEYS if feature.get(k)}
            slim["group"] = gid
            slim["docs"] = [
                {
                    "label": d["label"],
                    "kind": _kind_label(d["kind"]),
                    "url": _url(builder, fromdoc, d),
                }
                for d in sorted(feature["docs"], key=_kind_rank)
            ]
            features.append(slim)
    data = {
        "groups": [
            {k: g[k] for k in ("id", "title", "tagline", "slot")}
            | ({"style": g["style"]} if g["style"] else {})
            for g in catalog.groups
        ],
        "features": features,
        "tasks": [
            {"id": t["id"], "want": t["want"], "features": t["features"]}
            for t in catalog.tasks
        ],
    }
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/")


def _feature_fallback(feature, builder, fromdoc):
    head = esc(feature["title"])
    if feature["docs"]:
        head = f'<a href="{esc(_url(builder, fromdoc, feature["docs"][0]))}">{head}</a>'
    return (
        f'<li id="feature-{esc(feature["id"])}">{head}'
        f'<span class="mfm-fb-t">{_inline(feature["tagline"])}</span></li>'
    )


def _map_html(catalog, builder, fromdoc):
    parts = [
        '<div class="mfm mfm-explorer" data-mfm-root>',
        '<script type="application/json" id="mfm-data">',
        _payload(catalog, builder, fromdoc),
        '</script><div class="mfm-fallback">',
    ]
    for n, group in enumerate(catalog.groups, 1):
        items = "".join(
            _feature_fallback(f, builder, fromdoc)
            for f in catalog.by_group[group["id"]]
        )
        parts.append(
            f'<div class="mfm-fb-g" data-p="{group["slot"]}">'
            f'<h2 class="mfm-fb-h">{_badge(n)}{esc(group["title"])}</h2>'
            f'<p class="mfm-fb-tag">{_inline(group["tagline"])}</p><ul>{items}</ul></div>'
        )
    parts.append("</div></div>")
    return "".join(parts)


def _doc_list(docs, builder, fromdoc):
    items = "".join(
        f'<li><a href="{esc(_url(builder, fromdoc, d))}">{esc(d["label"])}</a>'
        f'<span class="mfm-kind">{esc(_kind_label(d["kind"]))}</span></li>'
        for d in docs
    )
    return f"<ul>{items}</ul>"


def _categories(catalog):
    categories = {}
    for task in catalog.tasks:
        categories.setdefault(task["category"], []).append(task)
    return categories


def _tasks_html(catalog, builder, fromdoc):
    numbered = {
        g["title"]: (n, g) for n, g in enumerate(catalog.groups, 1) if g["slot"]
    }
    by_id = {f["id"]: f for f in catalog.features}
    slot_of = {g["id"]: g["slot"] for g in catalog.groups}
    categories = _categories(catalog)
    marks = {}
    for category in categories:
        num, group = numbered.get(category, (0, None))
        marks[category] = (
            _badge(num) if group else _badge(0, "mfm-num mfm-num-start"),
            group["slot"] if group else "start",
        )
    jump = "".join(
        f'<a href="#tasks-{_slug(category)}" data-p="{slot}" '
        f'title="{esc(category)}"><span class="mfm-sr">{esc(category)}</span>{mark}</a>'
        for category, (mark, slot) in marks.items()
    )
    rows = [
        '<div class="mfm mfm-tasks">',
        f'<nav class="mfm-tjump" aria-label="Jump to a task category">{jump}</nav>',
        '<div class="mfm-tfilter" hidden><label class="mfm-search">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/>'
        '<path d="m20 20-4.6-4.6"/></svg><span class="mfm-sr">Filter tasks</span>'
        '<input type="search" placeholder="Filter tasks, e.g. battery" autocomplete="off">'
        '</label><span class="mfm-tcount"></span></div>',
        '<table class="mfm-table"><caption class="mfm-sr">Tasks and the monee '
        "features that solve them</caption>",
        '<colgroup><col class="mfm-col-goal"><col class="mfm-col-how">'
        '<col class="mfm-col-docs"></colgroup>',
        '<thead><tr><th scope="col">Goal</th><th scope="col">How</th>'
        '<th scope="col">Docs</th></tr></thead>',
    ]
    for category, tasks in categories.items():
        mark, slot = marks[category]
        rows.append(
            f'<tbody data-p="{slot}"><tr class="mfm-cat">'
            f'<th colspan="3" scope="rowgroup" id="tasks-{_slug(category)}">{mark}'
            f'{esc(category)}<span class="mfm-cat-n"><span class="mfm-sr">, </span>'
            f'<span class="mfm-cat-c">{len(tasks)}</span>'
            '<span class="mfm-sr"> tasks</span></span></th></tr>'
        )
        for task in tasks:
            chips = "".join(
                f'<a class="mfm-chip" href="#feature-{esc(fid)}" data-fid="{esc(fid)}" '
                f'data-p="{slot_of.get(by_id[fid]["group"], 0)}">'
                f"{esc(by_id[fid]['title'])}</a>"
                for fid in task["features"]
            )
            rows.append(
                f'<tr id="{esc(task["id"])}"><td class="mfm-goal">{_inline(task["want"])}</td>'
                f'<td class="mfm-how"><p>{_inline(task["how"])}</p>'
                f'<p class="mfm-chips">{chips}</p></td>'
                f'<td class="mfm-tdocs">{_doc_list(task["docs"], builder, fromdoc)}</td></tr>'
            )
        rows.append("</tbody>")
    rows.append("</table></div>")
    return "".join(rows)


def _literal_nodes(code):
    out = []
    for n, piece in enumerate(p for p in re.split(r"(?<=[_=(])", code) if p):
        if n:
            out.append(nodes.raw("", r"\allowbreak{}", format="latex"))
        out.append(nodes.literal(piece, piece))
    return out


def _inline_nodes(text):
    out = []
    for n, part in enumerate(re.split(r"`([^`]+)`", text or "")):
        if n % 2:
            out.extend(_literal_nodes(part))
        elif part:
            out.append(nodes.Text(part))
    return out


def _ref_node(builder, fromdoc, ref, label):
    # LaTeX only labels sections, so rubric anchors fall back to the page.
    anchor = ref["anchor"] if ref["is_section"] else None
    return make_refnode(
        builder, fromdoc, ref["doc"], anchor or "", nodes.Text(label), label
    )


def _map_nodes(catalog, builder, fromdoc):
    out = []
    for n, group in enumerate(catalog.groups, 1):
        prefix = f"{n}. " if group["slot"] else ""
        out.append(nodes.rubric("", prefix + group["title"]))
        if group["tagline"]:
            out.append(nodes.paragraph("", "", *_inline_nodes(group["tagline"])))
        items = nodes.bullet_list()
        for feature in catalog.by_group[group["id"]]:
            para = nodes.paragraph()
            if feature["docs"]:
                para += _ref_node(
                    builder, fromdoc, feature["docs"][0], feature["title"]
                )
            else:
                para += nodes.Text(feature["title"])
            if feature["tagline"]:
                para += nodes.Text(": ")
                para.extend(_inline_nodes(feature["tagline"]))
            items += nodes.list_item("", para)
        out.append(items)
    return out


def _tasks_nodes(catalog, builder, fromdoc):
    by_id = {f["id"]: f for f in catalog.features}
    out = []
    for category, tasks in _categories(catalog).items():
        out.append(nodes.rubric("", category))
        group = nodes.tgroup(cols=3)
        for width in (25, 50, 25):
            group += nodes.colspec(colwidth=width)
        head = nodes.row()
        for title in ("Goal", "How", "Docs"):
            head += nodes.entry("", nodes.paragraph("", title))
        group += nodes.thead("", head)
        body = nodes.tbody()
        for task in tasks:
            how = [nodes.paragraph("", "", *_inline_nodes(task["how"]))]
            if task["features"]:
                titles = ", ".join(by_id[fid]["title"] for fid in task["features"])
                how.append(nodes.paragraph("", f"Features: {titles}"))
            docs = [
                nodes.paragraph("", "", _ref_node(builder, fromdoc, d, d["label"]))
                for d in task["docs"]
            ]
            body += nodes.row(
                "",
                nodes.entry("", nodes.paragraph("", "", *_inline_nodes(task["want"]))),
                nodes.entry("", *how),
                nodes.entry("", *docs),
            )
        group += body
        out.append(nodes.table("", group, classes=["longtable", "colwidths-given"]))
    return out


def _render(app, doctree, docname):
    placeholders = list(doctree.findall(feature_map_node))
    if not placeholders:
        return
    catalog = _catalog(app)
    builder = app.builder
    for node in placeholders:
        kind = node["feature_map"]
        if not (catalog.features if kind == "map" else catalog.tasks):
            new = []
        elif builder.format == "html":
            markup = (_map_html if kind == "map" else _tasks_html)(
                catalog, builder, docname
            )
            new = [nodes.raw("", markup, format="html", feature_map=kind)]
        else:
            new = (_map_nodes if kind == "map" else _tasks_nodes)(
                catalog, builder, docname
            )
        node.replace_self(new)


def _add_assets(app, pagename, templatename, context, doctree):
    if doctree is None:
        return
    if any(n.get("feature_map") for n in doctree.findall(nodes.raw)):
        app.add_css_file(f"{STATIC_DIR}/feature_map.css")
        app.add_js_file(f"{STATIC_DIR}/feature_map.js", loading_method="defer")


def setup(app):
    app.add_config_value("feature_map_data", "_data/features.yaml", "env", [str])
    app.add_directive("feature-map", FeatureMapDirective)
    app.add_directive("feature-tasks", FeatureTasksDirective)
    app.add_env_collector(_HeadingCollector)
    app.connect("env-get-outdated", _outdated)
    app.connect("env-check-consistency", _check_consistency)
    app.connect("doctree-resolved", _render)
    app.connect("html-page-context", _add_assets)
    return {
        "version": "1",
        "env_version": 1,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
