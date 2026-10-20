"""Export the local corpus as JSONL shards for the agent systems (App. C.3).

"The local corpus is mounted as JSONL shard files indexed by paper ID and
primary category." Layout of an export directory::

    <out_dir>/
      README.md                   # human/agent-readable description of the layout
      manifest.json               # format, counts, per-category shard lists
      paper_index.tsv             # paper_id <TAB> category <TAB> shard <TAB> line <TAB> byte_offset
      shards/<category>/shard-00000.jsonl
      shards/<category>/shard-00001.jsonl
      ...

Each shard line is one :class:`~scholarpath.types.Paper` record (``paper_id``,
``title``, ``abstract``, dates, categories, and the section ``chunks`` when
``include_chunks``). Papers are grouped by primary category (first arXiv
category; ``uncategorized`` when missing) and sorted by paper id inside a
category. ``paper_index.tsv`` is sorted by paper id, so agents can look a
paper up with ``grep``/``rg`` and seek to its byte offset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from scholarpath.data.corpus import Corpus
from scholarpath.types import Paper

FORMAT = "scholarpath-corpus-shards/v1"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def category_dir(category: Optional[str]) -> str:
    """Filesystem-safe directory name of a category (``math.OC`` stays ``math.OC``)."""
    name = _SAFE.sub("_", (category or "").strip()).strip("._")
    return name or "uncategorized"


def primary_category(paper: Paper) -> str:
    return paper.primary_category or (paper.categories[0] if paper.categories else "") or "uncategorized"


@dataclass
class CorpusExport:
    root: Path
    num_papers: int
    shards: List[str] = field(default_factory=list)  # relative shard paths
    categories: Dict[str, int] = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def index_path(self) -> Path:
        return self.root / "paper_index.tsv"


def _papers(source: Union[Corpus, Iterable[Paper], Dict[str, Paper]]) -> List[Paper]:
    if isinstance(source, Corpus):
        return list(source)
    if isinstance(source, dict):
        return list(source.values())
    return list(source)


def export_corpus_shards(
    corpus: Union[Corpus, Iterable[Paper], Dict[str, Paper]],
    out_dir: Union[str, Path],
    shard_size: int = 5000,
    include_chunks: bool = True,
    readonly: bool = False,
) -> CorpusExport:
    """Write the corpus as category-partitioned JSONL shards plus a manifest.

    With ``readonly`` the exported files are made read-only (the agents get a
    read-only view of the corpus).
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    root = Path(out_dir)
    (root / "shards").mkdir(parents=True, exist_ok=True)
    by_cat: Dict[str, List[Paper]] = {}
    for p in _papers(corpus):
        by_cat.setdefault(primary_category(p), []).append(p)
    index_rows: List[tuple] = []
    shards: List[str] = []
    manifest_cats: Dict[str, Any] = {}
    for cat in sorted(by_cat):
        papers = sorted(by_cat[cat], key=lambda p: p.paper_id)
        cdir = category_dir(cat)
        cat_shards: List[str] = []
        for s in range(0, len(papers), shard_size):
            rel = f"shards/{cdir}/shard-{s // shard_size:05d}.jsonl"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            offset = 0
            with open(path, "wb") as f:
                for line_no, p in enumerate(papers[s : s + shard_size]):
                    data = (json.dumps(p.to_dict(include_chunks=include_chunks), ensure_ascii=False) + "\n").encode()
                    f.write(data)
                    index_rows.append((p.paper_id, cat, rel, line_no, offset))
                    offset += len(data)
            cat_shards.append(rel)
        shards.extend(cat_shards)
        manifest_cats[cat] = {"directory": f"shards/{cdir}", "num_papers": len(papers), "shards": cat_shards}
    index_rows.sort(key=lambda r: r[0])
    with open(root / "paper_index.tsv", "w", encoding="utf-8") as f:
        f.write("paper_id\tcategory\tshard\tline\tbyte_offset\n")
        for pid, cat, rel, line_no, off in index_rows:
            f.write(f"{pid}\t{cat}\t{rel}\t{line_no}\t{off}\n")
    manifest = {
        "format": FORMAT,
        "num_papers": len(index_rows),
        "shard_size": shard_size,
        "include_chunks": include_chunks,
        "fields": ["paper_id", "title", "abstract", "year", "date", "categories", "primary_category",
                   "authors", "arxiv_id", "s2_id", "doi", "chunks", "metadata"],
        "index": "paper_index.tsv",
        "categories": manifest_cats,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (root / "README.md").write_text(_readme(manifest), encoding="utf-8")
    if readonly:
        for path in root.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
    return CorpusExport(root, len(index_rows), shards, {c: v["num_papers"] for c, v in manifest_cats.items()})


def _readme(manifest: Dict[str, Any]) -> str:
    cats = "\n".join(f"- `{c}`: {v['num_papers']} papers in `{v['directory']}/`"
                     for c, v in manifest["categories"].items())
    return (
        "# Local paper corpus (read-only)\n\n"
        f"{manifest['num_papers']} papers stored as JSON-lines shards, one paper per line, grouped by primary "
        "category and sorted by paper id.\n\n"
        "* `paper_index.tsv` -- `paper_id, category, shard, line, byte_offset` for every paper (sorted by id).\n"
        "* `shards/<category>/shard-NNNNN.jsonl` -- paper records with fields "
        f"{', '.join('`' + f + '`' for f in manifest['fields'])}.\n"
        "  `chunks` holds the section chunks of the full text (`heading`, `text`).\n\n"
        "Cite papers by their `paper_id`. Categories:\n\n" + cats + "\n"
    )


def load_corpus_shards(root: Union[str, Path], name: Optional[str] = None) -> Corpus:
    """Read an export back into a :class:`~scholarpath.data.corpus.Corpus`."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{root} is not a {FORMAT} export")
    papers: List[Paper] = []
    for cat in manifest["categories"].values():
        for rel in cat["shards"]:
            with open(root / rel, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        papers.append(Paper.from_dict(json.loads(line)))
    return Corpus(papers, name=name or root.name)


def read_paper(root: Union[str, Path], paper_id: str) -> Paper:
    """Random access to one exported paper through ``paper_index.tsv`` offsets."""
    root = Path(root)
    with open(root / "paper_index.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, _cat, rel, _line_no, off = line.rstrip("\n").split("\t")
            if pid == paper_id:
                with open(root / rel, "rb") as sf:
                    sf.seek(int(off))
                    return Paper.from_dict(json.loads(sf.readline().decode("utf-8")))
    raise KeyError(paper_id)
