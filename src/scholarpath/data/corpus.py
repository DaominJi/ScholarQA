"""Corpus container and JSONL I/O.

The processed on-disk format used by every script is a (optionally
gzipped) JSON-lines file of :class:`~scholarpath.types.Paper` records::

    {"paper_id": "2101.00001", "title": "...", "abstract": "...",
     "date": "2021-01-01", "year": 2021, "categories": ["cs.DB"],
     "primary_category": "cs.DB", "arxiv_id": "2101.00001",
     "chunks": [{"chunk_id": "2101.00001#0", "paper_id": "2101.00001",
                 "heading": "Introduction", "text": "...", "order": 0}, ...]}

Full-text chunks may also live in a separate ``chunks.jsonl(.gz)`` file
(one :class:`~scholarpath.types.Chunk` per line), attached with
:meth:`Corpus.attach_chunks`. Queries use the same JSONL convention with
:class:`~scholarpath.types.Query` records.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union

from scholarpath.types import Chunk, Paper, Query

PathLike = Union[str, Path]


def _open(path: PathLike, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8") if "t" in mode else gzip.open(path, mode)
    return open(path, mode, encoding="utf-8") if "t" in mode else open(path, mode)


def read_jsonl(path: PathLike, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    with _open(path, "rt") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: PathLike, records: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_json(path: PathLike) -> Any:
    with _open(path, "rt") as f:
        return json.load(f)


def write_json(path: PathLike, obj: Any, indent: Optional[int] = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "wt") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


class Corpus:
    """An ordered collection of papers keyed by ``paper_id``."""

    def __init__(self, papers: Optional[Iterable[Paper]] = None, name: str = "corpus"):
        self.name = name
        self.papers: Dict[str, Paper] = {}
        for p in papers or []:
            self.add(p)

    # ------------------------------------------------------------ basics
    def add(self, paper: Paper) -> None:
        self.papers[paper.paper_id] = paper

    def __len__(self) -> int:
        return len(self.papers)

    def __iter__(self) -> Iterator[Paper]:
        return iter(self.papers.values())

    def __contains__(self, pid: str) -> bool:
        return pid in self.papers

    def __getitem__(self, pid: str) -> Paper:
        return self.papers[pid]

    def get(self, pid: str) -> Optional[Paper]:
        return self.papers.get(pid)

    @property
    def ids(self) -> List[str]:
        return list(self.papers)

    def texts(self, ids: Optional[Sequence[str]] = None) -> List[str]:
        ids = list(ids) if ids is not None else self.ids
        return [self.papers[i].text for i in ids]

    @property
    def num_chunks(self) -> int:
        return sum(len(p.chunks) for p in self.papers.values())

    # ---------------------------------------------------------- subsets
    def subset(self, ids: Iterable[str], name: Optional[str] = None) -> "Corpus":
        return Corpus((self.papers[i] for i in ids if i in self.papers), name=name or self.name)

    def filter(self, pred: Callable[[Paper], bool], name: Optional[str] = None) -> "Corpus":
        return Corpus((p for p in self.papers.values() if pred(p)), name=name or self.name)

    def sample(self, n: int, seed: int = 0, name: Optional[str] = None) -> "Corpus":
        ids = sorted(self.papers)
        rng = random.Random(seed)
        if n < len(ids):
            ids = rng.sample(ids, n)
        return self.subset(sorted(ids), name=name)

    def before(self, date: str, name: Optional[str] = None) -> "Corpus":
        """Papers whose first-version date is strictly before ``date`` (ISO)."""
        return self.filter(lambda p: bool(p.date) and p.date < date, name=name)

    # --------------------------------------------------------------- io
    def save(self, path: PathLike, include_chunks: bool = True) -> int:
        return write_jsonl(path, (p.to_dict(include_chunks=include_chunks) for p in self.papers.values()))

    @classmethod
    def load(
        cls,
        path: PathLike,
        limit: Optional[int] = None,
        chunks_path: Optional[PathLike] = None,
        name: Optional[str] = None,
    ) -> "Corpus":
        corpus = cls((Paper.from_dict(d) for d in read_jsonl(path, limit=limit)),
                     name=name or Path(path).name.split(".")[0])
        if chunks_path is not None and Path(chunks_path).exists():
            corpus.attach_chunks(chunks_path)
        return corpus

    def attach_chunks(self, chunks_path: PathLike) -> int:
        """Attach chunks from a ``chunks.jsonl(.gz)`` file; returns #attached."""
        n = 0
        for d in read_jsonl(chunks_path):
            c = Chunk.from_dict(d)
            p = self.papers.get(c.paper_id)
            if p is not None:
                p.chunks.append(c)
                n += 1
        for p in self.papers.values():
            p.chunks.sort(key=lambda c: c.order)
        return n

    def save_chunks(self, path: PathLike) -> int:
        return write_jsonl(path, (c.to_dict() for p in self.papers.values() for c in p.chunks))


def load_queries(path: PathLike) -> List[Query]:
    return [Query.from_dict(d) for d in read_jsonl(path)]


def save_queries(path: PathLike, queries: Iterable[Query]) -> int:
    return write_jsonl(path, (q.to_dict() for q in queries))
