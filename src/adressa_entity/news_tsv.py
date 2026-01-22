from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


NEWS_TSV_NUM_COLUMNS = 8


@dataclass(frozen=True)
class NewsTsvRow:
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities: str
    abstract_entities: str

    @classmethod
    def from_tsv_line(cls, line: str, *, line_num: int, path: str | Path) -> "NewsTsvRow":
        parts = line.rstrip("\n").split("\t")
        if len(parts) != NEWS_TSV_NUM_COLUMNS:
            raise ValueError(
                f"Invalid news.tsv row at {path}:{line_num}: expected {NEWS_TSV_NUM_COLUMNS} columns, got {len(parts)}"
            )
        return cls(*parts)  # type: ignore[misc]

    def to_tsv_line(self) -> str:
        return "\t".join(
            [
                self.news_id,
                self.category,
                self.subcategory,
                self.title,
                self.abstract,
                self.url,
                self.title_entities,
                self.abstract_entities,
            ]
        )


def iter_news_tsv(path: str | Path) -> Iterator[NewsTsvRow]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            yield NewsTsvRow.from_tsv_line(line, line_num=line_num, path=path)


def write_news_tsv(path: str | Path, rows: Iterable[NewsTsvRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.to_tsv_line())
            f.write("\n")


def read_news_ids_and_titles(path: str | Path, *, limit: int | None = None) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for idx, row in enumerate(iter_news_tsv(path)):
        if limit is not None and idx >= limit:
            break
        items.append((row.news_id, row.title))
    return items

