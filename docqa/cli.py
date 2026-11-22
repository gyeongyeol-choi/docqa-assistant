"""Command-line interface for DocQA Assistant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docqa.config import Settings
from docqa.service import DocQAService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docqa",
        description="Personal & technical document Q&A assistant.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index documents in a directory")
    p_index.add_argument("docs_dir", nargs="?", default=None, help="Directory to index")

    p_ask = sub.add_parser("ask", help="Ask a question against indexed documents")
    p_ask.add_argument("question", help="Question to ask")
    p_ask.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    p_ask.add_argument("--json", action="store_true", help="Output raw JSON")

    sub.add_parser("docs", help="List currently indexed documents")

    p_serve = sub.add_parser("serve", help="Start the FastAPI web server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    service = DocQAService(settings=settings)

    if args.command == "index":
        target = Path(args.docs_dir) if args.docs_dir else settings.docs_dir
        summary = service.index_directory(target)
        print(
            f"Indexed {summary['chunks']} chunks from {summary['documents']} documents "
            f"({summary['docs_dir']}) -> {summary['index_path']}"
        )
        return 0

    if args.command == "docs":
        sources = service.list_sources()
        if not sources:
            print("No indexed documents. Run `docqa index <dir>` first.")
            return 0
        for item in sources:
            print(f"- {item['source']} ({item['chunks']} chunks)")
        return 0

    if args.command == "ask":
        result = service.ask(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0
        print(f"\n[{result.provider}] Answer:\n{result.answer}\n")
        if result.citations:
            print("Sources:")
            for c in result.citations:
                print(f"  [{c.index}] {c.source} > {c.section} (score={c.score})")
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("docqa.api:app", host=args.host, port=args.port, reload=False)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
