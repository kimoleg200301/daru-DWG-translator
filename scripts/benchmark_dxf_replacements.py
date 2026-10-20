"""Compare DXF exact-label lookup with the former sequential replacement loop.

This measures local text replacement only, excluding APIs, fonts, and file I/O.
Run: python scripts/benchmark_dxf_replacements.py --items 3000 --repeat 3
"""

import argparse
import sys
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daru.dxf.entities import TextReplacer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=3000)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.items <= 0 or args.repeat <= 0:
        parser.error("--items and --repeat must be positive")
    pairs = [
        (f"Source item {index:05d}", f"Translated item {index:05d}")
        for index in range(args.items)
    ]

    def sequential() -> list[str]:
        result = []
        for text, _ in pairs:
            for source, translated in pairs:
                text = text.replace(source, translated)
            result.append(text)
        return result

    def indexed() -> list[str]:
        replacer = TextReplacer(pairs)
        return [replacer.translate(text) for text, _ in pairs]

    assert sequential() == indexed()
    old = min(timeit.repeat(sequential, number=1, repeat=args.repeat))
    new = min(timeit.repeat(indexed, number=1, repeat=args.repeat))
    print(f"{args.items} labels / {args.items} pairs")
    print(f"Sequential: {old:.6f}s; indexed: {new:.6f}s; ratio: {old / new:.1f}x")


if __name__ == "__main__":
    main()
