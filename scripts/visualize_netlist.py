#!/usr/bin/env python3
"""
Generate a PNG netlist diagram from one or more Verilog files using yosys show.

Usage:
    python scripts/visualize_netlist.py design.v [extra.v ...] [-o output.png] [-m top_module]

Requires: yosys + graphviz (dot) installed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.evaluator import VerilogEvaluator


def main():
    parser = argparse.ArgumentParser(description="Visualize Verilog netlist as PNG")
    parser.add_argument("files", nargs="+", help="Verilog source files")
    parser.add_argument("-o", "--output", default="netlist.png", help="Output PNG path (default: netlist.png)")
    parser.add_argument("-m", "--module", default=None, help="Top module name (optional)")
    args = parser.parse_args()

    sources = {}
    for fpath in args.files:
        p = Path(fpath)
        if not p.exists():
            print(f"ERROR: file not found: {fpath}", file=sys.stderr)
            sys.exit(1)
        sources[p.name] = p.read_text()

    evaluator = VerilogEvaluator()
    result = evaluator.visualize(sources, top_module=args.module)

    if not result.success:
        print(f"Visualization failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    import base64
    png_bytes = base64.b64decode(result.image_b64)
    out = Path(args.output)
    out.write_bytes(png_bytes)
    print(f"Netlist diagram saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
