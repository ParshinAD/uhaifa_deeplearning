"""Execute a .ipynb in-process and write outputs back into it.

WHY NOT nbconvert/nbclient: the ``allen`` campaign env has no jupyter kernel
(ipykernel and nbclient are not installed) and mutating that env just to render
a notebook is not worth the reproducibility risk. This emulates the kernel
contract closely enough for this project's notebooks: code cells run
top-to-bottom in ONE shared namespace; stdout, the last bare expression's
repr (text + HTML when available, e.g. DataFrames), and any matplotlib figures
left open at the end of a cell are captured into the cell's outputs.

Usage:  $PY layering/tools/exec_notebook.py <notebook.ipynb>
Exits non-zero (and stores the traceback in the failing cell) on any error.
"""
from __future__ import annotations

import ast
import base64
import io
import sys
import traceback

import matplotlib
matplotlib.use("Agg")                      # before any pyplot import in cells
import matplotlib.pyplot as plt
import nbformat


def run(path: str) -> int:
    nb = nbformat.read(path, as_version=4)
    # clear ALL previous outputs up front, so a mid-run failure cannot leave
    # stale outputs from an earlier execution after the failing cell
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    ns: dict = {"__name__": "__main__"}
    ec = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        ec += 1
        cell.execution_count = ec
        tree = ast.parse(cell.source or "")
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop(-1).value)
        buf = io.StringIO()
        old_stdout, sys.stdout = sys.stdout, buf
        try:
            exec(compile(tree, f"<cell {ec}>", "exec"), ns)
            result = (eval(compile(last_expr, f"<cell {ec}>", "eval"), ns)
                      if last_expr is not None else None)
        except Exception as exc:
            sys.stdout = old_stdout
            err = traceback.format_exc()
            partial = buf.getvalue()          # keep what the cell printed before dying
            if partial:
                cell.outputs.append(nbformat.v4.new_output(
                    "stream", name="stdout", text=partial))
            cell.outputs.append(nbformat.v4.new_output(
                "error", ename=type(exc).__name__, evalue=str(exc),
                traceback=err.splitlines()))
            nbformat.write(nb, path)
            print(err, file=sys.stderr)
            print(f"FAILED at code cell {ec} of {path}", file=sys.stderr)
            return 1
        finally:
            sys.stdout = old_stdout
        text = buf.getvalue()
        if text:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
        if result is not None:
            data = {"text/plain": repr(result)}
            html = getattr(result, "_repr_html_", None)
            if callable(html):
                try:
                    data["text/html"] = html()
                except Exception:
                    pass
            cell.outputs.append(nbformat.v4.new_output(
                "execute_result", data=data, execution_count=ec))
        for num in plt.get_fignums():
            fig = plt.figure(num)
            png = io.BytesIO()
            fig.savefig(png, format="png", bbox_inches="tight")
            b64 = base64.b64encode(png.getvalue()).decode("ascii")
            cell.outputs.append(nbformat.v4.new_output(
                "display_data", data={"image/png": b64}))
        plt.close("all")
    nbformat.write(nb, path)
    print(f"executed {ec} code cells -> {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
