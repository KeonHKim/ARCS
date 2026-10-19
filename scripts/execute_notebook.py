"""Execute and checkpoint the shipped benchmark notebook, including its plots.

Default: a normal Jupyter kernel. --in-process: sequential IPython execution
without kernel sockets, useful in restricted headless build environments.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import io
from contextlib import redirect_stdout
import os
import sys
import time
import traceback
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "examples/01_circular_benchmark.ipynb"
nb = None
started = None
TASK_LOG_STREAM = sys.stdout

def on_start(cell, cell_index):
    print(f"Cell {cell_index + 1}/{len(nb.cells)}: {cell.cell_type}", file=TASK_LOG_STREAM, flush=True)


def on_executed(cell, cell_index, execute_reply=None):
    nbformat.write(nb, PATH)
    print(f"Completed code cell {cell_index + 1} ({time.perf_counter() - started:.1f} s elapsed)", file=TASK_LOG_STREAM, flush=True)


def execute_in_process():
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(ROOT)
    shell = TerminalInteractiveShell.instance()
    # The terminal defaults to text/plain, while a notebook needs rich output.
    shell.display_formatter.active_types = list(shell.display_formatter.formatters)
    execution_count = 0
    for index, cell in enumerate(nb.cells):
        on_start(cell, index)
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        execution_count += 1
        cell.execution_count = execution_count
        cell.outputs = []
        cell.metadata["execution"] = {"iopub.execute_input": datetime.now(timezone.utc).isoformat()}
        buffer = io.StringIO()
        class Tee:
            def write(self, value):
                buffer.write(value)
                original.write(value)
                return len(value)
            def flush(self):
                original.flush()
        original = TASK_LOG_STREAM
        with capture_output(stdout=False) as captured, redirect_stdout(Tee()):
            result = shell.run_cell(cell.source, store_history=True)
        if buffer.getvalue():
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=buffer.getvalue()))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            print(captured.stderr, end="", file=TASK_LOG_STREAM, flush=True)
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
        cell.metadata["execution"]["iopub.status.idle"] = datetime.now(timezone.utc).isoformat()
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__, evalue=str(error),
                                                       traceback=traceback.format_exception(error)))
        on_executed(cell, index)
        if error is not None:
            raise RuntimeError(f"Notebook cell {index + 1} failed") from error


def main():
    global nb, started, PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-process", action="store_true", help="Execute in IPython without Jupyter kernel sockets")
    parser.add_argument("--notebook", type=Path, default=PATH,
                        help="Notebook to execute; relative paths are resolved under the source directory")
    args = parser.parse_args()
    PATH = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    nb = nbformat.read(PATH, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    started = time.perf_counter()
    try:
        if args.in_process:
            execute_in_process()
        else:
            client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                                    resources={"metadata": {"path": str(ROOT)}},
                                    on_cell_start=on_start, on_cell_executed=on_executed)
            client.execute()
        nb.metadata["numerical_output_provenance"] = {
            "executed_version": __import__('arcs').__version__,
            "release_version": __import__('arcs').__version__,
            "all_code_cells_freshly_executed": True,
            "execution_mode": "in-process IPython" if args.in_process else "Jupyter kernel",
        }
    finally:
        nbformat.write(nb, PATH)
    
    outputs = [o for cell in nb.cells if cell.cell_type == "code" for o in cell.get("outputs", [])]
    summary = {
        "executed_version": __import__('arcs').__version__,
        "executed_code_cells": sum(cell.cell_type == "code" for cell in nb.cells),
        "embedded_png_plots": sum("image/png" in o.get("data", {}) for o in outputs),
        "errors": sum(o.get("output_type") == "error" for o in outputs),
        "execution_mode": "in-process IPython" if args.in_process else "Jupyter kernel",
        "frequency_samples": 100,
        "frequency_band_Hz": [10.0, 1e13],
        "elapsed_seconds": time.perf_counter() - started,
    }
    capture_lines = ["Captured output from the completed notebook.", ""]
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            capture_lines.append(f"Cell {index + 1}, execution {cell.execution_count}")
            for output in cell.outputs:
                if output.output_type == "stream":
                    capture_lines.append(output.name + ":\n" + output.text)
                elif output.output_type == "error":
                    capture_lines.extend(output.traceback)
    stem = "notebook" if PATH.name == "01_circular_benchmark.ipynb" else PATH.stem
    (ROOT / f"results/{stem}_captured_output.log").write_text("\n".join(capture_lines) + "\n")
    (ROOT / f"results/{stem}_execution.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), file=TASK_LOG_STREAM, flush=True)


if __name__ == "__main__":
    main()
