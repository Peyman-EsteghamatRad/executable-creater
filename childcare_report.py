"""
Evaluation CSV Parser
=====================
Parses a Mantau-style evaluation export CSV into a structured JSON file.

CSV Layout (1-indexed):
  Line 1   : Subject name (e.g. "FO - JHOAN DAVID ARMIJOS MOSQUERA")
  Line 10  : Column headers (keys for each field)
  Line 12  : Selection-field labels (e.g. "1 - No logra", "3 - En proceso", …)
  Line 13+ : One data entry per row; entries end at the first blank line
  Lines 4-7: Metadata (ignored for JSON output)

For selection fields the row contains 0/1 flags; we find the "1" and map it to
the corresponding label from line 12.
"""

import csv
import json
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox
import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_raw_rows(filepath: str) -> list[list[str]]:
    """Read the CSV with semicolons as delimiter, keeping empty cells intact.
    Tries UTF-8 first, then falls back to Latin-1 / CP-1252 for accented chars."""
    for encoding in ("utf-8-sig", "latin-1", "cp1252"):
        rows: list[list[str]] = []
        try:
            with open(filepath, newline="", encoding=encoding) as fh:
                reader = csv.reader(fh, delimiter=";", quotechar='"')
                for row in reader:
                    rows.append(row)
            return rows
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the CSV file with any supported encoding.")


def strip_quotes(value: str) -> str:
    """Remove surrounding whitespace."""
    return value.strip()


def is_blank_row(row: list[str]) -> bool:
    return all(cell.strip() == "" for cell in row)

def create_png(data: dict):
    def parse_score(val):
        match = re.match(r"(\d+)\s*-", str(val))
        return int(match.group(1)) if match else None

    def get_category_groups(entry):
        """Group scored fields between Observation keys."""
        keys = list(entry.keys())
        obs_indices = [i for i, k in enumerate(keys) if "Observaciones" in k]
        groups = []
        prev = 0
        for obs_idx in obs_indices:
            group = [k for k in keys[prev:obs_idx] if parse_score(entry[k]) is not None]
            groups.append(group)
            prev = obs_idx + 1
        return groups

    for kid in data:
        entries = kid["entries"]
        kid_name = kid.get("name", "Kid")
        groups = get_category_groups(entries[0])
        dates = [e["Fecha de evaluación"] for e in entries]

        fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5))
        if len(groups) == 1:
            axes = [axes]
        fig.suptitle(kid_name, fontsize=13)

        colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"]

        for i, (ax, group) in enumerate(zip(axes, groups)):
            avgs = []
            for entry in entries:
                scores = [parse_score(entry[k]) for k in group if parse_score(entry[k]) is not None]
                avgs.append(np.mean(scores) if scores else 0)

            ax.bar(dates, avgs, color=colors[:len(dates)])
            ax.set_title(f"Categoría {i+1}", fontsize=10)
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Promedio")
            ax.set_ylim(0, 10)
            ax.set_yticks([1, 3, 5, 7, 9])
            ax.set_yticklabels(["No logra", "En proceso", "Alcanza", "Logra", "Supera"], fontsize=8)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            for bar, avg in zip(ax.patches, avgs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{avg:.1f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        out = f"nna_stats_{kid_name.replace(' ', '_')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

# ──────────────────────────────────────────────────────────────────────────────
# Core parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_evaluation_csv(filepath: str) -> dict:
    rows = read_raw_rows(filepath)

    # ── Subject name (line 1, 0-indexed: row 0) ──────────────────────────────
    subject_name = strip_quotes(rows[0][0]) if rows else "Unknown"

    # ── Column headers – line 10 (0-indexed: row 9) ──────────────────────────
    headers = [strip_quotes(h) for h in rows[9]] if len(rows) > 9 else []

    # ── Selection labels – line 12 (0-indexed: row 11) ───────────────────────
    labels = [strip_quotes(l) for l in rows[11]] if len(rows) > 11 else []

    # ── Data entries – start at line 13 (0-indexed: row 12) ──────────────────
    # They end at the first blank row.
    entry_rows = []
    for row in rows[12:]:
        if is_blank_row(row):
            break
        entry_rows.append(row)

    entries = []
    for row in entry_rows:
        entry = {}

        # Evaluator name: columns 3 (Firstname) and 4 (Lastname)
        firstname = strip_quotes(row[3]) if len(row) > 3 else ""
        lastname  = strip_quotes(row[4]) if len(row) > 4 else ""
        entry["evaluator"] = f"{firstname} {lastname}".strip()

        # Walk every column that has a non-empty header
        col = 0
        cat = 1
        while col < len(headers):
            key = headers[col]
            if not key:
                col += 1
                continue

            cell_value = strip_quotes(row[col]) if col < len(row) else ""

            # ── Detect selection-field block ──────────────────────────────────
            # A selection block starts when the *next* column has an empty
            # header AND a non-empty label (the label row tells us the options).
            # We collect consecutive option columns until we hit a non-empty
            # header again (or run out of columns).
            next_col = col + 1
            option_cols = []
            while next_col < len(headers) and headers[next_col] == "":
                if next_col < len(labels) and labels[next_col] != "":
                    option_cols.append(next_col)
                next_col += 1

            if option_cols:
                # This is a selection field.
                # The current column (col) may itself be an option if its label
                # is set; collect all option positions including col.
                all_option_cols = []

                # Check whether col itself is one of the options
                if col < len(labels) and labels[col] != "":
                    all_option_cols.append(col)
                all_option_cols.extend(option_cols)

                # Find which option has the value "1"
                selected_label = None
                for oc in all_option_cols:
                    oc_value = strip_quotes(row[oc]) if oc < len(row) else ""
                    oc_label = labels[oc] if oc < len(labels) else ""
                    if oc_value == "1" and oc_label:
                        selected_label = oc_label
                        break

                # If none of the "option" cols had the label, the current cell
                # might be a plain text value (e.g. "Fecha de evaluación").
                if selected_label is None and cell_value not in ("0", "1", ""):
                    entry[key] = cell_value
                else:
                    entry[key] = selected_label  # may be None if unanswered

                col = next_col  # jump past the whole option block
            else:
                # Plain text / numeric field
                if key == "Observaciones":
                    key = str(cat) + "-" + key  # Prefix category number to distinguish blocks
                    cat += 1  # Increment category for the next block of fields
                entry[key] = cell_value
                col += 1

        entries.append(entry)

    return {
        "name": subject_name,
        "entries": entries,
    }


# ──────────────────────────────────────────────────────────────────────────────
# GUI + main
# ──────────────────────────────────────────────────────────────────────────────

def select_file_and_parse():
    root = tk.Tk()
    root.withdraw()  # Hide the empty Tk window

    filepaths = filedialog.askopenfilenames(
        title="Select Evaluation CSV(s)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if not filepaths:
        messagebox.showinfo("Cancelled", "No file selected. Exiting.")
        return

    
    results = []
    errors = []
    for filepath in filepaths:
        try:
            results.append(parse_evaluation_csv(filepath))
        except Exception as exc:
            errors.append(f"{os.path.basename(filepath)}: {exc}")

    if errors:
        messagebox.showwarning("Some files failed", "\n".join(errors))

    if not results:
        return

    # Save JSON next to the first selected CSV
    base = os.path.splitext(filepaths[0])[0]
    output_path = base + "_parsed.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    # Also print to console for quick inspection
    print(json.dumps(results, ensure_ascii=False, indent=2))

    names = "\n".join(r["name"] for r in results)
    messagebox.showinfo(
        "Done",
        f"Parsed {len(results)} file(s):\n{names}\n\n"
        f"JSON saved to:\n{output_path}",
    )

    create_png(results)


if __name__ == "__main__":
    select_file_and_parse()