# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

"""Read templates and choose the spec file for each routine and datatype."""

import argparse
from pathlib import Path

from utils import read_json


def expand_template(template, datatype):
    """Expand datatype-specific fields in a routine template."""

    def expand(value):
        if isinstance(value, dict):
            return {
                key: expand(
                    item[datatype]
                    if isinstance(item, dict) and datatype in item
                    else item
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand({**template, "datatype": datatype})


def read_specs(path):
    """Read all routine/datatype entries, expanding any templates in the file."""
    try:
        entries = read_json(path)
        if not isinstance(entries, list) or not entries:
            raise ValueError("expected a nonempty list of specs")

        result = {}
        for entry in entries:
            datatypes = entry["datatype"]
            template = isinstance(datatypes, list)
            if not template:
                datatypes = [datatypes]
            if not datatypes:
                raise ValueError("expected at least one datatype")

            for datatype in datatypes:
                if not all(isinstance(s, str) and s for s in (entry["routine"], datatype)):
                    raise ValueError("expected routine and datatype names")
                key = f"{entry['routine']}_{datatype}"
                if key in result:
                    raise ValueError(f"duplicate spec {key}")
                result[key] = expand_template(entry, datatype) if template else entry
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{path}: {error}") from error


def list_specs(templates_dir, defaults_dir=None, target_dir=None):
    """Choose one file per spec, with later directories taking precedence."""
    if not Path(templates_dir).is_dir():
        raise ValueError(f"template directory does not exist: {templates_dir}")
    if target_dir is not None:
        target_dir = Path(target_dir)
        if target_dir.name not in read_json(Path(__file__).with_name("systems.json")):
            raise ValueError(f"unknown target: {target_dir.name}")

    selected = {}
    for directory in (templates_dir, defaults_dir, target_dir):
        if directory is None:
            continue
        directory = Path(directory)
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"not a spec directory: {directory}")

        layer = {}
        for path in sorted(directory.glob("*.json")):
            for key, entry in read_specs(path).items():
                if directory != Path(templates_dir) and path.stem != key:
                    raise ValueError(f"{path}: filename must match spec key {key}")
                if directory == target_dir and entry.get("target") != target_dir.name:
                    raise ValueError(f"{path}: target must match directory {target_dir.name}")
                if key in layer:
                    raise ValueError(f"duplicate spec {key}: {layer[key]} and {path}")
                layer[key] = path

        # A later directory replaces the whole spec from an earlier one.
        selected.update(layer)

    return dict(sorted(selected.items()))


def load_spec(path, key):
    entries = read_specs(path)
    if key not in entries:
        raise ValueError(f"{path}: expected one spec for {key}, found none")
    return entries[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list key=source_path entries")
    listing.add_argument("--templates-dir", required=True, type=Path)
    listing.add_argument("--defaults-dir", type=Path)
    listing.add_argument("--target-dir", type=Path)
    args = parser.parse_args()

    try:
        selected = list_specs(args.templates_dir, args.defaults_dir, args.target_dir)
    except ValueError as error:
        parser.error(str(error))

    # Forward slashes survive the Windows shell when Make reuses these paths.
    for key, path in selected.items():
        print(f"{key}={path.as_posix()}")


if __name__ == "__main__":
    main()
