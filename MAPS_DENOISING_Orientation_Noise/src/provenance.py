#!/usr/bin/env python3
"""
Experiment configs and run provenance.

Every experiment is one YAML file under configs/ (see configs/README.md). A
script that takes --config reads its own section of that file -- `noise:` for
add_ebsd_noise.py, `train:` for train.py -- and uses it as its defaults, so
the precedence is

    command line  >  config file  >  the script's built-in default

Every results file then records which code produced it: the git commit, plus
"-dirty" when src/ or configs/ had uncommitted changes at the time, because a
bare hash says nothing about edits that were never committed.
"""

import argparse
import datetime
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parent


def git_info():
    """Commit of the running code, and the uncommitted files in src/ and configs/.

    The dirty check is limited to those two folders on purpose: the nightly
    sync log is tracked and always modified, so a repo-wide check would mark
    every run dirty and the flag would mean nothing.
    """
    def git(*cmd):
        return subprocess.run(["git", "-C", str(SRC), *cmd], capture_output=True,
                              text=True, check=True).stdout
    try:
        commit = git("rev-parse", "HEAD").strip()
        changed = [line[3:] for line in
                   git("status", "--porcelain", "--", ".", "../configs").splitlines()]
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty_files": []}
    return {"commit": commit + ("-dirty" if changed else ""), "dirty_files": changed}


def provenance(config=None, args=None):
    """Everything needed to tell later which code, settings and machine made a result."""
    return {
        "git": git_info(),
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": config,
        "args": dict(vars(args)) if args is not None else None,
    }


def append_provenance(path, record):
    """Add one record to the JSON list at `path`, creating it if needed.

    A list rather than a single record because one output folder can be
    written by several runs -- a resumed training job, or a noise set built in
    --start/--limit chunks -- and each of them should stay on file.
    """
    path = Path(path)
    records = json.loads(path.read_text()) if path.exists() else []
    records.append(record)
    path.write_text(json.dumps(records, indent=2, default=str))


def parse_with_config(parser, section, argv=None):
    """Parse the command line, taking defaults from `section` of a --config file.

    Returns (args, config) where config is {"path", "content"} or None. Keys in
    the section are the script's long option names with '-' written as '_'
    (`scatter_deg`, `no_misfit`). A key the script does not know is an error,
    so a typo cannot silently fall back to the built-in default.
    """
    parser.add_argument("--config", default=None,
                        help=f"YAML experiment file; its `{section}:` section sets "
                             "the defaults, and flags given here override it")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    path = pre.parse_known_args(argv)[0].config
    if path is None:
        return parser.parse_args(argv), None

    path = Path(path).resolve()
    content = yaml.safe_load(path.read_text()) or {}
    values = content.get(section)
    if not isinstance(values, dict):
        parser.error(f"{path} has no `{section}:` section")

    actions = {a.dest: a for a in parser._actions}
    unknown = sorted(set(values) - set(actions))
    if unknown:
        parser.error(f"{path}: unknown {section} setting(s) {unknown}")
    for key, value in values.items():
        action = actions[key]
        # argparse checks `choices` only for values typed on the command line,
        # never for defaults, so check them here instead.
        if action.choices is not None and value not in action.choices:
            parser.error(f"{path}: {key}={value!r} is not one of {list(action.choices)}")
        # A required option set by the config no longer has to be typed.
        action.required = False
    parser.set_defaults(**values)
    return parser.parse_args(argv), {"path": str(path), "content": content}
