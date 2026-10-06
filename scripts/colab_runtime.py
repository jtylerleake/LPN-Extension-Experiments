#!/usr/bin/env python3
"""Run this repo on Google Colab GPUs through the `colab` CLI (run from WSL).

The `colab` CLI only exists on Linux/macOS, so this driver runs in WSL
against the Windows checkout (/mnt/c/...). Stdlib only -- no project
dependencies are needed locally.

    python3 scripts/colab_runtime.py up --gpu A100          # new VM + sync code + check deps
    python3 scripts/colab_runtime.py run -- -m pytest -q     # foreground, output streamed
    python3 scripts/colab_runtime.py run -- scripts/gpu_smoke.py
    python3 scripts/colab_runtime.py launch arch1 -- scripts/train.py --config ...   # background job
    python3 scripts/colab_runtime.py jobs | logs arch1 --follow | kill arch1
    python3 scripts/colab_runtime.py fetch                   # VM outputs -> ./outputs/
    python3 scripts/colab_runtime.py down                    # release the VM (stops billing)

How it works:
- `sync` tars the working tree (tracked + untracked-but-not-ignored files under
  SYNC_PATHS, so uncommitted edits go too), uploads it, and replaces
  REMOTE_ROOT with it. A BUNDLE.json manifest (git commit, dirty flag, bundle
  hash) is written alongside so every run can log which code it ran.
- Everything executes as a *subprocess* of the session's kernel, never inside
  it: the kernel persists across `colab exec` calls, so in-kernel imports
  would keep serving stale modules after a re-sync.
- `colab exec` times out after 30 s by default and dies with the connection,
  so anything long goes through `launch`: a detached job (own process group)
  writing JOBS_DIR/<name>.log, with its exit code in <name>.exit when done.
- Jobs get LPN_OUTPUT_ROOT in their environment (VM-local /content/outputs,
  or Google Drive with --drive after you've run `colab drivemount` yourself;
  it's interactive). `fetch` pulls an output directory back to ./outputs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SYNC_PATHS = ["src", "scripts", "tests", "pyproject.toml", "docker/requirements/colab.txt"]
DEFAULT_SESSION = "lpn"

REMOTE_ROOT = "/content/lpn"
JOBS_DIR = "/content/jobs"
VM_OUTPUTS = "/content/outputs"
DRIVE_MOUNT = "/content/drive"
DRIVE_OUTPUTS = f"{DRIVE_MOUNT}/MyDrive/lpn_interp_outputs"

# import name -> pip name; Colab preinstalls all of these today, `setup` only
# installs whatever is missing (never upgrades/downgrades Colab's own builds).
REQUIRED_MODULES = {
    "torch": "torch", "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
    "seaborn": "seaborn", "pydantic": "pydantic", "yaml": "pyyaml", "tqdm": "tqdm", "pytest": "pytest",
}

RC_TAG = "__RC__="
JSON_TAG = "__JSON__="


# --------------------------------------------------------------------------- colab CLI plumbing

def colab(session: str | None, *args: str, check: bool = True) -> int:
    cmd = ["colab", *args] + (["-s", session] if session else [])
    rc = subprocess.run(cmd).returncode
    if check and rc != 0:
        sys.exit(f"`{' '.join(cmd)}` failed with exit code {rc}")
    return rc


def remote_python(session: str, code: str, timeout: float = 120.0, echo: bool = True) -> tuple[int, list]:
    """Run `code` in the session's kernel; stream its output.

    Snippets end by printing RC_TAG<int> (and optionally JSON_TAG<json>
    lines); we parse those instead of relying on `colab exec`'s own exit
    status. Returns (rc, json_payloads).
    """
    cmd = ["colab", "exec", "-s", session, "--timeout", str(timeout)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    proc.stdin.write(code)
    proc.stdin.close()
    rc, payloads = None, []
    for line in proc.stdout:
        stripped = line.strip()
        if stripped.startswith(RC_TAG):
            rc = int(stripped[len(RC_TAG):])
        elif stripped.startswith(JSON_TAG):
            payloads.append(json.loads(stripped[len(JSON_TAG):]))
        elif echo:
            print(line, end="", flush=True)
    cli_rc = proc.wait()
    if rc is None:  # snippet never finished: exec timeout, dead session, or a Python error in the snippet
        rc = cli_rc or 1
    return rc, payloads


def snippet(body: str, **consts) -> str:
    """Python source: constants (repr-quoted, so arbitrary args are safe) + body.

    The body runs inside a function, so it can `return` early instead of
    raising SystemExit in the kernel and leaves no globals behind in the
    persistent kernel. Any uncaught error prints its traceback and RC 1.
    """
    source = "".join(f"{k} = {v!r}\n" for k, v in consts.items()) + body
    indented = "".join(f"    {line}\n" if line.strip() else "\n" for line in source.splitlines())
    return (
        f"def _lpn_snippet():\n{indented}"
        "try:\n    _lpn_snippet()\nexcept BaseException:\n"
        f"    import traceback; traceback.print_exc(); print('{RC_TAG}1')\n"
        "finally:\n    del _lpn_snippet\n"
    )


# --------------------------------------------------------------------------- remote snippets

_ENV = """
import os, sys
def job_env(output_root):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REMOTE_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["LPN_OUTPUT_ROOT"] = output_root
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("MPLBACKEND", None)  # the kernel's inline backend doesn't exist outside it
    return env
"""

_CHECK_DRIVE = """
if OUTPUT_ROOT.startswith(DRIVE_MOUNT) and not os.path.isdir(os.path.join(DRIVE_MOUNT, "MyDrive")):
    print("Google Drive isn't mounted on this VM. Run `colab drivemount -s " + SESSION + "` in your "
          "WSL terminal first (it's interactive), or drop --drive.")
    print("__RC__=2"); return
os.makedirs(OUTPUT_ROOT, exist_ok=True)
"""

_JOB_STATE = """
import json, os, signal
def job_state(name):
    base = os.path.join(JOBS_DIR, name)
    if os.path.exists(base + ".exit"):
        code = open(base + ".exit").read().strip()
        return "exited(" + code + ")" if code != "0" else "done"
    try:
        os.kill(int(open(base + ".pid").read()), 0)
        return "running"
    except (OSError, ValueError):
        return "died"  # killed without writing an exit code (e.g. VM/kernel reset)
def running_jobs():
    if not os.path.isdir(JOBS_DIR):
        return []
    names = sorted(f[:-4] for f in os.listdir(JOBS_DIR) if f.endswith(".pid"))
    return [n for n in names if job_state(n) == "running"]
"""

EXTRACT = _JOB_STATE + """
import shutil, tarfile
busy = running_jobs()
if busy and not FORCE:
    print("Refusing to replace the code under running job(s): " + ", ".join(busy)
          + " (lazy imports would read the new files). Wait, `kill` them, or pass --force.")
    print("__RC__=3"); return
shutil.rmtree(REMOTE_ROOT, ignore_errors=True)
os.makedirs(REMOTE_ROOT)
with tarfile.open(TARBALL) as t:
    try:
        t.extractall(REMOTE_ROOT, filter="data")
    except TypeError:  # Python < 3.12
        t.extractall(REMOTE_ROOT)
os.remove(TARBALL)
with open(os.path.join(REMOTE_ROOT, "BUNDLE.json"), "w") as f:
    json.dump(MANIFEST, f)
print("synced", MANIFEST["files"], "files to", REMOTE_ROOT, "(commit", MANIFEST["commit"][:8] + ("+dirty" if MANIFEST["dirty"] else "") + ")")
print("__RC__=0")
"""

SETUP = """
import importlib, subprocess, sys
missing = []
for module, pip_name in REQUIRED.items():
    try:
        m = importlib.import_module(module)
        print(f"  {module:<11} {getattr(m, '__version__', '?')}")
    except ImportError:
        missing.append(pip_name)
if missing:
    print("installing missing:", " ".join(missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
import pydantic, torch
assert int(pydantic.VERSION.split(".")[0]) >= 2, "pydantic v2 required"
print("  cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("__RC__=0")
"""

FOREGROUND = _ENV + _CHECK_DRIVE + """
import subprocess
proc = subprocess.Popen([sys.executable, "-u", *ARGV], cwd=REMOTE_ROOT, env=job_env(OUTPUT_ROOT),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
print("__RC__=" + str(proc.wait()))
"""

LAUNCH = _ENV + _CHECK_DRIVE + _JOB_STATE + """
import shlex, subprocess, time
os.makedirs(JOBS_DIR, exist_ok=True)
base = os.path.join(JOBS_DIR, NAME)
if os.path.exists(base + ".pid") and job_state(NAME) == "running":
    print("job " + NAME + " is already running; `kill` it or pick another name")
    print("__RC__=4"); return
for ext in (".log", ".exit", ".pid", ".json"):
    if os.path.exists(base + ext):
        os.remove(base + ext)
command = shlex.join([sys.executable, "-u", *ARGV])
shell = f"cd {shlex.quote(REMOTE_ROOT)} && {command} > {shlex.quote(base + '.log')} 2>&1; echo $? > {shlex.quote(base + '.exit')}"
proc = subprocess.Popen(["bash", "-c", shell], env=job_env(OUTPUT_ROOT), start_new_session=True,
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
open(base + ".pid", "w").write(str(proc.pid))
json.dump({"argv": ARGV, "output_root": OUTPUT_ROOT, "started": time.strftime("%Y-%m-%d %H:%M:%S")},
          open(base + ".json", "w"))
print("launched " + NAME + " (pid " + str(proc.pid) + "): " + command)
print("__RC__=0")
"""

JOBS = _JOB_STATE + """
rows = []
if os.path.isdir(JOBS_DIR):
    for f in sorted(os.listdir(JOBS_DIR)):
        if f.endswith(".pid"):
            name = f[:-4]
            meta = json.load(open(os.path.join(JOBS_DIR, name + ".json"))) if os.path.exists(os.path.join(JOBS_DIR, name + ".json")) else {}
            log = os.path.join(JOBS_DIR, name + ".log")
            last = open(log, errors="replace").read().strip().splitlines()[-1:] if os.path.exists(log) else []
            rows.append({"name": name, "state": job_state(name), "started": meta.get("started", "?"),
                         "argv": " ".join(meta.get("argv", [])), "last": last[0][:120] if last else ""})
print("__JSON__=" + json.dumps(rows))
print("__RC__=0")
"""

TAIL = _JOB_STATE + """
from collections import deque
log = os.path.join(JOBS_DIR, NAME + ".log")
if not os.path.exists(os.path.join(JOBS_DIR, NAME + ".pid")):
    print("no such job: " + NAME); print("__RC__=5"); return
size = os.path.getsize(log) if os.path.exists(log) else 0
if OFFSET is None:
    lines = deque(open(log, errors="replace"), maxlen=LINES) if size else []
    text = "".join(lines)
else:
    with open(log, errors="replace") as f:
        f.seek(min(OFFSET, size))
        text = f.read()
print("__JSON__=" + json.dumps({"text": text, "offset": size, "state": job_state(NAME)}))
print("__RC__=0")
"""

KILL = _JOB_STATE + """
if job_state(NAME) != "running":
    print(NAME + " is not running (" + job_state(NAME) + ")"); print("__RC__=0"); return
os.killpg(int(open(os.path.join(JOBS_DIR, NAME + ".pid")).read()), signal.SIGTERM)
open(os.path.join(JOBS_DIR, NAME + ".exit"), "w").write("killed")
print("sent SIGTERM to " + NAME)
print("__RC__=0")
"""

PACK = """
import os, tarfile
src = os.path.join(OUTPUT_ROOT, SUBDIR) if SUBDIR else OUTPUT_ROOT
if not os.path.isdir(src):
    print("nothing to fetch: " + src + " doesn't exist"); print("__RC__=6"); return
skip = lambda ti: None if (NO_CHECKPOINTS and ti.name.endswith((".pt", ".tmp"))) else ti
with tarfile.open(TARBALL, "w:gz") as t:
    t.add(src, arcname=SUBDIR or ".", filter=skip)
print("packed " + src + " (" + str(os.path.getsize(TARBALL) // 1024) + " KiB)")
print("__RC__=0")
"""


# --------------------------------------------------------------------------- local helpers

def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def bundle_files() -> list[str]:
    listed = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *SYNC_PATHS)
    return sorted(p for p in set(listed.split("\0")) if p and (REPO / p).is_file())


def build_bundle(dest: Path) -> dict:
    files = bundle_files()
    digest = hashlib.sha256()
    with tarfile.open(dest, "w:gz") as tar:
        for rel in files:
            digest.update(rel.encode() + (REPO / rel).read_bytes())
            tar.add(REPO / rel, arcname=rel)
    commit = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--", *SYNC_PATHS).strip())
    return {"commit": commit, "dirty": dirty, "sha256": digest.hexdigest()[:16], "files": len(files),
            "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def output_root(args) -> str:
    return DRIVE_OUTPUTS if args.drive else VM_OUTPUTS


def require_ok(rc: int) -> None:
    if rc != 0:
        sys.exit(rc)


# --------------------------------------------------------------------------- commands

def cmd_up(args) -> None:
    new = ["new", "--gpu", args.gpu] if args.gpu != "cpu" else ["new"]
    colab(args.session, *new, *(["--high-mem"] if args.high_mem else []))
    cmd_sync(args)
    cmd_setup(args)
    print(f"\nSession '{args.session}' ready. Remember `down` when finished (idle VMs burn compute units).")


def cmd_sync(args) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "bundle.tar.gz"
        manifest = build_bundle(local)
        remote = "/content/lpn_bundle.tar.gz"
        colab(args.session, "upload", str(local), remote)
    rc, _ = remote_python(args.session, snippet(
        EXTRACT, REMOTE_ROOT=REMOTE_ROOT, JOBS_DIR=JOBS_DIR, TARBALL=remote, MANIFEST=manifest,
        FORCE=getattr(args, "force", False)))
    require_ok(rc)


def cmd_setup(args) -> None:
    require_ok(remote_python(args.session, snippet(SETUP, REQUIRED=REQUIRED_MODULES), timeout=600)[0])


def cmd_run(args) -> None:
    if not args.argv:
        sys.exit("nothing to run: pass a script and its args after --")
    if not args.no_sync:
        cmd_sync(args)
    code = snippet(FOREGROUND, ARGV=args.argv, REMOTE_ROOT=REMOTE_ROOT, OUTPUT_ROOT=output_root(args),
                   DRIVE_MOUNT=DRIVE_MOUNT, SESSION=args.session)
    require_ok(remote_python(args.session, code, timeout=args.timeout)[0])


def cmd_launch(args) -> None:
    if not args.argv:
        sys.exit("nothing to launch: pass a script and its args after --")
    if not args.no_sync:
        cmd_sync(args)
    code = snippet(LAUNCH, NAME=args.name, ARGV=args.argv, REMOTE_ROOT=REMOTE_ROOT, JOBS_DIR=JOBS_DIR,
                   OUTPUT_ROOT=output_root(args), DRIVE_MOUNT=DRIVE_MOUNT, SESSION=args.session)
    require_ok(remote_python(args.session, code)[0])
    print(f"follow with: python3 scripts/colab_runtime.py logs {args.name} --follow")


def cmd_jobs(args) -> None:
    rc, payloads = remote_python(args.session, snippet(JOBS, JOBS_DIR=JOBS_DIR), echo=False)
    require_ok(rc)
    rows = payloads[0] if payloads else []
    if not rows:
        print("no jobs")
    for r in rows:
        print(f"{r['name']:<16} {r['state']:<12} {r['started']}  {r['argv']}\n{'':<16} last: {r['last']}")


def cmd_logs(args) -> None:
    def tail(offset):
        rc, payloads = remote_python(args.session, snippet(
            TAIL, NAME=args.name, JOBS_DIR=JOBS_DIR, LINES=args.lines, OFFSET=offset), echo=True)
        require_ok(rc)
        return payloads[0]

    out = tail(None)
    print(out["text"], end="")
    while args.follow and out["state"] == "running":
        time.sleep(args.interval)
        out = tail(out["offset"])
        print(out["text"], end="", flush=True)
    print(f"[{args.name}: {out['state']}]")


def cmd_kill(args) -> None:
    require_ok(remote_python(args.session, snippet(KILL, NAME=args.name, JOBS_DIR=JOBS_DIR))[0])


def cmd_fetch(args) -> None:
    remote = "/content/lpn_fetch.tar.gz"
    rc, _ = remote_python(args.session, snippet(
        PACK, OUTPUT_ROOT=output_root(args), SUBDIR=args.subdir, TARBALL=remote,
        NO_CHECKPOINTS=args.no_checkpoints), timeout=600)
    require_ok(rc)
    dest = (REPO / args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "fetch.tar.gz"
        colab(args.session, "download", remote, str(local))
        with tarfile.open(local) as t:
            t.extractall(dest, filter="data")
    print(f"fetched into {dest}")


def cmd_down(args) -> None:
    colab(args.session, "stop")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--session", default=DEFAULT_SESSION, help=f"colab session name (default {DEFAULT_SESSION})")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_drive(p):
        p.add_argument("--drive", action="store_true",
                       help=f"use {DRIVE_OUTPUTS} as the output root (needs `colab drivemount` first)")
        return p

    p = sub.add_parser("up", help="create a session, sync code, check dependencies")
    p.add_argument("--gpu", default="T4", help="T4, L4, G4, A100, H100, or cpu (default T4)")
    p.add_argument("--high-mem", action="store_true")
    p.set_defaults(func=cmd_up)

    p = sub.add_parser("sync", help="upload the working tree to the VM")
    p.add_argument("--force", action="store_true", help="sync even while jobs are running")
    p.set_defaults(func=cmd_sync)

    sub.add_parser("setup", help="check/install Python dependencies on the VM").set_defaults(func=cmd_setup)

    p = with_drive(sub.add_parser("run", help="sync, then run `python -u ARGV...` in the foreground"))
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds (default 3600); use `launch` for long jobs")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_run)

    p = with_drive(sub.add_parser("launch", help="sync, then start `python -u ARGV...` as a detached background job"))
    p.add_argument("name")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_launch)

    sub.add_parser("jobs", help="list background jobs").set_defaults(func=cmd_jobs)

    p = sub.add_parser("logs", help="show a job's log")
    p.add_argument("name")
    p.add_argument("-n", "--lines", type=int, default=40)
    p.add_argument("-f", "--follow", action="store_true", help="keep polling until the job ends")
    p.add_argument("--interval", type=float, default=20.0)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("kill", help="SIGTERM a running job")
    p.add_argument("name")
    p.set_defaults(func=cmd_kill)

    p = with_drive(sub.add_parser("fetch", help="download an output directory into ./outputs"))
    p.add_argument("subdir", nargs="?", default="", help="subdirectory of the output root (default: all)")
    p.add_argument("--dest", default="outputs")
    p.add_argument("--no-checkpoints", action="store_true", help="skip *.pt files")
    p.set_defaults(func=cmd_fetch)

    sub.add_parser("down", help="stop the session and release the VM").set_defaults(func=cmd_down)
    sub.add_parser("status", help="colab status for the session").set_defaults(
        func=lambda a: colab(a.session, "status"))

    args = parser.parse_args(argv)
    if getattr(args, "argv", None) and args.argv[0] == "--":
        args.argv = args.argv[1:]
    args.func(args)


if __name__ == "__main__":
    main()
