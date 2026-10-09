#!/usr/bin/env python3
"""Run this repo on Google Colab GPUs through the `colab` CLI (run from WSL).

The `colab` CLI only exists on Linux/macOS, so this driver runs in WSL
against the Windows checkout (/mnt/c/...). Stdlib only -- no project
dependencies are needed locally.

    python3 colab/colab_runtime.py up --gpu A100 --drive  # new VM + mount Drive + sync code + check deps
    python3 colab/colab_runtime.py run -- -m pytest -q     # foreground, output streamed
    python3 colab/colab_runtime.py launch stage1 -- scripts/train.py --config src/configs/stage1_frozen.yaml
    python3 colab/colab_runtime.py watch                   # poll jobs; on VM loss, prints recovery steps
    python3 colab/colab_runtime.py jobs | logs stage1 --follow | kill stage1
    python3 colab/colab_runtime.py resume                  # after a disconnect, on a fresh `up`
    python3 colab/colab_runtime.py fetch                   # outputs -> ./outputs/
    python3 colab/colab_runtime.py down                    # release the VM (stops billing)

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
- Jobs get LPN_OUTPUT_ROOT in their environment: Google Drive for sessions
  started with `up --drive` (or after `mount`), else VM-local /content/outputs.
  `fetch` pulls an output directory back to ./outputs/.
- Resume after a disconnect: every `launch` is also recorded locally in
  .colab/state.json, which outlives the VM. `resume` relaunches each
  recorded job that hasn't finished, with the same command. The scripts
  continue from their own checkpoints on Drive, so only work since the last
  checkpoint is redone.
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
SYNC_PATHS = ["src", "scripts", "tests", "colab", "pyproject.toml"]
DEFAULT_SESSION = "lpn"

REMOTE_ROOT = "/content/lpn"
JOBS_DIR = "/content/jobs"
VM_OUTPUTS = "/content/outputs"
DRIVE_MOUNT = "/content/drive"
DRIVE_OUTPUTS = f"{DRIVE_MOUNT}/MyDrive/lpn_interp_outputs"

# What the VM must have; `setup` hands this file to colab/vm_setup.py on the VM.
VM_REQUIREMENTS = "colab/requirements.txt"

RC_TAG = "__RC__="
JSON_TAG = "__JSON__="


# --------------------------------------------------------------------------- colab CLI plumbing

def colab(session: str | None, *args: str, check: bool = True) -> int:
    cmd = ["colab", *args] + (["-s", session] if session else [])
    rc = subprocess.run(cmd, check=False).returncode
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
    env["PYTHONPATH"] = REMOTE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
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


# --------------------------------------------------------------------------- local state (survives the VM)
#
# .colab/state.json remembers each session's settings (gpu, drive) and every
# job launched on it (argv, drive, last known state). The VM forgets all of
# this when it's reclaimed; this file is what lets `resume` relaunch the
# unfinished jobs on a fresh VM. The jobs themselves resume from their own
# checkpoints on Drive (see src/training/train.py).

STATE_FILE = REPO / ".colab" / "state.json"
INTERRUPTED_RC = "75"  # train.py: stopped by SIGTERM after checkpointing


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"sessions": {}, "jobs": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def session_settings(session: str) -> dict:
    return load_state()["sessions"].get(session, {})


def use_drive(args) -> bool:
    """--drive / --no-drive win; otherwise the session's setting from `up --drive` / `mount`."""
    if getattr(args, "drive", None) is not None:
        return args.drive
    return session_settings(args.session).get("drive", False)


def output_root(drive: bool) -> str:
    return DRIVE_OUTPUTS if drive else VM_OUTPUTS


def require_ok(rc: int) -> None:
    if rc != 0:
        sys.exit(rc)


def remote_jobs(session: str) -> list[dict] | None:
    """Job rows from the VM, or None if the session can't be reached."""
    rc, payloads = remote_python(session, snippet(JOBS, JOBS_DIR=JOBS_DIR), echo=False)
    return payloads[0] if rc == 0 and payloads else None


def record_remote_states(session: str, rows: list[dict]) -> None:
    state = load_state()
    jobs = state["jobs"].get(session, {})
    for row in rows:
        if row["name"] in jobs:
            jobs[row["name"]]["state"] = row["state"]
    save_state(state)


DRIVE_CHECK = """
import os
print("__JSON__=" + ("true" if os.path.isdir(os.path.join(DRIVE_MOUNT, "MyDrive")) else "false"))
print("__RC__=0")
"""


def drive_mounted(session: str) -> bool:
    rc, payloads = remote_python(session, snippet(DRIVE_CHECK, DRIVE_MOUNT=DRIVE_MOUNT), echo=False)
    return rc == 0 and bool(payloads and payloads[0])


def ensure_drive(session: str) -> None:
    """Mount Drive if it isn't. `colab drivemount` needs a human (browser consent),
    so this only works when run from an interactive terminal."""
    if drive_mounted(session):
        return
    if not sys.stdin.isatty():
        sys.exit(f"Drive isn't mounted on '{session}'. In your WSL terminal run:\n"
                 f"  python3 colab/colab_runtime.py -s {session} mount")
    print("Mounting Google Drive: approve the consent prompt below.")
    colab(session, "drivemount", DRIVE_MOUNT)
    if not drive_mounted(session):
        sys.exit("Drive still isn't mounted; try again with `mount`.")
    print(f"Drive mounted; outputs go to {DRIVE_OUTPUTS}")


# --------------------------------------------------------------------------- commands

def cmd_up(args) -> None:
    new = ["new", "--gpu", args.gpu] if args.gpu != "cpu" else ["new"]
    colab(args.session, *new, *(["--high-mem"] if args.high_mem else []))
    state = load_state()
    state["sessions"][args.session] = {"gpu": args.gpu, "high_mem": args.high_mem, "drive": args.drive}
    save_state(state)
    if args.drive:
        ensure_drive(args.session)
    cmd_sync(args)
    cmd_setup(args)
    pending = [n for n, j in state["jobs"].get(args.session, {}).items() if j.get("state") != "done"]
    if pending:
        print(f"\nUnfinished jobs from before: {', '.join(pending)}. Run `resume` to relaunch them.")
    print(f"\nSession '{args.session}' ready. Remember `down` when finished (idle VMs burn compute units).")


def cmd_mount(args) -> None:
    ensure_drive(args.session)
    state = load_state()
    state["sessions"].setdefault(args.session, {})["drive"] = True
    save_state(state)


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
    """Check/install the VM's packages from colab/requirements.txt (needs the code synced first)."""
    code = snippet(FOREGROUND, ARGV=["colab/vm_setup.py", VM_REQUIREMENTS], REMOTE_ROOT=REMOTE_ROOT,
                   OUTPUT_ROOT=VM_OUTPUTS, DRIVE_MOUNT=DRIVE_MOUNT, SESSION=args.session)
    require_ok(remote_python(args.session, code, timeout=600)[0])


def cmd_run(args) -> None:
    if not args.argv:
        sys.exit("nothing to run: pass a script and its args after --")
    if not args.no_sync:
        cmd_sync(args)
    code = snippet(FOREGROUND, ARGV=args.argv, REMOTE_ROOT=REMOTE_ROOT, OUTPUT_ROOT=output_root(use_drive(args)),
                   DRIVE_MOUNT=DRIVE_MOUNT, SESSION=args.session)
    require_ok(remote_python(args.session, code, timeout=args.timeout)[0])


def launch_job(session: str, name: str, argv: list[str], drive: bool) -> None:
    code = snippet(LAUNCH, NAME=name, ARGV=argv, REMOTE_ROOT=REMOTE_ROOT, JOBS_DIR=JOBS_DIR,
                   OUTPUT_ROOT=output_root(drive), DRIVE_MOUNT=DRIVE_MOUNT, SESSION=session)
    require_ok(remote_python(session, code)[0])
    state = load_state()
    state["jobs"].setdefault(session, {})[name] = {
        "argv": argv, "drive": drive, "state": "running",
        "launched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_state(state)


def cmd_launch(args) -> None:
    if not args.argv:
        sys.exit("nothing to launch: pass a script and its args after --")
    drive = use_drive(args)
    if not drive:
        print("note: outputs go to the VM's own disk, so this job can't resume after the VM is lost. "
              "Use `up --drive` / `mount` (or --drive) for long runs.")
    if not args.no_sync:
        cmd_sync(args)
    launch_job(args.session, args.name, args.argv, drive)
    print(f"follow with: python3 colab/colab_runtime.py logs {args.name} --follow")


def cmd_jobs(args) -> None:
    rows = remote_jobs(args.session)
    if rows is None:
        sys.exit(f"couldn't reach session '{args.session}'")
    record_remote_states(args.session, rows)
    remote_names = {r["name"] for r in rows}
    for r in rows:
        print(f"{r['name']:<16} {r['state']:<12} {r['started']}  {r['argv']}\n{'':<16} last: {r['last']}")
    for name, job in load_state()["jobs"].get(args.session, {}).items():
        if name not in remote_names:  # launched on an earlier VM
            print(f"{name:<16} {'not on VM':<12} last known: {job['state']}  (`resume` relaunches unfinished jobs)")
    if not rows and not load_state()["jobs"].get(args.session):
        print("no jobs")


def resume_decision(job: dict, remote_state: str | None, named: bool, include_failed: bool) -> str | None:
    """Why a job should be skipped, or None to relaunch it. Pure, so it's unit-testable."""
    state = remote_state or job.get("state", "")
    if remote_state == "running":
        return "still running"
    if state == "done":
        return "already done"
    if state in ("killed", "exited(killed)") and not named:
        return "was killed on purpose (name it to resume)"
    if state.startswith("exited(") and state != f"exited({INTERRUPTED_RC})" and not (named or include_failed):
        return f"failed ({state}); check `logs`, then name it or pass --include-failed"
    return None  # no record on this VM (new VM), died, or interrupted after checkpointing


def cmd_resume(args) -> None:
    jobs = load_state()["jobs"].get(args.session, {})
    if args.names:
        unknown = set(args.names) - set(jobs)
        if unknown:
            sys.exit(f"no record of job(s): {', '.join(sorted(unknown))}")
    rows = remote_jobs(args.session)
    if rows is None:
        sys.exit(f"couldn't reach session '{args.session}'; start one with `up` first")
    record_remote_states(args.session, rows)
    remote = {r["name"]: r["state"] for r in rows}
    jobs = load_state()["jobs"].get(args.session, {})

    todo = []
    for name, job in jobs.items():
        if args.names and name not in args.names:
            continue
        reason = resume_decision(job, remote.get(name), bool(args.names), args.include_failed)
        if reason:
            print(f"skip {name}: {reason}")
        else:
            todo.append(name)
    if not todo:
        print("nothing to resume")
        return
    if any(jobs[n]["drive"] for n in todo):
        ensure_drive(args.session)
    cmd_sync(args)
    for name in todo:
        print(f"relaunching {name}: {' '.join(jobs[name]['argv'])}")
        launch_job(args.session, name, jobs[name]["argv"], jobs[name]["drive"])


def cmd_forget(args) -> None:
    state = load_state()
    jobs = state["jobs"].get(args.session, {})
    for name in args.names:
        print(f"forgot {name}" if jobs.pop(name, None) else f"no record of {name}")
    save_state(state)


def cmd_watch(args) -> None:
    """Poll every job until none is running. Each poll is kernel activity, which
    also helps keep Colab from treating the session as idle."""
    last: dict[str, str] = {}
    while True:
        rows = remote_jobs(args.session)
        stamp = datetime.now().strftime("%H:%M:%S")
        if rows is None:
            settings = session_settings(args.session)
            up = f"up --gpu {settings.get('gpu', 'A100')}" + (" --drive" if settings.get("drive") else "")
            sys.exit(f"[{stamp}] lost contact with session '{args.session}' (VM reclaimed?). "
                     "To pick up where it left off:\n"
                     f"  python3 colab/colab_runtime.py -s {args.session} {up}\n"
                     f"  python3 colab/colab_runtime.py -s {args.session} resume")
        record_remote_states(args.session, rows)
        watched = [r for r in rows if not args.names or r["name"] in args.names]
        for r in watched:
            if last.get(r["name"]) != r["state"] or r["state"] == "running":
                print(f"[{stamp}] {r['name']:<16} {r['state']:<12} {r['last']}", flush=True)
            last[r["name"]] = r["state"]
        if not any(r["state"] == "running" for r in watched):
            print(f"[{stamp}] no running jobs")
            return
        time.sleep(args.interval)


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
    state = load_state()
    if args.name in state["jobs"].get(args.session, {}):
        state["jobs"][args.session][args.name]["state"] = "killed"
        save_state(state)


def cmd_fetch(args) -> None:
    remote = "/content/lpn_fetch.tar.gz"
    rc, _ = remote_python(args.session, snippet(
        PACK, OUTPUT_ROOT=output_root(use_drive(args)), SUBDIR=args.subdir, TARBALL=remote,
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
    parser.add_argument("-s", "--session", default=DEFAULT_SESSION,
                        help=f"colab session name (default {DEFAULT_SESSION})")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_drive(p):
        p.add_argument("--drive", action=argparse.BooleanOptionalAction, default=None,
                       help=f"output root {DRIVE_OUTPUTS} (default: the session's setting from `up --drive`/`mount`)")
        return p

    p = sub.add_parser("up", help="create a session, (mount Drive,) sync code, check dependencies")
    p.add_argument("--gpu", default="T4", help="T4, L4, G4, A100, H100, or cpu (default T4)")
    p.add_argument("--high-mem", action="store_true")
    p.add_argument("--drive", action="store_true", help="mount Google Drive (interactive) and write outputs there")
    p.set_defaults(func=cmd_up)

    sub.add_parser("mount", help="mount Google Drive on the session (interactive)").set_defaults(func=cmd_mount)

    p = sub.add_parser("sync", help="upload the working tree to the VM")
    p.add_argument("--force", action="store_true", help="sync even while jobs are running")
    p.set_defaults(func=cmd_sync)

    sub.add_parser("setup", help="check/install the VM's packages (colab/requirements.txt)").set_defaults(func=cmd_setup)

    p = with_drive(sub.add_parser("run", help="sync, then run `python -u ARGV...` in the foreground"))
    p.add_argument("--timeout", type=float, default=3600.0,
                   help="seconds (default 3600); use `launch` for long jobs")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_run)

    p = with_drive(sub.add_parser("launch", help="sync, then start `python -u ARGV...` as a detached background job"))
    p.add_argument("name")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_launch)

    sub.add_parser("jobs", help="list background jobs (this VM's and earlier ones')").set_defaults(func=cmd_jobs)

    p = sub.add_parser("resume", help="relaunch unfinished jobs (e.g. on a fresh VM after a disconnect)")
    p.add_argument("names", nargs="*", help="only these jobs (also allows resuming killed/failed ones)")
    p.add_argument("--include-failed", action="store_true", help="also relaunch jobs that exited with an error")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("forget", help="drop jobs from the local registry")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("watch", help="poll jobs until none is running; says how to recover if the VM is lost")
    p.add_argument("names", nargs="*")
    p.add_argument("--interval", type=float, default=300.0, help="seconds between polls (default 300)")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("logs", help="show a job's log")
    p.add_argument("name")
    p.add_argument("-n", "--lines", type=int, default=40)
    p.add_argument("-f", "--follow", action="store_true", help="keep polling until the job ends")
    p.add_argument("--interval", type=float, default=20.0)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("kill", help="SIGTERM a running job (training checkpoints before exiting)")
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
