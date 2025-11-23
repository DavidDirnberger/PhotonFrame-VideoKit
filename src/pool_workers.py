#!/usr/bin/env python3
# pool_workers.py — persistente CLI-Worker (Directory-Shards) mit starkem Logging
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple, cast

import ai_backend as ab
import consoleOutput as co
import mem_guard as mg
from loghandler import print_log


class _ProcLike(Protocol):
    # returncode ist bei laufendem Prozess None, ansonsten int
    returncode: Optional[int]

    def poll(self) -> Optional[int]: ...
    def terminate(self) -> None: ...


@dataclass
class _Shard:
    idx: int
    in_dir: Path
    out_dir: Path
    items: List[Path]
    stems: set[str] = field(default_factory=set)
    attempts: int = 0
    proc: Optional[_ProcLike] = None
    log_path: Optional[Path] = None
    t_start: float = 0.0


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # hardlink
        return
    except Exception:
        pass
    try:
        dst.symlink_to(src)  # symlink
        return
    except Exception:
        pass
    shutil.copy2(src, dst)  # copy (fallback)


def _list_frames(d: Path) -> List[Path]:
    return sorted(d.glob("frame_*.png"))


def _split_round_robin(frames: List[Path], k: int) -> List[List[Path]]:
    k = max(1, min(k, len(frames)))
    buckets: List[List[Path]] = [[] for _ in range(k)]
    for i, p in enumerate(frames):
        buckets[i % k].append(p)
    return buckets


def _prepare_shards(
    in_dir: Path, out_dir: Path, workers: int
) -> Tuple[List[_Shard], Path]:
    frames = _list_frames(in_dir)
    tmp_root = out_dir / "__pool_tmp__"
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Grunddiagnostik
    try:
        can_r_in = os.access(in_dir, os.R_OK)
        can_w_out = os.access(out_dir, os.W_OK)
        print_log(
            f"[PT-pool] PREP in_dir={in_dir} r={can_r_in} n={len(frames)}  out_dir={out_dir} w={can_w_out}",
            "_pytorch",
        )
        smp = [p.name for p in frames[:5]]
        if smp:
            print_log(f"[PT-pool] PREP samples_in={smp}", "_pytorch")
    except Exception as e:
        print_log(f"[PT-pool] PREP diag error: {e!r}", "_pytorch")

    buckets = _split_round_robin(frames, workers)
    shards: List[_Shard] = []
    for i, items in enumerate(buckets, 1):
        s_in = tmp_root / f"shard_{i:02d}" / "in"
        s_out = tmp_root / f"shard_{i:02d}" / "out"
        s_in.mkdir(parents=True, exist_ok=True)
        s_out.mkdir(parents=True, exist_ok=True)

        linked = 0
        copied = 0
        for src in items:
            dst = s_in / src.name
            before = dst.exists()
            _link_or_copy(src, dst)
            # grobe Heuristik: Hardlink → gleiche inode (sofern unterstützt)
            try:
                if not before and src.exists() and dst.exists():
                    if (
                        hasattr(os, "stat")
                        and os.stat(src).st_ino == os.stat(dst).st_ino
                    ):
                        linked += 1
                    else:
                        copied += 1
            except Exception:
                copied += 1

        stems = {p.stem for p in items}
        shards.append(
            _Shard(idx=i, in_dir=s_in, out_dir=s_out, items=items, stems=stems)
        )
        print_log(
            f"[PT-pool] shard#{i} size={len(items)} in={s_in} out={s_out} linked={linked} copied={copied} "
            f"samples={[p.name for p in items[:3]]}",
            "_pytorch",
        )
    return shards, tmp_root


def _diagnose_shards_state(
    shards: List[_Shard],
    global_out: Path,
    esrgan_root: Path,
    tmp_root: Path,
    *,
    tag: str = "pool-shards",
) -> None:
    try:
        print_log(f"[PT-diag] SHARDS tag={tag} tmp_root={tmp_root}", "_pytorch")
        # global basics
        try:
            can_w = os.access(global_out, os.W_OK)
            n_out = len(list(global_out.glob("*_out.png")))
            n_seq = len(list(global_out.glob("frame_*.png")))
            print_log(
                f"[PT-diag] OUTDIR={global_out} write={can_w} *_out={n_out} frame_*= {n_seq}",
                "_pytorch",
            )
        except Exception as e:
            print_log(f"[PT-diag] OUTDIR diag err: {e!r}", "_pytorch")

        # results/ short count (manche Runner schreiben dort hin)
        res = esrgan_root / "results"
        if res.exists():
            try:
                n_res = len(list(res.rglob("*_out.png")))
                print_log(
                    f"[PT-diag] results/ present files_out={n_res} path={res}",
                    "_pytorch",
                )
            except Exception as e:
                print_log(f"[PT-diag] results diag err: {e!r}", "_pytorch")

        # Per-shard inventory
        for sh in shards:
            try:
                nin = len(list(sh.in_dir.glob("frame_*.png")))
                nout = len(list(sh.out_dir.glob("*.png")))
                smp_in = [p.name for p in list(sh.in_dir.glob("frame_*.png"))[:3]]
                smp_out = [p.name for p in list(sh.out_dir.glob("*.png"))[:3]]
                print_log(
                    f"[PT-diag] shard#{sh.idx} nin={nin} nout={nout} "
                    f"in_dir={sh.in_dir} out_dir={sh.out_dir} smp_in={smp_in} smp_out={smp_out}",
                    "_pytorch",
                )
            except Exception as e:
                print_log(f"[PT-diag] shard#{sh.idx} err: {e!r}", "_pytorch")
    except Exception as e:
        print_log(f"[PT-diag] SHARDS fatal: {e!r}", "_pytorch")


def _snapshot_outputs(
    global_out: Path, esrgan_root: Path, *, tag: str = "pool-snapshot"
) -> None:
    try:
        n_out = len(list(global_out.glob("*_out.png")))
        n_seq = len(list(global_out.glob("frame_*.png")))
        res = esrgan_root / "results"
        n_res = len(list(res.rglob("*_out.png"))) if res.exists() else 0
        smp = [p.name for p in list(global_out.glob("*_out.png"))[:4]]
        print_log(
            f"[PT-snap] {tag} up_dir={global_out} *_out={n_out} frame_*= {n_seq} results_out={n_res} smp={smp}",
            "_pytorch",
        )
    except Exception as e:
        print_log(f"[PT-snap] err: {e!r}", "_pytorch")


def _spawn_one(
    sh: _Shard,
    venv_python: Path,
    esr_script: Path,
    esrgan_root: Path,
    make_args: Callable[[Path, Path], List[str]],
    env: Dict[str, str],
) -> None:
    args = [
        str(venv_python),
        str(esr_script),
        *map(str, make_args(sh.in_dir, sh.out_dir)),
    ]
    sh.log_path = sh.out_dir / f"shard_{sh.idx:02d}.log"
    sh.t_start = time.time()
    printable = " ".join(args)
    print_log(
        f"[PT-pool] SPAWN shard#{sh.idx} attempts={sh.attempts} cmd: {printable}",
        "_pytorch",
    )
    try:
        sh.proc = cast(
            _ProcLike,
            mg.popen(
                args,
                cwd=esrgan_root,
                text=True,
                env=ab.build_esrgan_env(env),
                log_to=str(sh.log_path),
            ),
        )
    except Exception as e:
        sh.proc = None
        co.print_warning(f"[PT-pool] spawn failed for shard#{sh.idx}: {e!r}")
        print_log(f"[PT-pool] spawn EXC shard#{sh.idx}: {e!r}", "_pytorch")


def _count_outputs(p: Path) -> int:
    try:
        return len(list(p.glob("*_out.png"))) + len(list(p.glob("frame_*.png")))
    except Exception:
        return 0


def _count_outputs_global_for_shard(
    sh: _Shard, global_out: Path, esrgan_root: Path
) -> int:
    """
    Zählt produzierte Outputs eines Shards im *globalen* up_dir (und ggf. esrgan_root/results),
    gefiltert nach den Stems der Items des Shards. Double-Counting wird vermieden.
    """
    try:
        stems = sh.stems or {p.stem for p in sh.items}
        found: set[str] = set()
        # global up_dir
        for st in stems:
            if (global_out / f"{st}_out.png").exists():
                found.add(st)
        # optional: esrgan_root/results (manche Runner schreiben temporär dorthin)
        results = esrgan_root / "results"
        if results.exists():
            for st in stems:
                if (results / f"{st}_out.png").exists():
                    found.add(st)
        return len(found)
    except Exception:
        return 0


def _shard_summary(
    sh: _Shard, global_out: Path, esrgan_root: Path
) -> Dict[str, object]:
    dur = (time.time() - sh.t_start) if sh.t_start else 0.0
    return {
        "idx": sh.idx,
        "attempts": sh.attempts,
        "in": str(sh.in_dir),
        "out": str(sh.out_dir),
        "n_in": len(sh.items),
        "n_out": _count_outputs_global_for_shard(sh, global_out, esrgan_root),
        "rc": (
            None if (sh.proc is None or sh.proc.poll() is None) else sh.proc.returncode
        ),
        "dt": f"{dur:.1f}s",
        "log": (str(sh.log_path) if sh.log_path else None),
    }


def run_sharded_dir_job_with_retries(
    *,
    venv_python: Path,
    esr_script: Path,
    esrgan_root: Path,
    in_dir: Path,
    out_dir: Path,
    make_args: Callable[[Path, Path], List[str]],
    total_frames: int,
    workers: int,
    env: Optional[Dict[str, str]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    max_retries: int = 2,
    per_shard_timeout: Optional[int] = None,
) -> bool:
    """
    Startet bis zu `workers` persistente CLI-Jobs auf Shard-Verzeichnisse und
    überwacht Fortschritt/RC. Bei RC!=0 werden Shards bis `max_retries` neu
    gestartet. Liefert True, wenn *alle* Shards erfolgreich waren.
    """
    env = dict(env or os.environ)
    mg.install_global_cancel_handlers()
    print_log(
        f"[PT-pool] BEGIN in={in_dir} out={out_dir} workers={workers} total_frames≈{total_frames} "
        f"retries={max_retries} timeout={per_shard_timeout}",
        "_pytorch",
    )

    if workers <= 1 or total_frames <= 1:
        print_log(
            "[PT-pool] trivial case (workers<=1 or total_frames<=1) → let upper layer handle serial",
            "_pytorch",
        )
        return False  # bewusst: Pool nicht sinnvoll

    shards, tmp_root = _prepare_shards(in_dir, out_dir, workers)
    _diagnose_shards_state(shards, out_dir, esrgan_root, tmp_root, tag="after-prepare")

    if sum(len(s.items) for s in shards) <= 0:
        print_log("[PT-pool] no frames to process → abort", "_pytorch")
        return False

    pending = shards[:]
    running: List[_Shard] = []
    done: List[_Shard] = []
    failed: List[_Shard] = []

    # Prime
    while pending and len(running) < workers:
        sh = pending.pop(0)
        sh.attempts = 1
        _spawn_one(sh, venv_python, esr_script, esrgan_root, make_args, env)
        running.append(sh)

    last_progress_emit = 0.0
    with mg.escape_cancel_guard(nonintrusive=True):
        while running or pending:
            if mg.is_cancelled():
                for sh in running:
                    try:
                        if sh.proc:
                            sh.proc.terminate()
                    except Exception:
                        pass
                mg.kill_all()
                print_log("[PT-pool] CANCEL received → killed", "_pytorch")
                break

            # poll
            new_running: List[_Shard] = []
            for sh in running:
                rc = None if sh.proc is None else sh.proc.poll()
                if rc is None:
                    new_running.append(sh)
                    continue

                # finished
                dur = time.time() - (sh.t_start or time.time())
                n_out = _count_outputs_global_for_shard(sh, out_dir, esrgan_root)
                _snapshot_outputs(out_dir, esrgan_root, tag="loop")
                print_log(
                    f"[PT-pool] shard#{sh.idx} finished rc={getattr(sh.proc, 'returncode', None)} "
                    f"dt={dur:.1f}s produced={n_out}/{len(sh.items)} log={sh.log_path}",
                    "_pytorch",
                )

                if getattr(sh.proc, "returncode", 1) == 0 and n_out >= 1:
                    done.append(sh)
                else:
                    if sh.attempts <= max_retries:
                        sh.attempts += 1
                        print_log(
                            f"[PT-pool] shard#{sh.idx} RETRY {sh.attempts}/{max_retries}",
                            "_pytorch",
                        )
                        # outputs wegräumen (damit CLI nicht stolpert)
                        try:
                            for p in list(sh.out_dir.glob("*")):
                                if p.is_file():
                                    p.unlink()
                        except Exception:
                            pass
                        _spawn_one(
                            sh, venv_python, esr_script, esrgan_root, make_args, env
                        )
                        new_running.append(sh)
                    else:
                        failed.append(sh)

            running = new_running

            # start more if free slots
            while pending and len(running) < workers:
                sh = pending.pop(0)
                sh.attempts = 1
                _spawn_one(sh, venv_python, esr_script, esrgan_root, make_args, env)
                running.append(sh)

            # progress
            now = time.time()
            if progress_cb and (now - last_progress_emit) >= 0.25:
                produced = sum(
                    _count_outputs_global_for_shard(sh, out_dir, esrgan_root)
                    for sh in shards
                )
                progress_cb(min(produced, total_frames), total_frames)
                last_progress_emit = now

            time.sleep(0.05)

    _diagnose_shards_state(shards, out_dir, esrgan_root, tmp_root, tag="before-summary")

    # Summary
    summary = {
        "done": [_shard_summary(s, out_dir, esrgan_root) for s in done],
        "failed": [_shard_summary(s, out_dir, esrgan_root) for s in failed],
        "running": [_shard_summary(s, out_dir, esrgan_root) for s in running],
        "pending": [dict(idx=s.idx, n_in=len(s.items)) for s in pending],
    }
    print_log(f"[PT-pool] SUMMARY: {summary}", "_pytorch")

    # Erfolg nur, wenn keine Shards gescheitert sind
    ok = len(failed) == 0 and all(
        _count_outputs_global_for_shard(s, out_dir, esrgan_root) >= 1 for s in shards
    )
    if ok:
        print_log(f"[PT-pool] ALL SHARDS OK → outputs in {out_dir}", "_pytorch")
    else:
        co.print_warning("[PT-pool] some shards failed (see logs above)")
    return ok
