"""Profiling hook for the M6 HTTP serving work, loaded via PYTHONPATH=benchmarks/scripts/profiling.

Active only when DLI_PROF_OUT is set, and only in `python -m dli.server` (role server: the API
process), the engine process it spawns (role engine) or the benchmark client
(`python -m dli.bench.campaign`, role client). Writes <DLI_PROF_OUT>/<role>-<pid>.json every 2 s
and at exit. It perturbs what it measures (any extra Python thread contends for the GIL, the 2 ms
stack sampler most of all), so compare runs made with the same settings, not with uninstrumented
ones. Contents:
* per-thread CPU seconds (from /proc), keyed by Python thread name;
* server / engine: every engine step's wall time and shape, engine-thread time between steps, and
  the time spent in AsyncEngine._dispatch or _deliver / _publish / _drain_commands;
* optional (DLI_PROF_SAMPLE=1): stack samples of the main (event loop) and engine threads every
  2 ms, aggregated by leaf function and by inclusive dli function.
DLI_PROF_SWITCH=<seconds> sets sys.setswitchinterval in the server.
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import sys
import threading
import time

_OUT = os.environ.get("DLI_PROF_OUT")
_argv = " ".join(getattr(sys, "orig_argv", sys.argv))
_ROLE = (
    "server"
    if "dli.server" in _argv
    else "client"
    if "dli.bench.campaign" in _argv
    else "engine"
    if "spawn_main" in _argv and os.environ.get("DLI_PROF_OUT")
    else None
)

if _OUT and _ROLE:
    os.makedirs(_OUT, exist_ok=True)
    _tck = os.sysconf("SC_CLK_TCK")
    _t0 = time.perf_counter()
    _cpu: dict[str, float] = {}
    _data: dict[str, object] = {"role": _ROLE, "pid": os.getpid()}
    _leaf: collections.Counter[str] = collections.Counter()
    _incl: collections.Counter[str] = collections.Counter()
    _samples = collections.Counter()

    def _thread_cpu() -> None:
        names = {t.native_id: t.name for t in threading.enumerate()}
        for tid in os.listdir("/proc/self/task"):
            try:
                with open(f"/proc/self/task/{tid}/stat") as f:
                    fields = f.read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            name = names.get(int(tid), f"native-{tid}")
            _cpu[f"{name}:{tid}"] = (int(fields[11]) + int(fields[12])) / _tck

    def _dump() -> None:
        _thread_cpu()
        _data["wall_s"] = time.perf_counter() - _t0
        _data["thread_cpu_s"] = dict(_cpu)
        _data["samples"] = dict(_samples)
        _data["leaf"] = _leaf.most_common(60)
        _data["inclusive"] = _incl.most_common(60)
        with open(os.path.join(_OUT, f"{_ROLE}-{os.getpid()}.json"), "w") as f:
            json.dump(_data, f)

    def _key(fr: object) -> str:
        co = fr.f_code  # type: ignore[attr-defined]
        fn = co.co_filename
        for marker in ("site-packages/", "/lib/python3.12/", "/src/"):
            if marker in fn:
                fn = fn.split(marker, 1)[1]
                break
        return f"{fn}:{co.co_name}"

    def _background() -> None:
        sample = os.environ.get("DLI_PROF_SAMPLE") == "1" and _ROLE == "server"
        main_id = threading.main_thread().ident
        last_dump = time.perf_counter()
        while True:
            time.sleep(0.002 if sample else 0.5)
            if sample:
                engine_id = next(
                    (t.ident for t in threading.enumerate() if t.name == "dli-engine"), None
                )
                frames = sys._current_frames()
                for label, ident in (("loop", main_id), ("engine", engine_id)):
                    fr = frames.get(ident) if ident else None
                    if fr is None:
                        continue
                    _samples[label] += 1
                    _leaf[f"{label} {_key(fr)}"] += 1
                    seen = set()
                    while fr is not None:
                        k = _key(fr)
                        if "dli/" in k and k not in seen:
                            _incl[f"{label} {k}"] += 1
                            seen.add(k)
                        fr = fr.f_back
            if time.perf_counter() - last_dump > 2.0:
                _dump()
                last_dump = time.perf_counter()

    atexit.register(_dump)
    threading.Thread(target=_background, name="dli-prof", daemon=True).start()

    if _ROLE in ("server", "engine"):
        sw = os.environ.get("DLI_PROF_SWITCH")
        if sw:
            sys.setswitchinterval(float(sw))
        _data["switchinterval"] = sys.getswitchinterval()

        from dli.engine import engine as _E
        from dli.server import async_engine as _A

        _steps: list[list[float]] = []
        _data["steps"] = _steps
        _totals: dict[str, list[float]] = {}
        _data["engine_thread_calls"] = _totals
        _orig_step = _E.LLMEngine.step

        def _step(self: object) -> object:
            t0 = time.perf_counter()
            out = _orig_step(self)  # type: ignore[arg-type]
            t1 = time.perf_counter()
            st = self.stats.steps[-1]  # type: ignore[attr-defined]
            _steps.append(
                [
                    t0 - _t0,
                    t1 - t0,
                    st.num_prefill_tokens,
                    st.num_decode_tokens,
                    st.num_seqs,
                    st.schedule_s,
                    st.forward_s,
                    st.postprocess_s,
                ]
            )
            return out

        _E.LLMEngine.step = _step  # type: ignore[method-assign]

        def _timed(cls: type, name: str) -> None:
            orig = getattr(cls, name)
            acc = _totals.setdefault(name, [0.0, 0])

            def wrapper(*a: object, **k: object) -> object:
                t = time.perf_counter()
                try:
                    return orig(*a, **k)
                finally:
                    acc[0] += time.perf_counter() - t
                    acc[1] += 1

            setattr(cls, name, wrapper)

        # _dispatch (per token) became _deliver (per step) in M6; time whichever exists.
        for _n in ("_dispatch", "_deliver", "_publish", "_drain_commands", "_account"):
            if hasattr(_A.AsyncEngine, _n):
                _timed(_A.AsyncEngine, _n)
