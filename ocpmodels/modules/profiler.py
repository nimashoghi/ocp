import argparse
import contextlib
import copy
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from torch.autograd.profiler import record_function as record_function
from torch.profiler import ProfilerActivity, profile, schedule
from typing_extensions import TypedDict

log = logging.getLogger(__name__)


class ProfilerScheduleConfig(TypedDict):
    wait: int
    warmup: int
    active: int
    repeat: int


class ProfilerConfig(TypedDict):
    activities: List[Literal["CPU", "CUDA"]]
    with_stack: bool
    with_modules: bool
    record_shapes: bool
    schedule: ProfilerScheduleConfig

    export_path: str
    cuda_launch_blocking: bool


DEFAULT_PROFILER_CONFIG: ProfilerConfig = {
    "activities": ["CPU", "CUDA"],
    "with_stack": False,
    "with_modules": False,
    "record_shapes": False,
    "schedule": {
        "wait": 0,
        "warmup": 10,
        "active": 10,
        "repeat": 1,
    },
    "export_path": "./traces",
    "cuda_launch_blocking": True,
}


@dataclass
class ActiveProfilerState:
    profiler: profile
    config: ProfilerConfig
    max_num_steps: Optional[int]


_active_profiler: Optional[ActiveProfilerState] = None


def active_profiler() -> Optional[ActiveProfilerState]:
    global _active_profiler
    return _active_profiler


def _construct_profiler(profiler_config: ProfilerConfig, model_name: str):
    activities: List[ProfilerActivity] = []
    for activity_str in profiler_config["activities"]:
        if activity_str == "CPU":
            activity = ProfilerActivity.CPU
        elif activity_str == "CUDA":
            activity = ProfilerActivity.CUDA
        else:
            raise ValueError(f"Unknown profiler activity {activity_str}")
        activities.append(activity)

    def on_trace_ready(profiler: profile) -> None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        trace_filename = f"trace-{timestamp}.pt.trace.json"
        path = Path(profiler_config["export_path"]) / model_name
        path.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(path / trace_filename))

    return profile(
        activities=activities,
        with_stack=profiler_config["with_stack"],
        with_modules=profiler_config["with_modules"],
        record_shapes=profiler_config["record_shapes"],
        schedule=schedule(
            wait=profiler_config["schedule"]["wait"],
            warmup=profiler_config["schedule"]["warmup"],
            active=profiler_config["schedule"]["active"],
            repeat=profiler_config["schedule"]["repeat"],
        ),
        on_trace_ready=on_trace_ready,
    )


def profiler_initialize(args: argparse.Namespace, config: Dict[str, Any]):
    if not args.profile:
        return

    # If we're not in debug mode, warn the user
    if not args.debug:
        log.warning("Profiling is enabled but debug mode is not enabled.")

    # Make sure this is not distributed training
    if args.distributed:
        raise ValueError("Profiling is not supported in distributed training")

    profiler_config: ProfilerConfig = copy.deepcopy(DEFAULT_PROFILER_CONFIG)
    profiler_config.update(config.get("task", {}).get("profiler", {}))

    if args.profiler_cuda_blocking is not None:
        profiler_config["cuda_launch_blocking"] = args.profiler_cuda_blocking

    if profiler_config["cuda_launch_blocking"]:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        log.critical("Setting CUDA_LAUNCH_BLOCKING=1 for profiling")

    profiler = _construct_profiler(profiler_config, config["model"]["name"])
    max_num_steps: Optional[int] = None
    if profiler_config["schedule"]["repeat"] > 0:
        max_num_steps = (
            profiler_config["schedule"]["wait"]
            + profiler_config["schedule"]["warmup"]
            + profiler_config["schedule"]["active"]
        ) * profiler_config["schedule"]["repeat"]

    global _active_profiler
    _active_profiler = ActiveProfilerState(
        profiler, profiler_config, max_num_steps
    )


@contextlib.contextmanager
def profiler_context():
    profiler_state = active_profiler()
    if profiler_state is None:
        yield
        return

    with profiler_state.profiler:
        yield
