import os
import json
import time
import argparse
from typing import Callable, Never, Optional, Generator
from prometheus_client import REGISTRY, start_http_server
from . import metrics
from dataclasses import dataclass

type Timestamp = float
type Seconds = float

@dataclass
class DirInfo:
    path: str
    size: int
    latest_mtime: Timestamp
    oldest_mtime: Timestamp
    entries_count: int
    processing_time: Seconds

ONE_S_IN_NS = 1_000_000_000


class BudgetedDirInfoWalker:
    def __init__(self, iops_budget: int=100):
        """
        iops_budget is number of io operations allowed every second.
        """
        self.iops_budget = iops_budget
        # Use _ns to avoid subtractive messiness possible when using floats
        self._last_iops_reset_time = time.monotonic_ns()
        self._io_calls_since_last_reset = 0

    def do_iops_action[R, **P](self, func: Callable[P, R], *args, **kwargs) -> R:
        """
        Perform an action that does IO, waiting if necessary so it is within budget.

        All IO performed should be wrapped with this function, so we do not exceed
        our budget. Each call to this function is treated as one IO.
        """
        if time.monotonic_ns() - self._last_iops_reset_time > ONE_S_IN_NS:
            # One second has passed since last time we reset the budget clock
            # So we reset it again now, regardless of how many iops have happened
            self._io_calls_since_last_reset = 0
            self._last_iops_reset_time = time.monotonic_ns()

        if self._io_calls_since_last_reset > self.iops_budget:
            # We are over budget, so we wait for 1s + 1ns since last reset
            # IO can be performed once this is wait is done. We reset the budget clock
            # after our wait.
            wait_period_in_s = (
                ONE_S_IN_NS - (time.monotonic_ns() - self._last_iops_reset_time) + 1
            ) / ONE_S_IN_NS
            time.sleep(wait_period_in_s)
            self._io_calls_since_last_reset = 0
            self._last_iops_reset_time = time.monotonic_ns()

        return_value = func(*args, **kwargs)
        self._io_calls_since_last_reset += 1
        return return_value

    def get_dir_info(self, path: str) -> Optional[DirInfo]:
        start_time = time.monotonic()
        try:
            self_statinfo = os.stat(path)
        except FileNotFoundError:
            # Directory was deleted from the time it was listed and now
            return None

        # Get absolute path of all children of directory
        children = [
            os.path.abspath(os.path.join(path, c))
            for c in self.do_iops_action(os.listdir, path)
        ]
        # Split into files and directories for different kinds of traversal.
        # We count symlinks as files, but do not resolve them when checking size -
        # but do include them in the mtime calculation.
        files = [
            c
            for c in children
            if self.do_iops_action(os.path.isfile, c)
            or self.do_iops_action(os.path.islink, c)
        ]
        dirs = [c for c in children if self.do_iops_action(os.path.isdir, c)]

        total_size = self_statinfo.st_size
        latest_mtime = self_statinfo.st_mtime
        oldest_mtime = self_statinfo.st_mtime
        entries_count = len(files) + 1  # Include this directory as an entry

        for f in files:
            # Do not follow symlinks, as that may lead to double counting a symlinked
            # file's size.
            try:
                stat_info = self.do_iops_action(os.stat, f, follow_symlinks=False)
            except FileNotFoundError:
                # File might have been deleted from the time we listed it anda now
                continue
            total_size += stat_info.st_size
            if latest_mtime < stat_info.st_mtime:
                latest_mtime = stat_info.st_mtime
            if oldest_mtime > stat_info.st_mtime:
                oldest_mtime = stat_info.st_mtime

        for d in dirs:
            dirinfo = self.get_dir_info(d)
            if dirinfo is None:
                # The directory was deleted between the time the listing
                # was done and now.
                continue
            total_size += dirinfo.size
            entries_count += dirinfo.entries_count
            if latest_mtime < dirinfo.latest_mtime:
                latest_mtime = dirinfo.latest_mtime
            if oldest_mtime > dirinfo.latest_mtime:
                oldest_mtime = dirinfo.latest_mtime

        return DirInfo(
            path=os.path.basename(path),
            size=total_size,
            latest_mtime=latest_mtime,
            oldest_mtime=oldest_mtime,
            entries_count=entries_count,
            processing_time=time.monotonic() - start_time,
        )

    def get_subdirs_info(self, dir_path: str) -> Generator[DirInfo | None, None, None]:
        try:
            children = [
                os.path.abspath(os.path.join(dir_path, c))
                for c in self.do_iops_action(os.listdir, dir_path)
            ]

            dirs = [c for c in children if self.do_iops_action(os.path.isdir, c)]

            for c in dirs:
                yield self.get_dir_info(c)
        except OSError as e:
            if e.errno == 116:
                # See https://github.com/yuvipanda/prometheus-dirsize-exporter/issues/6
                # Stale file handle, often because the file we were looking at
                # changed in the NFS server via another client in such a way that
                # a new inode was created. This is a race, so let's just ignore and
                # not report any data for this file. If this file was recreated,
                # our next run should catch it
                return None
            # Any other errors should just be propagated
            raise
        except PermissionError as e:
            if e.errno == 13:
                # See https://github.com/yuvipanda/prometheus-dirsize-exporter/issues/5
                # A file we are trying to open is owned in such a way that we don't have
                # access to it. Ideally this should not really happen, but when it does,
                # we just ignore it and continue.
                return None
            # Any other permission error should be propagated
            raise
        except FileNotFoundError as e:
            # File has been renamed or deleted between list and getting information
            # We can skip this silently
            return None


def load_state(state_file: str) -> dict:
    """
    Load the previously saved state from disk, if it exists.

    Returns an empty dict if the file does not exist or can not be read.
    """
    if not state_file or not os.path.exists(state_file):
        return {}
    try:
        with open(state_file) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not load state file {state_file}, ignoring it: {e}")
        return {}


def save_state(state_file: str, state: dict):
    """
    Save the current state to disk, so it can be reloaded on next startup.

    Written atomically (via a temporary file + rename) to avoid leaving a
    corrupted state file behind if the process is killed mid-write.
    """
    if not state_file:
        return
    tmp_state_file = f"{state_file}.tmp"
    try:
        with open(tmp_state_file, "w") as f:
            json.dump(state, f)
        os.replace(tmp_state_file, state_file)
    except OSError as e:
        print(f"Could not save state file {state_file}: {e}")


def remove_stale_entries(
    base_dir: str, state: dict, stale_paths: set[str], disable_total_size: bool
):
    """
    Remove directories that no longer exist (deleted between iterations) from
    both the saved state and the exposed prometheus metrics, so they do not
    linger forever with stale values.
    """
    for path in stale_paths:
        del state[base_dir][path]
        gauges = [metrics.LATEST_MTIME, metrics.OLDEST_MTIME, metrics.ENTRIES_COUNT, metrics.PROCESSING_TIME, metrics.LAST_UPDATED]
        if not disable_total_size:
            gauges.append(metrics.TOTAL_SIZE)
        for gauge in gauges:
            try:
                gauge.remove(path, base_dir)
            except KeyError:
                # This directory never had a value set for this particular
                # metric (e.g. processing time when detailed metric is
                # disabled), nothing to remove.
                pass
        print(f"Removed stale entry for {path} in {base_dir} (directory no longer exists)")


def apply_state_to_metrics(
    state: dict, enable_detailed_processing_time_metric: bool, disable_total_size: bool
):
    """
    Push previously saved state into the prometheus metrics, so there is no
    gap in the exposed metrics when the exporter restarts.
    """
    for base_dir, subdirs in state.items():
        for path, info in subdirs.items():
            if not disable_total_size:
                metrics.TOTAL_SIZE.labels(directory=path, base=base_dir).set(info["size"])
            metrics.LATEST_MTIME.labels(directory=path, base=base_dir).set(
                info["latest_mtime"]
            )
            if "oldest_mtime" in info:
                metrics.OLDEST_MTIME.labels(directory=path, base=base_dir).set(
                    info["oldest_mtime"]
                )
            metrics.ENTRIES_COUNT.labels(directory=path, base=base_dir).set(
                info["entries_count"]
            )
            if enable_detailed_processing_time_metric and "processing_time" in info:
                metrics.PROCESSING_TIME.labels(directory=path, base=base_dir).set(
                    info["processing_time"]
                )
            if "last_updated" in info:
                metrics.LAST_UPDATED.labels(directory=path, base=base_dir).set(
                    info["last_updated"]
                )


def main() -> Never:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "parent_dir",
        help="The directories, comma separated, to whose subdirectories will have their information exported",
    )
    argparser.add_argument(
        "iops_budget", help="Number of IO operations allowed per second", type=int
    )
    argparser.add_argument(
        "wait_time_minutes",
        help="Number of minutes to wait before data collection runs",
        type=int,
    )
    # Don't report amount of time it took to process each directory by
    # default. This is highly variable, and probably causes prometheus to
    # not compress metrics very well. Not particularly useful outside of
    # debugging the exporter itself.
    argparser.add_argument(
        "--enable-detailed-processing-time-metric",
        help="Report amount of time it took to process each directory",
        action="store_true"
    )
    argparser.add_argument(
        "--port", help="Port for the server to listen on", type=int, default=8000
    )
    argparser.add_argument(
        "--state-file",
        help="Path to a JSON file used to persist directory metrics between "
        "restarts, so there is no gap in exposed metrics when the exporter "
        "restarts. Disabled by default; state is only saved/loaded if this "
        "option is explicitly provided.",
        default=None,
    )

    argparser.add_argument(
        "--disable-total-size",
        help="Disable total size metric reporting",
        action="store_true"

    )

    args = argparser.parse_args()

    if not args.disable_total_size:
        REGISTRY.register(metrics.TOTAL_SIZE)

    state = load_state(args.state_file)
    if state:
        apply_state_to_metrics(
            state, args.enable_detailed_processing_time_metric, args.disable_total_size
        )
        print(f"Loaded previous state from {args.state_file}")

    start_http_server(args.port)
    while True:
        walker = BudgetedDirInfoWalker(args.iops_budget)
        for base_dir in args.parent_dir.split(','):
            state.setdefault(base_dir, {})
            seen_paths = set()
            for subdir_info in walker.get_subdirs_info(base_dir):
                if subdir_info is None:
                    continue
                seen_paths.add(subdir_info.path)
                if not args.disable_total_size:
                    metrics.TOTAL_SIZE.labels(directory=subdir_info.path, base=base_dir).set(subdir_info.size)
                metrics.LATEST_MTIME.labels(directory=subdir_info.path, base=base_dir).set(subdir_info.latest_mtime)
                metrics.OLDEST_MTIME.labels(directory=subdir_info.path, base=base_dir).set(subdir_info.oldest_mtime)
                metrics.ENTRIES_COUNT.labels(directory=subdir_info.path, base=base_dir).set(
                    subdir_info.entries_count
                )
                if args.enable_detailed_processing_time_metric:
                    metrics.PROCESSING_TIME.labels(directory=subdir_info.path, base=base_dir).set(
                        subdir_info.processing_time
                    )
                last_updated = time.time()
                metrics.LAST_UPDATED.labels(directory=subdir_info.path, base=base_dir).set(last_updated)
                state[base_dir][subdir_info.path] = {
                    "size": subdir_info.size,
                    "latest_mtime": subdir_info.latest_mtime,
                    "oldest_mtime": subdir_info.oldest_mtime,
                    "entries_count": subdir_info.entries_count,
                    "processing_time": subdir_info.processing_time,
                    "last_updated": last_updated,
                }
                print(f"Updated values for {subdir_info.path} in {base_dir}")
            # Any directory that was known in a previous iteration but was not
            # seen in this one has been deleted: drop it from the state and
            # from the exposed metrics so it does not linger with stale data.
            stale_paths = set(state[base_dir].keys()) - seen_paths
            if stale_paths:
                remove_stale_entries(base_dir, state, stale_paths, args.disable_total_size)
        save_state(args.state_file, state)
        time.sleep(args.wait_time_minutes * 60)


if __name__ == "__main__":
    main()
