"""Own the shared TransferQueue controller for a multi-job Weave launcher."""

from __future__ import annotations

import argparse
import signal
import threading

import ray
import transfer_queue as tq
from omegaconf import OmegaConf

_shutdown = threading.Event()


def _request_shutdown(_signum: int, _frame: object) -> None:
    _shutdown.set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    ray.init(address=args.ray_address, namespace="transfer_queue")
    conf = OmegaConf.create({"backend": {"SimpleStorage": {"num_data_storage_units": 2}}})
    tq.init(conf)
    print("TRANSFER_QUEUE_OWNER_READY", flush=True)
    _shutdown.wait()
    tq.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
