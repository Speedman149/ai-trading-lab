import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from app.data.errors import DataError
from app.ml import storage
from app.ml.service import execute

EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research")
JOBS = {}


class ResearchBusy(Exception):
    pass


def initialize():
    with storage.LOCK:
        for path in (storage.folder() / "jobs").glob("*.json"):
            import json

            job = json.loads(path.read_text())
            if job["status"] in ("queued", "running"):
                job.update(
                    status="interrupted",
                    progress="Backend restarted; submit training again.",
                    error="Training was interrupted by a backend restart.",
                )
                storage.atomic_json(path, job)
            JOBS[job["id"]] = job


def save(job):
    storage.atomic_json(storage.folder() / "jobs" / f"{job['id']}.json", job)


def submit(config):
    with storage.LOCK:
        if any(j["status"] in ("queued", "running") for j in JOBS.values()):
            raise ResearchBusy("A training job is already running. Wait for it to finish.")
        identifier = uuid.uuid4().hex[:24]
        job = {
            "id": identifier,
            "status": "queued",
            "progress": "Queued",
            "created_at": datetime.now(UTC).isoformat(),
            "result_id": None,
            "error": None,
        }
        JOBS[identifier] = job
        save(job)

    def progress(message):
        with storage.LOCK:
            job["progress"] = message
            save(job)

    def work():
        with storage.LOCK:
            job["status"] = "running"
            save(job)
        try:
            result = execute(config, progress)
            with storage.LOCK:
                job.update(
                    status="completed", result_id=result["id"], progress="Training and evaluation complete"
                )
                save(job)
        except Exception as exc:
            logging.getLogger(__name__).exception("Research job failed")
            with storage.LOCK:
                job.update(
                    status="failed",
                    error=str(exc)
                    if isinstance(exc, (DataError, FileNotFoundError))
                    else "Training failed; see backend logs.",
                    progress="Training failed",
                )
                save(job)

    EXECUTOR.submit(work)
    return job.copy()


def retrieve(identifier):
    with storage.LOCK:
        return JOBS[storage.safe_id(identifier)].copy()
