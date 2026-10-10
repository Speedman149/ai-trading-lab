import json
import re
import tempfile
from pathlib import Path
from threading import RLock

from app.core.config import ROOT

LOCK = RLock()


def folder():
    path = ROOT / "artifacts" / "research"
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_id(identifier):
    if not re.fullmatch(r"[0-9a-f]{24}", identifier):
        raise FileNotFoundError("Unknown model version")
    return identifier


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, allow_nan=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def result(identifier):
    return json.loads((folder() / safe_id(identifier) / "result.json").read_text())


def registry():
    path = folder() / "registry.json"
    return json.loads(path.read_text()) if path.exists() else {"active": {}, "history": []}


def key(config):
    import hashlib

    fields = config.model_dump(mode="json")
    costs = {
        k: v for k, v in fields["backtest"].items() if k not in ("start", "end", "strategy", "fast", "slow")
    }
    return hashlib.sha256(
        json.dumps(
            {"backtest": costs, "algorithm": config.algorithm, "horizon": config.horizon}, sort_keys=True
        ).encode()
    ).hexdigest()[:24]


def versions():
    saved = []
    state = registry()
    active = set(state["active"].values())
    for path in folder().glob("*/result.json"):
        data = json.loads(path.read_text())
        saved.append(
            {
                "id": data["id"],
                "created_at": data["created_at"],
                "algorithm": data["config"]["algorithm"],
                "asset": data["config"]["backtest"]["asset"],
                "provider": data["config"]["backtest"]["provider"],
                "eligible": data["promotion"]["eligible"],
                "active": data["id"] in active,
                "rollback_available": any(
                    data["id"] in (event["active"], event["previous"]) for event in state["history"]
                ),
                "net_return": data["holdout"]["strategy"]["metrics"]["net_return"],
                "parent_id": data["config"]["parent_id"],
            }
        )
    return sorted(saved, key=lambda item: item["created_at"], reverse=True)


def promote(identifier):
    from app.ml.schemas import ResearchRequest

    with LOCK:
        data = result(identifier)
        config = ResearchRequest.model_validate(data["config"])
        state = registry()
        family = key(config)
        current = state["active"].get(family)
        if not data["promotion"]["eligible"]:
            raise ValueError("Candidate did not pass the predefined validation criteria.")
        if current != data["promotion"]["active_model_at_training"]:
            raise ValueError(
                "Active model changed since evaluation. Retrain against the current active version."
            )
        state["active"][family] = identifier
        state["history"].append(
            {"family": family, "previous": current, "active": identifier, "action": "promote"}
        )
        atomic_json(folder() / "registry.json", state)
        return state


def rollback(identifier):
    from app.ml.schemas import ResearchRequest

    with LOCK:
        data = result(identifier)
        family = key(ResearchRequest.model_validate(data["config"]))
        state = registry()
        if not any(
            event["family"] == family and identifier in (event["active"], event["previous"])
            for event in state["history"]
        ):
            raise ValueError("Rollback requires a previously active version in this model family.")
        previous = state["active"].get(family)
        state["active"][family] = identifier
        state["history"].append(
            {"family": family, "previous": previous, "active": identifier, "action": "rollback"}
        )
        atomic_json(folder() / "registry.json", state)
        return state


def search_result(identifier):
    path = folder() / "searches" / f"{safe_id(identifier)}.json"
    return json.loads(path.read_text())


def searches():
    saved = []
    for path in (folder() / "searches").glob("*.json"):
        data = json.loads(path.read_text())
        selected = next((trial for trial in data["trials"] if trial["trial"] == data["selected_trial"]), None)
        saved.append(
            {
                "parameters": selected["parameters"] if selected else None,
                "metrics": selected["metrics"] if selected else None,
                "id": data["id"],
                "created_at": data["created_at"],
                "status": data["status"],
                "asset": data["config"]["research"]["backtest"]["asset"],
                "provider": data["config"]["research"]["backtest"]["provider"],
                "trials": len(data["trials"]),
                "objective": data["config"]["objective"],
                "mode": data["config"].get("mode", "random"),
                "winner_id": data["winner_id"],
            }
        )
    return sorted(saved, key=lambda value: value["created_at"], reverse=True)
