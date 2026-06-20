"""Split-10 tests 1-4 + R9 (no key) and 7-8 (@api): the full stack over HTTP.

The contract assertions run against the real app via ``TestClient`` (deterministic,
no socket flakiness); a single ``live_server`` smoke proves ``uvicorn`` actually
boots and serves over a real socket. Together they cover R3 (full-stack boot),
R4 (missing-key path), R5 (0-item → N/A end to end), and R9 (the adversarial
remove-the-sample check).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_SAMPLE = (
    REPO_ROOT / "api" / "src" / "frugalroute_api" / "data" / "sample_run.json"
)


# --- Test 1 (R3): the api boots and is healthy, no key configured ------------
def test_live_server_boots_over_http(live_server) -> None:
    """A real uvicorn process answers /api/health 200 with has_api_key:false."""
    resp = live_server.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["has_api_key"] is False  # native backend, no ANTHROPIC_API_KEY


def test_live_server_serves_committed_sample(live_server) -> None:
    """The running stack serves the committed proof over real HTTP (no key)."""
    resp = live_server.get("/api/eval/sample")
    assert resp.status_code == 200
    body = resp.json()
    assert {r["strategy"] for r in body["reports"]} >= {"cascade", "predictive"}


def test_client_health_no_key(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["has_api_key"] is False


# --- Test 2 (R1): Frontier proof — headline + 6-row leaderboard --------------
def test_eval_sample_headline_matches_committed(client: TestClient) -> None:
    served = client.get("/api/eval/sample").json()
    committed = json.loads(COMMITTED_SAMPLE.read_text(encoding="utf-8"))
    for served_r, committed_r in zip(
        served["reports"], committed["reports"], strict=True
    ):
        assert served_r["retention_at_target"] == committed_r["retention_at_target"]
        assert (
            served_r["cost_reduction_at_target"]
            == committed_r["cost_reduction_at_target"]
        )


def test_leaderboard_has_six_rows(client: TestClient) -> None:
    """always-cheap · always-strong · random · cascade · predictive · oracle."""
    served = client.get("/api/eval/sample").json()
    cascade = next(r for r in served["reports"] if r["strategy"] == "cascade")
    predictive = next(r for r in served["reports"] if r["strategy"] == "predictive")
    rows = set(cascade["baselines"]) | {
        "oracle",
        cascade["strategy"],
        predictive["strategy"],
    }
    assert rows == {
        "always_cheap",
        "always_strong",
        "random",
        "oracle",
        "cascade",
        "predictive",
    }
    assert cascade["points"] and predictive["points"]


# --- Test 3 (R4): no-key single query → the honest missing-key path ----------
def test_route_no_key_is_missing_key_with_proof_hint(client: TestClient) -> None:
    resp = client.post(
        "/api/route", json={"strategy": "cascade", "example_id": "gsm8k-1142"}
    )
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["type"] == "missing-key"
    # The message points the UI at the precomputed proof escape hatch.
    assert "/api/eval/sample" in err["message"]


def test_proof_path_reachable_without_key(client: TestClient) -> None:
    """The 'View the Proof' escape lands on a populated Frontier even with no key."""
    resp = client.get("/api/eval/sample")
    assert resp.status_code == 200
    assert resp.json()["reports"][0]["points"]


# --- Test 4 (R5): 0-item / empty bundle → N/A, never a fake number -----------
def test_empty_bundle_serves_na_not_fake(
    client: TestClient, override_settings: Callable[..., None], tmp_path: Path
) -> None:
    """A zero-item run (empty points, nan headline) is served verbatim — the UI
    renders N/A — never an invented number."""
    empty = tmp_path / "empty.json"
    empty.write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "strategy": "cascade",
                        "points": [],
                        "baselines": {
                            "always_cheap": {
                                "quality": float("nan"),
                                "quality_spread": 0.0,
                                "cost": float("nan"),
                                "cost_spread": 0.0,
                            },
                            "always_strong": {
                                "quality": float("nan"),
                                "quality_spread": 0.0,
                                "cost": float("nan"),
                                "cost_spread": 0.0,
                            },
                            "random": {
                                "quality": float("nan"),
                                "quality_spread": 0.0,
                                "cost": float("nan"),
                                "cost_spread": 0.0,
                            },
                        },
                        "oracle": {
                            "quality": float("nan"),
                            "quality_spread": 0.0,
                            "cost": float("nan"),
                        },
                        "retention_at_target": float("nan"),
                        "retention_at_target_spread": float("nan"),
                        "cost_reduction_at_target": float("nan"),
                        "cost_reduction_at_target_spread": float("nan"),
                        "n_refused": 0,
                        "prompt_version": "v1",
                        "model_tiers": ["claude-haiku-4-5", "claude-opus-4-8"],
                        "n_runs": 0,
                    }
                ],
                "benchmark": "gsm8k",
                "frozen_split": {"n_test": 0, "n_calibration": 0, "small_n": True},
                "generated_at": "2026-06-20T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    override_settings(sample_run_path=empty)
    resp = client.get("/api/eval/sample")
    assert resp.status_code == 200
    report = resp.json()["reports"][0]
    assert report["points"] == []  # nothing to plot → the UI shows N/A
    # nan is serialized as null over JSON (never 0 / a fabricated number).
    assert report["retention_at_target"] is None


# --- R9 (adversarial): remove the committed sample → honest N/A, then recover -
def test_missing_sample_is_honest_404(
    client: TestClient, override_settings: Callable[..., None], tmp_path: Path
) -> None:
    override_settings(sample_run_path=tmp_path / "gone.json")
    resp = client.get("/api/eval/sample")
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["type"] == "not-found"
    assert "eval" in err["message"].lower()  # offers the 'run an eval' path


def test_recovers_when_sample_restored(client: TestClient) -> None:
    """With the committed sample present (default settings) the Frontier is back."""
    resp = client.get("/api/eval/sample")
    assert resp.status_code == 200
    assert resp.json()["reports"][0]["points"]


# --- Tests 7-8 (@api): live stream route + quick eval round-trip -------------
@pytest.mark.azure
def test_live_stream_route_end_to_end(live_server_azure) -> None:
    """A live SSE route streams to a terminal `done` carrying a well-formed result."""
    url = "/api/route/stream?strategy=cascade&example_id=gsm8k-1142"
    saw_done = False
    with httpx.stream("GET", live_server_azure.base_url + url, timeout=120.0) as resp:
        assert resp.status_code == 200
        event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event == "done":
                payload = json.loads(line.split(":", 1)[1].strip())
                assert payload["query"]
                assert payload["tier_used"]
                assert isinstance(payload["cost_usd"], (int, float))
                saw_done = True
                break
    assert saw_done, "stream never reached a terminal done event"


@pytest.mark.azure
def test_live_quick_eval_round_trips(live_server_azure) -> None:
    """POST /api/eval {quick:true} returns a bundle the Frontier can render."""
    resp = live_server_azure.post(
        "/api/eval", json={"strategy": "cascade", "benchmark": "gsm8k", "quick": True}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reports"][0]["points"]
    assert body["reports"][0]["model_tiers"]
