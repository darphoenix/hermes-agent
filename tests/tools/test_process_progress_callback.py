import time

from tools.environments.base import set_process_progress_callback
from tools.environments.local import LocalEnvironment


def test_long_process_progress_callback_can_cancel(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    reviews = []
    try:
        set_process_progress_callback(
            lambda progress: reviews.append(progress) or True,
            initial_delay=1,
            interval=10,
        )
        started = time.monotonic()
        result = env.execute("sleep 5", timeout=8)
    finally:
        set_process_progress_callback(None)
        env.cleanup()

    assert result["returncode"] == 125
    assert "conscience progress review" in result["output"]
    assert reviews and reviews[0]["elapsed_seconds"] >= 1
    assert time.monotonic() - started < 4
