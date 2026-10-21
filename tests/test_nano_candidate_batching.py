import asyncio

import numpy as np

from app.services.voice.clone_service import VoiceCloneService, _gather_cancel_on_error


class DeterministicVoiceCloneService(VoiceCloneService):
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = []

    async def _generate_nano(
        self, target_text, prompt_text, reference_path, cfg, seed,
        retry_badcase, retry_max_times, retry_ratio_threshold,
        reference_mode, reference_latents=None,
    ):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append((target_text, seed, reference_latents))
        await asyncio.sleep(0.01)
        self.active -= 1
        return np.full(4, (seed % 9) / 10.0, dtype=np.float32)


def test_nano_candidate_segments_generate_concurrently_and_join_in_text_order():
    service = DeterministicVoiceCloneService()
    chunks = ["first", "second", "third"]
    audio = asyncio.run(service._generate_nano_candidate(
        chunks=chunks,
        prompt_text="reference transcript",
        reference_path=None,
        cfg=2.0,
        candidate_seed=100,
        retry_badcase=True,
        retry_max_times=3,
        retry_ratio_threshold=6.0,
        reference_mode="hifi",
        reference_latents=b"cached-latents",
        sample_rate=24000,
        silence_ms=0,
    ))

    assert service.max_active == len(chunks)
    assert {call[0]: call[1] for call in service.calls} == {
        "first": 100,
        "second": 101,
        "third": 102,
    }
    assert all(call[2] == b"cached-latents" for call in service.calls)
    expected = np.concatenate([
        np.full(4, seed / 10.0, dtype=np.float32) for seed in (1, 2, 3)
    ])
    np.testing.assert_array_equal(audio, expected)


def test_nano_batch_failure_cancels_sibling_tasks():
    async def run():
        sibling_cancelled = asyncio.Event()

        async def fail():
            await asyncio.sleep(0.01)
            raise ValueError("expected test failure")

        async def sibling():
            try:
                await asyncio.Event().wait()
            finally:
                sibling_cancelled.set()

        try:
            await _gather_cancel_on_error([fail(), sibling()])
        except ValueError as exc:
            assert str(exc) == "expected test failure"
        else:
            raise AssertionError("the failing task did not propagate its exception")
        assert sibling_cancelled.is_set()

    asyncio.run(run())
