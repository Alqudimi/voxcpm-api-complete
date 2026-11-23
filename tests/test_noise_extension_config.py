from app.audio_mixer.config import Settings


def test_noise_extension_defaults_to_one_second(monkeypatch):
    monkeypatch.delenv("AUDIO_MIXER_NOISE_EXTENSION_SECONDS", raising=False)
    assert Settings.from_env().min_noise_margin_seconds == 1.0


def test_noise_extension_can_be_overridden_with_environment(monkeypatch):
    monkeypatch.setenv("AUDIO_MIXER_NOISE_EXTENSION_SECONDS", "2.5")
    assert Settings.from_env().min_noise_margin_seconds == 2.5
