import json
import random
import subprocess
from pathlib import Path

from .errors import ProcessingError, ValidationError

# صيغ الإخراج الآمنة والمختبرة. الإدخال يقبله FFmpeg بالكامل عبر ffprobe.
OUTPUT_FORMATS = {
    "wav": {
        "suffix": ".wav",
        "media_type": "audio/wav",
        "codec_args": ["-c:a", "pcm_s16le", "-f", "wav"],
    },
    "ogg": {
        "suffix": ".ogg",
        "media_type": "audio/ogg",
        "codec_args": ["-c:a", "libvorbis", "-q:a", "5", "-f", "ogg"],
    },
    "opus": {
        "suffix": ".opus",
        "media_type": "audio/ogg",
        "codec_args": ["-c:a", "libopus", "-b:a", "128k", "-vbr", "on", "-f", "opus"],
    },
    "mp3": {
        "suffix": ".mp3",
        "media_type": "audio/mpeg",
        "codec_args": ["-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3"],
    },
    "flac": {
        "suffix": ".flac",
        "media_type": "audio/flac",
        "codec_args": ["-c:a", "flac", "-f", "flac"],
    },
    "m4a": {
        "suffix": ".m4a",
        "media_type": "audio/mp4",
        "codec_args": ["-c:a", "aac", "-b:a", "192k", "-f", "ipod"],
    },
    "aac": {
        "suffix": ".aac",
        "media_type": "audio/aac",
        "codec_args": ["-c:a", "aac", "-b:a", "192k", "-f", "adts"],
    },
    "webm": {
        "suffix": ".webm",
        "media_type": "audio/webm",
        "codec_args": ["-c:a", "libopus", "-b:a", "128k", "-f", "webm"],
    },
}


class FFmpeg:
    def __init__(self, ffmpeg="ffmpeg", ffprobe="ffprobe"):
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    @staticmethod
    def normalize_output_format(output_format="wav"):
        value = (output_format or "wav").lower().strip().lstrip(".")
        if value not in OUTPUT_FORMATS:
            supported = ", ".join(sorted(OUTPUT_FORMATS))
            raise ValidationError(
                f"Unsupported output format '{value}'. Supported formats: {supported}"
            )
        return value

    @staticmethod
    def output_spec(output_format="wav"):
        return OUTPUT_FORMATS[FFmpeg.normalize_output_format(output_format)]

    def _run(self, args, timeout=120):
        try:
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
            )
        except FileNotFoundError as e:
            raise ProcessingError("FFmpeg executable is unavailable") from e
        except subprocess.TimeoutExpired as e:
            raise ProcessingError("FFmpeg operation timed out") from e

    def capabilities(self):
        v = self._run([self.ffmpeg, "-version"])
        enc = self._run([self.ffmpeg, "-hide_banner", "-encoders"], 30)
        mux = self._run([self.ffmpeg, "-hide_banner", "-muxers"], 30)
        available = v.returncode == 0
        return {
            "ffmpeg_version": v.stdout.splitlines()[0] if v.stdout else "unknown",
            "available": available,
            "audio_encoders": enc.stdout.splitlines()[:200],
            "muxers": mux.stdout.splitlines()[:200],
            "supported_input_formats": "all formats supported by the installed FFmpeg/ffprobe",
            "supported_output_formats": sorted(OUTPUT_FORMATS) if available else [],
        }

    def probe(self, path: Path):
        r = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "format=duration,format_name:stream=codec_name,codec_type,sample_rate,channels,channel_layout,bits_per_sample,bit_rate",
                "-of",
                "json",
                str(path),
            ],
            30,
        )
        if r.returncode != 0:
            raise ValidationError("Invalid or unsupported audio file")
        try:
            d = json.loads(r.stdout)
            s = d.get("streams", [{}])[0]
            f = d.get("format", {})
            dur = float(f.get("duration") or s.get("duration") or 0)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            raise ValidationError("Audio metadata could not be read") from e
        if s.get("codec_type") != "audio" or dur <= 0:
            raise ValidationError("Audio stream is missing or has no duration")
        return {
            "duration": dur,
            "sample_rate": int(s.get("sample_rate") or 0),
            "channels": int(s.get("channels") or 0),
            "codec": s.get("codec_name"),
            "format": f.get("format_name"),
            "bit_depth": int(s.get("bits_per_sample") or 0),
            "bit_rate": int(s.get("bit_rate") or 0),
        }

    def mix(
        self,
        source: Path,
        noise: Path | None,
        output: Path,
        source_gain: float = 0,
        noise_gain: float = -18,
        crossfade: float = 0.5,
        variation_seed: int | None = None,
        output_format: str = "wav",
        noise_extension_seconds: float = 2.0,
        timeout=120,
        source_metadata=None,
        noise_metadata=None,
    ):
        output_format = self.normalize_output_format(output_format)
        output_spec = self.output_spec(output_format)
        # Callers that have already validated inputs can pass metadata to avoid
        # spawning ffprobe again on the latency-sensitive mixing path.
        source_metadata = source_metadata or self.probe(source)
        dur = source_metadata["duration"]
        args = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
        ]
        if noise:
            args += ["-stream_loop", "-1", "-i", str(noise)]
        # Extend the source with silence so the noise tail remains audible.
        extension = max(0.0, noise_extension_seconds) if noise else 0.0
        output_duration = dur + extension
        prepared_duration = output_duration if noise else 0.0
        noise_offset = 0.0
        if noise and variation_seed is not None:
            noise_metadata = noise_metadata or self.probe(noise)
            noise_duration = noise_metadata["duration"]
            rng = random.Random(variation_seed)
            max_offset = max(0.0, noise_duration - min(noise_duration, prepared_duration))
            noise_offset = rng.uniform(0.0, max_offset) if max_offset else 0.0
        if noise:
            fc = f"[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={source_gain}dB,apad=pad_dur={extension:.6f},atrim=duration={output_duration:.6f}[s];[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,atrim=start={noise_offset:.6f}:duration={prepared_duration:.6f},asetpts=N/SR/TB,volume={noise_gain}dB[n];[s][n]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.88[out]"
        else:
            fc = f"[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={source_gain}dB,alimiter=limit=0.88[out]"
        args += [
            "-filter_complex",
            fc,
            "-map",
            "[out]",
            "-t",
            f"{output_duration:.6f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            *output_spec["codec_args"],
            str(output),
        ]
        r = self._run(args, max(timeout, int(dur * 4) + 30))
        if r.returncode != 0:
            raise ProcessingError(
                f"Audio mixing failed for format {output_format}: {r.stderr[-500:]}"
            )
        return {
            "source_duration": dur,
            "output_duration": output_duration,
            "extension_seconds": extension,
            "prepared_noise_duration": prepared_duration,
            "variation_seed": variation_seed,
            "noise_offset_seconds": round(noise_offset, 6),
            "variation_method": "seeded_noise_window_offset",
            "output_format": output_format,
            "output_codec": output_spec["codec_args"][1],
        }
