#!/usr/bin/env python3
"""yt-voice-agent: script.txt -> Hinglish AI voiceover -> merged into your video.

Free: uses edge-tts (Microsoft Edge neural voices, no API key).

script.txt : sections separated by a blank line. Section N is spoken over scene N.
scenes.txt : one duration (seconds) per line, one per scene. Optional.
             Without it the voice is just laid over the video from the start.

Usage:
  python voice.py                          # uses input/ folder, writes output/
  python voice.py --voice hi-IN-SwaraNeural
  python voice.py --rate +5%               # speak faster
  python voice.py --list                   # list Indian voices
"""
import argparse, asyncio, os, re, subprocess, sys, json, base64, time, urllib.request, urllib.error

import edge_tts

DEFAULT_VOICE = "hi-IN-MadhurNeural"   # natural Hindi; write script in Devanagari (English terms can stay in Roman)


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ffmpeg error:\n{' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def duration(path):
    out = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path])
    return float(json.loads(out)["format"]["duration"])


async def tts(text, voice, rate, path, tries=5):
    for k in range(tries):
        try:
            await edge_tts.Communicate(text, voice, rate=rate).save(path)
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                return
        except Exception as e:
            print(f"  retry {k + 1}/{tries}: {type(e).__name__}")
        await asyncio.sleep(3 * (k + 1))
    sys.exit("edge-tts se audio nahi mila. Thodi der baad dobara chalao.")


def sarvam_tts(text, speaker, pace, path, tries=4):
    """Sarvam AI Bulbul v3 - most natural Hindi/Hinglish. Needs SARVAM_API_KEY."""
    body = json.dumps({"text": text, "language_code": "hi-IN", "model": "bulbul:v3",
                       "speaker": speaker, "pace": pace, "speech_sample_rate": 44100,
                       "output_audio_codec": "wav"}).encode()
    for k in range(tries):
        req = urllib.request.Request("https://api.sarvam.ai/text-to-speech", data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "api-subscription-key": os.environ["SARVAM_API_KEY"]})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            with open(path, "wb") as f:
                f.write(base64.b64decode(data["audios"][0]))
            return
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")[:300]
            if e.code in (400, 401, 403):
                sys.exit(f"Sarvam error {e.code}: {msg}")
            print(f"  retry {k + 1}/{tries}: HTTP {e.code} {msg}")
        except Exception as e:
            print(f"  retry {k + 1}/{tries}: {type(e).__name__}")
        time.sleep(3 * (k + 1))
    sys.exit("Sarvam se audio nahi mila.")


def pace_from_rate(rate):
    m = re.match(r"([+-]?\d+)%", rate or "+0%")
    return round(max(0.5, min(2.0, 1 + (int(m.group(1)) if m else 0) / 100)), 2)


async def list_voices():
    for v in await edge_tts.list_voices():
        if v["Locale"] in ("hi-IN", "en-IN"):
            print(f'{v["ShortName"]:28} {v["Gender"]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="input/script.txt")
    ap.add_argument("--scenes", default="input/scenes.txt")
    ap.add_argument("--video", default="input/video.mp4")
    ap.add_argument("--out", default="output")
    ap.add_argument("--engine", default=os.environ.get("ENGINE") or "auto",
                    help="auto (sarvam if SARVAM_API_KEY set, else edge), sarvam, edge")
    ap.add_argument("--voice", default=os.environ.get("VOICE") or "")
    ap.add_argument("--rate", default=os.environ.get("RATE") or "+0%")
    ap.add_argument("--max-speedup", type=float, default=1.35,
                    help="if a section is longer than its scene, speed it up at most this much")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        asyncio.run(list_voices()); return
    if a.engine == "auto":
        a.engine = "sarvam" if os.environ.get("SARVAM_API_KEY") else "edge"
    if a.engine == "sarvam" and not os.environ.get("SARVAM_API_KEY"):
        sys.exit("SARVAM_API_KEY set nahi hai.")
    if not a.voice:
        a.voice = "shubh" if a.engine == "sarvam" else DEFAULT_VOICE

    text = open(a.script, encoding="utf-8").read().strip()
    left = re.findall(r"\[[^\]]+\]", text)
    if left:
        sys.exit(f"script.txt mein placeholders bache hain: {', '.join(sorted(set(left)))}\n"
                 "Inhe apne real values se replace karo, warna voice inhe padh degi.")
    sections = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "tmp"); os.makedirs(tmp, exist_ok=True)

    scenes = None
    if a.scenes and os.path.exists(a.scenes):
        scenes = [float(x) for x in open(a.scenes).read().split()]
        if len(scenes) != len(sections):
            sys.exit(f"script.txt mein {len(sections)} sections hain lekin scenes.txt mein {len(scenes)} durations. "
                     "Dono barabar hone chahiye.")

    print(f"engine={a.engine} voice={a.voice} rate={a.rate} sections={len(sections)}")
    parts = []
    for i, s in enumerate(sections, 1):
        if a.engine == "sarvam":
            raw = os.path.join(tmp, f"raw_{i:02d}.wav")
            sarvam_tts(s.strip(), a.voice, pace_from_rate(a.rate), raw)
        else:
            raw = os.path.join(tmp, f"raw_{i:02d}.mp3")
            asyncio.run(tts(" ".join(s.split()), a.voice, a.rate, raw))
        if scenes is None:
            parts.append(raw); continue
        d = scenes[i - 1]
        alen = duration(raw)
        target = max(1.0, d - 0.6)            # 0.3s lead-in + 0.3s breathing room
        tempo = 1.0
        if alen > target:
            tempo = min(a.max_speedup, alen / target)
            if alen / target > a.max_speedup:
                print(f"  ! section {i}: voice {alen:.1f}s vs scene {d:.1f}s - "
                      f"cut ho jayega. Text chhota karo ya scene lamba karo.")
        out = os.path.join(tmp, f"sec_{i:02d}.wav")
        af = (f"atempo={tempo:.4f}," if tempo > 1.0 else "") + \
             f"adelay=300|300,apad=whole_dur={d},atrim=0:{d}"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af", af, "-ar", "48000", "-ac", "2", out])
        print(f"  section {i}: voice {alen:.1f}s, scene {d:.1f}s, speed x{tempo:.2f}")
        parts.append(out)

    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        for p in parts: f.write(f"file '{os.path.abspath(p)}'\n")
    voice = os.path.join(a.out, "voice.m4a")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
        "-c:a", "aac", "-b:a", "160k", voice])
    print(f"voice: {voice} ({duration(voice):.1f}s)")

    if a.video and os.path.exists(a.video):
        final = os.path.join(a.out, "final.mp4")
        vlen, alen = duration(a.video), duration(voice)
        vf = []
        if alen > vlen + 0.05:  # voice longer: hold last frame
            vf = ["-vf", f"tpad=stop_mode=clone:stop_duration={alen - vlen + 0.5:.2f}"]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", a.video, "-i", voice,
               "-map", "0:v", "-map", "1:a"] + \
              (vf + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"] if vf else ["-c:v", "copy"]) + \
              ["-c:a", "aac", "-b:a", "160k"] + ([] if vf else ["-af", "apad", "-shortest"]) + [final]
        sh(cmd)
        print(f"final video: {final} ({duration(final):.1f}s)")


if __name__ == "__main__":
    main()
