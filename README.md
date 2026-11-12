# yt-voice-agent

Free Hinglish AI voiceover for YouTube videos. Script do, video do, voice ke saath final video lo.

- Voice: [edge-tts](https://github.com/rany2/edge-tts) (Microsoft Edge neural voices, free, no API key)
- Chalta hai: GitHub Actions pe (free) ya Termux/PC pe

## Files

| File | Kya hai |
|---|---|
| `input/script.txt` | Bolne wali lines. Har section ke beech ek khaali line. Section 1 = scene 1. |
| `input/scenes.txt` | Har scene ki length seconds mein, ek line mein ek. Voice har scene ke start pe sync hoti hai. |
| `input/video.mp4` | Bina awaaz wali video |
| `voice.py` | Voice banata hai aur video mein jodta hai |

## GitHub pe (phone se bhi)

1. `input/script.txt` kholo → ✏️ edit → `[APNA RATIO]`, `[APNE TICKS]`, `[APNA R:R]` apne numbers se badlo → Commit.
2. Commit karte hi **Actions** tab mein "Make voiceover" chalega (2-3 min).
3. Kaam khatam hone pe `output/final.mp4` repo mein aa jayega (Artifacts → voiceover mein bhi).

Voice ya speed badalni ho: Actions → Make voiceover → **Run workflow** → voice/rate bharo.

## Termux / PC pe

```bash
pkg install python ffmpeg      # Termux (PC pe: ffmpeg install karo)
pip install -r requirements.txt
python voice.py                          # output/final.mp4
python voice.py --voice hi-IN-MadhurNeural --rate +8%
python voice.py --list                   # saari Indian voices
```


## Realistic voice: Sarvam AI (recommended)

Sarvam Bulbul v3 Indian model hai, Hinglish natural bolta hai. Signup pe ₹1,000 free credit milta hai, aur ek ~3 min video ~₹8 ka padta hai (lagbhag 100+ video free).

1. https://dashboard.sarvam.ai pe signup karo → API key copy karo.
2. GitHub repo → Settings → Secrets and variables → Actions → New repository secret
   - Name: `SARVAM_API_KEY`  Value: tumhari key
3. Bas. Ab har run Sarvam se banega (secret na ho toh free edge-tts pe chalega).

Voices (Run workflow → voice): male `shubh` (default), `aditya`, `rahul`, `amit`, `kabir`, `rohan`; female `priya`, `neha`, `shreya`, `ishita`, `pooja`, `kavya`.

## Free voices (edge-tts)

| Voice | |
|---|---|
| `hi-IN-MadhurNeural` | male, default — sabse natural |
| `hi-IN-SwaraNeural` | female |
| `en-IN-PrabhatNeural` / `en-IN-NeerjaNeural` | Indian English, Hindi ko robotic padhte hain |

**Script Devanagari mein likho** (English trading words jaise wall, delta, tape Roman mein reh sakte hain). Roman Hindi pe voice bhains jaisi lagti hai.

## Sync

Agar kisi section ki voice scene se lambi hai toh woh max 1.35x tak tez ho jayegi aur log mein warning aayegi. Zyada lambi ho toh text chhota karo ya `scenes.txt` mein us scene ka time badhao (aur video bhi utni lambi honi chahiye).

## Note

edge-tts Microsoft ki official API nahi hai. Free hai, lekin kabhi Microsoft side se badlav hua toh kuch din ke liye toot sakta hai; tab `pip install -U edge-tts` karo.
