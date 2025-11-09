Excellent — here’s a new companion document specifically focused on understanding and controlling audio generation behavior in your ACE-Step API using the /generate endpoint parameters.

This guide explains how every key parameter affects music quality, style, coherence, emotion, and diversity — so you can precisely tune the output.

⸻

📘 Download link:
👉 Download ACE_STEP_AUDIO_CONTROL_GUIDE.md￼

⸻

🎛️ ACE-Step Audio Generation Control Guide

(Fine-tuning output quality, emotion, and style using /generate parameters)

This guide is designed to help you master control over how ACE-Step produces music — balancing creativity, coherence, and fidelity — by manipulating the FastAPI parameters in the /generate endpoint.

⸻

🧠 1️⃣ Core Generation Overview

ACE-Step uses text-to-music diffusion — transforming textual and lyrical input into waveforms through iterative refinement.
Each parameter in /generate adjusts one part of that diffusion and conditioning process.

Think of it as:
	•	🎤 Prompt → Defines the musical identity
	•	📝 Lyrics → Structures rhythm, tone, emotion
	•	⚙️ Guidance → Controls fidelity vs creativity
	•	🧮 Steps + Seeds → Affect consistency and detail
	•	🧠 ERG / OSS → Manage variation and stability

⸻

🎚️ 2️⃣ Master Parameters Breakdown

🎶 Audio Duration (audio_duration)
	•	Controls: Total length (seconds)
	•	Effect: Longer durations = more musical structure but higher VRAM use
	•	Recommended ranges:
	•	30–60 → short clips
	•	90–120 → full songs
	•	Tip: A5000 GPUs handle up to ~90s comfortably

⸻

🧾 Prompt (prompt)
	•	Controls: Global style, mood, instrumentation, and tempo
	•	Examples:
	•	"lofi chill beat, rainy night, warm piano, deep bass"
	•	"epic orchestral, cinematic tension, strings and brass"
	•	"k-pop, upbeat, dance, electronic synth"

Prompt writing tips:
	•	Use comma-separated tags (acts like “embedding hints”).
	•	Avoid full sentences.
	•	Place mood or genre keywords first.
	•	Add up to 10 descriptors.

⸻

🎤 Lyrics (lyrics)
	•	Controls: Phonetic structure, rhythm, emotional tone
	•	Supports [intro], [verse], [chorus], [bridge], [outro]
	•	Multilingual — English, Chinese, Korean, Japanese, Spanish, etc.
	•	Tip: Keep ~8–12 syllables per line for rhythmic coherence.

⸻

⚙️ 3️⃣ Diffusion and Guidance Parameters

🪄 Inference Steps (infer_step)
	•	Controls: How many diffusion refinements are applied
	•	Range: 20–120
	•	Effect:
	•	Higher = more coherent & detailed
	•	Lower = faster, sometimes noisier
	•	Recommended: 50–70 for A5000

⸻

🎚️ Guidance Scale (guidance_scale)
	•	Controls: Strength of prompt & lyric adherence
	•	Range: 1–25
	•	Effect:
	•	Low (5–10): loose, creative, more noise
	•	Medium (12–17): balanced clarity
	•	High (18–25): strict fidelity, robotic if too high
	•	Recommended: 15.0

⸻

⚙️ CFG Type (cfg_type)
	•	Options: "apg", "cfg", "cfg_star"
	•	Effects:
	•	"apg" — Adaptive guidance (balanced realism & creativity) ✅ Recommended
	•	"cfg" — Classic classifier-free guidance (strong adherence)
	•	"cfg_star" — Advanced guidance for stable instrumentals

⸻

🔄 Scheduler Type (scheduler_type)
	•	Options: "euler", "heun", "pingpong"
	•	Effects:
	•	"euler" — Fast, clean diffusion ✅ Recommended default
	•	"heun" — Smoother audio, slower
	•	"pingpong" — Dynamic variation, creative artifacts

⸻

🌊 Omega Scale (omega_scale)
	•	Controls: Balance of smoothness vs expressiveness
	•	Typical: 8–12
	•	Higher values: more dynamic range and contrast
	•	Lower values: smoother, but flatter sound

⸻

🧮 4️⃣ Seed and Sampling Controls

🎲 Seeds (actual_seeds)
	•	Controls: Random noise initialization for deterministic output
	•	Example: [135767468]
	•	Use cases:
	•	Same seed = reproducible track
	•	Multiple seeds = batch variations
	•	Tip: Change seed to explore creative variants of same lyrics/prompt.

⸻

🧭 OSS Steps (oss_steps)
	•	Controls: Which diffusion steps are used explicitly
	•	Default: [] (all steps)
	•	Example: [1, 5, 10, 20, 30, 40, 50, 60]
	•	Useful for partial generation or debug control.
	•	Advanced users can use this to shorten inference time.

⸻

🧠 5️⃣ ERG (Entropy Rectifying Guidance)

ERG adds controlled stochasticity into text/lyric embeddings, encouraging diversity and emotion.

Parameter	Meaning	Effect
use_erg_tag	Randomizes prompt embeddings	More diverse genre texture
use_erg_lyric	Randomizes lyric embeddings	More expressive phrasing
use_erg_diffusion	Adds variety in denoising	Less repetitive loops

💡 Recommended settings:
All three set to true for organic, expressive generation.
Set to false for precise reproductions or benchmarking.

⸻

🌀 6️⃣ Advanced Guidance Dynamics

guidance_interval
	•	Fraction of steps where guidance is active (0.0–1.0)
	•	Example: 0.5 → guidance used during 50% of diffusion
	•	Higher values: more faithful to prompt
	•	Lower values: freer improvisation

⸻

guidance_interval_decay
	•	How fast guidance weakens during diffusion
	•	Range: 0.0–1.0
	•	0.0: Constant adherence
	•	0.5+: Gradual shift to creativity
	•	Combine with min_guidance_scale for controlled fade-outs.

⸻

min_guidance_scale
	•	Minimum guidance strength after decay
	•	Range: 0–10
	•	Example: min_guidance_scale: 3.0 keeps it semi-grounded

⸻

🎛️ Double Condition Guidance

Allows separate control for tags and lyrics.

Parameter	Role	Recommended
guidance_scale_text	Extra scale for tag embeddings	0.0 (disabled) or >1.0 for double guidance
guidance_scale_lyric	Extra scale for lyric embeddings	Use if lyrics dominate over style


⸻

🎧 7️⃣ Precision and Performance

Parameter	Default	Effect
bf16	true	Uses bfloat16 for faster inference (safe for A5000)
torch_compile	false	Compiles model for speed, requires Triton (enable if stable)
device_id	0	GPU index


⸻

🗂️ 8️⃣ Output Control

Parameter	Description	Example
output_path	Destination path for saved file	/workspace/ace-step-api/generated/song.wav
Format	Default is .wav	Conversion to .mp3 supported manually
Multiple outputs	Use different seeds or filenames	Creates separate files per run


⸻

🔬 9️⃣ Recommended Parameter Recipes

🎵 Balanced Realism (Default)

{
  "infer_step": 60,
  "guidance_scale": 15.0,
  "cfg_type": "apg",
  "scheduler_type": "euler",
  "omega_scale": 10.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true
}

Produces emotionally rich, rhythmic, and coherent tracks.

⸻

🎶 Experimental / Freestyle

{
  "guidance_scale": 9.0,
  "guidance_interval": 0.3,
  "guidance_interval_decay": 0.5,
  "min_guidance_scale": 2.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true
}

Looser adherence, more creativity, ideal for “jam session” vibes.

⸻

🎧 High Fidelity / Precise

{
  "guidance_scale": 18.0,
  "cfg_type": "cfg_star",
  "use_erg_tag": false,
  "use_erg_lyric": false,
  "use_erg_diffusion": false
}

Tight synchronization to prompt/lyrics — best for structured songs or covers.

⸻

📊 10️⃣ Optimization for RTX A5000

Setting	Recommended	Notes
bf16: true	✅ Faster inference	
audio_duration: ≤ 90	✅ Prevents OOM	
infer_step: 50–60	✅ Balance of speed/detail	
use_erg_diffusion: true	✅ Adds richness without extra memory	
torch_compile: false	⚠️ Optional, only if stable	


⸻

💡 11️⃣ Practical Tips
	•	Keep audio_duration and infer_step proportional (longer → more steps).
	•	Use same seed to compare parameter differences.
	•	Lower guidance_scale = freer sound.
	•	Raise omega_scale = more punchy transient detail.
	•	Use ERG on when exploring new genres or moods.
	•	Save your successful parameter sets in JSON for reuse.

⸻

✅ 12️⃣ Summary

Category	What It Controls
🎶 prompt, lyrics	Core musical content
⚙️ infer_step, guidance_scale, cfg_type	Diffusion accuracy
🧠 use_erg_*, guidance_interval_*	Expression and variation
🧮 actual_seeds, oss_steps	Reproducibility and sampling
💾 output_path	Save location
🎧 bf16, torch_compile	Performance and precision


⸻

🎵 With this guide, you can now precisely craft musical emotion, structure, and diversity — directly from your FastAPI endpoint.

Would you like me to also generate a “parameter preset library” JSON file (e.g., presets.json) that you can load dynamically in your app to switch between Balanced, Freestyle, and Precision modes?