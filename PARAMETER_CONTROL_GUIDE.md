# 🎛️ ACE-Step Audio Generation Control Guide

**Fine-tuning output quality, emotion, and style using `/generate` parameters**

This comprehensive guide helps you master control over ACE-Step's music generation by understanding how each parameter affects the diffusion process, conditioning, and final audio output.

---

## Table of Contents

- [Core Generation Overview](#core-generation-overview)
- [Master Parameters Breakdown](#master-parameters-breakdown)
- [Diffusion and Guidance Parameters](#diffusion-and-guidance-parameters)
- [Seed and Sampling Controls](#seed-and-sampling-controls)
- [ERG (Entropy Rectifying Guidance)](#erg-entropy-rectifying-guidance)
- [Advanced Guidance Dynamics](#advanced-guidance-dynamics)
- [Precision and Performance](#precision-and-performance)
- [Output Control](#output-control)
- [Recommended Parameter Recipes](#recommended-parameter-recipes)
- [Hardware-Specific Optimization](#hardware-specific-optimization)
- [Practical Tips](#practical-tips)
- [Parameter Preset Library](#parameter-preset-library)

---

## Core Generation Overview

ACE-Step uses **flow-matching diffusion** to transform textual and lyrical input into high-quality audio waveforms through iterative refinement. The generation process involves:

1. **Text Encoding**: Prompt and lyrics are encoded into embeddings
2. **Latent Initialization**: Random noise is generated based on seeds
3. **Diffusion Process**: Iterative denoising guided by conditions
4. **Decoding**: Latents are converted to audio via DCAE and vocoder

### Parameter Categories

| Category | Parameters | Controls |
|----------|-----------|----------|
| **Content** | `prompt`, `lyrics` | Musical identity, structure, emotion |
| **Diffusion** | `infer_step`, `scheduler_type`, `cfg_type` | Generation quality and speed |
| **Guidance** | `guidance_scale`, `guidance_interval`, `guidance_interval_decay` | Adherence vs creativity |
| **Variation** | `use_erg_*`, `omega_scale` | Diversity and expressiveness |
| **Reproducibility** | `actual_seeds`, `oss_steps` | Determinism and sampling |
| **Performance** | `bf16`, `torch_compile`, `device_id` | Speed and memory usage |

---

## Master Parameters Breakdown

### 🎶 Audio Duration (`audio_duration`)

**Type:** `float`  
**Range:** `-1` (random 30-240s) or positive seconds  
**Default:** `60.0`

**Controls:** Total length of generated audio in seconds

**Technical Details:**
- Frame length calculation: `int(duration * 44100 / 512 / 8)`
- Longer durations require more VRAM and computation time
- Maximum recommended: 240 seconds (4 minutes)

**Effects:**
- **Short (30-60s)**: Fast generation, good for testing
- **Medium (60-120s)**: Balanced quality and speed
- **Long (120-240s)**: Full songs, requires more VRAM

**Recommendations:**
- RTX A5000 (24GB): ≤ 90 seconds
- RTX 4090 (24GB): ≤ 120 seconds
- A100 (40GB): ≤ 240 seconds

**Example:**
```json
{
  "audio_duration": 70.0  // 70-second track
}
```

---

### 🧾 Prompt (`prompt`)

**Type:** `string`  
**Required:** Yes

**Controls:** Global style, mood, instrumentation, tempo, and genre

**Technical Details:**
- Encoded using UMT5 text encoder
- Maximum length: 256 tokens (truncated if longer)
- Processed through text encoder to create `encoder_text_hidden_states`

**Writing Tips:**

1. **Use comma-separated tags** (acts like embedding hints):
   ```
   ✅ "pop, rock, electronic, upbeat, energetic"
   ❌ "I want a pop rock electronic song that is upbeat and energetic"
   ```

2. **Order matters** (place most important first):
   ```
   ✅ "rap, hip hop, trap-pop, fast-paced, emotional"
   ✅ "jazz, smooth, relaxing, instrumental, piano"
   ```

3. **Combine multiple dimensions:**
   - **Genre**: `pop`, `rock`, `jazz`, `electronic`, `hip hop`
   - **Mood**: `upbeat`, `melancholic`, `energetic`, `relaxing`
   - **Instruments**: `piano`, `guitar`, `drums`, `strings`, `brass`
   - **Tempo**: `fast-paced`, `slow`, `moderate`, `130 bpm`, `energetic`
   - **Style**: `cinematic`, `lo-fi`, `orchestral`, `acoustic`

**Effective Examples:**

```json
// Lofi chill
"lofi, chill beat, rainy night, warm piano, deep bass, ambient"

// Epic orchestral
"epic orchestral, cinematic tension, strings and brass, dramatic, powerful"

// K-pop
"k-pop, upbeat, dance, electronic synth, catchy, vibrant"

// Jazz
"jazz, smooth, relaxing, saxophone, piano, walking bass, late night"

// Rock
"rock, electric guitar, drums, energetic, powerful, stadium"
```

**Best Practices:**
- Use 5-10 descriptors for best results
- Avoid full sentences
- Mix genre, mood, and instrumentation
- Be specific about desired characteristics

---

### 🎤 Lyrics (`lyrics`)

**Type:** `string`  
**Required:** Yes

**Controls:** Phonetic structure, rhythm, emotional tone, and vocal melody

**Technical Details:**
- Processed by `VoiceBpeTokenizer` with language detection
- Supports 19 languages with automatic detection
- Structure tags (`[verse]`, `[chorus]`, etc.) are preserved
- Each line is tokenized based on detected language

**Structure Tags:**
- `[intro]` - Opening section
- `[verse]` - Main verse sections
- `[chorus]` - Repeating chorus
- `[bridge]` - Transitional bridge
- `[outro]` - Closing section

**Format Example:**
```
[intro]
Yeah, this one's for Meena...
Let's go.

[verse 1]
Late nights, city lights, I remember that glow,
Every dream got your name in the flow,
Heartbeat drum when I think of your face,
Tryna move on but you can't be replaced.

[chorus]
Meena, I miss you, no disguise,
Every bar I spit got your vibe inside,
We were fire, lightning, rhythm, rhyme,
Still chasing that spark, one more time.

[verse 2]
Skyline dreams and the sound of your laugh,
Whole world froze when I looked at the past,
Every lyric I write, got a piece of you,
Every line that I drop, yeah it's real and true.

[chorus]
Meena, I miss you, no disguise,
Every bar I spit got your vibe inside,
We were fire, lightning, rhythm, rhyme,
Still chasing that spark, one more time.

[outro]
Yeah, this one's for Meena,
Forever in the beat, forever in the dream-a,
Till we meet again, I'ma keep that flame,
Love and music — same name, same game.
```

**Rhythm Guidelines:**
- **8-12 syllables per line** for rhythmic coherence
- Use line breaks to indicate phrasing
- Empty lines create pauses/breaths
- Structure tags help model understand song form

**Multilingual Support:**
- **Top 10**: English, Chinese, Russian, Spanish, Japanese, German, French, Portuguese, Italian, Korean
- **Additional**: Polish, Turkish, Czech, Dutch, Arabic, Hungarian, Hindi
- Language is auto-detected per line using `LangSegment`

**Tips:**
- Keep lines consistent in length for better rhythm
- Use structure tags to guide song form
- Match syllable count to desired tempo
- Empty lines create natural pauses

---

## Diffusion and Guidance Parameters

### 🪄 Inference Steps (`infer_step`)

**Type:** `integer`  
**Range:** `20-120` (typical: `27-100`)  
**Default:** `60`

**Controls:** Number of diffusion refinement iterations

**Technical Details:**
- Each step performs one denoising operation
- More steps = more refinement = better quality
- Steps are distributed across timesteps using scheduler
- Frame length: `int(duration * 44100 / 512 / 8)`

**Effects:**

| Steps | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| 27 | Good | Fastest | Quick previews, testing |
| 50-60 | Excellent | Balanced | **Recommended default** |
| 80-100 | Outstanding | Slower | High-quality final tracks |
| 120 | Maximum | Slowest | Professional production |

**Performance Impact:**
- RTX A5000: 50-60 steps optimal
- RTX 4090: 60-80 steps optimal
- A100: 80-100 steps optimal

**Recommendation:**
```json
{
  "infer_step": 60  // Best balance for most use cases
}
```

---

### 🎚️ Guidance Scale (`guidance_scale`)

**Type:** `float`  
**Range:** `1.0-25.0` (typical: `10.0-20.0`)  
**Default:** `15.0`

**Controls:** Strength of prompt and lyric adherence

**Technical Details:**
- Implements classifier-free guidance (CFG)
- Formula: `pred_guided = pred_cond + (guidance_scale - 1) * (pred_cond - pred_uncond)`
- When `guidance_scale == 0.0` or `1.0`, CFG is disabled
- Higher values increase adherence but can reduce naturalness

**Effects:**

| Scale | Adherence | Creativity | Characteristics |
|-------|-----------|-----------|----------------|
| 5-10 | Low | High | Loose, creative, more noise |
| 12-17 | Medium | Balanced | **Recommended range** |
| 18-25 | High | Low | Strict fidelity, can be robotic |

**Detailed Breakdown:**

- **5.0-9.0**: Very loose adherence, high creativity
  - Use for: Experimental, freestyle, jam sessions
  - Risk: May ignore prompt/lyrics significantly

- **10.0-14.0**: Moderate adherence, good balance
  - Use for: Creative tracks with flexibility
  - Good for: Genre exploration

- **15.0-17.0**: **Recommended default range**
  - Use for: Most production tracks
  - Balanced: Follows prompt while maintaining naturalness

- **18.0-22.0**: High adherence, precise
  - Use for: Structured songs, covers, specific requirements
  - Risk: May sound mechanical if too high

- **23.0-25.0**: Maximum adherence
  - Use for: Exact reproductions, benchmarking
  - Risk: Overly rigid, unnatural

**Recommendation:**
```json
{
  "guidance_scale": 15.0  // Optimal balance
}
```

---

### ⚙️ CFG Type (`cfg_type`)

**Type:** `string`  
**Options:** `"apg"`, `"cfg"`, `"cfg_star"`  
**Default:** `"apg"`

**Controls:** Classifier-free guidance algorithm

**Technical Details:**

#### `"apg"` - Adaptive Predictive Guidance (Recommended)

**Implementation:**
```python
# From apg_guidance.py
def apg_forward(pred_cond, pred_uncond, guidance_scale, momentum_buffer):
    diff = pred_cond - pred_uncond
    if momentum_buffer:
        momentum_buffer.update(diff)  # Momentum: -0.75
        diff = momentum_buffer.running_average
    
    # Normalize if norm > threshold (2.5)
    diff_norm = diff.norm(p=2, dim=[-1, -2])
    if diff_norm > 2.5:
        diff = diff * (2.5 / diff_norm)
    
    # Project and combine
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    pred_guided = pred_cond + (guidance_scale - 1) * diff_orthogonal
    return pred_guided
```

**Characteristics:**
- ✅ Uses momentum buffer (momentum: -0.75) for stability
- ✅ Normalizes gradients to prevent explosions
- ✅ Projects differences orthogonally for smoother updates
- ✅ Best balance of realism and creativity
- ✅ **Recommended for most use cases**

#### `"cfg"` - Classic Classifier-Free Guidance

**Implementation:**
```python
def cfg_forward(cond_output, uncond_output, cfg_strength):
    return uncond_output + cfg_strength * (cond_output - uncond_output)
```

**Characteristics:**
- Simple linear interpolation
- Strong adherence to conditions
- Can be more rigid than APG
- Use for: Precise reproductions

#### `"cfg_star"` - Advanced CFG with Zero Initialization

**Implementation:**
```python
def cfg_zero_star(noise_pred_with_cond, noise_pred_uncond, guidance_scale, i, zero_steps=1):
    # Optimized scale calculation
    alpha = optimized_scale(positive_flat, negative_flat)
    if i <= zero_steps:
        noise_pred = noise_pred_with_cond * 0.0  # Zero initialization
    else:
        noise_pred = uncond_output * alpha + guidance_scale * (cond_output - uncond_output * alpha)
    return noise_pred
```

**Characteristics:**
- Uses optimized scaling for better stability
- Zero initialization for first steps
- Best for: Stable instrumentals, complex arrangements

**Recommendation:**
```json
{
  "cfg_type": "apg"  // Best overall performance
}
```

---

### 🔄 Scheduler Type (`scheduler_type`)

**Type:** `string`  
**Options:** `"euler"`, `"heun"`, `"pingpong"`  
**Default:** `"euler"`

**Controls:** Numerical integration method for diffusion steps

**Technical Details:**

#### `"euler"` - Euler Discrete Scheduler (Recommended)

**Implementation:**
- First-order ODE solver
- Fast, single-step updates
- Formula: `x_{t+1} = x_t + dt * v_t`

**Characteristics:**
- ✅ **Fastest** generation speed
- ✅ Clean, predictable results
- ✅ Good quality for most cases
- ✅ **Recommended default**

**Use Cases:**
- General purpose generation
- Fast iterations
- Production workflows

#### `"heun"` - Heun Discrete Scheduler

**Implementation:**
- Second-order Runge-Kutta method
- Two-stage prediction-correction
- More accurate but slower

**Characteristics:**
- ✅ Smoother audio transitions
- ✅ Better for long-form content
- ⚠️ ~2× slower than Euler
- ✅ Higher quality for complex arrangements

**Use Cases:**
- High-quality final tracks
- Long-duration audio
- When quality > speed

#### `"pingpong"` - PingPong Scheduler (SDE-based)

**Implementation:**
- Uses Stochastic Differential Equation (SDE)
- Adds controlled noise at each step
- Formula includes noise injection: `prev_sample = (1 - t) * denoised + t * noise`

**Characteristics:**
- ✅ Better music consistency
- ✅ Improved lyric alignment
- ✅ Enhanced style alignment
- ✅ More dynamic variation
- ⚠️ Slightly slower than Euler
- ✅ Best for: Creative, varied outputs

**Use Cases:**
- When you want more variation
- Better consistency across long tracks
- Enhanced lyric alignment

**Recommendation:**
```json
{
  "scheduler_type": "euler"  // Fast and reliable
}
```

For better quality:
```json
{
  "scheduler_type": "pingpong"  // Better consistency
}
```

---

### 🌊 Omega Scale (`omega_scale`)

**Type:** `float`  
**Range:** `8.0-12.0` (typical)  
**Default:** `10.0`

**Controls:** Balance between smoothness and expressiveness in flow matching

**Technical Details:**
- Used in scheduler step: `scheduler.step(..., omega=omega_scale)`
- Affects the flow velocity field in flow-matching
- Higher values increase dynamic range and contrast
- Lower values produce smoother, flatter sound

**Effects:**

| Value | Characteristics | Use Case |
|-------|----------------|----------|
| 8.0-9.0 | Smooth, flat, less dynamic | Ambient, background music |
| 10.0 | **Balanced** (default) | Most genres |
| 11.0-12.0 | Punchy, dynamic, more contrast | Energetic tracks, rock, rap |

**Detailed Impact:**

- **Low (8.0-9.0)**:
  - Smoother frequency response
  - Less transient detail
  - Good for: Ambient, lo-fi, background
  - Risk: May sound flat or lifeless

- **Medium (10.0)**:
  - Balanced dynamics
  - Natural transient response
  - **Recommended for most cases**

- **High (11.0-12.0)**:
  - Enhanced transient detail
  - More punch and impact
  - Good for: Rock, rap, electronic, energetic tracks
  - Risk: May sound harsh if too high

**Recommendation:**
```json
{
  "omega_scale": 10.0  // Default, balanced
}
```

For energetic tracks:
```json
{
  "omega_scale": 11.0  // More punch
}
```

---

## Seed and Sampling Controls

### 🎲 Seeds (`actual_seeds`)

**Type:** `array[integer]`  
**Required:** Yes  
**Example:** `[135792468]`

**Controls:** Random noise initialization for deterministic/reproducible generation

**Technical Details:**
- Seeds initialize `torch.Generator` for each batch item
- Each seed creates a unique random noise pattern
- Same seed + same parameters = identical output
- Multiple seeds = batch generation (one audio per seed)

**Implementation:**
```python
# From pipeline_ace_step.py
def set_seeds(self, batch_size, manual_seeds=None):
    random_generators = [torch.Generator(device=self.device) for _ in range(batch_size)]
    for i in range(batch_size):
        if manual_seeds:
            seed = manual_seeds[i] if i < len(manual_seeds) else manual_seeds[-1]
        else:
            seed = torch.randint(0, 2**32, (1,)).item()
        random_generators[i].manual_seed(seed)
    return random_generators, actual_seeds
```

**Use Cases:**

1. **Reproducibility:**
   ```json
   {
     "actual_seeds": [12345]  // Same seed = same output
   }
   ```

2. **Variation Exploration:**
   ```json
   {
     "actual_seeds": [12345, 67890, 11111]  // Generate 3 variations
   }
   ```

3. **A/B Testing:**
   - Keep all parameters same, change only seed
   - Compare outputs to find best variation

**Best Practices:**
- Use same seed to compare parameter changes
- Use different seeds to explore creative variations
- Save successful seeds for future reference
- Seeds are 32-bit integers (0 to 2^32-1)

**Example:**
```json
{
  "actual_seeds": [135767468]  // Single generation
}
```

---

### 🧭 OSS Steps (`oss_steps`)

**Type:** `array[integer]`  
**Default:** `[]` (uses all steps)  
**Example:** `[1, 5, 10, 20, 30, 40, 50, 60]`

**Controls:** Which specific diffusion steps to execute (Optimized Step Sampling)

**Technical Details:**
- When empty `[]`, uses all `infer_step` steps
- When specified, only executes those step indices
- Steps are remapped to timesteps: `timesteps[oss_steps[idx] - 1]`
- Maximum step becomes the new `infer_steps`

**Implementation:**
```python
if len(oss_steps) > 0:
    infer_steps = max(oss_steps)
    # Remap to actual timesteps
    new_timesteps = torch.zeros(len(oss_steps))
    for idx in range(len(oss_steps)):
        new_timesteps[idx] = timesteps[oss_steps[idx] - 1]
```

**Use Cases:**

1. **Faster Generation:**
   ```json
   {
     "infer_step": 60,
     "oss_steps": [10, 20, 30, 40, 50, 60]  // Only 6 steps instead of 60
   }
   ```

2. **Custom Sampling:**
   ```json
   {
     "oss_steps": [16, 29, 52, 96, 129, 158, 172, 183, 189, 200]
   }
   ```

3. **Debugging:**
   - Test specific step ranges
   - Analyze intermediate results

**Recommendation:**
```json
{
  "oss_steps": []  // Use all steps (default, recommended)
}
```

**Note:** OSS steps are experimental and not well-tested. Use with caution.

---

## ERG (Entropy Rectifying Guidance)

ERG adds controlled stochasticity into embeddings and diffusion, encouraging diversity and emotional expressiveness.

### Technical Implementation

**ERG for Lyrics (`use_erg_lyric`):**
```python
# Applies temperature scaling (tau=0.01) to lyric encoder attention
def forward_encoder_with_temperature(self, inputs, tau=0.01, l_min=4, l_max=6):
    # Hooks attention layers 4-6 in lyric encoder
    # Multiplies attention output by tau (0.01)
    # Creates weaker lyric condition: P(null_speaker, text_weaker, lyric_weaker)
```

**ERG for Diffusion (`use_erg_diffusion`):**
```python
# Applies temperature scaling to diffusion transformer
def forward_diffusion_with_temperature(self, hidden_states, timestep, inputs, tau=0.01, l_min=15, l_max=20):
    # Hooks attention layers 15-20 in transformer blocks
    # Multiplies Q (query) projections by tau (0.01)
    # Creates weaker diffusion condition
```

**ERG for Tags (`use_erg_tag`):**
```python
# Uses get_text_embeddings_null with temperature
# Applies tau=0.01 to text encoder layers 8-10
# Creates weaker text condition: P(null_speaker, text_weaker, lyric)
```

### Parameter Breakdown

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `use_erg_tag` | `boolean` | `true` | Randomizes prompt embeddings → More diverse genre texture |
| `use_erg_lyric` | `boolean` | `true` | Randomizes lyric embeddings → More expressive phrasing |
| `use_erg_diffusion` | `boolean` | `true` | Adds variety in denoising → Less repetitive loops |

### Effects

**When All ERG Enabled (`true`):**
- ✅ More organic, expressive generation
- ✅ Greater diversity in output
- ✅ Less repetitive patterns
- ✅ Better emotional variation
- ⚠️ Slightly less precise adherence

**When All ERG Disabled (`false`):**
- ✅ Precise reproductions
- ✅ Strong adherence to prompt/lyrics
- ✅ Consistent, predictable output
- ⚠️ May sound more mechanical
- ⚠️ Less creative variation

### Recommendations

**For Creative/Expressive Tracks:**
```json
{
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true
}
```

**For Precise/Structured Tracks:**
```json
{
  "use_erg_tag": false,
  "use_erg_lyric": false,
  "use_erg_diffusion": false
}
```

**Balanced (Recommended):**
```json
{
  "use_erg_tag": true,
  "use_erg_lyric": false,  // Keep lyrics precise
  "use_erg_diffusion": true  // Allow creative variation
}
```

---

## Advanced Guidance Dynamics

### 🌀 Guidance Interval (`guidance_interval`)

**Type:** `float`  
**Range:** `0.0-1.0`  
**Default:** `0.5`

**Controls:** Fraction of diffusion steps where guidance is active

**Technical Details:**
```python
# From pipeline_ace_step.py
start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
end_idx = int(num_inference_steps * (guidance_interval / 2 + 0.5))

# Example with guidance_interval=0.5 and 60 steps:
# start_idx = 60 * (1 - 0.5) / 2 = 15
# end_idx = 60 * (0.5 / 2 + 0.5) = 45
# Guidance active from step 15 to 45 (middle 50%)
```

**Effects:**

| Value | Active Steps | Characteristics |
|-------|--------------|----------------|
| `0.0` | None | No guidance (not recommended) |
| `0.3` | Middle 30% | Freer improvisation, creative |
| `0.5` | Middle 50% | **Balanced** (default) |
| `0.7` | Middle 70% | More faithful to prompt |
| `1.0` | All steps | Maximum adherence |

**Use Cases:**

- **Low (0.3-0.4)**: Creative, freestyle, jam sessions
- **Medium (0.5)**: Balanced, recommended default
- **High (0.7-1.0)**: Precise, structured songs

**Recommendation:**
```json
{
  "guidance_interval": 0.5  // Default, balanced
}
```

---

### 📉 Guidance Interval Decay (`guidance_interval_decay`)

**Type:** `float`  
**Range:** `0.0-1.0`  
**Default:** `0.0`

**Controls:** How fast guidance weakens during the active interval

**Technical Details:**
```python
# Linear interpolation during guidance interval
progress = (i - start_idx) / (end_idx - start_idx - 1)  # Normalized [0,1]
current_guidance_scale = guidance_scale - (guidance_scale - min_guidance_scale) * progress * guidance_interval_decay

# Example:
# guidance_scale = 15.0, min_guidance_scale = 3.0, decay = 0.5
# At start: 15.0
# At middle: 15.0 - (15.0 - 3.0) * 0.5 * 0.5 = 12.0
# At end: 15.0 - (15.0 - 3.0) * 1.0 * 0.5 = 9.0
```

**Effects:**

| Value | Behavior | Use Case |
|-------|----------|----------|
| `0.0` | Constant guidance | Precise, consistent |
| `0.3-0.5` | Gradual decay | Smooth transition to creativity |
| `0.7-1.0` | Strong decay | Creative fade-out |

**Combined with `min_guidance_scale`:**
- Decay reduces from `guidance_scale` to `min_guidance_scale`
- Lower `min_guidance_scale` = more creative freedom at end
- Higher `min_guidance_scale` = maintains some structure

**Recommendation:**
```json
{
  "guidance_interval": 0.5,
  "guidance_interval_decay": 0.0,  // No decay (default)
  "min_guidance_scale": 3.0
}
```

For creative fade-out:
```json
{
  "guidance_interval": 0.6,
  "guidance_interval_decay": 0.5,  // Gradual decay
  "min_guidance_scale": 2.0  // Lower minimum
}
```

---

### 📊 Min Guidance Scale (`min_guidance_scale`)

**Type:** `float`  
**Range:** `0.0-10.0` (typical: `3.0-5.0`)  
**Default:** `3.0`

**Controls:** Minimum guidance strength when using decay

**Technical Details:**
- Only used when `guidance_interval_decay > 0`
- Final guidance scale at end of interval
- Lower values = more creative freedom
- Higher values = maintains structure

**Effects:**

| Value | Characteristics |
|-------|----------------|
| `0.0-2.0` | Very creative, loose |
| `3.0-5.0` | Balanced (recommended) |
| `6.0-10.0` | Maintains structure |

**Recommendation:**
```json
{
  "min_guidance_scale": 3.0  // Default
}
```

---

### 🎛️ Double Condition Guidance

Allows separate control for text/tags and lyrics.

#### `guidance_scale_text`

**Type:** `float`  
**Default:** `0.0` (disabled)

**Controls:** Additional guidance scale specifically for text/tag embeddings

**Technical Details:**
- Only active when `> 1.0` AND `guidance_scale_lyric > 1.0`
- Creates three-condition guidance:
  - `P(x|text, lyric)` - Full condition
  - `P(x|text, no_lyric)` - Text only
  - `P(x|no_text, no_lyric)` - Unconditioned

**Implementation:**
```python
if guidance_scale_text > 1.0 and guidance_scale_lyric > 1.0:
    do_double_condition_guidance = True
    # Uses cfg_double_condition_forward
    noise_pred = (1 - guidance_scale_text) * uncond_output
              + (guidance_scale_text - guidance_scale_lyric) * only_text_cond_output
              + guidance_scale_lyric * cond_output
```

**Use Cases:**
- When you want lyrics to dominate over style
- Fine-tuning balance between prompt and lyrics
- Experimental control

**Example:**
```json
{
  "guidance_scale": 15.0,
  "guidance_scale_text": 8.0,   // Moderate text influence
  "guidance_scale_lyric": 12.0  // Stronger lyric influence
}
```

#### `guidance_scale_lyric`

**Type:** `float`  
**Default:** `0.0` (disabled)

**Controls:** Additional guidance scale specifically for lyric embeddings

**Recommendation:**
```json
{
  "guidance_scale_text": 0.0,  // Disabled (default)
  "guidance_scale_lyric": 0.0   // Disabled (default)
}
```

Use double condition guidance only for advanced control scenarios.

---

## Precision and Performance

### `bf16` (Bfloat16 Precision)

**Type:** `boolean`  
**Default:** `true`

**Controls:** Use bfloat16 instead of float32 for faster inference

**Technical Details:**
- Reduces memory usage by ~50%
- Faster computation on modern GPUs
- Minimal quality loss
- Not supported on MPS (macOS), falls back to float32

**Effects:**
- ✅ ~2× faster inference
- ✅ ~50% less VRAM usage
- ✅ Negligible quality difference
- ⚠️ Not available on macOS MPS

**Recommendation:**
```json
{
  "bf16": true  // Recommended for all CUDA GPUs
}
```

**Exception (macOS):**
```json
{
  "bf16": false  // Required on macOS
}
```

---

### `torch_compile` (Model Compilation)

**Type:** `boolean`  
**Default:** `false`

**Controls:** Use `torch.compile()` for additional speedup

**Technical Details:**
- Compiles model to optimized graph
- First run is slower (compilation)
- Subsequent runs are faster
- Requires Triton on Windows

**Effects:**
- ✅ 10-30% additional speedup
- ✅ Optimized memory access
- ⚠️ First generation slower (compilation)
- ⚠️ Requires Triton on Windows: `pip install triton-windows`

**Recommendation:**
```json
{
  "torch_compile": false  // Default, enable if stable
}
```

If you want to enable:
```bash
# Windows only
pip install triton-windows

# Then in request
{
  "torch_compile": true
}
```

---

### `device_id` (GPU Selection)

**Type:** `integer`  
**Default:** `0`

**Controls:** Which GPU to use (for multi-GPU systems)

**Technical Details:**
- Sets `CUDA_VISIBLE_DEVICES` environment variable
- `0` = first GPU, `1` = second GPU, etc.
- Only relevant for multi-GPU systems

**Example:**
```json
{
  "device_id": 0  // Use first GPU
}
```

---

## Output Control

### `output_path` (Save Location)

**Type:** `string` (optional)  
**Default:** `null` (auto-generated)

**Controls:** Where to save the generated audio file

**Technical Details:**
- If `null`: Auto-generates in `./outputs/` directory
- If filename only: Saves to `./outputs/{filename}`
- If full path: Saves to specified location
- Directory is created automatically if it doesn't exist

**Examples:**

```json
// Auto-generated (recommended)
{
  "output_path": null
}
// Result: ./outputs/output_{uuid}.wav

// Custom filename
{
  "output_path": "my_song.wav"
}
// Result: ./outputs/my_song.wav

// Full path
{
  "output_path": "/workspace/generated/rap_for_meena.wav"
}
// Result: /workspace/generated/rap_for_meena.wav
```

**Best Practices:**
- Use `null` for automatic naming
- Include `.wav` extension
- Ensure directory is writable
- Use unique names to avoid overwrites

---

## Recommended Parameter Recipes

### 🎵 Balanced Realism (Default Recommended)

**Best for:** Most production tracks, general purpose

```json
{
  "infer_step": 60,
  "guidance_scale": 15.0,
  "cfg_type": "apg",
  "scheduler_type": "euler",
  "omega_scale": 10.0,
  "guidance_interval": 0.5,
  "guidance_interval_decay": 0.0,
  "min_guidance_scale": 3.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true
}
```

**Characteristics:**
- ✅ Emotionally rich and rhythmic
- ✅ Good coherence and naturalness
- ✅ Balanced creativity and adherence
- ✅ Suitable for most genres

---

### 🎶 Experimental / Freestyle

**Best for:** Creative exploration, jam sessions, experimental tracks

```json
{
  "infer_step": 50,
  "guidance_scale": 9.0,
  "cfg_type": "apg",
  "scheduler_type": "pingpong",
  "omega_scale": 11.0,
  "guidance_interval": 0.3,
  "guidance_interval_decay": 0.5,
  "min_guidance_scale": 2.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true
}
```

**Characteristics:**
- ✅ Looser adherence to prompt/lyrics
- ✅ High creativity and variation
- ✅ More dynamic and expressive
- ✅ Good for: Experimental, freestyle, creative exploration

---

### 🎧 High Fidelity / Precise

**Best for:** Structured songs, covers, precise requirements

```json
{
  "infer_step": 80,
  "guidance_scale": 18.0,
  "cfg_type": "cfg_star",
  "scheduler_type": "heun",
  "omega_scale": 10.0,
  "guidance_interval": 0.7,
  "guidance_interval_decay": 0.0,
  "min_guidance_scale": 5.0,
  "use_erg_tag": false,
  "use_erg_lyric": false,
  "use_erg_diffusion": false
}
```

**Characteristics:**
- ✅ Tight synchronization to prompt/lyrics
- ✅ High quality and coherence
- ✅ Precise and structured
- ✅ Best for: Professional production, structured songs

---

### ⚡ Fast Preview

**Best for:** Quick testing, iterations, previews

```json
{
  "infer_step": 27,
  "guidance_scale": 15.0,
  "cfg_type": "apg",
  "scheduler_type": "euler",
  "omega_scale": 10.0,
  "guidance_interval": 0.5,
  "guidance_interval_decay": 0.0,
  "min_guidance_scale": 3.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true
}
```

**Characteristics:**
- ✅ Fastest generation (~3-5 seconds for 30s audio)
- ✅ Good enough quality for previews
- ✅ Quick iteration cycles
- ✅ Use for: Testing, rapid prototyping

---

### 🎤 Rap / Hip-Hop Optimized

**Best for:** Rap, hip-hop, trap, fast-paced tracks

```json
{
  "infer_step": 60,
  "guidance_scale": 15.0,
  "cfg_type": "apg",
  "scheduler_type": "pingpong",
  "omega_scale": 11.0,
  "guidance_interval": 0.5,
  "guidance_interval_decay": 0.0,
  "min_guidance_scale": 3.0,
  "use_erg_tag": true,
  "use_erg_lyric": false,  // Keep lyrics precise for rap
  "use_erg_diffusion": true
}
```

**Characteristics:**
- ✅ Better lyric alignment (pingpong scheduler)
- ✅ More punchy dynamics (higher omega)
- ✅ Precise lyric delivery
- ✅ Good for: Rap, hip-hop, vocal-focused tracks

---

### 🎹 Instrumental / Ambient

**Best for:** Instrumental tracks, ambient, background music

```json
{
  "infer_step": 60,
  "guidance_scale": 12.0,
  "cfg_type": "apg",
  "scheduler_type": "euler",
  "omega_scale": 9.0,
  "guidance_interval": 0.5,
  "guidance_interval_decay": 0.2,
  "min_guidance_scale": 2.0,
  "use_erg_tag": true,
  "use_erg_lyric": true,
  "use_erg_diffusion": true
}
```

**Characteristics:**
- ✅ Smoother, less dynamic (lower omega)
- ✅ More creative freedom
- ✅ Good for: Ambient, lo-fi, background music

---

## Hardware-Specific Optimization

### RTX A5000 (24GB VRAM)

**Recommended Settings:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `audio_duration` | ≤ 90s | Prevents VRAM overflow |
| `bf16` | `true` | Faster, uses less memory |
| `infer_step` | 50-60 | Good quality/speed balance |
| `guidance_scale` | 12-16 | Natural sound tone |
| `torch_compile` | `false` | Optional, enable if stable |
| Concurrent requests | 1 | 24GB VRAM limit |

**Optimal Recipe:**
```json
{
  "audio_duration": 70.0,
  "infer_step": 60,
  "guidance_scale": 15.0,
  "bf16": true,
  "torch_compile": false
}
```

---

### RTX 4090 (24GB VRAM)

**Recommended Settings:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `audio_duration` | ≤ 120s | More headroom than A5000 |
| `bf16` | `true` | Essential for speed |
| `infer_step` | 60-80 | Can handle more steps |
| `torch_compile` | `true` | Good performance boost |

**Optimal Recipe:**
```json
{
  "audio_duration": 90.0,
  "infer_step": 70,
  "guidance_scale": 15.0,
  "bf16": true,
  "torch_compile": true
}
```

---

### A100 (40GB VRAM)

**Recommended Settings:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `audio_duration` | ≤ 240s | Can handle full songs |
| `infer_step` | 80-100 | Maximum quality |
| `bf16` | `true` | Still recommended |
| `torch_compile` | `true` | Excellent performance |

**Optimal Recipe:**
```json
{
  "audio_duration": 180.0,
  "infer_step": 100,
  "guidance_scale": 15.0,
  "bf16": true,
  "torch_compile": true
}
```

---

## Practical Tips

### 1. Parameter Relationships

**Duration and Steps:**
- Longer duration → Use more steps for quality
- Short clips (30s) → 27-50 steps sufficient
- Full songs (120s+) → 80-100 steps recommended

**Guidance and ERG:**
- High guidance + ERG enabled = Balanced adherence and creativity
- Low guidance + ERG enabled = Very creative, loose
- High guidance + ERG disabled = Precise, structured

**Omega and Genre:**
- Energetic genres (rock, rap) → Higher omega (11-12)
- Smooth genres (jazz, ambient) → Lower omega (8-9)
- Balanced genres (pop) → Default omega (10)

### 2. Iteration Workflow

1. **Start with defaults** (Balanced Realism recipe)
2. **Test with same seed** to compare parameter changes
3. **Adjust one parameter at a time** to understand effects
4. **Save successful parameter sets** in JSON files
5. **Use fast preview** (27 steps) for rapid testing

### 3. Quality vs Speed Trade-offs

| Priority | Settings |
|----------|----------|
| **Maximum Quality** | `infer_step: 100`, `scheduler_type: "heun"`, `guidance_scale: 18.0` |
| **Balanced** | `infer_step: 60`, `scheduler_type: "euler"`, `guidance_scale: 15.0` |
| **Maximum Speed** | `infer_step: 27`, `scheduler_type: "euler"`, `guidance_scale: 15.0` |

### 4. Common Patterns

**Problem: Output too robotic**
- Solution: Lower `guidance_scale` (12-14), enable all ERG

**Problem: Output ignores prompt**
- Solution: Raise `guidance_scale` (18-20), disable ERG

**Problem: Output too repetitive**
- Solution: Enable `use_erg_diffusion`, use `scheduler_type: "pingpong"`

**Problem: Output lacks energy**
- Solution: Raise `omega_scale` (11-12), use `scheduler_type: "pingpong"`

**Problem: Generation too slow**
- Solution: Lower `infer_step` (27-50), use `bf16: true`, `scheduler_type: "euler"`

### 5. Seed Management

- **Save successful seeds** for reproducible results
- **Use different seeds** to explore variations
- **Compare seeds** with same parameters to find best output
- **Document seed + parameters** for future reference

---

## Parameter Preset Library

Save these presets as JSON files for easy reuse:

### `presets/balanced.json`
```json
{
  "name": "Balanced Realism",
  "description": "Default recommended settings for most use cases",
  "parameters": {
    "infer_step": 60,
    "guidance_scale": 15.0,
    "cfg_type": "apg",
    "scheduler_type": "euler",
    "omega_scale": 10.0,
    "guidance_interval": 0.5,
    "guidance_interval_decay": 0.0,
    "min_guidance_scale": 3.0,
    "use_erg_tag": true,
    "use_erg_lyric": true,
    "use_erg_diffusion": true
  }
}
```

### `presets/freestyle.json`
```json
{
  "name": "Experimental Freestyle",
  "description": "Creative, loose, high variation",
  "parameters": {
    "infer_step": 50,
    "guidance_scale": 9.0,
    "cfg_type": "apg",
    "scheduler_type": "pingpong",
    "omega_scale": 11.0,
    "guidance_interval": 0.3,
    "guidance_interval_decay": 0.5,
    "min_guidance_scale": 2.0,
    "use_erg_tag": true,
    "use_erg_lyric": true,
    "use_erg_diffusion": true
  }
}
```

### `presets/precise.json`
```json
{
  "name": "High Fidelity Precise",
  "description": "Structured, precise, high quality",
  "parameters": {
    "infer_step": 80,
    "guidance_scale": 18.0,
    "cfg_type": "cfg_star",
    "scheduler_type": "heun",
    "omega_scale": 10.0,
    "guidance_interval": 0.7,
    "guidance_interval_decay": 0.0,
    "min_guidance_scale": 5.0,
    "use_erg_tag": false,
    "use_erg_lyric": false,
    "use_erg_diffusion": false
  }
}
```

### `presets/fast.json`
```json
{
  "name": "Fast Preview",
  "description": "Quick testing and iterations",
  "parameters": {
    "infer_step": 27,
    "guidance_scale": 15.0,
    "cfg_type": "apg",
    "scheduler_type": "euler",
    "omega_scale": 10.0,
    "guidance_interval": 0.5,
    "guidance_interval_decay": 0.0,
    "min_guidance_scale": 3.0,
    "use_erg_tag": true,
    "use_erg_lyric": true,
    "use_erg_diffusion": true
  }
}
```

### `presets/rap.json`
```json
{
  "name": "Rap/Hip-Hop Optimized",
  "description": "Optimized for rap with better lyric alignment",
  "parameters": {
    "infer_step": 60,
    "guidance_scale": 15.0,
    "cfg_type": "apg",
    "scheduler_type": "pingpong",
    "omega_scale": 11.0,
    "guidance_interval": 0.5,
    "guidance_interval_decay": 0.0,
    "min_guidance_scale": 3.0,
    "use_erg_tag": true,
    "use_erg_lyric": false,
    "use_erg_diffusion": true
  }
}
```

---

## Summary Table

| Category | Parameter | Recommended Value | Effect |
|----------|-----------|-------------------|--------|
| **Content** | `prompt` | Comma-separated tags | Defines style and mood |
| | `lyrics` | Structured with tags | Controls rhythm and emotion |
| | `audio_duration` | 60-90s (A5000) | Length of output |
| **Diffusion** | `infer_step` | 60 | Quality vs speed |
| | `scheduler_type` | `"euler"` | Fast and reliable |
| | `cfg_type` | `"apg"` | Best balance |
| **Guidance** | `guidance_scale` | 15.0 | Adherence strength |
| | `guidance_interval` | 0.5 | Active step range |
| | `guidance_interval_decay` | 0.0 | Decay rate |
| | `min_guidance_scale` | 3.0 | Minimum strength |
| **Variation** | `use_erg_tag` | `true` | Genre diversity |
| | `use_erg_lyric` | `true` | Phrasing variation |
| | `use_erg_diffusion` | `true` | Denoising variety |
| | `omega_scale` | 10.0 | Dynamic range |
| **Reproducibility** | `actual_seeds` | `[12345]` | Deterministic output |
| | `oss_steps` | `[]` | Use all steps |
| **Performance** | `bf16` | `true` | Faster inference |
| | `torch_compile` | `false` | Optional speedup |

---

## Next Steps

1. ✅ Start with **Balanced Realism** preset
2. ✅ Test with different seeds to explore variations
3. ✅ Adjust parameters based on desired output
4. ✅ Save successful parameter combinations
5. ✅ Experiment with different presets for different use cases

---

**Last Updated:** 2025-01-09  
**Guide Version:** 1.0

For API documentation, see [README.md](./README.md).  
For setup instructions, see [setup.md](./setup.md).

