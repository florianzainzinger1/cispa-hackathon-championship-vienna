# 🛡️ BitMark Watermark Removal Pipeline

**CISPA European AI & Cybersecurity Championship - Vienna 2025**  
**Task 3: Watermark Removal Attack**

---

## 📋 Projektübersicht

Dieses Projekt implementiert einen **6-schrittigen Watermark-Removal-Pipeline** für die Entfernung von BitMark-Wasserzeichen aus KI-generierten Bildern. Die Pipeline kombiniert mehrere Angriffsstrategien, um die Erkennbarkeit von Wasserzeichen signifikant zu reduzieren, während die visuelle Qualität der Bilder erhalten bleibt.

### Zielmetrik
- **Primary Metric**: 1 - TPR @ 1% FPR (True Positive Rate bei 1% False Positive Rate)
- **Quality Constraint**: MSE ≤ 0.08 (Mean Squared Error zum Original)
- **Dataset**: 100 watermarked PNG Bilder (512×512×3)

---

## 🏗️ Architektur

### 6-Schritt Attack Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WATERMARK REMOVAL PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  1. Fast-Sand │───▶│  2. Image-GS │───▶│  3. Diffusion│              │
│  │   (Inpaint)   │    │ (2D Gaussian)│    │   (img2img)  │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  4. Rotation  │───▶│   5. Blur    │───▶│  6. ColorFix │──▶ OUTPUT   │
│  │  (0.5°-2.0°)  │    │  (Gaussian)  │    │ (Histogram)  │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Beschreibung der Angriffsschritte

| Schritt | Name | Beschreibung | Library |
|---------|------|--------------|---------|
| 1 | **Fast-Sand** | Iteratives Patch-Inpainting mit Stable Diffusion. Ersetzt zufällige Patches im Bild. | `diffusers` |
| 2 | **Image-GS** | Re-Repräsentation als 2D Gaussians und Re-Rendering. | `gsplat` |
| 3 | **Diffusion** | Globale img2img Rekonstruktion mit niedriger Stärke. | `diffusers` |
| 4 | **Rotation** | Kleine Rotation (0.5°-2°) um Bit-Alignment zu brechen. | `scipy` |
| 5 | **Blur** | Gaussian Blur um hochfrequente Watermark-Reste zu entfernen. | `opencv` |
| 6 | **Color-Fix** | Histogram Matching zum Original um MSE zu minimieren. | `scikit-image` |

---

## 📁 Projektstruktur

```
task_3_watermark_removal/
├── src/                          # Hauptquellcode
│   ├── __init__.py               # Package exports
│   ├── config.py                 # AttackConfig Dataclass & Presets
│   ├── pipeline.py               # AttackPipeline Orchestrator
│   └── attacks/                  # Attack Module
│       ├── __init__.py
│       ├── base.py               # Abstrakte BaseAttack Klasse
│       ├── fast_sand.py          # Schritt 1: Patch Inpainting
│       ├── image_gs.py           # Schritt 2: 2D Gaussian Splatting
│       ├── diffusion.py          # Schritt 3: img2img Reconstruction
│       ├── rotation.py           # Schritt 4: Small Rotation
│       ├── blur.py               # Schritt 5: Gaussian Blur
│       └── color_fix.py          # Schritt 6: Histogram Matching
│
├── scripts/                      # Ausführbare Skripte
│   ├── run_attack.py             # Haupt-CLI zum Ausführen der Pipeline
│   └── test_bitmark.py           # Test BitMark Watermark Detection
│
├── repo/                         # Externe Repositories
│   ├── image-gs/                 # 2D Gaussian Splatting
│   ├── BitMark/                  # BitMark Watermark Detection
│   ├── diffusers/                # Hugging Face Diffusers
│   └── impossibility-watermark/  # Sand-Vision Attack Reference
│
├── Dataset/                      # 100 watermarked Bilder
├── output/                       # Angegriffene Bilder
├── logs/                         # SLURM Job Logs
│
├── run_attack.sh                 # SLURM Job Script
├── submit.py                     # Leaderboard Submission
├── submit.sh                     # SLURM Submit Script
├── submission.npz                # Finales Submission File
│
├── pyproject.toml                # Python Projekt Konfiguration
└── README.md                     # Diese Dokumentation
```

---

## 🔧 Technologie-Stack

### Kernbibliotheken

| Bibliothek | Version | Verwendung |
|------------|---------|------------|
| **PyTorch** | 2.5.1+cu121 | Deep Learning Framework, GPU Acceleration |
| **Diffusers** | 0.36.0 | Stable Diffusion Inpaint & img2img Pipelines |
| **Transformers** | 4.49.0 | CLIP Text Encoder für Diffusion |
| **Accelerate** | 1.10.1 | Mixed Precision Training, Device Management |
| **gsplat** | 1.4.0 | 2D Gaussian Splatting CUDA Kernels |

### Bildverarbeitung

| Bibliothek | Version | Verwendung |
|------------|---------|------------|
| **NumPy** | 2.2.6 | Array Operations |
| **Pillow** | 11.3.0 | Bildlade/Speicher |
| **OpenCV** | 4.12.0 | Gaussian Blur, Image Processing |
| **SciPy** | 1.16.3 | Image Rotation (ndimage) |
| **scikit-image** | 0.25.2 | Histogram Matching (exposure.match_histograms) |

### Infrastruktur

| Tool | Verwendung |
|------|------------|
| **UV** | Python Package Manager (Alternative zu pip/conda) |
| **SLURM** | HPC Job Scheduler auf JURECA |
| **CUDA 12** | GPU Acceleration |
| **cuDNN 9.5** | Deep Learning Optimizations |

---

## ⚙️ Konfiguration & Presets

Die Pipeline unterstützt verschiedene **Presets** für unterschiedliche Trade-offs zwischen Angriffsintensität und MSE:

```python
PRESETS = {
    "minimal": {
        # Nur CPU-basierte Attacks (schnell, niedriger MSE)
        "enable_fast_sand": False,
        "enable_image_gs": False,
        "enable_diffusion": False,
        "enable_rotation": True,
        "enable_blur": True,
        "enable_color_fix": True,
    },
    "no_gs": {
        # Alle außer Image-GS (empfohlen wenn gsplat nicht verfügbar)
        "enable_fast_sand": True,
        "enable_image_gs": False,
        "enable_diffusion": True,
        "enable_rotation": True,
        "enable_blur": True,
        "enable_color_fix": True,
    },
    "full": {
        # Alle 6 Schritte aktiviert
        "enable_fast_sand": True,
        "enable_image_gs": True,
        "enable_diffusion": True,
        "enable_rotation": True,
        "enable_blur": True,
        "enable_color_fix": True,
    },
    "safe_mse": {
        # Konservative Parameter für MSE ≤ 0.08
        "sand_iterations": 1,
        "sand_coverage": 0.2,
        "diffusion_strength": 0.05,
        "rotation_angle": 0.5,
        "blur_sigma": 0.3,
    },
    "aggressive": {
        # Aggressive Parameter (höherer MSE, stärkere Watermark Removal)
        "sand_iterations": 5,
        "sand_coverage": 0.6,
        "diffusion_strength": 0.2,
        "rotation_angle": 2.0,
        "blur_sigma": 1.5,
    }
}
```

---

## 🚀 Verwendung

### 1. Environment Setup

```bash
# UV Environment aktivieren
cd /p/project1/training2562/syed1/Hackathon/task_3_watermark_removal
source .venv/bin/activate

# CUDA Module laden (auf JURECA)
module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12
```

### 2. Pipeline ausführen

```bash
# Mit Standard-Preset (no_gs)
python scripts/run_attack.py --preset no_gs --output submission.npz

# Mit aggressiveren Parametern
python scripts/run_attack.py --preset aggressive --output submission.npz

# Mit spezifischen Overrides
python scripts/run_attack.py \
    --preset no_gs \
    --sand-iterations 2 \
    --diffusion-strength 0.1 \
    --output submission.npz
```

### 3. Via SLURM auf GPU-Node

```bash
# Edit run_attack.sh um Preset zu wählen
sbatch run_attack.sh

# Progress verfolgen
tail -f logs/progress_<JOB_ID>.log
```

### 4. Submission zum Leaderboard

```bash
# Lokale Submission (von Rechner mit Internet)
curl -X POST \
    -H "X-API-Key: YOUR_API_TOKEN" \
    -F "file=@submission.npz" \
    http://35.192.205.84:80/submit/09-watermark-removal
```

---

## 📊 Ergebnisse

### Typische MSE-Werte nach Preset

| Preset | Ø MSE | Bilder > 0.08 | Status |
|--------|-------|---------------|--------|
| minimal | 0.014 | 0/100 | ✓ Safe |
| safe_mse | 0.037 | 0/100 | ✓ Safe |
| no_gs | 0.062 | 29/100 | ⚠ Grenzwertig |
| aggressive | 0.14+ | 80+/100 | ✗ Über Limit |

### Beispiel Pipeline Output

```
[  1/100] 1.png
    [fast_sand] MSE=0.0528 ✓
    [diffusion] MSE=0.0576 ✓
    [rotation] MSE=0.0577 ✓
    [blur] MSE=0.0572 ✓
    [color_fix] MSE=0.0370 ✓

MSE Summary: min=0.0132, max=0.1406, mean=0.0624
```

---

## 🔬 Technische Details

### Fast-Sand Attack

Der Fast-Sand Attack basiert auf dem "Watermarks in the Sand" Paper und verwendet iteratives Patch-Inpainting:

```python
def apply(self, image, original):
    for iteration in range(self.config.sand_iterations):
        # Generiere zufällige Patch-Maske
        mask = self._create_random_mask(image.shape, self.config.sand_coverage)
        
        # Inpaint mit Stable Diffusion
        result = self.inpaint_pipeline(
            prompt="high quality photo",
            image=image,
            mask_image=mask,
            num_inference_steps=20,
            strength=0.5
        ).images[0]
        
        image = np.array(result)
    return image
```

### Diffusion img2img

Globale Rekonstruktion mit niedriger Stärke erhält Bildinhalt während Watermark-Struktur destabilisiert wird:

```python
result = self.img2img_pipeline(
    prompt="high quality photo",
    image=pil_image,
    num_inference_steps=self.config.diffusion_steps,
    strength=self.config.diffusion_strength,  # typisch: 0.05-0.15
    guidance_scale=1.0
).images[0]
```

### Color-Fix für MSE-Optimierung

Histogram Matching stellt Farbverteilung des Originals wieder her:

```python
from skimage.exposure import match_histograms

def apply(self, image, original):
    # Mindestens 50% Gewicht auf das Original
    matched = match_histograms(image, original, channel_axis=2)
    return matched.astype(np.uint8)
```

---

## 🖥️ HPC Infrastruktur

### JURECA Cluster Specs

| Resource | Specification |
|----------|---------------|
| **GPU Nodes** | 4× NVIDIA A100 (40GB HBM2e) |
| **CPU Nodes** | 2× AMD EPYC 7742 (128 cores @ 2.25GHz) |
| **Memory** | 512GB DDR4 |
| **Login Nodes** | 2× NVIDIA Quadro RTX8000 |

### SLURM Job Konfiguration

```bash
#SBATCH --account=training2562
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --reservation=cispahack
```

---

## 📚 Referenzen

1. **BitMark**: Kerner et al., "BitMark: Watermarking Bitwise Autoregressive Image Generative Models" (NeurIPS 2025)
2. **Watermarks in the Sand**: Zhang et al., "On the Impossibility of Model-Agnostic Explainable Watermarking"
3. **2D Gaussian Splatting**: image-gs Repository
4. **Stable Diffusion**: Hugging Face Diffusers Library

---

## 👥 Team

**Team !!1337** - CISPA European AI & Cybersecurity Championship Vienna 2025

---

## 📝 Lizenz

Dieses Projekt wurde für den CISPA Hackathon entwickelt. Externe Repositories (BitMark, image-gs, diffusers) unterliegen ihren jeweiligen Lizenzen.
