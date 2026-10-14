#!/usr/bin/env python3
"""
Hauptskript zum Ausführen der Attack-Pipeline.

Verwendung:
    python scripts/run_attack.py --preset full
    python scripts/run_attack.py --preset full --no-image-gs
    python scripts/run_attack.py --enable-sand --enable-diffusion --diffusion-strength 0.2
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Füge src zum Path hinzu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import AttackConfig, PRESETS, AttackPipeline


def load_dataset(dataset_dir: str):
    """Lädt alle Bilder aus dem Dataset."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset nicht gefunden: {dataset_dir}")
    
    image_files = sorted(dataset_path.glob("*.png"))
    
    if len(image_files) == 0:
        raise ValueError(f"Keine PNG-Bilder in {dataset_dir}")
    
    print(f"Gefunden: {len(image_files)} Bilder in {dataset_dir}")
    
    images, names = [], []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        images.append(np.array(img))
        names.append(img_path.name)
    
    return images, names


def create_submission(images, names, output_path: str):
    """Erstellt submission.npz Datei."""
    images_array = np.stack(images, axis=0).astype(np.uint8)
    
    # WICHTIG: Namen als Integer (OHNE .png Extension)!
    # Basierend auf task_template.py Zeile 91
    names_clean = [int(os.path.splitext(n)[0]) for n in names]
    names_array = np.array(names_clean)
    
    # Validierung
    expected_shape = (100, 512, 512, 3)
    if images_array.shape != expected_shape:
        print(f"⚠ Warnung: Shape ist {images_array.shape}, erwartet {expected_shape}")
    
    if len(names_array) != 100:
        print(f"⚠ Warnung: {len(names_array)} Bilder, erwartet 100")
    
    np.savez_compressed(output_path, images=images_array, names=names_array)
    print(f"✓ Submission gespeichert: {output_path}")
    print(f"  Shape: {images_array.shape}")
    print(f"  Names: {len(names_array)} Einträge")


def submit_to_server(submission_path: str, api_key: str, config: AttackConfig):
    """Sendet Submission an Leaderboard-Server."""
    import requests
    
    url = f"{config.server_url}/submit/{config.task_id}"
    print(f"Sende an {url}...")
    
    response = requests.post(
        url,
        files={"file": open(submission_path, "rb")},
        headers={"X-API-Key": api_key},
    )
    
    print(f"Server-Antwort: {response.json()}")
    return response.json()


def print_config(config: AttackConfig):
    """Zeigt aktuelle Konfiguration an."""
    print("\n" + "=" * 60)
    print("PIPELINE KONFIGURATION")
    print("=" * 60)
    
    steps = [
        ("1. Fast-Sand", config.enable_fast_sand, f"iter={config.sand_iterations}, coverage={config.sand_coverage}"),
        ("2. Image-GS", config.enable_image_gs, f"gaussians={config.gs_num_gaussians}"),
        ("3. Diffusion", config.enable_diffusion, f"strength={config.diffusion_strength}, steps={config.diffusion_steps}"),
        ("4. Rotation", config.enable_rotation, f"angle={config.rotation_angle}°"),
        ("5. Blur", config.enable_blur, f"σ={config.blur_sigma}"),
        ("6. Color-Fix", config.enable_color_fix, "histogram matching"),
    ]
    
    for name, enabled, params in steps:
        status = "✓ ON " if enabled else "✗ OFF"
        print(f"  {status}  {name}: {params}")
    
    print(f"\n  Max MSE: {config.max_mse}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BitMark Watermark Removal Attack")
    
    # Basics
    parser.add_argument("--dataset", default="Dataset", help="Pfad zum Dataset")
    parser.add_argument("--output", default="submission.npz", help="Output-Datei")
    parser.add_argument("--preset", default="full", choices=list(PRESETS.keys()))
    parser.add_argument("--api-key", help="Team API Key")
    parser.add_argument("--submit", action="store_true", help="An Server senden")
    
    # Step Toggles
    parser.add_argument("--enable-sand", action="store_true")
    parser.add_argument("--no-sand", action="store_true")
    parser.add_argument("--enable-image-gs", action="store_true")
    parser.add_argument("--no-image-gs", action="store_true")
    parser.add_argument("--enable-diffusion", action="store_true")
    parser.add_argument("--no-diffusion", action="store_true")
    parser.add_argument("--enable-rotation", action="store_true")
    parser.add_argument("--no-rotation", action="store_true")
    parser.add_argument("--enable-blur", action="store_true")
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--enable-color-fix", action="store_true")
    parser.add_argument("--no-color-fix", action="store_true")
    
    # Parameter
    parser.add_argument("--sand-iterations", type=int)
    parser.add_argument("--diffusion-strength", type=float)
    parser.add_argument("--rotation-angle", type=float)
    parser.add_argument("--blur-sigma", type=float)
    
    args = parser.parse_args()
    
    # Config aus Preset laden (deepcopy um Original nicht zu modifizieren)
    import copy
    config = copy.deepcopy(PRESETS[args.preset])
    
    # Toggles anwenden
    if args.enable_sand: config.enable_fast_sand = True
    if args.no_sand: config.enable_fast_sand = False
    if args.enable_image_gs: config.enable_image_gs = True
    if args.no_image_gs: config.enable_image_gs = False
    if args.enable_diffusion: config.enable_diffusion = True
    if args.no_diffusion: config.enable_diffusion = False
    if args.enable_rotation: config.enable_rotation = True
    if args.no_rotation: config.enable_rotation = False
    if args.enable_blur: config.enable_blur = True
    if args.no_blur: config.enable_blur = False
    if args.enable_color_fix: config.enable_color_fix = True
    if args.no_color_fix: config.enable_color_fix = False
    
    # Parameter anwenden
    if args.sand_iterations: config.sand_iterations = args.sand_iterations
    if args.diffusion_strength: config.diffusion_strength = args.diffusion_strength
    if args.rotation_angle: config.rotation_angle = args.rotation_angle
    if args.blur_sigma: config.blur_sigma = args.blur_sigma
    
    if args.api_key: config.api_key = args.api_key
    
    # Config anzeigen
    print_config(config)
    
    # Pipeline erstellen
    pipeline = AttackPipeline(config)
    print(f"\nAktive Schritte: {pipeline.get_enabled_steps()}")
    
    # Dataset laden
    print("\n" + "=" * 60)
    print("DATASET LADEN")
    print("=" * 60)
    images, names = load_dataset(args.dataset)
    
    # Modelle vorladen
    print("\n" + "=" * 60)
    print("MODELLE LADEN")
    print("=" * 60)
    pipeline.preload_models()
    
    # Attack ausführen
    print("\n" + "=" * 60)
    print("ATTACK AUSFÜHREN")
    print("=" * 60)
    
    start_time = time.time()
    attacked_images, all_mse = pipeline.attack_batch(images, names, verbose=True)
    elapsed = time.time() - start_time
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    
    final_mses = [list(m.values())[-1] for m in all_mse]
    print(f"Zeit: {elapsed:.1f}s ({elapsed/len(images):.2f}s pro Bild)")
    print(f"MSE: min={min(final_mses):.4f}, max={max(final_mses):.4f}, mean={np.mean(final_mses):.4f}")
    print(f"Bilder mit MSE > {config.max_mse}: {sum(1 for m in final_mses if m > config.max_mse)}")
    
    # Submission erstellen
    print("\n" + "=" * 60)
    print("SUBMISSION ERSTELLEN")
    print("=" * 60)
    create_submission(attacked_images, names, args.output)
    
    # Optional: An Server senden
    if args.submit and config.api_key:
        print("\n" + "=" * 60)
        print("AN SERVER SENDEN")
        print("=" * 60)
        submit_to_server(args.output, config.api_key, config)
    
    # Modelle freigeben
    pipeline.unload_models()
    
    print("\n✓ Fertig!")


if __name__ == "__main__":
    main()
