"""
Image-GS Attack: Bild als 2D Gaussians re-repräsentieren.

Referenz:
- Paper: https://arxiv.org/abs/2407.01866
- Repo: siehe repo/image-gs/
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
from PIL import Image
from .base import BaseAttack


class ImageGSAttack(BaseAttack):
    """Image-GS Re-Representation Attack."""
    
    name = "image_gs"
    
    def is_enabled(self) -> bool:
        return self.config.enable_image_gs
    
    def _check_gsplat(self) -> bool:
        """Prüft ob gsplat installiert ist."""
        try:
            import gsplat
            return True
        except ImportError:
            return False
    
    def _get_repo_path(self) -> str:
        """Gibt Pfad zum image-gs Repo zurück."""
        # Handle relative and absolute paths
        repo_path = self.config.gs_repo_path
        if not os.path.isabs(repo_path):
            # Get the project root (where src/ is located)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            repo_path = os.path.join(project_root, repo_path)
        return repo_path
    
    def apply(self, image: np.ndarray, original: np.ndarray) -> np.ndarray:
        # Prüfe ob gsplat verfügbar
        if not self._check_gsplat():
            print("      ⚠ gsplat nicht installiert - überspringe Image-GS")
            return image
        
        repo_path = self._get_repo_path()
        main_script = os.path.join(repo_path, "main.py")
        
        if not os.path.exists(main_script):
            print(f"      ⚠ image-gs Repo nicht gefunden unter {repo_path}")
            return image
        
        # Temporäres Verzeichnis für dieses Bild
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.png")
            exp_dir = os.path.join(tmpdir, "exp")
            log_root = os.path.join(tmpdir, "logs")
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(log_root, exist_ok=True)
            
            # Input speichern
            Image.fromarray(image).save(input_path)
            
            try:
                # Training: Bild → Gaussians
                cmd_train = [
                    sys.executable,
                    main_script,
                    "--input_path", input_path,
                    "--exp_name", "gs_attack",
                    "--log_root", log_root,
                    "--num_gaussians", str(self.config.gs_num_gaussians),
                    "--max_steps", str(self.config.gs_iterations),
                ]
                if self.config.gs_quantize:
                    cmd_train.append("--quantize")
                
                print(f"      Training Gaussians...")
                env = os.environ.copy()
                env["PYTHONPATH"] = repo_path + ":" + env.get("PYTHONPATH", "")
                
                result = subprocess.run(
                    cmd_train, 
                    check=True, 
                    capture_output=True, 
                    timeout=300,
                    cwd=repo_path,
                    env=env
                )
                
                # Find the log directory created by image-gs
                # The naming convention in image-gs is complex, let's find the latest dir
                log_dirs = []
                for root, dirs, files in os.walk(log_root):
                    for d in dirs:
                        log_dirs.append(os.path.join(root, d))
                
                if not log_dirs:
                    print(f"      ⚠ No log directory created by Image-GS")
                    return image
                
                # Get the actual log dir (should be the deepest one with rendered images)
                actual_log_dir = None
                for ld in sorted(log_dirs, key=lambda x: len(x), reverse=True):
                    # Look for checkpoint or rendered image
                    for f in os.listdir(ld):
                        if f.endswith('.png') or f.endswith('.pt'):
                            actual_log_dir = ld
                            break
                    if actual_log_dir:
                        break
                
                if not actual_log_dir:
                    # Just use the deepest directory
                    actual_log_dir = max(log_dirs, key=lambda x: len(x))
                
                # Rendering: Gaussians → Bild
                cmd_render = [
                    sys.executable,
                    main_script,
                    "--input_path", input_path,
                    "--exp_name", "gs_attack",
                    "--log_root", log_root,
                    "--num_gaussians", str(self.config.gs_num_gaussians),
                    "--eval",
                    "--render_height", str(image.shape[0]),
                ]
                if self.config.gs_quantize:
                    cmd_render.append("--quantize")
                
                print(f"      Rendering...")
                subprocess.run(
                    cmd_render, 
                    check=True, 
                    capture_output=True, 
                    timeout=120,
                    cwd=repo_path,
                    env=env
                )
                
                # Ergebnis laden - search for rendered output
                for root, dirs, files in os.walk(log_root):
                    for f in files:
                        if 'render' in f.lower() and f.endswith('.png'):
                            output_path = os.path.join(root, f)
                            result_img = np.array(Image.open(output_path).convert("RGB"))
                            # Resize if needed
                            if result_img.shape[:2] != image.shape[:2]:
                                result_img = np.array(
                                    Image.fromarray(result_img).resize((image.shape[1], image.shape[0]))
                                )
                            return result_img
                
                # If no rendered file found, look for any PNG that's not the input
                for root, dirs, files in os.walk(log_root):
                    for f in files:
                        if f.endswith('.png') and f != 'input.png':
                            output_path = os.path.join(root, f)
                            result_img = np.array(Image.open(output_path).convert("RGB"))
                            if result_img.shape[:2] != image.shape[:2]:
                                result_img = np.array(
                                    Image.fromarray(result_img).resize((image.shape[1], image.shape[0]))
                                )
                            return result_img
                
                print(f"      ⚠ Output nicht gefunden")
                return image
                
            except subprocess.TimeoutExpired:
                print(f"      ⚠ Image-GS Timeout")
                return image
            except subprocess.CalledProcessError as e:
                print(f"      ⚠ Image-GS Fehler: {e.stderr.decode() if e.stderr else str(e)}")
                return image
            except Exception as e:
                print(f"      ⚠ Image-GS Fehler: {e}")
                return image
