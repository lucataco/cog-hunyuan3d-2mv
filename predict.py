from cog import BasePredictor, Input, Path as CogPath
import os
import time
import uuid
import torch
import shutil
import random
import subprocess
from PIL import Image
from pathlib import Path

MODEL_CACHE = "Hunyuan3D-2mv"
MODEL_URL = "https://weights.replicate.delivery/default/tencent/Hunyuan3D-2mv/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.MAX_SEED = 1e7
        self.SAVE_DIR = "outputs"
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        # Download weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Import model-specific libraries
        from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
        from hy3dgen.shapegen.pipelines import export_to_trimesh
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.rembg import BackgroundRemover

        # Load background remover
        self.rmbg_worker = BackgroundRemover()
        
        # Load shape generation model
        model_path = 'Hunyuan3D-2mv'
        os.environ['HY3DGEN_MODELS'] = model_path
        subfolder = 'hunyuan3d-dit-v2-mv-fast'
        self.i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=False,
            device=self.device,
        )
        
        # Load post-processing tools
        self.floater_remove_worker = FloaterRemover()
        self.degenerate_face_remove_worker = DegenerateFaceRemover()
        self.face_reduce_worker = FaceReducer()
        self.export_to_trimesh = export_to_trimesh

    def gen_save_folder(self, max_size=200):
        """Generate a unique folder to save results"""
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        # Get all folder paths
        dirs = [f for f in Path(self.SAVE_DIR).iterdir() if f.is_dir()]

        # If folder count exceeds max_size, delete oldest folder
        if len(dirs) >= max_size:
            oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
            shutil.rmtree(oldest_dir)
            print(f"Removed the oldest folder: {oldest_dir}")

        # Generate a new uuid folder name
        new_folder = os.path.join(self.SAVE_DIR, str(uuid.uuid4()))
        os.makedirs(new_folder, exist_ok=True)
        print(f"Created new folder: {new_folder}")

        return new_folder

    def export_mesh(self, mesh, save_folder, textured=False, file_type='glb'):
        """Export mesh to a file"""
        if textured:
            path = os.path.join(save_folder, f'textured_mesh.{file_type}')
        else:
            # Set mesh color to gray instead of default white
            gray_color = [0.5, 0.5, 0.5, 1.0]  # RGBA for medium gray
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
                mesh.visual.face_colors = [int(c * 255) for c in gray_color]
            
            path = os.path.join(save_folder, f'gray_mesh.{file_type}')
            
        if file_type not in ['glb', 'obj']:
            mesh.export(path)
        else:
            mesh.export(path, include_normals=textured)
            
        return path

    def randomize_seed(self, seed, randomize_seed):
        """Randomize seed if requested"""
        if randomize_seed:
            seed = random.randint(0, int(self.MAX_SEED))
        return int(seed)

    def predict(
        self,
        front_image: CogPath = Input(description="Front view image"),
        back_image: CogPath = Input(description="Back view image", default=None),
        left_image: CogPath = Input(description="Left view image", default=None),
        right_image: CogPath = Input(description="Right view image", default=None),
        steps: int = Input(description="Number of inference steps", default=30, ge=1, le=100),
        guidance_scale: float = Input(description="Guidance scale", default=5.0),
        seed: int = Input(description="Random seed", default=1234),
        octree_resolution: int = Input(description="Octree resolution", default=256, ge=16, le=512),
        remove_background: bool = Input(description="Remove image background", default=True),
        num_chunks: int = Input(description="Number of chunks", default=200000, ge=1000, le=5000000),
        randomize_seed: bool = Input(description="Randomize seed", default=True),
        target_face_num: int = Input(description="Target number of faces for mesh simplification", default=10000, ge=100, le=1000000),
        file_type: str = Input(description="Output file type", choices=["glb", "obj", "ply", "stl"], default="glb"),
    ) -> CogPath:
        """Generate 3D model from input images"""
        # Check that at least one image is provided
        if all(img is None for img in [front_image, back_image, left_image, right_image]):
            raise ValueError("Please provide at least one view image.")

        # Seed handling
        seed = self.randomize_seed(seed, randomize_seed)
        
        # Process input images
        image = {}
        if front_image:
            image['front'] = Image.open(front_image)
        if back_image:
            image['back'] = Image.open(back_image)
        if left_image:
            image['left'] = Image.open(left_image)
        if right_image:
            image['right'] = Image.open(right_image)
            
        # Remove background if requested
        if remove_background:
            for k, v in image.items():
                if v.mode == "RGB":
                    img = self.rmbg_worker(v.convert('RGB'))
                    image[k] = img
        
        # Generate 3D shape
        start_time = time.time()
        generator = torch.Generator()
        generator = generator.manual_seed(seed)
        
        outputs = self.i23d_worker(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type='mesh'
        )
        
        print(f"Shape generation took {time.time() - start_time:.2f} seconds")
        
        # Convert to trimesh
        mesh = self.export_to_trimesh(outputs)[0]
        
        # Post-processing
        # Reduce faces if needed
        mesh = self.face_reduce_worker(mesh, target_face_num)
        
        # Generate texture if requested and available
        save_folder = self.gen_save_folder()
        
        # Export untextured mesh
        untextured_path = self.export_mesh(mesh, save_folder, textured=False, file_type=file_type)
        result = CogPath(untextured_path)
            
        return result