# Hunyuan3D-2mv

A Cog implementation of [Tencent's Hunyuan3D-2mv](https://github.com/Tencent/Hunyuan3D), which generates high-quality 3D models from multiple view images.

## Model Description

Hunyuan3D-2mv is a multi-view 3D generation model that can create detailed 3D meshes from up to four input images (front, back, left, and right views). The model uses advanced techniques to generate consistent 3D shapes with accurate geometry.

## Examples

Generate a 3D model from a single front view:
```bash
cog predict -i front_image=@dog-front.png
```

Generate a 3D model using multiple views:
```bash
cog predict -i front_image=@object-front.png -i back_image=@object-back.png -i left_image=@object-left.png -i right_image=@object-right.png
```

## Input Parameters

- `front_image`: (Required) Front view image of the object
- `back_image`: (Optional) Back view image
- `left_image`: (Optional) Left view image
- `right_image`: (Optional) Right view image
- `steps`: Number of inference steps (default: 30, range: 1-100)
- `guidance_scale`: Guidance scale (default: 5.0)
- `seed`: Random seed (default: 1234)
- `octree_resolution`: Octree resolution (default: 256, range: 16-512)
- `remove_background`: Whether to remove image background (default: true)
- `num_chunks`: Number of chunks (default: 200000, range: 1000-5000000)
- `randomize_seed`: Whether to randomize seed (default: true)
- `target_face_num`: Target number of faces for mesh simplification (default: 10000, range: 100-1000000)
- `file_type`: Output file type, one of "glb", "obj", "ply", "stl" (default: "glb")

## Output

The model outputs a 3D mesh file in the specified format (GLB, OBJ, PLY, or STL). The mesh is processed with face reduction to the specified target face number and is exported with a neutral gray color.
