import argparse
import os
import time
import requests
from fastapi import FastAPI, HTTPException, Body
import uvicorn

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()


def img_to_3d(image_path, output_glb):
    image = Image.open(image_path)
    # image.save(os.path.join(output_folder, "preview.png"))
    # Run the pipeline
    outputs = pipeline.run(
        image,
        seed=1,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3,
        },
    )
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,  # Ratio of triangles to remove in the simplification process
        texture_size=1024,  # Size of the texture used for the GLB
    )
    glb.export(output_glb)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", default=9071, type=int)
    parser.add_argument("--save_folder", type=str, default='../neural-subnet/generate/outputs')

    args = parser.parse_args()
    return args


args = get_args()

app = FastAPI()


@app.post("/generate_from_text")
async def text_to_3d(prompt: str = Body(), steps: int = None, seed: int = None):
    start = time.time()

    output_folder = os.path.join(args.save_folder, "text_to_3d")
    os.makedirs(output_folder, exist_ok=True)

    url = "http://localhost:9072/text_to_image"
    payload = {"prompt": prompt, "steps": steps, "seed": seed}

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"err to request text_to_image. {response.text}")
        return {"success": False, "path": output_folder}
    print(f"response: {response.text}")

    img_to_3d(os.path.join(output_folder, "mesh.png"), os.path.join(output_folder, "mesh.glb"))

    print(f"Successfully generated: {output_folder}")
    print(f"Generation time: {time.time() - start}")

    return {"success": True, "path": output_folder}


@app.post("/generate_from_image")
async def image_to_3d(image_path: str):
    start = time.time()

    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail="Image file not found")

    output_folder = os.path.join(args.save_folder, "image_to_3d")
    os.makedirs(output_folder, exist_ok=True)

    img_to_3d(image_path, os.path.join(output_folder, "mesh.glb"))
    print(f"Successfully generated: {output_folder}")
    print(f"Generation time: {time.time() - start}")

    return {"success": True, "path": output_folder}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
