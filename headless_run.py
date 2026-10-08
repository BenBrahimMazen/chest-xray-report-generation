# Headless executor: runs the WHOLE notebook on a Modal A10G with no
# browser and no client connection. nbconvert executes every cell and
# writes the outputs INTO the .ipynb, which is then saved to the volume
# — so the deliverable notebook-with-outputs exists even if no laptop
# is ever connected. The notebook is read from (and the executed copy
# written back to) the volume root; the volume is otherwise EMPTY, so
# Stage 1.3 imports nothing and every stage executes for real.
import modal

app = modal.App("projet-llm-headless")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision", "ipywidgets", "pandas", "matplotlib", "pillow", "tqdm", "requests",
        "transformers>=4.50.0,<5.0.0", "accelerate", "peft", "bitsandbytes",
        "qwen-vl-utils", "evaluate", "rouge_score", "bert-score", "nltk",
        "sentencepiece", "scikit-learn",
    )
    .pip_install("jupyter")  # nbconvert + ipykernel
)

volume = modal.Volume.from_name("projet-artefacts", create_if_missing=True)


@app.function(
    gpu="A10G", cpu=4, memory=16384, timeout=86400,
    image=image,
    volumes={"/root/lab/volume": volume},
)
def run_notebook():
    import os, shutil, subprocess

    src = "/root/lab/volume/medical_report_generation.ipynb"
    os.makedirs("/root/work", exist_ok=True)
    work = "/root/work/medical_report_generation.ipynb"
    shutil.copy2(src, work)

    print("nbconvert starting: full pipeline from scratch (expect 9-13 h)", flush=True)
    rc = subprocess.run(
        [
            "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--inplace", "--allow-errors", work,
            "--ExecutePreprocessor.startup_timeout=600",
            "--ExecutePreprocessor.timeout=85000",
        ]
    ).returncode
    print(f"nbconvert finished with exit code {rc}", flush=True)

    shutil.copy2(work, "/root/lab/volume/medical_report_generation.ipynb")
    print("EXECUTED notebook (with all outputs) saved to volume root", flush=True)


# NOTE ON LAUNCH (learned the hard way 2026-10-06):
#   - `modal run` (even with -d/--detach) ties the app to the local CLI
#     process: when that process dies, the app stops. NOT good enough for an
#     overnight run behind household wifi that shuts down.
#   - Correct pattern: DEPLOY the app (server-side, permanent), then trigger
#     the run with a fire-and-forget spawn. The spawned call belongs to the
#     deployed app — no client dependency at all:
#       modal deploy headless_run.py
#       python -c "import modal; modal.Function.from_name('projet-llm-headless', 'run_notebook').spawn()"
