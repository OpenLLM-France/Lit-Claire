import os
import sys
import json
import glob
import shutil
from pathlib import Path
from typing import Optional

wd = os.path.dirname(os.path.realpath(__file__))

from utils.run_command import run_command
from utils.hf_upload_model import upload_to_huggingface_hub

path_lit_gpt_script = os.path.join(wd, "lit_gpt", "scripts")
assert os.path.isdir(path_lit_gpt_script), f"Cannot find {path_lit_gpt_script}"

script_convert_model = os.path.join(path_lit_gpt_script, "convert_lit_checkpoint.py")
script_split_large_model = os.path.join(wd, "utils", "hf_split_large_model.py")
script_merge_lora = os.path.join(wd, "utils", "merge_lora.py")
assert os.path.isfile(script_convert_model), f"Cannot find {script_convert_model}"
assert os.path.isfile(script_split_large_model), f"Cannot find {script_split_large_model}"
assert os.path.isfile(script_merge_lora), f"Cannot find {script_merge_lora}"


def convert_lit_checkpoint(
    input_path: Path,
    output_dir: Path,
    hf_files_dir = Path(wd) / "hf_files" / "falcon_v01",
    checkpoint_dir: Optional[Path] = None,
    repo_id: Optional[str] = None,
    merge_lora: Optional[bool] = None,
    overwrite_existing: bool = False,
    message: Optional[str] = None,
    clean: bool = False,
):
    if not input_path.is_file():
        raise FileNotFoundError(f"Cannot find input chekpoint file {input_path}")

    if not hf_files_dir.is_dir():
        raise FileNotFoundError(f"Cannot find input folder {hf_files_dir}")

    input_dir = input_path.parent

    if checkpoint_dir is None:
        hparams = input_dir / "hparams.json" 
        if not hparams.exists():
            raise FileNotFoundError(f"Cannot find hyperparameter file {input_dir}/hparams.json")
        with open(hparams, "r") as f:
            hparams = json.load(f)
        assert "checkpoint_dir" in hparams, f"Cannot find 'checkpoint_dir' in {hparams}"
        checkpoint_dir = Path(hparams["checkpoint_dir"])
        if merge_lora is None:
            merge_lora = hparams.get("use_lora", False)

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Cannot find base checkpoint folder {checkpoint_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Copy HuggingFace files from foundation model
    if checkpoint_dir != output_dir:
        for file in (
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ):
            if not overwrite_existing and (output_dir / file).exists():
                continue
            if not (checkpoint_dir / file).exists():
                continue
            shutil.copy2(checkpoint_dir / file, output_dir / file)

    # Copy HuggingFace files specific to the new model (always overwrite)
    for file in os.listdir(hf_files_dir):
        if not (hf_files_dir / file).is_file():
            continue
        shutil.copy2(hf_files_dir / file, output_dir / file)

    lit_model_path = output_dir / "lit_model.pth" if merge_lora else None
    pytorch_model_path = output_dir / "pytorch_model.bin"
    has_splitted_torch_model = len(glob.glob(str(output_dir / "pytorch_model-*.bin")))

    if not clean and merge_lora:
        shutil.copy2(input_path, output_dir / "lit_lora.pth")

    if overwrite_existing or not has_splitted_torch_model or not clean:

        if overwrite_existing or not os.path.isfile(pytorch_model_path) or not clean:

            if merge_lora:
                if overwrite_existing or not lit_model_path.exists():

                    # Merge LoRA weights by running
                    #
                    # python merge_lora.py \
                    # --lora_path <<..>>/pytorch_model.bin \
                    # --checkpoint_dir <<..>>/lit_model.pth \
                    # --save_path <<..>>/lit_model.pth

                    run_command(
                        [
                            sys.executable,  # python
                            script_merge_lora,
                            "--lora_path",
                            str(input_path),
                            "--checkpoint_dir",
                            str(checkpoint_dir),
                            "--save_path",
                            str(lit_model_path),
                        ],
                        need_gpu=True,
                    )

                assert os.path.isfile(lit_model_path)

                input_path = lit_model_path

            # Convert the model by running
            #
            # python lit_gpt/scripts/convert_lit_checkpoint.py \
            # --checkpoint_path <<..>>/lit_model.pth \
            # --output_path <<..>>/pytorch_model.bin \
            # --config_path <<..>>/lit_config.json

            if overwrite_existing or (not os.path.isfile(pytorch_model_path) and not has_splitted_torch_model):

                try:
                    run_command(
                        [
                            sys.executable,  # python
                            script_convert_model,
                            "--checkpoint_path",
                            str(input_path),
                            "--output_path",
                            str(pytorch_model_path),
                            "--config_path",
                            str(checkpoint_dir / "lit_config.json"),
                        ],
                    )
                except Exception as err:
                    if os.path.isfile(pytorch_model_path):
                        os.remove(pytorch_model_path)
                    raise(err)

        assert os.path.isfile(pytorch_model_path) or has_splitted_torch_model

        if lit_model_path and os.path.isfile(lit_model_path) and clean:
            os.remove(lit_model_path)

        # Split the model if it's too big
        #
        # Running something like
        # srun --ntasks=1 --gres=gpu:1 --constraint=a100 --cpus-per-task=8 \
        #   python split_model.py \
        #   --folder_path $WORK/../commun/Claire/checkpoints/OpenLLM-France/Claire-7B

        if overwrite_existing or not has_splitted_torch_model:
            run_command(
                [
                    sys.executable,  # python
                    script_split_large_model,
                    "--folder_path",
                    str(output_dir),
                ],
                need_gpu=True,
            )

        # Remove the big model if it has been splitted into smaller ones
        if has_splitted_torch_model and os.path.isfile(pytorch_model_path):
            os.remove(pytorch_model_path)

    if repo_id:
        upload_to_huggingface_hub(repo_id=repo_id, input_dir=output_dir, message=message)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_lit_checkpoint, as_positional=False)
