import json
import os
import time

import requests
import yaml
from loguru import logger
from huggingface_hub import HfApi

from train_lora import LoraTrainingArguments, train_lora
from utils.constants import model2base_model, model2size
from utils.flock_api import get_task, submit_task

HF_USERNAME = os.environ["HF_USERNAME"]


def get_modelsize(max_params: int) -> dict:
    model2size_out = {k: v for k, v in model2size.items() if v <= max_params}
    return model2size_out

def check_training_parameter(all_training_args: list, max_params: int) -> dict:

    model_sizes = get_modelsize(max_params)
    #print(model_sizes)

    arg_out = []
    for training_args in all_training_args:
        mode_id=list(training_args)[0]
        if mode_id in model_sizes:
            arg_out.append(training_args)
        else:
            logger.info(f"model: {mode_id} has exceed max size {max_params}")
    
    #logger.info(f"Models within the max_params: {arg_out.keys()}")
    return arg_out


def training_automation(task_id: str, training_args: list, training_data_path=None, dry_run=False):

    task = get_task(task_id)
    max_params = task["data"]["max_params"]
    context_length = task["data"]["context_length"]

    if training_data_path is None:
    # download in chunks
        data_url = task["data"]["training_set_url"]
        response = requests.get(data_url, stream=True)
        training_data_path = "demo_data.jsonl"
        with open(training_data_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    all_training_args_list = check_training_parameter(training_args, max_params)

    hg_repo_id_list =[]

    # train all feasible models and merge
    for all_training_args in all_training_args_list:
        model_id = list(all_training_args.keys())[0]
        logger.info(f"Start to train the model {model_id}...")
        # if OOM, proceed to the next model
        try:
            train_lora(
                model_id=model_id,
                context_length=context_length,
                training_args=LoraTrainingArguments(**all_training_args[model_id]),
                data_file_path=training_data_path
            )
        except RuntimeError as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
            continue

        # generate a random repo id based on timestamp
        hg_repo_id = f"{model_id.replace('/', '-')}-" + str(int(time.time()))
        hg_repo_id_list.append(hg_repo_id)

        if not dry_run:

            try:
                logger.info("Start to push the lora weight to the hub...")
                api = HfApi(token=os.environ["HF_TOKEN"])
                api.create_repo(
                    f"{HF_USERNAME}/{hg_repo_id}",
                    repo_type="model",
                )
                api.upload_folder(
                    folder_path="outputs",
                    repo_id=f"{HF_USERNAME}/{hg_repo_id}",
                    repo_type="model",
                )
                # submit
                submit_task(
                    task_id, f"{HF_USERNAME}/{hg_repo_id}", model2base_model[model_id]
                )
                logger.info("Task submitted successfully")
            except Exception as e:
                logger.error(f"Error: {e}")
                logger.info("Proceed to the next model...")
            finally:
                # cleanup merged_model and output
                os.system("rm -rf merged_model")
                os.system("rm -rf outputs")
                continue
    return hg_repo_id_list


if __name__ == "__main__":
    task_id = os.environ["TASK_ID"]
    # load trainin args
    # define the path of the current file
    current_folder = os.path.dirname(os.path.realpath(__file__))
    with open(f"{current_folder}/training_args.yaml", "r") as f:
        all_training_args = yaml.safe_load(f)

    task = get_task(task_id)
    # log the task info
    logger.info(json.dumps(task, indent=4))
    # download data from a presigned url

    # filter out the model within the max_params
    training_automation(task_id, all_training_args, training_data_path=None)
    


    
