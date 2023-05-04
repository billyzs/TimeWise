#!/usr/bin/env python3

import functools
import os
import sys
from pathlib import Path
import pickle
import logging

import gin
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=16'
import jax

from datetime import datetime
from jax import numpy as jp
import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.training.agents.apg import train as apg
import wandb


wandb_init = gin.external_configurable(wandb.init, name="wandb_init")


def progress(num_steps, metrics):
    logging.info(f"{num_steps=}")
    logging.info(f"{metrics=}")
    try:
        metrics_cpu = {k:float(v) for k,v in metrics.items()}
        wandb.log(metrics_cpu, step=num_steps)
    except Exception as e:
        logging.exception(e)


@gin.configurable
def training_main(
        env_name: str,
        backend: str,
        experiment_name: str,
        log_store_dir: str,
        train_config: dict,
):
    exp_dir = f"{log_store_dir}/{experiment_name}/{env_name}/{backend}"
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    seed = train_config.get("seed")
    env = envs.get_environment(env_name=env_name,
                               backend=backend)
    train_fn = functools.partial(
        ppo.train,
        **train_config,
    )

    # no way to save intermediate params, only at very end
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
    )
    params_path = f"{exp_dir}/params.pickle"
    model.save_params(params_path, params)
    wandb.save(params_path)
    wandb.save(f"{exp_dir}/*.log")
    return env_name, backend, make_inference_fn, params


def eval_main(
    env_name,
    backend,
    make_inference_fn,
    params,
):
    env = envs.get_environment(env_name=env_name, backend=backend)
    inference_fn = make_inference_fn(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)
    for rollout_idx in range(10):
        logging.info(f"{rollout_idx=}")
        rollout = []
        rng = jax.random.PRNGKey(seed=rollout_idx)
        state = jit_env_reset(rng=rng)
        act_rng, rng = jax.random.split(rng)
        done = False
        rollout_len = 0
        while not done:
            rollout_len += 1
            act, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_env_step(state, act)
            rollout.append(state.pipeline_state)
            done = state.done.any() or rollout_len > 2000
        logging.info(f"{rollout_idx=} {len(rollout)=}")
        # TODO bzs save render to json
        wandb.Html(html.render(env.sys.replace(dt=env.dt), rollout))


if __name__ == "__main__":
    gin_config_file = sys.argv[1]
    logging.basicConfig(level=logging.INFO)
    gin.parse_config_files_and_bindings([
        gin_config_file
    ],
        bindings=None,
        skip_unknown=False,
        print_includes_and_imports=True,
    )
    wandb_init(
        config=gin.get_bindings(training_main),
        notes=gin.config_str(),
        settings=wandb.Settings(start_mode="fork"),
    )
    try:
        logging.info(gin.config_str())
        env_name, backend, make_inference_fn, params = training_main()
        eval_main(env_name, backend, make_inference_fn, params)
    except KeyboardInterrupt:
        logging.warning("user quit")
    finally:
        wandb.finish()
