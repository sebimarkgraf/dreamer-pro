import collections
import functools
import logging
import os
import pathlib
import sys
import warnings
import time

import gym

import wandb

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf

import agent
import elements
import common
import hydra
from omegaconf import OmegaConf


def assert_train_on_gpu(config):
  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  # assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))

def make_env(mode, config):
  suite = config.env.suite
  domain, task = config.env.task.split('-')
  if suite == 'alr':
    distraction = config.env.distraction
    env = common.make_alr_env(domain, task, distraction)
  elif suite == 'dmc':
    env = common.DMC(task, config.action_repeat, config.image_size)
    env = common.NormalizeAction(env)
  elif suite == 'nat':
    bg_path = config.bg_path_train if mode == 'train' else config.bg_path_test
    env = common.NAT(
        task, config.action_repeat, config.image_size,
        bg_path=bg_path, random_bg=config.random_bg, max_videos=config.max_videos)
    env = common.NormalizeAction(env)
  elif suite == 'atari':
    env = common.Atari(
        task, config.action_repeat, config.image_size, config.grayscale,
        life_done=False, sticky_actions=True, all_actions=True)
    env = common.OneHotAction(env)
  else:
    raise NotImplementedError(suite)
  env = gym.wrappers.StepAPICompatibility(env, output_truncation_bool=True)
  env = gym.wrappers.TimeLimit(env, max_episode_steps=config.time_limit)
  env = common.RewardObs(env)
  env = common.ResetObs(env)
  env = gym.wrappers.StepAPICompatibility(env, output_truncation_bool=False)
  return env

def per_episode(ep, mode, logger, start_time, train_replay, eval_replay):
  length = len(ep['reward']) - 1
  score = float(ep['reward'].astype(np.float64).sum())
  total_time = time.time() - start_time
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  replay_ = dict(train=train_replay, eval=eval_replay)[mode]
  replay_.add(ep)
  logger.scalar(f'{mode}_transitions', replay_.num_transitions)
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_hours', total_time / 3600)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_eps', replay_.num_episodes)
  logger.write()

@hydra.main(version_base="1.1", config_path="./config", config_name="config")
def main(config):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  logging.getLogger().setLevel('ERROR')
  warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

  config = OmegaConf.to_container(config, resolve=True)
  config = elements.Config(config)

  logdir = pathlib.Path(config.logdir).expanduser()
  config = config.update(
      steps=config.steps // config.action_repeat,
      eval_every=config.eval_every // config.action_repeat,
      log_every=config.log_every // config.action_repeat,
      time_limit=config.time_limit // config.action_repeat,
      prefill=config.prefill // config.action_repeat)




  wandb.init(sync_tensorboard=True)

  print('Logdir', logdir)
  train_replay = common.Replay(logdir / 'train_replay', config.replay_size)
  eval_replay = common.Replay(logdir / 'eval_replay', config.time_limit or 1)
  step = elements.Counter(train_replay.total_steps)
  outputs = [
      elements.TerminalOutput(),
      elements.JSONLOutput(logdir),
      elements.TensorBoardOutput(logdir),
  ]
  logger = elements.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)
  should_train = elements.Every(config.train_every)
  should_log = elements.Every(config.log_every)

  print('Create envs.')
  train_envs = [make_env('train', config) for _ in range(config.num_envs)]
  eval_envs = [make_env('eval', config) for _ in range(config.num_envs)]
  action_space = train_envs[0].action_space['action']
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train', logger=logger, start_time=start_time, train_replay=train_replay, eval_replay=eval_replay))
  train_driver.on_step(lambda _: step.increment())
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval', logger=logger, start_time=start_time, train_replay=train_replay, eval_replay=eval_replay))

  start_time = time.time()

  prefill = max(0, config.prefill - train_replay.total_steps)
  if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(action_space)
    train_driver(random_agent, steps=1, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

  print('Create agent.')
  train_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, logger, action_space, step, train_dataset)

  if config.pretrain:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      agnt.train(next(train_dataset))

  def train_step(tran):
    if should_train(step):
      for _ in range(config.train_steps):
        _, mets = agnt.train(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.write(fps=True)
  train_driver.on_step(train_step)

  while step < config.steps:
    logger.write()
    print('Start evaluation.')
    eval_policy = functools.partial(agnt.policy, mode='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
    print('Start training.')
    train_driver(agnt.policy, steps=config.eval_every)
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == "__main__":
  main()