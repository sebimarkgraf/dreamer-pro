import os
import threading
from pathlib import Path

import gym
import numpy as np

import common

import distractor_dmc2gym as dmc2gym
from wrap_your_gym import ObsDict, ActionDict
from gym.wrappers import StepAPICompatibility
import mbrl_envs



background_path = Path(__file__).parent / ".." / ".." / "background"

def make_mbrl_env(domain, task, distraction, seed, image_size, distraction_location):
    print("Making MBRL ENV")
    env = mbrl_envs.make(domain_name=domain,
                         task_name=task,
                         seed=seed,
                         action_repeat=2,
                         obs_type=mbrl_envs.ObsTypes.IMAGE,
                         img_size=image_size,
                         distraction_location=distraction_location,
                         distraction_type=distraction,
                         no_lists=True,
                         distractor_paths={
                             "kinetics400": str(background_path / "kinetics400"),
                             "pre_rendered_disks_medium": str(background_path / "disks_medium")
                         })
    env = ObsDict(env, key="image")
    env = ActionDict(env)
    env = StepAPICompatibility(env, output_truncation_bool=False)

    return env



def make_alr_env(domain, task, distraction):
    print("Making ALR ENV")
    env = dmc2gym.make(
        domain_name=domain,
        task_name=task,
        frame_skip=2,
        height=64,
        width=64,
        camera_id=0,
        from_pixels=True,
        environment_kwargs=None,
        visualize_reward=False,
        channels_first=False,
        distraction_source=distraction,
        distraction_location='background'
    )
    env = ObsDict(env, key="image")
    env = ActionDict(env)
    env = StepAPICompatibility(env, output_truncation_bool=False)

    return env


class DMC:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        os.environ['MUJOCO_GL'] = 'egl'
        ################################################################################
        # Implementation of Cheetah/Walker Run Sparse follows
        # https://github.com/younggyoseo/RE3
        self._name = name
        if name == 'cheetah_run_sparse':
            name = 'cheetah_run'
        if name == 'walker_run_sparse':
            name = 'walker_run'
        ################################################################################
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return gym.spaces.Dict({'action': action})

    def step(self, action):
        action = action['action']
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            ################################################################################
            step_reward = time_step.reward
            if step_reward is not None:
                if self._name in ('cheetah_run_sparse', 'walker_run_sparse'):
                    reward += 0 if step_reward < 0.25 else step_reward
                else:
                    reward += step_reward
            ################################################################################
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


################################################################################
# Implementation of natural background DMC follows
# https://github.com/VinAIResearch/TPC-tensorflow
class NAT:

    def __init__(
            self, name, action_repeat=1, size=(64, 64), camera=None,
            bg_path=None, random_bg=False, max_videos=100):
        os.environ['MUJOCO_GL'] = 'egl'
        ################################################################################
        # Implementation of Cheetah/Walker Run Sparse follows
        # https://github.com/younggyoseo/RE3
        self._name = name
        if name == 'cheetah_run_sparse':
            name = 'cheetah_run'
        if name == 'walker_run_sparse':
            name = 'walker_run'
        ################################################################################
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        ################################################################################
        assert bg_path is not None
        files = [
            os.path.join(bg_path, f) for f in os.listdir(bg_path)
            if os.path.isfile(os.path.join(bg_path, f))]
        self._bg_source = common.RandomVideoSource(
            size, files, random_bg, max_videos, grayscale=False)
        ################################################################################

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return gym.spaces.Dict({'action': action})

    def step(self, action):
        action = action['action']
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            ################################################################################
            step_reward = time_step.reward
            if step_reward is not None:
                if self._name in ('cheetah_run_sparse', 'walker_run_sparse'):
                    reward += 0 if step_reward < 0.25 else step_reward
                else:
                    reward += step_reward
            ################################################################################
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        ################################################################################
        self._bg_source.reset()
        ################################################################################
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        ################################################################################
        obs = self._env.physics.render(*self._size, camera_id=self._camera)
        mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
        bg = self._bg_source.get_image()
        obs[mask] = bg[mask]
        return obs
        ################################################################################


################################################################################


class Atari:
    LOCK = threading.Lock()

    def __init__(
            self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
            life_done=False, sticky_actions=True, all_actions=False):
        assert size[0] == size[1]
        import gym.wrappers
        import gym.envs.atari
        if name == 'james_bond':
            name = 'jamesbond'
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name, obs_type='image', frameskip=1,
                repeat_action_probability=0.25 if sticky_actions else 0.0,
                full_action_space=all_actions)
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
        env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale)
        self._env = env
        self._grayscale = grayscale

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'image': self._env.observation_space,
            'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
        })

    @property
    def action_space(self):
        return gym.spaces.Dict({'action': self._env.action_space})

    def close(self):
        return self._env.close()

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        obs = {'image': image, 'ram': self._env.env._get_ram()}
        return obs

    def step(self, action):
        action = action['action']
        image, reward, done, info = self._env.step(action)
        if self._grayscale:
            image = image[..., None]
        obs = {'image': image, 'ram': self._env.env._get_ram()}
        return obs, reward, done, info

    def render(self, mode):
        return self._env.render(mode)


class Dummy:

    def __init__(self):
        pass

    @property
    def observation_space(self):
        image = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        return gym.spaces.Dict({'image': image})

    @property
    def action_space(self):
        action = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        return gym.spaces.Dict({'action': action})

    def step(self, action):
        obs = {'image': np.zeros((64, 64, 3))}
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        obs = {'image': np.zeros((64, 64, 3))}
        return obs



class NormalizeAction:

    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.action_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})


class OneHotAction:

    def __init__(self, env, key='action'):
        assert isinstance(env.action_space[key], gym.spaces.Discrete)
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space[self._key].n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env, key='reward'):
        super().__init__(env)
        assert key not in env.observation_space.spaces
        self._key = key
        reward_space = gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
        self.observation_space = {
            **env.observation_space.spaces, self._key: reward_space
        }

    def step(self, action):
        obs, reward, *others = self.env.step(action)
        return {**obs, 'reward': reward}, reward, *others

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return {**obs, 'reward': 0.0}, info


class ResetObs(gym.Wrapper):
    def __init__(self, env, key='reset'):
        super().__init__(env)
        assert key not in env.observation_space
        self._key = key
        self.observation_space = gym.spaces.Dict({
            **self.env.observation_space,
            self._key: gym.spaces.Box(0, 1, (), dtype=bool)
        })

    def step(self, action):
        obs, *others = self.env.step(action)
        obs['reset'] = np.array(False, bool)
        return (obs, *others)

    def reset(self):
        obs, info = self.env.reset()
        obs['reset'] = np.array(True, bool)
        return obs, info
