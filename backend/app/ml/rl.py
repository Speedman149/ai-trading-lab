"""Gymnasium environment backed by the exact Backtrader broker, not a second simulator."""

import json
import math
from queue import Empty, Queue
from threading import Thread
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.backtesting.engine import run_engine
from app.ml.features import FEATURES


def equity_reward(previous, current, previous_drawdown, drawdown, penalty=0):
    return math.log(current / previous) - penalty * max(0, abs(drawdown) - abs(previous_drawdown))


class EpisodeClosed(Exception):
    pass


class TradingEnvironment(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, frame, features, request, drawdown_penalty=0):
        super().__init__()
        self.frame, self.request = frame, request
        self.lookup = {
            row.timestamp.isoformat(): row[FEATURES].to_numpy(dtype=np.float32)
            for _, row in features.iterrows()
        }
        self.penalty = drawdown_penalty
        self.action_space = spaces.Discrete(2)  # target cash / target long
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(FEATURES) + 4,), dtype=np.float32)
        self.worker = None
        self.outputs, self.actions = Queue(), Queue()
        self.last = None
        self.result = None

    def _packet(self, point, terminal, index):
        feature = self.lookup[self.frame.iloc[index].timestamp.isoformat()]
        value = point["equity"]
        peak = max(self.peak, value)
        self.peak = peak
        drawdown = value / peak - 1
        observation = np.concatenate(
            [
                feature,
                np.array(
                    [point["cash"] / value, point["exposure"], value / self.request.initial_cash, drawdown],
                    dtype=np.float32,
                ),
            ]
        )
        return observation, point, drawdown, terminal

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.outputs, self.actions = Queue(), Queue()
        self.peak = self.request.initial_cash
        self.result = None
        env = self

        class InteractivePolicy:
            index = -1
            point = None

            def observe(self, point, terminal):
                self.index += 1
                self.point = point
                if terminal:
                    env.outputs.put(env._packet(point, True, self.index))

            def target(self, history):
                env.outputs.put(env._packet(self.point, False, self.index))
                try:
                    action = env.actions.get(timeout=30)
                except Empty:
                    raise EpisodeClosed() from None
                if action is None:
                    raise EpisodeClosed()
                return float(action)

        def run():
            try:
                env.result = run_engine(env.frame, env.request, InteractivePolicy())
            except EpisodeClosed:
                pass
            except Exception as exc:  # noqa: BLE001 - propagate broker failures across the worker boundary
                env.outputs.put(exc)

        self.worker = Thread(target=run, daemon=True)
        self.worker.start()
        self.last = self._receive()
        return self.last[0], {"equity": self.last[1]["equity"]}

    def _receive(self):
        try:
            item = self.outputs.get(timeout=30)
        except Empty:
            self.close()
            raise RuntimeError("RL simulator timed out") from None
        if isinstance(item, Exception):
            self.close()
            raise item
        return item

    def step(self, action):
        if self.last is None or self.last[3]:
            raise RuntimeError("Reset the environment before stepping")
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0 (cash) or 1 (long)")
        self.actions.put(int(action))
        current = self._receive()
        reward = equity_reward(
            self.last[1]["equity"], current[1]["equity"], self.last[2], current[2], self.penalty
        )
        self.last = current
        if current[3]:
            self.worker.join(timeout=30)
            if self.worker.is_alive():
                raise RuntimeError("RL simulator did not finish")
        return (
            current[0],
            reward,
            current[3],
            False,
            {
                "equity": current[1]["equity"],
                "drawdown": current[2],
                "reward_net_log_return": reward,
            },
        )

    def close(self):
        if self.worker is not None and self.worker.is_alive():
            self.actions.put(None)
            self.worker.join(timeout=5)


class QModel:
    def __init__(self, volatility_threshold, q=None):
        self.volatility_threshold = float(volatility_threshold)
        self.q = np.zeros((32, 2)) if q is None else np.asarray(q, dtype=float)
        self.metadata = {}

    def state(self, observation):
        bits = [
            observation[1] > 0,
            observation[6] > 0,
            observation[3] > self.volatility_threshold,
            observation[len(FEATURES) + 1] > 0,
            observation[-1] < -0.05,
        ]
        return sum(int(bit) << index for index, bit in enumerate(bits))

    @classmethod
    def fit(cls, frame, features, training, config):
        # End at last training decision bar, never at a validation candle's close.
        start, end = int(training.bar_index.iloc[0]), int(training.bar_index.iloc[-1])
        episode = frame.iloc[start : end + 1].copy().reset_index(drop=True)
        episode.attrs = frame.attrs.copy()
        model = cls(training.volatility_20.median())
        rng = np.random.default_rng(config.seed)
        env = TradingEnvironment(episode, features, config.backtest, config.drawdown_penalty)
        rewards = []
        try:
            for iteration in range(config.episodes):
                observation, _ = env.reset(seed=config.seed + iteration)
                total, done = 0.0, False
                epsilon = max(0.05, 0.8 * (1 - iteration / config.episodes))
                while not done:
                    state = model.state(observation)
                    action = (
                        int(rng.integers(2)) if rng.random() < epsilon else int(np.argmax(model.q[state]))
                    )
                    following, reward, done, _, _ = env.step(action)
                    continuation = 0 if done else 0.95 * float(model.q[model.state(following)].max())
                    model.q[state, action] += 0.1 * (reward + continuation - model.q[state, action])
                    total += reward
                    observation = following
                rewards.append(total)
        finally:
            env.close()
        model.metadata = {
            "episodes": config.episodes,
            "episode_rewards": rewards,
            "trained_through": training.signal_time.iloc[-1].isoformat(),
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "reward": "net log equity change minus penalty * increase in absolute drawdown",
            "drawdown_penalty": config.drawdown_penalty,
            "warnings": [
                "Tabular Q-learning is an experimental 32-state policy; seed sensitivity and limited state coverage can dominate results."
            ],
        }
        return model

    def predict(self, rows):
        return None, None

    def policy(self, rows):
        model = self

        class Policy:
            def __init__(self):
                self.lookup = {
                    row.timestamp.isoformat(): row[FEATURES].to_numpy(dtype=float)
                    for _, row in rows.iterrows()
                }
                self.point = None
                self.peak = 0
                self.last_target = 0.0
                self.decisions = {}

            def observe(self, point, terminal):
                self.point = point
                self.peak = max(self.peak, point["equity"])

            def target(self, history):
                timestamp = history.iloc[-1].timestamp.isoformat()
                feature = self.lookup.get(timestamp)
                if feature is not None:
                    value = self.point["equity"]
                    observation = np.concatenate(
                        [
                            feature,
                            [self.point["cash"] / value, self.point["exposure"], 1, value / self.peak - 1],
                        ]
                    )
                    self.last_target = float(np.argmax(model.q[model.state(observation)]))
                    self.decisions[timestamp] = self.last_target
                return self.last_target

        return Policy()

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.json").write_text(
            json.dumps(
                {
                    "algorithm": "q_learning",
                    "volatility_threshold": self.volatility_threshold,
                    "q": self.q.tolist(),
                    "metadata": self.metadata,
                },
                allow_nan=False,
            )
        )

    @classmethod
    def load(cls, folder):
        saved = json.loads((folder / "model.json").read_text())
        model = cls(saved["volatility_threshold"], saved["q"])
        model.metadata = saved["metadata"]
        return model

    def importance(self):
        return []
