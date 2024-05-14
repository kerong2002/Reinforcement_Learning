import random
import numpy as np
from collections import deque
from sumtree import SumTree  # 使用 SumTree 來實現優先級隨機抽樣

class Buffer:
    def __init__(self, size: int, prioritized=None, *args, **kwargs):
        """
        初始化緩衝區

        Args:
            size (int): 緩衝區大小
            prioritized (dict, optional): 優先級抽樣的相關參數，預設為 None
            *args, **kwargs: 其他參數
        """
        assert size > 0
        # 如果未指定優先級，則使用 deque 來實現普通的循環緩衝區；否則使用 SumTree 實現優先級緩衝區
        self.buffer = (deque(maxlen=size) if prioritized is None else SumTree(maxlen=size, **prioritized))
        self.maxlen = size  # 緩衝區的最大長度
        self.prioritized = prioritized is not None  # 是否使用優先級抽樣
        self._temp_buffer = []  # 用於暫存的緩衝區

    @staticmethod
    def _to_numpy(batch):
        """
        轉換抽樣的批次數據為 NumPy 格式

        Args:
            batch (list): 抽樣的批次數據

        Returns:
            tuple: 轉換後的 NumPy 數據
        """
        obs, act, rew, obs_next, done = [], [], [], [], []
        for o, a, r, o_, d in batch:
            obs.append(np.concatenate(o))
            act.append(a)
            rew.append(r)
            obs_next.append(np.concatenate(o_))
            done.append(d)

        return (np.array(obs, copy=True, dtype=np.float32),
                np.array(act, copy=True, dtype=np.int64)[:, np.newaxis],
                np.array(rew, copy=True, dtype=np.float32)[:, np.newaxis],
                np.array(obs_next, copy=True, dtype=np.float32),
                np.array(done, copy=True, dtype=np.float32)[:, np.newaxis])

    @property
    def is_full(self):
        """
        判斷緩衝區是否已滿

        Returns:
            bool: True 表示緩衝區已滿，False 表示緩衝區未滿
        """
        return len(self.buffer) == self.maxlen

    def add(self, obs, act, rew, done, obs_next):
        """
        將新的遷移添加到緩衝區中

        Args:
            obs: 觀察
            act: 動作
            rew: 獎勵
            done: 是否終止
            obs_next: 下一個觀察

        Returns:
            None
        """
        self._temp_buffer.append((obs, act, rew, done))
        if len(self._temp_buffer) == 2:
            prev_obs, prev_act, prev_rew, prev_done = self._temp_buffer.pop(0)
            self.buffer.append((prev_obs, prev_act, prev_rew, obs, prev_done))
            if done:
                self.buffer.append((obs, act, rew, obs_next, done))
                self._temp_buffer = []

    def sample(self, batch_size):
        """
        從緩衝區中抽樣指定大小的批次數據

        Args:
            batch_size (int): 批次大小

        Returns:
            tuple: 抽樣的批次數據
        """
        batch = random.sample(self.buffer, batch_size)
        return self._to_numpy(batch)

    def prioritized_sample(self, batch_size):
        """
        從優先級緩衝區中抽樣指定大小的批次數據

        Args:
            batch_size (int): 批次大小

        Returns:
            tuple: 抽樣的批次數據和對應的索引
        """
        batch, indices = self.buffer.sample(batch_size)
        return self._to_numpy(batch), indices

    def update_priority(self, priorities, indices):
        """
        更新抽樣數據的優先級

        Args:
            priorities (list): 新的優先級列表
            indices (list): 抽樣數據的索引列表

        Returns:
            ndarray: 正規化後的權重
        """
        weights = []
        for prio, idx in zip(priorities, indices):
            weights.append(self.buffer.update_prio(prio, idx))
        weights = np.array(weights, dtype=np.float32)
        return weights / np.max(weights)

    def step(self):
        """
        更新優先級緩衝區的內部狀態
        """
        if self.prioritized:
            self.buffer.step_beta()

    def __len__(self):
        """
        返回緩衝區中的樣本數量

        Returns:
            int: 緩衝區中的樣本數量
        """
        return len(self.buffer)

    def __str__(self):
        """
        返回緩衝區的字符串表示形式

        Returns:
            str: 緩衝區的字符串表示形式
        """
        return '\n'.join(map(str, self.buffer))


class MultistepBuffer(Buffer):
    def __init__(self, size: int, n: int = 5, gamma: float = 0.9, prioritized=None):
        """
        初始化多步驟緩衝區

        Args:
            size (int): 緩衝區大小
            n (int, optional): 多步驟的步數，預設為 5
            gamma (float, optional): 折扣因子，預設為 0.9
            prioritized (dict, optional): 優先級抽樣的相關參數，預設為 None
        """
        super(MultistepBuffer, self).__init__(size, prioritized=prioritized)
        self.n = n  # 多步驟的步數
        self.gamma = gamma  # 折扣因子

    def _add_nstep(self, obs_next, done):
        """
        將多步驟遷移添加到緩衝區中

        Args:
            obs_next: 下一個觀察
            done: 是否終止

        Returns:
            None
        """
        obs, act, rew, _ = self._temp_buffer.pop(0)
        gamma = self.gamma
        for rec in self._temp_buffer:
            rew += gamma * rec[2]  # 計算多步驟遷移的獎勵
            gamma *= self.gamma
        self.buffer.append((obs, act, rew, obs_next, done))

    def add(self, obs, act, rew, done, obs_next):
        """
        將新的遷移添加到緩衝區中

        Args:
            obs: 觀察
            act: 動作
            rew: 獎勵
            done: 是否終止
            obs_next: 下一個觀察

        Returns:
            None
        """
        self._temp_buffer.append((obs, act, rew, done))
        if done:
            self._add_nstep(obs, False)
            while len(self._temp_buffer) > 0:
                self._add_nstep(obs_next, done)
        elif len(self._temp_buffer) > self.n:
            self._add_nstep(obs, done)
