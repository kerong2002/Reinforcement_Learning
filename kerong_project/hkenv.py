import gc
import gym
import cv2
import time
import enum
import random
import pyautogui
import threading
import numpy as np
from mss.windows import MSS as mss

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.


class Actions(enum.Enum):
    @classmethod
    def random(cls):
        return random.choice(list(cls))


class Move(Actions):
    NO_OP = 0
    HOLD_LEFT = 1
    HOLD_RIGHT = 2
    # LOOK_LEFT = 3
    # LOOK_RIGHT = 4


class Attack(Actions):
    NO_OP = 0
    ATTACK = 1
    UP_ATTACK = 2
    # SPELL = 3


class Displacement(Actions):
    NO_OP = 0
    TIMED_SHORT_JUMP = 1
    TIMED_LONG_JUMP = 2
    DASH = 3


class HKEnv(gym.Env):
    """
    environment that interacts with Hollow knight game,
    implementation follows the gym custom environment API
    """

    KEYMAPS = {  # map each action to its corresponding key to press
        Move.HOLD_LEFT: 'a',
        Move.HOLD_RIGHT: 'd',
        # Move.LOOK_LEFT: 'a',
        # Move.LOOK_RIGHT: 'd',
        Displacement.TIMED_SHORT_JUMP: 'space',
        Displacement.TIMED_LONG_JUMP: 'space',
        Displacement.DASH: 'k',
        Attack.ATTACK: 'j',
        Attack.UP_ATTACK: ('w', 'j'),
        # Attack.SPELL: ('s', 'i')
    }
    REWMAPS = {  # map each action to its corresponding reward
        Move.HOLD_LEFT: 0,
        Move.HOLD_RIGHT: 0,
        # Move.LOOK_LEFT: 0,
        # Move.LOOK_RIGHT: 0,
        Displacement.TIMED_SHORT_JUMP: 0,
        Displacement.TIMED_LONG_JUMP: 0,
        Displacement.DASH: 1e-3,
        Attack.ATTACK: 1e-3,
        Attack.UP_ATTACK: -1e-1
        # Attack.SPELL: 1e-3
    }
    HP_CKPT = np.array([52, 91, 129, 169, 207, 246, 286, 324, 363], dtype=int)
    ACTIONS = [Move, Attack, Displacement]

    def __init__(self, obs_shape=(160, 160), rgb=False, gap=0.165, w1=.8, w2=.8, w3=-0.0001):
        """
        :param obs_shape: the shape of observation returned by step and reset
        :param w1: the weight of negative reward when being hit
                (for example, w1=1. means give -1 reward when being hit)
        :param w2: the weight of positive reward when hitting enemy
                (for example, w2=1. means give +1 reward when hitting enemy)
        :param w3: the weight of positive reward when not hitting nor being hit
                (for example, w3=-0.0001 means give -0.0001 reward when neither happens
        """
        self.monitor = self._find_window()      # 切取出視窗
        self.holding = []
        self.prev_knight_hp = None
        self.prev_enemy_hp = None
        self.prev_action = -1
        total_actions = np.prod([len(Act) for Act in self.ACTIONS])
        if rgb:
            obs_shape = (3,) + obs_shape
        else:
            obs_shape = (1,) + obs_shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                dtype=np.uint8, shape=obs_shape)
        self.action_space = gym.spaces.Discrete(int(total_actions))
        self.rgb = rgb
        self.gap = gap
        self._prev_time = None

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self._hold_time = 0.2
        self._fail_hold_rew = -1e-4
        self._timer = None
        self._episode_time = None

    # def locate_on_screen(self,  region, confidence):
    #     result = None
    #     try:
    #         result = pyautogui.locateOnScreen('./locator/attuned.png', region=region, confidence=confidence)
    #         if result:
    #             return result
    #         else:
    #             return result
    #     except pyautogui.ImageNotFoundException:
    #         print("圖像未找到->警告！")
    #         return None
    @staticmethod
    def _find_window():
        """
        切換視窗，讓hollow knight 可以在最上層運作
        """
        window = pyautogui.getWindowsWithTitle('Hollow Knight')
        assert len(window) == 1, f'found {len(window)} windows called Hollow Knight {window}'
        window = window[0]
        try:
            window.activate()
        except Exception:
            window.minimize()
            window.maximize()
            window.restore()
        window.moveTo(0, 0)

        geo = None
        conf = 0.9
        while geo is None:
            geo = pyautogui.locateOnScreen('./locator/geo.png', confidence=conf)
            conf = max(0.90, conf * 0.90)
            time.sleep(0.1)
        loc = {
            'left': geo.left - 36,
            'top': geo.top - 97,
            'width': 1020,
            'height': 692
        }
        # with mss() as sct:
        #     monitor = {'left': 144, 'top': 35, 'width': 1020, 'height': 692}
        #     x = sct.grab(monitor)
        #     frame = np.asarray(x, dtype=np.uint8)
        # knight_hp_bar = frame[:, :, 0]
        # cv2.imshow('x', knight_hp_bar)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return loc

    def _timed_hold(self, key, seconds):
        """
        use a separate thread to hold a key for given seconds
        if the key is already holding, do nothing and return 1,

        :param key: the key to be pressed
        :param seconds: time to hold the key
        :return: 1 if already holding, 0 when success
        """

        def timer_thread():
            pyautogui.keyDown(key)
            time.sleep(seconds)
            pyautogui.keyUp(key)
            time.sleep(0.0005)

        if self._timer is None or not self._timer.is_alive():
            # timer available, do timed action
            # ignore if there is already a timed action in progress
            self._timer = threading.Thread(target=timer_thread)
            self._timer.start()
            return 0
        else:
            return 1

    def _step_actions(self, actions):
        """
        release all non-timed holding keys,
        press keys corresponding to given actions

        :param actions: a list of actions
        :return: reward for doing given actions
        """
        t = self.gap - (time.time() - self._prev_time)
        if t > 0:
            time.sleep(t)
        # print(t)
        self._prev_time = time.time()

        for key in self.holding:
            pyautogui.keyUp(key)
        self.holding = []
        action_rew = 0
        for act in actions:
            if not act.value:
                continue
            key = self.KEYMAPS[act]
            action_rew += self.REWMAPS[act]

            if act.name.startswith('HOLD'):
                pyautogui.keyDown(key)
                self.holding.append(key)
            elif act.name.startswith('TIMED'):
                action_rew += (self._fail_hold_rew *
                               self._timed_hold(key, act.value * self._hold_time))
            elif isinstance(key, tuple):
                with pyautogui.hold(key[0]):
                    pyautogui.press(key[1])
            else:
                pyautogui.press(key)
        return action_rew

    def _to_multi_discrete(self, num):
        """
        interpret the single number to a list of actions

        :param num: the number representing an action combination
        :return: list of action enums
        """
        num = int(num)
        chosen = []
        for Act in self.ACTIONS:
            num, mod = divmod(num, len(Act))
            chosen.append(Act(mod))
        return chosen

    def _find_menu(self):
        """
        找到可以進入挑戰的位置
        """
        monitor = self.monitor
        # print(monitor)
        debug_left = int(monitor['left'] + monitor['width'] // 2)
        debug_top = int(monitor['top'] + monitor['height'] // 4)
        debug_width = int(monitor['width'] // 2)
        debug_height = int(monitor['height'] // 2)
        # print("test1 ", debug_top)
        monitor = (monitor['left'] + monitor['width'] // 2,
                   monitor['top'] + monitor['height'] // 4,
                   monitor['width'] // 2,
                   monitor['height'] // 2)
        # print("test2")
        # print(monitor)
        # with mss() as sct:
        #     test = {'left': debug_left, 'top': debug_top, 'width': debug_width, 'height': debug_height}
        #     x = sct.grab(test)
        #     frame = np.asarray(x, dtype=np.uint8)
        # # print("test3")
        # knight_hp_bar = frame[:, :, 0]
        # cv2.imshow('x', knight_hp_bar)
        # # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        try:
            result = pyautogui.locateOnScreen('./locator/attuned.png', region=monitor, confidence=0.925)
            if result:
                print("圖像找到了！")
                return True
            else:
                print("圖像未找到！")
                return False
        except pyautogui.ImageNotFoundException:
            print("圖像未找到->警告！")
            return False
        # return pyautogui.locateOnScreen('./locator/attuned.png',
        #                                 region=monitor,
        #                                 confidence=0.925)

    def observe(self, force_gray=False):
        """
        take a screenshot and identify enemy and knight's HP

        :param force_gray: override self.rgb to force return gray obs
        :return: observation (a resized screenshot), knight HP, and enemy HP
        """
        with mss() as sct:
            frame = np.asarray(sct.grab(self.monitor), dtype=np.uint8)
        enemy_hp_bar = frame[-1, 187:826, :]
        if (np.all(enemy_hp_bar[..., 0] == enemy_hp_bar[..., 1]) and
                np.all(enemy_hp_bar[..., 1] == enemy_hp_bar[..., 2])):
            # hp bar found
            enemy_hp = (enemy_hp_bar[..., 0] < 3).sum() / len(enemy_hp_bar)
        else:
            enemy_hp = 1.
        knight_hp_bar = frame[64, :, 0]
        checkpoint1 = knight_hp_bar[self.HP_CKPT]
        checkpoint2 = knight_hp_bar[self.HP_CKPT - 1]
        knight_hp = ((checkpoint1 > 200) | (checkpoint2 > 200)).sum()
        rgb = not force_gray and self.rgb
        obs = cv2.cvtColor(frame[:672, ...],

                           (cv2.COLOR_BGRA2RGB if rgb
                            else cv2.COLOR_BGRA2GRAY))
        obs = cv2.resize(obs,
                         dsize=self.observation_space.shape[1:],
                         interpolation=cv2.INTER_AREA)
        # make channel first
        obs = np.rollaxis(obs, -1) if rgb else obs[np.newaxis, ...]
        return obs, knight_hp, enemy_hp

    def step(self, actions):
        action_rew = 0
        if actions == self.prev_action:
            action_rew -= 2e-5
        self.prev_action = actions
        actions = self._to_multi_discrete(actions)
        action_rew += self._step_actions(actions)
        obs, knight_hp, enemy_hp = self.observe()

        win = self.prev_enemy_hp < enemy_hp
        lose = knight_hp == 0
        done = win or lose

        if win:
            lose = False
            enemy_hp = 0.
        hurt = knight_hp < self.prev_knight_hp
        hit = enemy_hp < self.prev_enemy_hp

        reward = (
                - self.w1 * hurt
                + self.w2 * hit
                + action_rew
        )
        if not (hurt or hit):
            reward += self.w3
        if win:  # extra reward for winning based on conditions
            time_rew = 5. / (time.time() - self._episode_time)
            reward += knight_hp / 40. + time_rew
        elif lose:
            reward -= enemy_hp / 5.
        # print('reward', reward)
        # print()

        if done:
            self.cleanup()
        else:
            self.prev_knight_hp = knight_hp
            self.prev_enemy_hp = enemy_hp
        reward = np.clip(reward, -1.5, 1.5)
        return obs, reward, done, False, win

    def reset(self, seed=None, options=None):
        super(HKEnv, self).reset(seed=seed)
        self.cleanup()
        while True:
            # print(self._find_menu())
            if self._find_menu():
                break
            pyautogui.press('w')
            time.sleep(0.75)
        pyautogui.press('space')

        # wait for loading screen
        ready = False
        while True:
            obs, _, _ = self.observe(force_gray=True)
            is_loading = (obs < 20).sum() < 10
            if ready and not is_loading:
                break
            else:
                ready = is_loading
        time.sleep(2.25)
        self.prepare()
        return self.observe()[0], None

    def prepare(self):
        self.prev_knight_hp, self.prev_enemy_hp = len(self.HP_CKPT), 1.
        self._episode_time = time.time()
        self._prev_time = time.time()

    def close(self):
        self.cleanup()

    def cleanup(self):
        """
        do any necessary cleanup on the interaction
        should only be called before or after an episode
        """

        if self._timer is not None:
            self._timer.join()
        self.holding = []
        for key in self.KEYMAPS.values():
            if isinstance(key, tuple):
                for k in key:
                    pyautogui.keyUp(k)
            else:
                pyautogui.keyUp(key)
        self.prev_knight_hp = None
        self.prev_enemy_hp = None
        self.prev_action = -1
        self._timer = None
        self._episode_time = None
        self._prev_time = None
        gc.collect()


# 大表哥的环境
class HKEnvBV(HKEnv):
    REWMAPS = {  # map each action to its corresponding reward
        Move.HOLD_LEFT: 0,
        Move.HOLD_RIGHT: 0,
        # Move.LOOK_LEFT: 0,
        # Move.LOOK_RIGHT: 0,
        Displacement.TIMED_SHORT_JUMP: 0,
        Displacement.TIMED_LONG_JUMP: 0,
        Displacement.DASH: 1e-3,
        Attack.ATTACK: 1e-3,
        Attack.UP_ATTACK: -1e-1,
        # Attack.SPELL: 1e-3
    }


# 白给守卫的环境
class HKEnvCG(HKEnv):
    REWMAPS = {  # map each action to its corresponding reward
        Move.HOLD_LEFT: 1e-4,
        Move.HOLD_RIGHT: 1e-4,
        # Move.LOOK_LEFT: 0,
        # Move.LOOK_RIGHT: 0,
        Displacement.TIMED_SHORT_JUMP: -1e-1,
        Displacement.TIMED_LONG_JUMP: 1e-3,
        Displacement.DASH: -1e-1,
        Attack.ATTACK: 1e-3,
        Attack.UP_ATTACK: -1e-1,
        # Attack.SPELL: 1e-3
    }

    def observe(self, force_gray=False):
        """
        take a screenshot and identify enemy and knight's HP

        :param force_gray: override self.rgb to force return gray obs
        :return: observation (a resized screenshot), knight HP, and enemy HP
        """
        with mss() as sct:
            frame = np.asarray(sct.grab(self.monitor), dtype=np.uint8)

        knight_hp_bar = frame[64, :, 0]
        checkpoint1 = knight_hp_bar[self.HP_CKPT]
        checkpoint2 = knight_hp_bar[self.HP_CKPT - 1]
        knight_hp = ((checkpoint1 > 200) | (checkpoint2 > 200)).sum()
        if knight_hp == 0:
            time.sleep(1.5)
            with mss() as sct:
                frame = np.asarray(sct.grab(self.monitor), dtype=np.uint8)
            knight_hp_bar = frame[64, :, 0]
            checkpoint1 = knight_hp_bar[self.HP_CKPT]
            checkpoint2 = knight_hp_bar[self.HP_CKPT - 1]
            knight_hp = ((checkpoint1 > 200) | (checkpoint2 > 200)).sum()
            if knight_hp == 9:
                knight_hp = 0

        enemy_hp_bar = frame[-1, 187:826, :]
        if (np.all(enemy_hp_bar[..., 0] == enemy_hp_bar[..., 1]) and
                np.all(enemy_hp_bar[..., 1] == enemy_hp_bar[..., 2])):
            # hp bar found
            enemy_hp = (enemy_hp_bar[..., 0] < 3).sum() / len(enemy_hp_bar)
        else:
            enemy_hp = 1.

        rgb = not force_gray and self.rgb
        obs = cv2.cvtColor(frame[:672, ...],
                           (cv2.COLOR_BGRA2RGB if rgb
                            else cv2.COLOR_BGRA2GRAY))
        obs = cv2.resize(obs,
                         dsize=self.observation_space.shape[1:],
                         interpolation=cv2.INTER_AREA)
        # make channel first
        obs = np.rollaxis(obs, -1) if rgb else obs[np.newaxis, ...]
        return obs, knight_hp, enemy_hp


# 假骑士的环境
class HKEnvFK(HKEnvCG):
    REWMAPS = {  # map each action to its corresponding reward
        Move.HOLD_LEFT: 0,
        Move.HOLD_RIGHT: 0,
        # Move.LOOK_LEFT: 0,
        # Move.LOOK_RIGHT: 0,
        Displacement.TIMED_SHORT_JUMP: 0,
        Displacement.TIMED_LONG_JUMP: 0,
        Displacement.DASH: 1e-3,
        Attack.ATTACK: 1e-3,
        Attack.UP_ATTACK: -1e-1,
        # Attack.SPELL: -1e-1
    }


class HKEnvV2(HKEnv):
    REWMAPS = {  # map each action to its corresponding reward
        Move.HOLD_LEFT: 0,
        Move.HOLD_RIGHT: 0,
        # Move.LOOK_LEFT: 0,
        # Move.LOOK_RIGHT: 0,
        Displacement.TIMED_SHORT_JUMP: 0,
        Displacement.TIMED_LONG_JUMP: 0,
        # Displacement.DASH: -1e-5,
        Attack.ATTACK: -2e-6,
        # Attack.UP_ATTACK: 0,
        # Attack.SPELL: 0
    }

    def __init__(self, obs_shape=(192, 192), rgb=False, gap=0.17,
                 w1=0.8, w2=0.5, w3=-8e-5):
        super().__init__(obs_shape, rgb, gap, w1, w2, w3)
        self._hold_time = self.gap * 0.97
        self._fail_hold_rew = -1e-5

    def step(self, actions):
        action_rew = 0
        if actions == self.prev_action:
            action_rew -= 2e-5
        self.prev_action = actions
        actions = self._to_multi_discrete(actions)
        action_rew += self._step_actions(actions)
        obs, knight_hp, enemy_hp = self.observe()

        win = self.prev_enemy_hp < enemy_hp
        lose = knight_hp == 0
        done = win or lose

        if win:
            lose = False
            enemy_hp = 0.
        hurt = knight_hp < self.prev_knight_hp
        hit = enemy_hp < self.prev_enemy_hp

        reward = (
                - self.w1 * hurt
                + self.w2 * hit
                + action_rew
        )
        if not (hurt or hit):
            reward += self.w3

        if win:  # extra reward for winning based on conditions
            reward += knight_hp / 45.
        elif lose:
            reward -= enemy_hp / 5.

        # print('reward', reward)
        # print()

        if done:
            self.cleanup()
        else:
            self.prev_knight_hp = knight_hp
            self.prev_enemy_hp = enemy_hp
        reward = np.clip(reward, -1.5, 1.5)
        return obs, reward, done, False, win


class HKEnvSurvive(HKEnv):
    ACTIONS = [Move, Displacement]

    def step(self, actions):
        t = self.gap - (time.time() - self._prev_time)
        if t > 0:
            time.sleep(t)
        # print(t)
        self._prev_time = time.time()
        actions = self._to_multi_discrete(actions)
        self._step_actions(actions)
        obs, knight_hp, enemy_hp = self.observe()

        win = self.prev_enemy_hp < enemy_hp
        lose = knight_hp == 0
        done = win or lose

        hurt = knight_hp < self.prev_knight_hp
        self.prev_knight_hp = knight_hp

        rew = (-self.w1) if hurt else (knight_hp / 18. + 0.4)
        rew = np.clip(rew, -1.5, 1.5)
        return obs, rew, done, False, win


# def test_input_img():
#     HP_CKPT = np.array([52, 91, 129, 169, 207, 246, 286, 324], dtype=int)
#     window = pyautogui.getWindowsWithTitle('Hollow Knight')
#     assert len(window) == 1, f'found {len(window)} windows called Hollow Knight {window}'
#     window = window[0]
#     try:
#         window.activate()
#     except Exception:
#         window.minimize()
#         window.maximize()
#         window.restore()
#     window.moveTo(0, 0)
#
#     with mss() as sct:
#         monitor = {'left': 144, 'top': 35, 'width': 1020, 'height': 692}
#         x = sct.grab(monitor)
#         frame = np.asarray(x, dtype=np.uint8)
#
#     knight_hp_bar = frame[:, :, 0]
#     # checkpoint1 = knight_hp_bar[HP_CKPT]
#     # checkpoint2 = knight_hp_bar[HP_CKPT - 1]
#     # print(knight_hp_bar)
#     cv2.imshow('x', knight_hp_bar)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#
# if __name__ == '__main__':
#     test_input_img()
