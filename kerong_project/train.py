import gym  # 導入 OpenAI Gym，提供各種標準化的強化學習環境
import torch  # 導入 PyTorch 深度學習框架，用於建構神經網路和訓練模型
import psutil  # 導入 psutil，用於獲取系統資源使用情況
from torch.backends import cudnn  # 從 torch.backends 導入 cudnn，用於設置 CuDNN
import numpy as np  # 導入 NumPy，用於數值運算

import hkenv  # 導入自定義的 hkenv 模組，可能是與遊戲環境相關的功能
import models  # 導入自定義的 models 模組，可能是與神經網路模型相關的功能
import trainer  # 導入自定義的 trainer 模組，可能是與訓練過程相關的功能
import buffer  # 導入自定義的 buffer 模組，可能是與記憶緩衝區相關的功能
import keyboard  # 導入自定義的 keyboard 模組，可能是與鍵盤相關的功能

DEVICE = 'cuda'  # 設置使用的運算裝置為 CUDA
cudnn.benchmark = True  # 啟用 CuDNN 的自動調整功能，以提高訓練速度


def get_model(env: gym.Env, n_frames: int, file_path=''):
    """
    建立 DQN 模型

    Args:
        env (gym.Env): 遊戲環境
        n_frames (int): 幀數
        file_path (str, optional): 模型檔案路徑，預設為空字串

    Returns:
        torch.nn.Module: 建立好的 DQN 模型
    """
    c, *shape = env.observation_space.shape  # 獲取觀察數
    m = models.SimpleExtractor(shape, n_frames * c, activation='relu', sn=False)  # 建立特徵提取器
    m = models.DuelingMLP(m, env.action_space.n, activation='relu', noisy=True, sn=False)  # 建立 DuelingMLP 模型
    m = m.to(DEVICE)  # 將模型移到指定的運算裝置
    if len(file_path):  # 如果指定了模型檔案路徑，則載入參數
        tl = torch.load(file_path)
        missing, unexpected = m.load_state_dict(tl, strict=False)
        if len(missing):
            print('miss:', missing)
        if len(unexpected):
            print('unexpected:', unexpected)
    return m


def train(dqn, old_path='.//saved//1714233383CG//explorations'):
    """
    訓練 DQN 模型

    Args:
        dqn: Trainer 物件，用於訓練 DQN
        old_path (str, optional): 舊模型路徑，預設為指定路徑

    Returns:
        None
    """
    print('training started')
    if not len(old_path):  # 如果未指定舊模型路徑，則保存探索資料
        dqn.save_explorations(200)
    dqn.load_explorations(old_path)  # 載入舊的探索資料
    dqn.learn()  # 預熱 DQN 模型

    saved_rew = float('-inf')
    saved_train_rew = float('-inf')

    win_episode = []  # 儲存贏得的賽局
    best_train_update = []  # 儲存最佳訓練模型更新的賽局
    best_update = []  # 儲存最佳模型更新的賽局
    for i in range(1, 2):  # 迴圈遍歷每一個賽局
        print('episod', i)
        rew, loss, lr, w = dqn.run_episode()  # 執行一個賽局
        if w:  # 如果贏得了賽局，則記錄賽局編號
            win_episode.append(i)
        if rew > saved_train_rew and dqn.eps < 0.11:  # 如果得分超過最佳訓練得分，且 epsilon 小於 0.11
            print('new best train model found')
            saved_train_rew = rew  # 更新最佳訓練得分
            dqn.save_models('besttrain', online_only=True)  # 保存最佳訓練模型
            best_train_update.append(i)  # 儲存最佳訓練模型更新的賽局
        if i % 10 == 0:  # 每 10 個賽局執行一次
            dqn.run_episode(random_action=True)  # 執行一個賽局，隨機動作

            if i >= 100:  # 從第 100 個賽局開始
                eval_rew, _ = dqn.evaluate()  # 評估模型性能

                if eval_rew > saved_rew:  # 如果評估得分超過最佳得分
                    print('new best eval model found')
                    saved_rew = eval_rew  # 更新最佳得分
                    dqn.save_models('best', online_only=True)  # 保存最佳模型
                    best_update.append(i)  # 儲存最佳模型更新的賽局
        dqn.save_models('latest', online_only=True)  # 保存最新模型

        # 記錄賽局相關信息
        dqn.log({'reward': rew, 'loss': loss, 'total steps': dqn.steps}, i)
        print(f'episode {i} finished, total step {dqn.steps}, learned {dqn.learn_steps}, epsilon {dqn.eps}',
              f'total rewards {round(rew, 3)}, loss {round(loss, 3)}, current lr {round(lr, 8)}',
              f'total memory usage {psutil.virtual_memory().percent}%', sep='\n')
        print()
    dqn.save_models('latest', online_only=False)  # 保存最新模型（包括在本地）
    print(win_episode)  # 輸出贏得的賽局
    print(best_update)  # 輸出最佳模型更新的賽局
    print(best_train_update)  # 輸出最佳訓練模型更新的賽局
    np.savetxt(dqn.save_loc + 'win_episode.txt', np.array(win_episode), '%d')  # 保存贏得的賽局到檔案
    np.savetxt(dqn.save_loc + 'best_update.txt', np.array(best_update), '%d')  # 保存最佳模型更新的賽局到檔案
    np.savetxt(dqn.save_loc + 'best_train_update.txt', np.array(best_train_update), '%d')  # 保存最佳訓練模型更新的賽局到檔案


def main():
    # keyboard.main()

    # old_model_path = 'saved/1707617125BV/'
    n_frames = 4  # 設置幀數
    env = hkenv.HKEnvCG((160, 160), rgb=False, gap=0.165, w1=0.8, w2=0.8, w3=-0.0001)  # 建立遊戲環境
    # env = hkenv.HKEnvV2((192, 192), rgb=False, gap=0.17, w1=0.8, w2=0.5, w3=-8e-5)
    # m = get_model(env, n_frames, old_model_path + 'bestonline.pt')
    m = get_model(env, n_frames)  # 建立 DQN 模型
    replay_buffer = buffer.MultistepBuffer(180000, n=10, gamma=0.99, prioritized=None)  # 建立記憶緩衝區

    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=0,
                          eps_func=(lambda val, step: 0),
                          target_steps=8000,
                          learn_freq=4,
                          model=m,
                          lr=8e-5,
                          lr_decay=False,
                          criterion=torch.nn.MSELoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          drq=True,
                          svea=False,
                          reset=0,  # no reset
                          n_targets=1,
                          save_suffix='CG',
                          no_save=False)
    # train(dqn, old_model_path + 'explorations/')
    train(dqn)  # 訓練 DQN 模型


if __name__ == '__main__':
    main()  # 執行主函式
