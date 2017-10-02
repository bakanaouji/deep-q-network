import gym

from trainer import Trainer
from env_wrappers import wrap_dqn

def main():
    env = gym.make("PongNoFrameskip-v4")
    env = wrap_dqn(env)
    params = {}

    # 環境側のパラメータ
    params['frame_width'] = 84  # リサイズ後のフレーム幅
    params['frame_height'] = 84  # リサイズ後のフレーム高さ
    params['agent_history_length'] = 4  # Q_networkの入力として与える，直近のフレームの数

    # DQNのアルゴリズムのパラメータ
    params['learning_rate'] = 1e-4  # 学習率
    params['tmax'] = 2000000  # 学習をやめる行動数
    params['replay_memory_size'] = 10000  # SGDによる更新に用いるデータは，このサイズの直近のフレームデータからサンプルする
    params['exploration_fraction'] = 0.1  # εが初期値から最終値に線形減少するフレーム数．tmaxとの割合で決定．
    params['final_exploration'] = 0.01  # ε-greedyにおけるεの最終値
    params['learn_frequency'] = 4   # この行動回数ごとに学習
    params['replay_start_size'] = 10000  # 学習を始める前に，このフレーム数に対して一様ランダムに行動を選択する政策が実行され，その経験がReplay memoryに蓄えられる
    params['target_network_update_frequency'] = 1000  # target_networkが更新される頻度（パラメータの更新頻度数で計測）
    params['discount_factor'] = 0.99  # Q_learningの更新で用いられる割引率γ
    params['minibatch_size'] = 32  # SGDによる更新に用いる訓練データの数

    # 学習時の設定
    params['render'] = False # 描画をするかどうか
    params['save_network_frequency'] = 100000 # Q_networkを保存する頻度（フレーム数）

    trainer = Trainer(env, **params)
    # 学習実行
    trainer.learn()

if __name__ == '__main__':
    main()
