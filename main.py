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
    params['learning_rate'] = 0.00025  # RMSPropで使用される学習率
    params['tmax'] = 50000000  # 学習をやめる行動数
    params['replay_memory_size'] = 500000  # SGDによる更新に用いるデータは，このサイズの直近のフレームデータからサンプルする
    params['final_exploration_frame'] = 1000000  # εが初期値から最終値に線形減少した時に，最終値に到達するフレーム数
    params['final_exploration'] = 0.1  # ε-greedyにおけるεの最終値
    params['learn_frequency'] = 4   # この行動回数ごとに学習
    params['replay_start_size'] = 25000  # 学習を始める前に，このフレーム数に対して一様ランダムに行動を選択する政策が実行され，その経験がReplay memoryに蓄えられる
    params['target_network_update_frequency'] = 10000  # target_networkが更新される頻度（パラメータの更新頻度数で計測）
    params['discount_factor'] = 0.99  # Q_learningの更新で用いられる割引率γ
    params['gradient_momentum'] = 0.95  # RMSPropで使用されるGradient momuntum
    params['squared_gradient_momuntum'] = 0.95  # RMSPropで使用されるSquared gradient (denominator) momentum
    params['min_squared_gradient'] = 0.01  # RMSPropの更新の際，Squared gradient (denominator)に加算される定数
    params['minibatch_size'] = 32  # SGDによる更新に用いる訓練データの数

    trainer = Trainer(env, **params)
    # 学習実行
    trainer.train()

if __name__ == '__main__':
    main()
