import gym

from trainer import Trainer
from env_wrappers import wrap_dqn

def main():
    env = gym.make("PongNoFrameskip-v4")
    env = wrap_dqn(env)
    params = {}
    params['learning_rate'] = 0.00025  # RMSPropで使用される学習率
    params['gradient_momentum'] = 0.95  # RMSPropで使用されるGradient momuntum
    params['squared_gradient_momuntum'] = 0.95  # RMSPropで使用されるSquared gradient (denominator) momentum
    params['min_squared_gradient'] = 0.01  # RMSPropの更新の際，Squared gradient (denominator)に加算される定数
    params['initial_exploration'] = 1.0  # ε-greedyにおけるεの初期値
    params['final_exploration'] = 0.1  # ε-greedyにおけるεの最終値
    params['final_exploration_frame'] = 1000000  # εが初期値から最終値に線形減少した時に，最終値に到達するフレーム数
    params['replay_memory_size'] = 500000  # SGDによる更新に用いるデータは，このサイズの直近のフレームデータからサンプルする
    params['frame_width'] = 84  # リサイズ後のフレーム幅
    params['frame_height'] = 84  # リサイズ後のフレーム高さ
    params['agent_history_length'] = 4  # Q_networkの入力として与える，直近のフレームの数
    params['tmax'] = 50000000  # 学習をやめる行動数
    trainer = Trainer(env, **params)
    # 学習実行
    trainer.train()

if __name__ == '__main__':
    main()
