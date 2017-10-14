import gym
import argparse

from trainer import Trainer
from env_wrappers import wrap_dqn


def main():
    parser = argparse.ArgumentParser(description='Deep Q Network')
    # 環境側のパラメータ
    parser.add_argument('--env_name', default='PongNoFrameskip-v4', help='Environment name')
    parser.add_argument('--width', type=int, default=84, help='Width of resized frame')
    parser.add_argument('--height', type=int, default=84, help='Height of resized frame')

    # DQNのアルゴリズムのパラメータ
    parser.add_argument('--tmax', type=int, default=2000000, help='Number of action selections to finish learning.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of training cases over which each SGD update is computed.')
    parser.add_argument('--mem_size', type=int, default=10000,
                        help='Gradient updates are sampled from this number of most recent frames.')
    parser.add_argument('--history_len', type=int, default=4,
                        help='Number of most recent frames experienced '
                             'by the agent that are given as input to the Q-Network.')
    parser.add_argument('--update_freq', type=int, default=1000,
                        help='Frequency (measured in the number of action selections) '
                             'with which the target network is updated.')
    parser.add_argument('--discount_fact', type=float, default=0.99,
                        help='Discount factor gamma used in the Q-Learning update.')
    parser.add_argument('--action_repeat', type=int, default=4,
                        help='Repeat each action selected by the agent this many times.')
    parser.add_argument('--learn_freq', type=int, default=4,
                        help='Number of actions selected by the agent between successive SGD updates.')
    parser.add_argument('--learn_rate', type=float, default=1e-4, help='Learning rate used by Adam.')
    parser.add_argument('--fin_expl', type=float, default=0.01, help='Final value of ε in ε-greedy exploration.')
    parser.add_argument('--expl_frac', type=float, default=0.1,
                        help='Fraction of entire training period over which the value of ε is annealed.')
    parser.add_argument('--replay_st_size', type=int, default=10000,
                        help='Uniform random policy is run for this number of frames before learning starts '
                             'and the resulting experience is used to populate the replay memory.')
    parser.add_argument('--no_op_max', type=int, default=30,
                        help='Maximum number of "do nothing" actions to be performed '
                             'by the agent at the start of an episode.')

    # 学習時の設定
    parser.add_argument('--test', action='store_true', help='Whether to test')
    parser.set_defaults(test=False)
    parser.add_argument('--render', action='store_true', help='Wheter to render')
    parser.set_defaults(render=False)
    parser.add_argument('--save_network_freq', type=int, default=100000,
                        help='Frequency (measured in the number of action selections) '
                             'with which the Q-Network is saved.')
    parser.add_argument('--save_network_path', default='saved_networks', help='Path to save Q-Network.')
    parser.add_argument('--save_summary_path', default='summary', help='Path to save summary.')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env = wrap_dqn(env, args.history_len, args.action_repeat, args.no_op_max)

    # 学習実行
    trainer = Trainer(env, args)
    if args.test:
        trainer.test()
    else:
        trainer.learn()


if __name__ == '__main__':
    main()
