# DeepQNetwork

## このリポジトリについて
DeepQNetwork（http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html?foxtrotcallback=true)をTensorflowとKerasを用いて実装したコードを公開しています．

Open AI Gymが提供しているAtari 2600 gamesを学習させることを想定しています．
DeepQNetworkの学習と，学習後のモデルのテストを行うことができます．

かなりコードを整理したつもりですが，要望や意見がありましたら，遠慮なくissuesやpull reqにお願いします．

## 動作環境
実装はPython3.5で行いました．
主に，以下のライブラリが必要になります．
* Tensorflow
* Keras
* Numpy
* Open AI Gym
* cv2

GPU環境であってもCPU環境であっても実行ができるように実装を行ったつもりです．

Tensorflowは1.0.0以降，Kerasは2.0.0以降を用いることを推奨しています．

以下の環境で，動作ができることを確認しています．
* Windows10
* Python3.5（Anaconda4.2.0）
* Tensorflow-gpu 1.1.0rc1
* Keras 2.0.8
* Numpy 1.13.1
* gym 0.9.3
* opencv3 3.1.0

### Tensorflow-gpu
Tensorflow-gpuは最新版ではなぜか動かないことがあるため，以下のコマンドでバージョンを指定してインストールしました．
```
pip install –ignore-installed –upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.1.0rc1-cp35-cp35m-win_amd64.whl
```

### opencv3
opencv3は，anacondaならcondaでインストールすることができます．
```
conda install --channel https://conda.anaconda.org/menpo opencv3
```

### gym
windows環境でOpen AI GymのAtari 2600 Games環境を動かそうとするとzlibが必要になりますが，以下のコマンドを打つことで，zlib依存なしで環境を動かすことができます．
```
git clone https://github.com/Kojoley/atari-py.git
cd atari-py
python setup.py install
```

## 実行
### 学習実行
以下のコマンドを実行することで，学習が実行されます．
```
cd src
python main.py
```

デフォルトでは，pongを学習します．
学習する環境や学習に用いるパラメータを変更したい場合，まず，以下のコマンドを実行することでどんなパラメータがあるのかを確認してください．
```
python main.py --help
```

その後，変更したいパラメータのオプションを指定してコマンドを実行すると，パラメータを変更して実行することができます．
例えば，学習を打ち切る行動回数を指定したい場合，以下のようにします．
```
python main.py --tmax=1000000
```

各パラメータのデフォルト値については，src/main.pyを参照してください．

### 学習中の推移確認
tensorboardを利用することで，学習時に合計報酬や損失誤差がどのように変動しているかを確認することができます．
以下のコマンドを実行すると，tensorboardが起動します．
```
tensorboard --logdir=data/summary/【環境名】【オプション名】/
```
tensorboardによって，以下の推移が確認できます．
* Average_Loss：損失誤差のエピソード毎の平均値の推移
* Average_max_Q：Q値のエピソード毎の平均値の推移
* Duration：エピソードの長さの推移
* Total_Reward：エピソード毎の合計報酬の推移
* Total_Step：総ステップ数

### テスト実行
以下のコマンドを実行することで，学習済みモデルのテスト実行を行うことができます．
```
cd src
python main.py --test
```
学習済みモデルのパスをsave_network_pathオプションやsave_option_nameオプションで指定することを忘れないようにしてください．
