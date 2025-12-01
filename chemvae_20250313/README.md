# ChemVAE

## 参考

[「機械学習による分子最適化」サポートページ](https://github.com/kanojikajino/ml4chem)

## 動作環境

- Python
- CUDA

## 必要なライブラリ

- rdkit
- torch
- matplotlib
- tqdm
- pandas
- torchdrug
- botorch

## サンプル入力データ

```bash
wget https://ndownloader.figshare.com/files/13612760 -O train.smi
wget https://ndownloader.figshare.com/files/13612766 -O val.smi
wget https://ndownloader.figshare.com/files/13612757 -O test.smi
```

## 実行方法

### LSTM

```bash
python smiles_lstm_main.py
```

下記ファイルが出力される。

- smiles_lstm_learning_curve.pdf

### VAE

```bash
python smiles_vae_main.py
```

下記ファイルが出力される。

- smiles_vae_learning_curve.pdf
- reconstruction_rate_curve.pdf
- vae.pt
- smiles_vae.pkl

### VAE ベイズ最適化

```bash
python smiles_vae_bo_main.py
```

下記ファイルが出力される。

- smiles_vae_best_mol.pklz
- smiles_vae_bo_full.pklz
