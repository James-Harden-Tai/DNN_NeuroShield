# NeuroShield 神经网络抗提取防御系统演示

基于 EUROCRYPT 2024 论文的神经网络参数提取攻击演示平台。

## 🚀 快速部署

### 1. Streamlit Cloud 部署（推荐）
1. Fork 本项目到你的 GitHub 账户
2. 打开 [Streamlit Cloud](https://share.streamlit.io/)
3. 选择你的仓库，主文件填写 `dnn_attack_dashboard.py`
4. 点击 Deploy 即可在线体验

### 2. 本地运行
```bash
pip install -r requirements.txt
streamlit run dnn_attack_dashboard.py
```

### 3. Docker 部署
```bash
docker build -t neuroshield-demo .
docker run -p 8501:8501 neuroshield-demo
```

## 主要依赖
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- networkx
- Pillow

## 免责声明
本项目仅供学术研究与教学演示，禁止用于任何非法用途。

# hard-label-dnn-extraction
Supplementary code for the EUROCRYPT 2024 paper "Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting"

The code is split into two phases:
1. Signature recovery
2. Sign recovery
see the README file in each directory for a detailed explanation.

The `data/` directory contains a .keras file for the neural network that we used to illustrate the attack. This is a 'real' network which
was trained on the CIFAR-10 dataset, achieving 0.52 accuracy. It also contains arrays of precomputed dual points for this network, which
were generated using the code in `signature_recovery`.

