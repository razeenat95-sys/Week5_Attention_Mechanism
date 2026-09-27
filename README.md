# Week 5 - Attention Mechanism

Implementing and Visualizing Scaled Dot-Product Attention.

This project demonstrates Query (Q), Key (K), Value (V), scaled dot-product attention,
softmax attention weights, final attention output, heatmap visualization, and a PyTorch comparison.

## Structure
Attention_Mechanism/
- dataset/sample_sentences.txt
- dataset/attention_weights.csv
- dataset/attention_output.csv
- dataset/attention_heatmap.png
- notebook/Week5_Attention_Mechanism.ipynb
- attention.py
- README.md
- requirements.txt

## Run
`py -m pip install -r requirements.txt`
then:
`py attention.py`

Open the notebook and select the Python kernel, then Run All.

Formula:
Attention(Q,K,V) = softmax((QK^T) / sqrt(d_k)) V
