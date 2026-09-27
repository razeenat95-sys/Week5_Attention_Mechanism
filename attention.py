from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA=Path("dataset")
def softmax(x):
    x=x-np.max(x,axis=-1,keepdims=True)
    e=np.exp(x)
    return e/e.sum(axis=-1,keepdims=True)

def make_embeddings(sentences,d=8):
    out=[]
    for s in sentences:
        v=np.zeros(d)
        for i,w in enumerate(s.lower().split()):
            c=sum(map(ord,w))
            v[i%d]+=(c%97)/97
            v[(i*3+1)%d]+=len(w)/15
        n=np.linalg.norm(v)
        out.append(v/n if n else v)
    return np.array(out)

sentences=[x.strip() for x in (DATA/"sample_sentences.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
if len(sentences)!=5: raise ValueError("Use exactly five sentences.")
X=make_embeddings(sentences)
Q,K,V=X,X,X
scores=(Q@K.T)/np.sqrt(Q.shape[1])
weights=softmax(scores)
output=weights@V

pd.DataFrame(weights,index=[f"Sentence_{i}" for i in range(1,6)],
             columns=[f"Sentence_{i}" for i in range(1,6)]).to_csv(DATA/"attention_weights.csv")
pd.DataFrame(output,index=[f"Sentence_{i}" for i in range(1,6)],
             columns=[f"dim_{i}" for i in range(1,9)]).to_csv(DATA/"attention_output.csv")

plt.figure(figsize=(8,6))
plt.imshow(weights,aspect="auto")
plt.colorbar(label="Attention Weight")
plt.xticks(range(5),[f"S{i}" for i in range(1,6)])
plt.yticks(range(5),[f"S{i}" for i in range(1,6)])
plt.xlabel("Key sentences"); plt.ylabel("Query sentences")
plt.title("Scaled Dot-Product Attention Weights")
plt.tight_layout(); plt.savefig(DATA/"attention_heatmap.png",dpi=150); plt.close()

print("Week 5 Attention Mechanism completed successfully.")
print("Q, K, V shape:",Q.shape)
print("Attention weights shape:",weights.shape)
print("Attention output shape:",output.shape)
print("Saved dataset/attention_weights.csv")
print("Saved dataset/attention_output.csv")
print("Saved dataset/attention_heatmap.png")

try:
    import torch
    tq=torch.tensor(Q,dtype=torch.float32); tk=torch.tensor(K,dtype=torch.float32); tv=torch.tensor(V,dtype=torch.float32)
    tw=torch.softmax((tq@tk.T)/np.sqrt(Q.shape[1]),dim=-1)
    diff=np.max(np.abs(output-(tw@tv).numpy()))
    print(f"PyTorch comparison max difference: {diff:.8f}")
except ImportError:
    print("PyTorch not available.")
