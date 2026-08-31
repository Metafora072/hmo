# V6.1 27B 32K Keep-0.10 Result Tables

Source: `e1_main.jsonl`

Setting: Qwen3.5-27B, context length 32768, keep_ratio 0.10. LongBench task metrics follow the project primary metric: HotpotQA/NarrativeQA use F1, GovReport uses ROUGE-L, Needle/LongEval/LCC use Accuracy.

## Main Metric Table
| Dataset / Metric | n | Full KV | Budgeted Recent KV | Budgeted Uniform KV | H2O | SnapKV | StreamingLLM | DuoAttention | PyramidKV-lite | Quest-lite | SAGE-KV-lite | HMO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Needle Acc | 50 | **1.0000** | 0.0800 | 0.0400 | 0.7800 | **1.0000** | 0.0800 | 0.0000 | 0.9800 | **1.0000** | **1.0000** | **1.0000** |
| LongEval-Lines Acc | 50 | **1.0000** | 0.0000 | 0.0000 | 0.1200 | **1.0000** | 0.0000 | 0.0000 | 0.7800 | **1.0000** | **1.0000** | **1.0000** |
| HotpotQA F1 | 50 | 0.6795 | 0.5785 | 0.6190 | 0.6871 | 0.6787 | 0.5785 | 0.6472 | 0.6211 | 0.6787 | 0.6268 | 0.6741 |
| NarrativeQA F1 | 50 | 0.3140 | 0.2791 | 0.3068 | 0.2899 | 0.3105 | 0.2791 | 0.2922 | 0.2935 | 0.3118 | 0.3085 | 0.3162 |
| GovReport ROUGE-L | 50 | 0.1901 | 0.1707 | 0.1784 | 0.1842 | 0.1803 | 0.1707 | 0.1795 | 0.1767 | 0.1850 | 0.1755 | 0.1761 |
| LCC Acc | 18 | **0.5000** | 0.4444 | 0.2778 | 0.2778 | 0.4444 | 0.4444 | 0.2778 | 0.2222 | 0.4444 | 0.3333 | 0.4444 |

## HMO Deltas
| Dataset | HMO | Δ vs Full KV | Δ vs H2O | Δ vs SnapKV | Δ vs Quest-lite | Δ vs SAGE-KV-lite | Δ vs PyramidKV-lite | Δ vs Budgeted Recent KV | Δ vs Budgeted Uniform KV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Needle | 1.0000 | +0.0000 | +0.2200 | +0.0000 | +0.0000 | +0.0000 | +0.0200 | +0.9200 | +0.9600 |
| LongEval-Lines | 1.0000 | +0.0000 | +0.8800 | +0.0000 | +0.0000 | +0.0000 | +0.2200 | +1.0000 | +1.0000 |
| HotpotQA | 0.6741 | -0.0054 | -0.0130 | -0.0047 | -0.0047 | +0.0472 | +0.0530 | +0.0956 | +0.0551 |
| NarrativeQA | 0.3162 | +0.0022 | +0.0262 | +0.0057 | +0.0044 | +0.0076 | +0.0227 | +0.0371 | +0.0093 |
| GovReport | 0.1761 | -0.0139 | -0.0081 | -0.0042 | -0.0089 | +0.0006 | -0.0005 | +0.0054 | -0.0023 |
| LCC | 0.4444 | -0.0556 | +0.1667 | +0.0000 | +0.0000 | +0.1111 | +0.2222 | +0.0000 | +0.1667 |

## Macro Mean
| Method | Groups | Macro Mean |
|---|---:|---:|
| Full KV | 6 | 0.6139 |
| Budgeted Recent KV | 6 | 0.2588 |
| Budgeted Uniform KV | 6 | 0.2370 |
| H2O | 6 | 0.3898 |
| SnapKV | 6 | 0.6023 |
| StreamingLLM | 6 | 0.2588 |
| DuoAttention | 6 | 0.2328 |
| PyramidKV-lite | 6 | 0.5122 |
| Quest-lite | 6 | 0.6033 |
| SAGE-KV-lite | 6 | 0.5740 |
| HMO | 6 | 0.6018 |

## Resource And Action Summary
| Dataset | Method | n | Tracked GB | Budget GB | Peak VRAM GB | Kept KV | Skeleton | Refresh | Dropped |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Needle | Full KV | 50 | 2.124 | 0.000 | 59.39 | 0.0 | 0.0 | 0.0 | 0.0 |
| Needle | Budgeted Recent KV | 50 | 0.252 | 0.252 | 59.39 | 3840.9 | 0.0 | 0.0 | 28570.0 |
| Needle | Budgeted Uniform KV | 50 | 0.252 | 0.252 | 59.39 | 3840.9 | 0.0 | 0.0 | 28570.0 |
| Needle | H2O | 50 | 0.252 | 0.252 | 62.41 | 3840.9 | 0.0 | 0.0 | 28570.0 |
| Needle | SnapKV | 50 | 0.252 | 0.252 | 59.39 | 3840.9 | 0.0 | 0.0 | 28570.0 |
| Needle | StreamingLLM | 50 | 0.252 | 0.252 | 58.33 | 3840.9 | 0.0 | 0.0 | 28570.0 |
| Needle | DuoAttention | 50 | 0.252 | 0.252 | 58.33 | 3840.9 | 0.0 | 0.0 | 28570.0 |
| Needle | PyramidKV-lite | 50 | 0.252 | 0.252 | 59.39 | 3840.9 | 0.0 | 0.0 | 28570.0 |
| Needle | Quest-lite | 50 | 0.249 | 0.252 | 59.39 | 3806.0 | 0.0 | 0.0 | 28604.8 |
| Needle | SAGE-KV-lite | 50 | 0.252 | 0.252 | 59.39 | 3840.9 | 0.0 | 0.0 | 28570.0 |
| Needle | HMO | 50 | 0.252 | 0.252 | 59.74 | 2.0 | 61.0 | 1.0 | 0.0 |
| LongEval-Lines | Full KV | 50 | 2.113 | 0.000 | 58.29 | 0.0 | 0.0 | 0.0 | 0.0 |
| LongEval-Lines | Budgeted Recent KV | 50 | 0.271 | 0.271 | 58.29 | 4139.5 | 0.0 | 0.0 | 28109.0 |
| LongEval-Lines | Budgeted Uniform KV | 50 | 0.271 | 0.271 | 58.29 | 4139.5 | 0.0 | 0.0 | 28109.0 |
| LongEval-Lines | H2O | 50 | 0.271 | 0.271 | 62.35 | 4139.5 | 0.0 | 0.0 | 28109.0 |
| LongEval-Lines | SnapKV | 50 | 0.271 | 0.271 | 58.29 | 4139.5 | 0.0 | 0.0 | 28109.0 |
| LongEval-Lines | StreamingLLM | 50 | 0.271 | 0.271 | 58.29 | 4139.5 | 0.0 | 0.0 | 28109.0 |
| LongEval-Lines | DuoAttention | 50 | 0.271 | 0.271 | 58.29 | 4139.5 | 0.0 | 0.0 | 28109.0 |
| LongEval-Lines | PyramidKV-lite | 50 | 0.271 | 0.271 | 58.29 | 4139.5 | 0.0 | 0.0 | 28109.0 |
| LongEval-Lines | Quest-lite | 50 | 0.270 | 0.271 | 58.29 | 4120.5 | 0.0 | 0.0 | 28128.0 |
| LongEval-Lines | SAGE-KV-lite | 50 | 0.271 | 0.271 | 58.29 | 4139.5 | 0.0 | 0.0 | 28109.0 |
| LongEval-Lines | HMO | 50 | 0.271 | 0.271 | 58.65 | 2.0 | 58.0 | 3.0 | 0.0 |
| HotpotQA | Full KV | 50 | 0.969 | 0.000 | 53.94 | 0.0 | 0.0 | 0.0 | 0.0 |
| HotpotQA | Budgeted Recent KV | 50 | 0.150 | 0.150 | 53.94 | 2288.1 | 0.0 | 0.0 | 12497.2 |
| HotpotQA | Budgeted Uniform KV | 50 | 0.150 | 0.150 | 53.94 | 2288.1 | 0.0 | 0.0 | 12497.2 |
| HotpotQA | H2O | 50 | 0.150 | 0.150 | 55.80 | 2288.1 | 0.0 | 0.0 | 12497.2 |
| HotpotQA | SnapKV | 50 | 0.150 | 0.150 | 53.94 | 2288.1 | 0.0 | 0.0 | 12497.2 |
| HotpotQA | StreamingLLM | 50 | 0.150 | 0.150 | 53.94 | 2288.1 | 0.0 | 0.0 | 12497.2 |
| HotpotQA | DuoAttention | 50 | 0.150 | 0.150 | 53.94 | 2288.1 | 0.0 | 0.0 | 12497.2 |
| HotpotQA | PyramidKV-lite | 50 | 0.150 | 0.150 | 53.94 | 2288.3 | 0.0 | 0.0 | 12497.0 |
| HotpotQA | Quest-lite | 50 | 0.148 | 0.150 | 53.94 | 2255.4 | 0.0 | 0.0 | 12529.9 |
| HotpotQA | SAGE-KV-lite | 50 | 0.150 | 0.150 | 53.94 | 2288.1 | 0.0 | 0.0 | 12497.2 |
| HotpotQA | HMO | 50 | 0.150 | 0.150 | 54.10 | 2.3 | 25.4 | 1.2 | 0.5 |
| NarrativeQA | Full KV | 50 | 1.497 | 0.000 | 55.94 | 0.0 | 0.0 | 0.0 | 0.0 |
| NarrativeQA | Budgeted Recent KV | 50 | 0.206 | 0.206 | 55.94 | 3146.4 | 0.0 | 0.0 | 19694.8 |
| NarrativeQA | Budgeted Uniform KV | 50 | 0.206 | 0.206 | 55.94 | 3146.4 | 0.0 | 0.0 | 19694.8 |
| NarrativeQA | H2O | 50 | 0.206 | 0.206 | 58.82 | 3146.4 | 0.0 | 0.0 | 19694.8 |
| NarrativeQA | SnapKV | 50 | 0.206 | 0.206 | 55.94 | 3146.4 | 0.0 | 0.0 | 19694.8 |
| NarrativeQA | StreamingLLM | 50 | 0.206 | 0.206 | 55.95 | 3146.4 | 0.0 | 0.0 | 19694.8 |
| NarrativeQA | DuoAttention | 50 | 0.206 | 0.206 | 55.95 | 3146.4 | 0.0 | 0.0 | 19694.8 |
| NarrativeQA | PyramidKV-lite | 50 | 0.206 | 0.206 | 55.94 | 3146.4 | 0.0 | 0.0 | 19694.8 |
| NarrativeQA | Quest-lite | 50 | 0.204 | 0.206 | 55.94 | 3109.4 | 0.0 | 0.0 | 19731.8 |
| NarrativeQA | SAGE-KV-lite | 50 | 0.206 | 0.206 | 55.94 | 3146.4 | 0.0 | 0.0 | 19694.8 |
| NarrativeQA | HMO | 50 | 0.206 | 0.206 | 56.17 | 2.3 | 41.4 | 1.3 | 0.0 |
| GovReport | Full KV | 50 | 0.982 | 0.000 | 53.99 | 0.0 | 0.0 | 0.0 | 0.0 |
| GovReport | Budgeted Recent KV | 50 | 0.152 | 0.152 | 53.99 | 2323.0 | 0.0 | 0.0 | 12663.2 |
| GovReport | Budgeted Uniform KV | 50 | 0.152 | 0.152 | 53.99 | 2323.0 | 0.0 | 0.0 | 12663.2 |
| GovReport | H2O | 50 | 0.152 | 0.152 | 55.87 | 2323.0 | 0.0 | 0.0 | 12663.2 |
| GovReport | SnapKV | 50 | 0.152 | 0.152 | 53.99 | 2323.0 | 0.0 | 0.0 | 12663.2 |
| GovReport | StreamingLLM | 50 | 0.152 | 0.152 | 53.99 | 2323.0 | 0.0 | 0.0 | 12663.2 |
| GovReport | DuoAttention | 50 | 0.152 | 0.152 | 53.99 | 2323.0 | 0.0 | 0.0 | 12663.2 |
| GovReport | PyramidKV-lite | 50 | 0.152 | 0.152 | 53.99 | 2323.3 | 0.0 | 0.0 | 12662.8 |
| GovReport | Quest-lite | 50 | 0.150 | 0.152 | 53.99 | 2288.7 | 0.0 | 0.0 | 12697.5 |
| GovReport | SAGE-KV-lite | 50 | 0.152 | 0.152 | 53.99 | 2323.0 | 0.0 | 0.0 | 12663.2 |
| GovReport | HMO | 50 | 0.152 | 0.152 | 54.09 | 2.3 | 27.0 | 0.5 | 0.0 |
| LCC | Full KV | 18 | 1.045 | 0.000 | 54.22 | 0.0 | 0.0 | 0.0 | 0.0 |
| LCC | Budgeted Recent KV | 18 | 0.157 | 0.157 | 54.22 | 2402.7 | 0.0 | 0.0 | 13542.8 |
| LCC | Budgeted Uniform KV | 18 | 0.157 | 0.157 | 54.22 | 2402.7 | 0.0 | 0.0 | 13542.8 |
| LCC | H2O | 18 | 0.157 | 0.157 | 56.24 | 2402.7 | 0.0 | 0.0 | 13542.8 |
| LCC | SnapKV | 18 | 0.157 | 0.157 | 54.22 | 2402.7 | 0.0 | 0.0 | 13542.8 |
| LCC | StreamingLLM | 18 | 0.157 | 0.157 | 54.23 | 2402.7 | 0.0 | 0.0 | 13542.8 |
| LCC | DuoAttention | 18 | 0.157 | 0.157 | 54.23 | 2402.7 | 0.0 | 0.0 | 13542.8 |
| LCC | PyramidKV-lite | 18 | 0.157 | 0.157 | 54.22 | 2403.2 | 0.0 | 0.0 | 13542.3 |
| LCC | Quest-lite | 18 | 0.155 | 0.157 | 54.22 | 2367.0 | 0.0 | 0.0 | 13578.6 |
| LCC | SAGE-KV-lite | 18 | 0.157 | 0.157 | 54.22 | 2402.7 | 0.0 | 0.0 | 13542.8 |
| LCC | HMO | 18 | 0.157 | 0.157 | 54.42 | 2.2 | 28.3 | 1.1 | 0.0 |

