| Property | Value (computed from files) |
|---|---|
| Forged images (train / test) | 3249 / 812 |
| Authentic files (train / test) | 3,249 / 812 (test set is not the mirror of the test forgeries: 86/812 are sources of a test forgery) |
| Source datasets | CORD 1,000 · WildReceipt 1,696 · SROIE 946 · XFUND 419 |
| Document type / languages | receipt 89.7 %, form 10.3 %; 9 languages (en 2,642, id 1,000) |
| Generator (metadata label) | Gemini 2.5 Flash Image (`gemini-nano`) 3,639 · Ideogram v2 Edit (`qwen-inpaint`) 422 |
| Mask format | 8-bit PNG {0,255}; 100 % single rectangles (rectangularity 1.00) |
| Metadata box = mask box | 4,047 / 4,061 (12 rows with swapped provenance) |
| Image size | W median 750 (150–6,016), H median 1,296 (168–7,016); 0.03–34.8 MP |
| Tampered area / image | median 0.98 % (IQR 0.37–1.62 %); < 1 %: 50.7 %, 1–5 %: 49.0 %, > 5 %: 0.4 % |
| Mask height / width (median) | 33 px / 191 px |
| Verified forged–authentic pairs | 3,247 / 3,249 (train), 89 / 812 (test) |
| Box pixels that actually change (D > 8) | median 49 % (IQR 37–62 %); IoU({D>8}, box) median 0.48 |
| Pairs with changes outside the box (D > 8) | 33.5 % (Gemini 35.9 %, Ideogram 11.9 %) |
| Pairs with a global ±1–3 re-encoding trace (D > 0) | 48.6 % (vanishes for ε ≥ 4) |
| Test forgeries sharing a source document with training (pHash ≤ 20) | 72 / 812 (8.9 %) |
| Licence | card: CC BY-NC-SA 4.0 + gated; `dataset_info.json`: CC BY 4.0 (conflict) |