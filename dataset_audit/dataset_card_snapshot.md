# AIForge-Doc v1 — Dataset Card Snapshot

Kaynak: Hugging Face `Scam-AI/AIForge-Doc-v1` (dataset repo), erişim tarihi 2026-10-20 (UTC).
Repo revision (sha): `b8db5654e9be43b0afcc45f8d570daf8bef96d41` · lastModified: `2026-08-06T04:30:48.000Z` · downloads: 951 · likes: 5 · gated: auto
Hub tags: task_categories:image-classification, task_categories:image-segmentation, language:en, license:cc-by-nc-sa-4.0, size_categories:1K<n<10K, arxiv:2603.01433, arxiv:2604.25213, arxiv:2602.20569, region:us, scam-ai, document-forgery, document-forensics, ai-generated, gemini, ideogram, tampering-detection, document-fraud

## Yerel dosya listesi (indirme sonrası, kodla sayıldı)

| Dizin | Dosya sayısı | GB |
|---|---|---|
| `.` | 4 | 0.002 |
| `TestingSet/authentic` | 812 | 1.150 |
| `TestingSet/images` | 812 | 0.987 |
| `TestingSet/masks` | 812 | 0.003 |
| `TrainingSet/authentic` | 3249 | 3.893 |
| `TrainingSet/images` | 3249 | 3.939 |
| `TrainingSet/masks` | 3249 | 0.010 |
| **Toplam** | **12187** | **9.984** |

Uzantılar: {'': 1, '.json': 1, '.jsonl': 1, '.md': 1, '.png': 12183} · Üst düzey dosyalar: ['.gitattributes', 'README.md', 'dataset_info.json', 'metadata.jsonl']
Not: README'de listelenen `DATASHEET.md` repoda YOK. README'nin "Directory Layout" bölümünde `TrainingSet/authentic/` yer almıyor, ancak repoda 3.249 dosyayla mevcut.

## Hub API dosya listesi özeti (indirme öncesi, `/api/datasets/...?blobs=true`)

Toplam 12187 dosya, 9.984 GB (Hub tarafından raporlanan boyutlar).

## README.md (birebir kopya)

```markdown
---
license: cc-by-nc-sa-4.0
pretty_name: "AIForge-Doc v1 — AI-Forged Document Images"
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - image-classification
  - image-segmentation
tags:
  - scam-ai
  - document-forgery
  - document-forensics
  - ai-generated
  - gemini
  - ideogram
  - tampering-detection
  - document-fraud
gated: true
extra_gated_heading: "Access Scam.AI Research Dataset"
extra_gated_description: |
  This dataset is released by Scam.AI for academic research and non-commercial use.
  Please share a couple of details so we can understand how the community is using our work.
  Access is granted automatically after submission.
extra_gated_button_content: "Agree and access dataset"
extra_gated_fields:
  Full Name: text
  Email: text
  I agree to use this dataset for non-commercial research purposes only and to cite the corresponding paper if I publish results based on it: checkbox
---

# AIForge-Doc: A Benchmark of AI-Forged Document Images

[![Format: DocTamper-compatible](https://img.shields.io/badge/Format-DocTamper--compatible-blue)]()
[![Images: 4,061](https://img.shields.io/badge/Images-4%2C061-green)]()

**AIForge-Doc** is the first large-scale benchmark of AI-forged document images, targeting
financial and identity document fraud.  Every tampered image was produced by a
diffusion-model inpainting pipeline — a threat model that existing forgery detectors
cannot reliably handle.

---

## At a Glance

| Attribute | Value |
|---|---|
| Total forged images | **4,061** |
| Training split | 3,249 (80 %) |
| Testing split | 812 (20 %) |
| Authentic baseline images | 812 (mirror of test split) |
| AI inpainting tools used | 2 (Gemini 2.5 Flash Image, Ideogram v2 Edit) |
| Source datasets | CORD v2, WildReceipt, SROIE, XFUND |
| Document types | Receipts (89.7 %), Forms (10.3 %) |
| Languages | 9 (EN, ID, DE, IT, ES, FR, PT, ZH, JA) |
| Output format | DocTamper-compatible (binary grayscale masks) |
| Mask convention | `0` = authentic · `255` = tampered pixel |

---

## Directory Layout

```
AIForge-Doc/
├── TrainingSet/
│   ├── images/       # 000000001.png … 000003249.png
│   └── masks/        # same filenames; 0=authentic, 255=tampered
├── TestingSet/
│   ├── images/       # 000000001.png … 000000812.png
│   └── masks/
├── metadata.jsonl    # full provenance for every image (see schema below)
├── README.md         # this file
└── DATASHEET.md      # Datasheets for Datasets (Gebru et al., 2021)
```

File names are **9-digit zero-padded integers** (`000000001.png`), identical to the
DocTamper dataset convention so that existing evaluation pipelines require no modification.

---

## Provenance — metadata.jsonl Schema

Each line is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `spec_id` | str | Unique forgery spec identifier |
| `image_id` | str | Original image ID from source dataset |
| `source_dataset` | str | `cord` / `wildreceipt` / `sroie` / `xfund` |
| `doc_type` | str | `receipt` or `form` |
| `language` | str | ISO 639-1 language code |
| `field_name` | str | Annotation key of the tampered field |
| `original_value` | str | Ground-truth text before tampering |
| `forged_value` | str | Synthesised replacement text |
| `bbox_xyxy` | list[int] | Tampered region `[x1, y1, x2, y2]` in full-image pixels |
| `assigned_tool` | str | Inpainting model used (`gemini-nano` / `qwen-inpaint`) |
| `split` | str | `training` or `testing` |
| `new_id` | str | 9-digit zero-padded filename stem |
| `final_image_path` | str | Absolute path on generation machine |
| `final_mask_path` | str | Absolute path on generation machine |
| `generated_at` | str | ISO 8601 timestamp |

---

## Dataset Statistics

### By Inpainting Tool

| Tool | API Provider | Images | Share |
|---|---|---|---|
| Gemini 2.5 Flash Image (`gemini-nano`) | Google / OpenRouter | 3,639 | 89.6 % |
| Ideogram v2 Edit (`qwen-inpaint`) | fal.ai | 422 | 10.4 % |

### By Source Dataset

| Source | Document Type | Images | Languages |
|---|---|---|---|
| WildReceipt | Receipt | 1,696 | EN |
| CORD v2 | Receipt | 1,000 | ID |
| SROIE | Receipt | 946 | EN |
| XFUND | Form | 419 | DE, IT, ES, FR, PT, ZH, JA |

### By Language

| Language | Code | Images |
|---|---|---|
| English | `en` | 2,642 |
| Indonesian | `id` | 1,000 |
| Italian | `it` | 81 |
| German | `de` | 78 |
| Spanish | `es` | 67 |
| French | `fr` | 56 |
| Portuguese | `pt` | 62 |
| Chinese | `zh` | 38 |
| Japanese | `ja` | 37 |

### Most Commonly Tampered Field Types

| Field | Count |
|---|---|
| Telephone / store address | 1,388 |
| Free-form text | 880 |
| Store address | 457 |
| Form answer | 419 |
| Menu price | 404 |

---

## Forgery Generation Pipeline

Each forgery is created in four steps to prevent global image drift:

1. **Field selection** — A numeric or key field (price, date, ID, phone) is chosen from
   the source annotation and a plausible replacement value is generated.
2. **Context crop** — The bounding box is expanded 50 % on each side (minimum 100 px) to
   provide font and colour context for the inpainting model.
3. **Masked inpainting** — The context crop is inpainted with the replacement text using
   a diffusion-model API (white mask = tamper region, black = preserve).
4. **Patch-back** — Only the exact field bbox region is pasted back into the full image;
   the ground-truth mask marks those pixels as `255`.

This technique ensures that only the tampered field is replaced while the rest of the
document (background, typography, logos) remains authentic.

---

## Baseline Results

Evaluated on the **TestingSet** (812 forged + 812 authentic paired images).

### Image-Level Detection (AUC-ROC)

| Method | AUC | 95 % CI | AP |
|---|---|---|---|
| TruFor (Guillaro et al., 2023) | **0.751** | [0.726, 0.776] | 0.709 |
| DocTamper (Qu et al., 2023) | 0.563 | [0.535, 0.591] | 0.564 |
| GPT-4o (zero-shot) | 0.509 | [0.481, 0.537] | 0.516 |

### Pixel-Level Localisation (TruFor only)

| Metric | Value |
|---|---|
| IoU | 0.358 |
| F1 | 0.434 |
| Pixel-AUC | 0.916 |

**Key finding:** Even the best-performing detector (TruFor) achieves only 0.751 AUC —
well below the ≥ 0.95 reported on traditional Photoshop-tampered benchmarks.
DocTamper and GPT-4o are near-random, confirming that AI-generated forgeries represent
a qualitatively different and substantially harder threat model.

---

## Licence

The forged images are derived from:

- **CORD v2** — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **WildReceipt** — [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **SROIE** — ICDAR 2019 research use
- **XFUND** — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The AIForge-Doc dataset itself (forged images + masks + metadata) is released under
[cc-by-nc-sa-4.0](https://creativecommons.org/licenses/by/4.0/).  You are free to share and
adapt the material for any purpose provided you give appropriate credit.

---

## Citation

If you use AIForge-Doc in your research, please cite:

```bibtex
@dataset{aiforgedoc2026,
  title   = {{AIForge-Doc}: A Benchmark of AI-Forged Document Images},
  year    = {2026},
  note    = {Dataset paper under submission},
  url     = {https://github.com/YOUR_ORG/aiforge-doc}
}
```

---

## Contact

For questions about the dataset or to report issues, please open a GitHub issue or
contact the authors at [your-email@institution.edu].

---

## Related Research from Scam.AI

This dataset is part of Scam.AI's broader research portfolio on deepfake detection, synthetic media forensics, and adversarial robustness. Other relevant work from our group:

- **DOCFORGE-BENCH: A Comprehensive Benchmark for Document Forgery Detection and Analysis** — Zhao, Xia, Wei et al. (arXiv:2603.01433)
- **When the Forger Is the Judge: GPT-Image-2 Cannot Recognize Its Own Faked Documents** — Wu, Zhou, Ng et al. (arXiv:2604.25213)
- **AIForge-Doc: A Benchmark for Detecting AI-Forged Tampering in Financial and Form Documents** — Wu, Zhou, Xu et al. (arXiv:2602.20569)
- **Can Multi-modal (reasoning) LLMs detect document manipulation?** — Liang, Zewde, Singh et al. (Google Scholar)

Browse our full publications list and dataset catalog at **[scam.ai/research](https://www.scam.ai/en/research)**.

## About Scam.AI

Scam.AI builds detection systems for AI-driven fraud — deepfakes, document forgery, AI-generated synthetic media, and adversarial attacks against identity verification. Learn more at **[scam.ai](https://www.scam.ai)**.

```

## dataset_info.json (birebir kopya)

```json
{
  "name": "AIForge-Doc",
  "version": "1.0.0",
  "description": "First large-scale benchmark of AI-forged document images using diffusion-model inpainting",
  "created": "2026-02-23",
  "license": "CC BY 4.0",
  "format": "DocTamper-compatible",
  "mask_convention": {"0": "authentic pixel", "255": "tampered pixel"},
  "filename_format": "9-digit zero-padded integer (000000001.png)",

  "splits": {
    "TrainingSet": {
      "images": 3249,
      "masks": 3249
    },
    "TestingSet": {
      "images": 812,
      "masks": 812
    },
    "total_forged": 4061
  },

  "inpainting_tools": {
    "gemini-nano": {
      "description": "Gemini 2.5 Flash Image generation (Google, via OpenRouter)",
      "api": "openrouter.ai",
      "count": 3639,
      "share": 0.896
    },
    "qwen-inpaint": {
      "description": "Ideogram v2 Edit (fal.ai)",
      "api": "fal.ai/ideogram/v2/edit",
      "count": 422,
      "share": 0.104
    }
  },

  "source_datasets": {
    "wildreceipt": {
      "doc_type": "receipt",
      "count": 1696,
      "languages": ["en"],
      "license": "Apache 2.0",
      "citation": "Sun et al., 2021"
    },
    "cord": {
      "doc_type": "receipt",
      "count": 1000,
      "languages": ["id"],
      "license": "CC BY 4.0",
      "citation": "Park et al., 2019"
    },
    "sroie": {
      "doc_type": "receipt",
      "count": 946,
      "languages": ["en"],
      "license": "ICDAR 2019 research use",
      "citation": "Huang et al., 2019"
    },
    "xfund": {
      "doc_type": "form",
      "count": 419,
      "languages": ["de", "it", "es", "fr", "pt", "zh", "ja"],
      "license": "CC BY 4.0",
      "citation": "Xu et al., 2022"
    }
  },

  "language_distribution": {
    "en": 2642,
    "id": 1000,
    "it": 81,
    "de": 78,
    "es": 67,
    "pt": 62,
    "zh": 38,
    "ja": 37,
    "fr": 56
  },

  "document_type_distribution": {
    "receipt": 3642,
    "form": 419
  },

  "top_field_types": [
    {"field": "Telephone_key", "count": 931},
    {"field": "text", "count": 880},
    {"field": "Store_addr_key", "count": 457},
    {"field": "answer", "count": 419},
    {"field": "menu.price", "count": 404},
    {"field": "Prod_item_key", "count": 152},
    {"field": "total.cashprice", "count": 134},
    {"field": "menu.unitprice", "count": 127},
    {"field": "sub_total.subtotal_price", "count": 100},
    {"field": "menu.nm", "count": 77}
  ],

  "baseline_results": {
    "evaluation_set": "TestingSet",
    "n_forged": 812,
    "n_authentic": 812,
    "methods": {
      "TruFor": {
        "citation": "Guillaro et al., 2023",
        "image_level": {
          "auc": 0.7513,
          "auc_95ci": [0.7263, 0.7757],
          "ap": 0.7087,
          "acc_opt": 0.7309
        },
        "pixel_level": {
          "iou": 0.3578,
          "f1": 0.4335,
          "auc_px": 0.9157
        },
        "ci_method": "bootstrap (10,000 resamples)"
      },
      "DocTamper": {
        "citation": "Qu et al., 2023",
        "image_level": {
          "auc": 0.5631,
          "auc_95ci": [0.5348, 0.5907],
          "ap": 0.5642,
          "acc_opt": 0.5585
        },
        "pixel_level": {
          "iou": 0.0204,
          "f1": 0.0300,
          "auc_px": 0.6748
        },
        "ci_method": "bootstrap (10,000 resamples)"
      },
      "GPT-4o": {
        "citation": "OpenAI, 2024",
        "image_level": {
          "auc": 0.5089,
          "auc_95ci": [0.4808, 0.5370],
          "ap": 0.5162,
          "acc_opt": 0.5351
        },
        "pixel_level": null,
        "ci_method": "Hanley-McNeil (1982)"
      }
    }
  },

  "generation_pipeline": {
    "context_expansion_factor": 0.5,
    "context_expansion_min_px": 100,
    "mask_type": "binary (field bbox only)",
    "inpaint_resolution": "context crop only (not full image)",
    "paste_back": "exact field bbox region pasted back to full image"
  },

  "metadata_schema": {
    "spec_id": "string — unique forgery spec identifier",
    "image_id": "string — source dataset image ID",
    "source_dataset": "string — cord | wildreceipt | sroie | xfund",
    "doc_type": "string — receipt | form",
    "language": "string — ISO 639-1 code",
    "field_name": "string — annotation key of tampered field",
    "original_value": "string — ground-truth text before tampering",
    "forged_value": "string — synthesised replacement text",
    "bbox_xyxy": "list[int] — [x1, y1, x2, y2] in full-image pixels",
    "assigned_tool": "string — inpainting model used",
    "split": "string — training | testing",
    "new_id": "string — 9-digit zero-padded filename stem",
    "generated_at": "string — ISO 8601 timestamp"
  }
}

```

## metadata.jsonl — ilk 2 satır ve gerçek şema

Gerçek alanlar (4061 satır): ['spec_id', 'source_dataset', 'doc_type', 'language', 'field_name', 'original_value', 'forged_value', 'bbox_xyxy', 'assigned_tool', 'split', 'new_id', 'generated_at', 'image', 'mask', 'key']

README şemasında olup metadata.jsonl'da OLMAYAN alanlar: `image_id`, `final_image_path`, `final_mask_path`.
metadata.jsonl'da olup README şemasında OLMAYAN alanlar: `image`, `mask` (göreli yollar).

İlk 2 satır (ham JSONL):

```json
{"spec_id": "000000099", "source_dataset": "cord", "doc_type": "receipt", "language": "id", "field_name": "menu.nm", "original_value": "TROPICANA.CHO75", "forged_value": "TROPICANA.CHO40", "bbox_xyxy": [148, 894, 337, 923], "assigned_tool": "gemini-nano", "split": "training", "new_id": "000000001", "generated_at": "2026-02-22T00:00:00", "image": "TrainingSet/images/000000001.png", "mask": "TrainingSet/masks/000000001.png"}
{"spec_id": "000000974", "source_dataset": "cord", "doc_type": "receipt", "language": "id", "field_name": "menu.nm", "original_value": "6001-Plastic", "forged_value": "4748-Plastic", "bbox_xyxy": [129, 749, 340, 786], "assigned_tool": "gemini-nano", "split": "training", "new_id": "000000002", "generated_at": "2026-02-22T00:00:00", "image": "TrainingSet/images/000000002.png", "mask": "TrainingSet/masks/000000002.png"}
```

Not: yukarıdaki 'Gerçek alanlar' listesindeki `key` alanı metadata.jsonl'da bulunmaz; audit sırasında eklenen (split öneki + new_id) türetilmiş sütundur. Ham satırlarda 14 alan vardır.
