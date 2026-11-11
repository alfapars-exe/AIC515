# Atıf künyelerinin denetimi: Crossref, arXiv ve DBLP (11.11.2026)

Betik: `kod/tools/check_refs_dblp.py`. DBLP'nin arama API'si bot denetimi ve istek sınırı arkasında olduğundan (betik sorgusu yanıtsız; tarayıcıdan üçüncü istekten sonra 429) DBLP sorgulanmadı. DOI'li kayıtlar Crossref kaydıyla, arXiv kayıtları arXiv API'siyle karşılaştırıldı: başlık, yazar soyadları (sırasıyla) ve yıl. Ön baskı olarak atıf yapılan kaynakların hakemli sürümü ve DOI'siz kayıtlar Crossref başlık aramasıyla arandı.

**Özet.** 81 kayıt (64 tanesi bildiride atıf yapılan): 74 uyumlu, 0 fark, 7 bu kaynaklarla doğrulanamadı. Kaynak dağılımı: Crossref 57, DataCite 1, arXiv 16, — 7.

| Anahtar | Bildiride | Tür | Yıl | Denetim kaynağı | Kayıttaki yer ve yıl | Durum | Not |
|---|---|---|---|---|---|---|---|
| `wu2026aiforgedoc` | evet | misc | 2026 | arXiv | arXiv 2026 (v2) | uyumlu | — |
| `guillaro2023trufor` | evet | inproceedings | 2023 | Crossref | 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023 | uyumlu | — |
| `wu2026aiforgedocv2` | evet | misc | 2026 | arXiv | arXiv 2026 (v2) | uyumlu | — |
| `geirhos2020shortcut` | evet | article | 2020 | Crossref | Nature Machine Intelligence 2020 | uyumlu | — |
| `shang2026maskspixels` | evet | inproceedings | 2026 | Crossref | Lecture Notes in Computer Science 2026 | uyumlu | — |
| `sauvola2000binarization` | evet | article | 2000 | Crossref | Pattern Recognition 2000 | uyumlu | — |
| `mareen2024tgif` | evet | inproceedings | 2024 | Crossref | 2024 IEEE International Workshop on Information Forensics and Security (WIFS) 2024 | uyumlu | — |
| `ronneberger2015unet` | evet | inproceedings | 2015 | Crossref | Lecture Notes in Computer Science 2015 | uyumlu | — |
| `he2016resnet` | evet | inproceedings | 2016 | Crossref | 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2016 | uyumlu | — |
| `milletari2016vnet` | evet | inproceedings | 2016 | Crossref | 2016 Fourth International Conference on 3D Vision (3DV) 2016 | uyumlu | — |
| `loshchilov2019adamw` | evet | inproceedings | 2019 | — | — | doğrulanamadı | DOI ve arXiv numarası yok; Crossref başlık aramasında kayıt yok (yazılım, web kaynağı ya da DOI vermeyen yayın yeri); bib notu: src: ICLR 2019 conference page and OpenReview PDF ("Published as a conference paper at ICLR 2019"); arXiv 1711.05101; checked 2026-11-10 |
| `chen2018deeplabv3plus` | evet | inproceedings | 2018 | Crossref | Lecture Notes in Computer Science 2018 | uyumlu | — |
| `jocher2024yolo11` | evet | misc | 2024 | — | — | doğrulanamadı | DOI ve arXiv numarası yok; Crossref başlık aramasında kayıt yok (yazılım, web kaynağı ya da DOI vermeyen yayın yeri); bib notu: src: docs.ultralytics.com/models/yolo11 (official citation block; no paper exists; DOI "pending") |
| `akyon2022sahi` | evet | inproceedings | 2022 | Crossref | 2022 IEEE International Conference on Image Processing (ICIP) 2022 | uyumlu | — |
| `lin2014coco` | evet | inproceedings | 2014 | Crossref | Lecture Notes in Computer Science 2014 | uyumlu | — |
| `efron1979bootstrap` | evet | article | 1979 | Crossref | The Annals of Statistics 1979 | uyumlu | — |
| `holm1979` | evet | article | 1979 | — | — | doğrulanamadı | DOI ve arXiv numarası yok; Crossref başlık aramasında kayıt yok (yazılım, web kaynağı ya da DOI vermeyen yayın yeri); bib notu: src: JSTOR stable/4615733 (Scand. J. Statist. 6(2):65-70, 1979); checked 2026-11-10 |
| `qu2023doctamper` | evet | inproceedings | 2023 | Crossref | 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023 | uyumlu | — |
| `chen2024ffdn` | evet | inproceedings | 2024 | Crossref | Lecture Notes in Computer Science 2025 | uyumlu | Crossref yılı 2025 (basım yılı; bildiri kitabı yılı 2024) |
| `wong2025adcdnet` | evet | inproceedings | 2025 | Crossref | 2025 IEEE/CVF International Conference on Computer Vision (ICCV) 2025 | uyumlu | — |
| `dhouib2026diffnet` | evet | inproceedings | 2026 | Crossref | Lecture Notes in Computer Science 2026 | uyumlu | — |
| `docqt2026` | evet | inproceedings | 2026 | Crossref | Lecture Notes in Computer Science 2027 | uyumlu | Crossref yılı 2027 (basım yılı; bildiri kitabı yılı 2026) |
| `ctpnet2023` | evet | misc | 2023 | arXiv | arXiv 2023 (v2) | uyumlu | — |
| `song2024caftbnet` | evet | article | 2025 | Crossref | ACM Transactions on Multimedia Computing, Communications, and Applications 2025 | uyumlu | — |
| `dong2024tce` | evet | article | 2024 | Crossref | IEEE Transactions on Consumer Electronics 2024 | uyumlu | — |
| `li2025ditl2` | evet | inproceedings | 2025 | Crossref | Proceedings of the 33rd ACM International Conference on Multimedia 2025 | uyumlu | — |
| `wang2022tamperedic13` | evet | inproceedings | 2022 | Crossref | Lecture Notes in Computer Science 2022 | uyumlu | — |
| `qu2025ostf` | evet | inproceedings | 2025 | Crossref | Proceedings of the AAAI Conference on Artificial Intelligence 2025 | uyumlu | — |
| `wang2022tsroie` | evet | article | 2022 | — | — | doğrulanamadı | DOI doi.org'da kayıtlı; kayıt ajansı Crossref ve DataCite dışında; Crossref/DataCite sorgusu başarısız (HTTP 404); bib notu: src: DOAJ article record (Chinese Journal of Network and Information Security / Wangluo yu Xinxi Anquan Xuebao, ISSN 2096-109X, Jun 2022: vol. 8, pp. 29-40). Issue 3 and "(in Chinese)" from the reference list of Ju et al., Acta Electronica Sinica 54(3), 2026,  |
| `luo2023icdar` | evet | inproceedings | 2023 | Crossref | Lecture Notes in Computer Science 2023 | uyumlu | — |
| `luo2025rtm` | evet | article | 2025 | Crossref | Pattern Recognition 2025 | uyumlu | — |
| `xu2026realitygap` | evet | article | 2026 | Crossref | IEEE Transactions on Information Forensics and Security 2026 | uyumlu | — |
| `qu2024textsleuth` | evet | misc | 2024 | arXiv | arXiv 2024 (v3) | uyumlu | — |
| `qu2026textshieldr1` | evet | inproceedings | 2026 | Crossref | Proceedings of the AAAI Conference on Artificial Intelligence 2026 | uyumlu | — |
| `zeng2026docshield` | evet | misc | 2026 | arXiv | arXiv 2026 (v2) | uyumlu | — |
| `zhang2026scrf` | evet | inproceedings | 2026 | Crossref | Proceedings of the 34th ACM International Conference on Multimedia 2026 | uyumlu | — |
| `gentextforensics2026` | evet | inproceedings | 2026 | Crossref | Proceedings of the 34th ACM International Conference on Multimedia 2026 | uyumlu | — |
| `zhao2026docforgebench` | evet | misc | 2026 | arXiv | arXiv 2026 (v3) | uyumlu | — |
| `zhang2026gpt4oreceipt` | evet | misc | 2026 | arXiv | arXiv 2026 (v3) | uyumlu | — |
| `mllmdoc2025` | evet | misc | 2025 | arXiv | arXiv 2025 (v2) | uyumlu | — |
| `textfake2026` | evet | misc | 2026 | arXiv | arXiv 2026 (v1) | uyumlu | — |
| `raj2026images25` | evet | misc | 2026 | arXiv | arXiv 2026 (v3) | uyumlu | — |
| `helali2026processtwin` | evet | misc | 2026 | DataCite | Zenodo 2026 | uyumlu | — |
| `leong2026docdet` | evet | misc | 2026 | — | — | doğrulanamadı | DOI ve arXiv numarası yok; Crossref başlık aramasında kayıt yok (yazılım, web kaynağı ya da DOI vermeyen yayın yeri); bib notu: src: GitHub repository page (checked 2026-11-09). NON-ARCHIVAL: no paper, no DOI, no licence stated. |
| `smeu2025shortcuts` | evet | inproceedings | 2025 | Crossref | 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025 | uyumlu | — |
| `yermakov2026gend` | evet | inproceedings | 2026 | Crossref | 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026 | uyumlu | — |
| `freqbias2025` | evet | article | 2026 | Crossref | IEEE Transactions on Dependable and Secure Computing 2026 | uyumlu | — |
| `nebioglu2026inpx` | evet | misc | 2026 | arXiv | arXiv 2026 (v2) | uyumlu | — |
| `giakoumoglou2026diffusionprint` | evet | misc | 2026 | arXiv | arXiv 2026 (v1) | uyumlu | — |
| `mareen2026tgif2` | evet | article | 2026 | Crossref | Journal on Information Security 2026 | uyumlu | — |
| `pandolfini2025localized` | evet | misc | 2025 | arXiv | arXiv 2025 (v1) | uyumlu | — |
| `cai2026zooming` | evet | inproceedings | 2026 | Crossref | Proceedings of the AAAI Conference on Artificial Intelligence 2026 | uyumlu | — |
| `aoudi2026beyond` | evet | article | 2026 | Crossref | IEEE Access 2026 | uyumlu | — |
| `giftbench2026` | evet | misc | 2026 | arXiv | arXiv 2026 (v1) | uyumlu | — |
| `dai2015boxsup` | evet | inproceedings | 2015 | Crossref | 2015 IEEE International Conference on Computer Vision (ICCV) 2015 | uyumlu | — |
| `tian2021boxinst` | evet | inproceedings | 2021 | Crossref | 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2021 | uyumlu | — |
| `maierhein2024metrics` | evet | article | 2024 | Crossref | Nature Methods 2024 | uyumlu | — |
| `zhou2018rgbn` | evet | inproceedings | 2018 | Crossref | 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition 2018 | uyumlu | — |
| `redmon2016yolo` | evet | inproceedings | 2016 | Crossref | 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2016 | uyumlu | — |
| `cheng2023smallobject` | evet | article | 2023 | Crossref | IEEE Transactions on Pattern Analysis and Machine Intelligence 2023 | uyumlu | — |
| `he2017maskrcnn` | evet | inproceedings | 2017 | Crossref | 2017 IEEE International Conference on Computer Vision (ICCV) 2017 | uyumlu | — |
| `guangjian2026defakerone` | evet | misc | 2026 | arXiv | arXiv 2026 (v2) | uyumlu | — |
| `lundberg2017shap` | evet | inproceedings | 2017 | — | — | doğrulanamadı | DOI ve arXiv numarası yok; Crossref başlık aramasında kayıt yok (yazılım, web kaynağı ya da DOI vermeyen yayın yeri); bib notu: src: NeurIPS proceedings metadata (papers.nips.cc, paper hash 8a20a8621978632d76c43dfd28b67767): pages 4765-4774 [added 2026-11-10] |
| `samek2017evaluating` | evet | article | 2017 | Crossref | IEEE Transactions on Neural Networks and Learning Systems 2017 | uyumlu | — |
| `fsts2025` | hayır | inproceedings | 2025 | Crossref | Advances in Neural Information Processing Systems 38 2025 | uyumlu | — |
| `martineztornes2023receipt` | hayır | inproceedings | 2023 | Crossref | Lecture Notes in Computer Science 2023 | uyumlu | — |
| `kwon2022catnet` | hayır | article | 2022 | Crossref | International Journal of Computer Vision 2022 | uyumlu | — |
| `dong2023mvssnet` | hayır | article | 2023 | Crossref | IEEE Transactions on Pattern Analysis and Machine Intelligence 2023 | uyumlu | — |
| `liu2022psccnet` | hayır | article | 2022 | Crossref | IEEE Transactions on Circuits and Systems for Video Technology 2022 | uyumlu | — |
| `ma2023imlvit` | hayır | misc | 2023 | arXiv | arXiv 2023 (v4) | uyumlu | — |
| `zhu2025mesorch` | hayır | inproceedings | 2025 | Crossref | Proceedings of the AAAI Conference on Artificial Intelligence 2025 | uyumlu | — |
| `kwon2025safire` | hayır | inproceedings | 2025 | Crossref | Proceedings of the AAAI Conference on Artificial Intelligence 2025 | uyumlu | — |
| `qu2024omniiml` | hayır | inproceedings | 2026 | — | — | doğrulanamadı | DOI ve arXiv numarası yok; Crossref başlık aramasında kayıt yok (yazılım, web kaynağı ya da DOI vermeyen yayın yeri); bib notu: src: ICLR 2026 virtual poster page (iclr.cc/virtual/2026/poster/10007905); arXiv 2411.14823 [FIXED: preprint -> published ICLR 2026 version; published title adds "Interpretable"] |
| `cocoinpaint2025` | hayır | inproceedings | 2026 | Crossref | 2026 IEEE International Conference on Image Processing (ICIP) 2026 | uyumlu | — |
| `ahmed1974dct` | hayır | article | 1974 | Crossref | IEEE Transactions on Computers 1974 | uyumlu | — |
| `ojala2002lbp` | hayır | article | 2002 | Crossref | IEEE Transactions on Pattern Analysis and Machine Intelligence 2002 | uyumlu | — |
| `haralick1973glcm` | hayır | article | 1973 | Crossref | IEEE Transactions on Systems, Man, and Cybernetics 1973 | uyumlu | — |
| `cortes1995svm` | hayır | article | 1995 | Crossref | Machine Learning 1995 | uyumlu | — |
| `breiman2001rf` | hayır | article | 2001 | Crossref | Machine Learning 2001 | uyumlu | — |
| `deng2009imagenet` | hayır | inproceedings | 2009 | Crossref | 2009 IEEE Conference on Computer Vision and Pattern Recognition 2009 | uyumlu | — |
| `kendall1938tau` | hayır | article | 1938 | Crossref | Biometrika 1938 | uyumlu | — |
