# Data

The datasets are public and are not redistributed here. Download them and place them as follows
(paths are relative to `code/`, which reads `code/data/...`; create a symbolic link `code/data -> ../data` or copy this folder):

| File | Source | SHA-256 of the file we used |
|---|---|---|
| `raw/uci` | KONECT "UC Irvine messages" (opsahl-ucsocial), file `out.opsahl-ucsocial`, renamed | 72515c16504464736f35498047cf8e163df2e95f89ea3063d48aa367936bd951 |
| `raw/soc-sign-bitcoinalpha.csv` | SNAP, soc-sign-bitcoin-alpha (gunzip) | 1b2a970f327d0ceba0c57bd5919670257cbe4cc0704e2ddac09abc4b08e2ca4d |
| `raw/soc-sign-bitcoinotc.csv` | SNAP, soc-sign-bitcoin-otc (gunzip) | 76bd9d8f1d3ff9a1813d9fc8e6902a0ee4d0a2f8c1003842dbc9ec79149ab60c |
| `snap/email-Eu-core-temporal.txt` | SNAP, email-Eu-core-temporal (gunzip) | 1835c94dc6685dcf2a614a1b6a0275b323dd4002f7aa111fbe9f76db8864391f |
| `snap/sx-mathoverflow.txt` | SNAP, sx-mathoverflow (gunzip) | 2bf52f784cdba69730f912633aa1414b381059406ded2c38bc37658ead5b0773 |

Preprocessing is done on loading (`code/motif_bench.py`, function `load`): interactions are made undirected
(u, v) -> (min, max), self-loops are removed, and events are sorted by time. Edge weights/signs are ignored.

## Held-out graphs (Section 7.14; place in `data/other/`)

| File | Source | SHA-256 of the file we used |
|---|---|---|
| `other/radoslaw.txt` | KONECT "radoslaw_email" (manufacturing company e-mail), file `out.radoslaw_email_email`, renamed | 0e90889f3c9dc4752d0a36aea187ddcf9aa0fe70e2c5e496c9e17d417ab0b45b |
| `other/thiers13.txt` | SocioPatterns "High school 2013" contact data (`High-School_data_2013` / `tij_Thiers13.dat`), columns t i j (remove any header) | f86fc0324d8486f0df1715dee027154bc37c2f7b3ca0260469a9c76119ea7d58 (188,508 lines, 327 nodes, 5,818 pairs) |

The NRC implementation (`coupling2.py`, with `coupling.py` for the scales) was last modified on 2026-10-31, before either
held-out graph was loaded (Thiers13 on 2026-11-01, Radoslaw on 2026-11-02). `results/freeze_record.txt` lists the
SHA-256 checksums of the code, recorded on 2026-11-02 immediately before the Radoslaw run; the NRC files were unchanged
from 2026-10-31 to that point. The only later code changes are data loaders and a strict periodic-generator variant.
