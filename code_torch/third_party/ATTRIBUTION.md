# Third-party baseline code — provenance & license

The official test-time adaptation baselines are **vendored verbatim** here (both
MIT-licensed) and used through a thin graph wrapper.  Only two clearly marked
`# [graph-port]` changes were applied to each file, neither of which alters the
adaptation algorithm:

1. the BatchNorm type check is generalized from `BatchNorm2d` to also include
   `BatchNorm1d` (graph node features are 1D: `[num_nodes, channels]`);
2. the single-argument `model(x)` forward is satisfied by `GraphModelWrapper`
   (in `../baselines_official.py`), which stores `edge_index` and forwards
   `gnn(x, edge_index)`.

| File | Method | Upstream repo | Paper | License |
|---|---|---|---|---|
| `tent.py` | Tent | https://github.com/DequanWang/tent (`tent.py`, `master`) | Wang & Shelhamer et al., *Tent: Fully Test-Time Adaptation by Entropy Minimization*, ICLR 2021 | MIT |
| `eata.py` | EATA | https://github.com/mr-eggplant/EATA (`eata.py`, `main`) | Niu et al., *Efficient Test-Time Model Adaptation without Forgetting*, ICML 2022 | MIT |

Downloaded 2026-06-10.

The Matcha and GTrans baselines (graph-native methods) are implemented in
`../baselines_official.py` following their published graph algorithms and are
cited there; they are not vendored verbatim because their official code is
tightly coupled to dataset-specific PyG training scaffolding.

* Matcha — Wang et al., *Matcha: Mitigating Graph Structure Shifts with
  Test-Time Adaptation* (graph-aware reliability masking + entropy minimization).
* GTrans — Jin et al., *Empowering Graph Representation Learning with Test-Time
  Graph Transformation*, ICLR 2023 (https://github.com/ChandlerBang/GTrans).

---

## MIT License (applies to `tent.py` and `eata.py`)

```
MIT License

Copyright (c) 2021 Dequan Wang and Evan Shelhamer   (tent.py)
Copyright (c) 2023 Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen,
                   Shijian Zheng, Peilin Zhao, Mingkui Tan   (eata.py)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
