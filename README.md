# Good-Enough Data Augmentation

A simple rule-based data augmentation scheme aimed at encouraging generalization in sequence-to-sequence models.

Jacob Andreas, ACL 2020. https://arxiv.org/abs/1904.09545

**Data**:

- Semantic parsing dataset (Finnegan-Dolak et al.): 
  https://github.com/jkkummerfeld/text2sql-data

- SCAN dataset (Lake and Baroni): 
  https://github.com/brendenlake/SCAN

- Language modeling: Under `data/lm`. All data is from Wikipedia, except for the
  Na data, which is derived from "Cross-Lingual Word Embeddings for Low-Resource
  Language Modeling" (Adams et al. 2017). Note different train/test splits for
  Na.

- Human sequence-to-sequence learning (Lake, Linzen and Baroni):
  Fig 2 in https://arxiv.org/pdf/1901.04587.pdf

- COGS dataset (Kim and Linzen): 
  https://github.com/najoungkim/COGS

**To use on a new dataset**:

1. Point `torchdec` at https://github.com/jacobandreas/torchdec.
2. Create a new data loader under `data` (look at `data/colors.py` for a minimal
   example).
3. Update `get_dataset` in `train.py` to use the new loader.
4. Run the experiment pipeline (look at `exp/scan_jump/retrieval/run.sh` for an
   example). 

The `wug_size` and `wug_count` flags (defined in `data/builder.py`) determine
the number and size of the fragments that will be extracted from each template.
the `template_sim` flag determines whether the whole string or a fixed-size
window will be used for evaluating template similarity; `sim_window_size`
determines the window size. The number and diversity of generated templates can
be further controlled using the `variants` and `n_sample` flags.
