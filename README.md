# Good-Enough Data Augmentation

A simple rule-based data augmentation scheme aimed at encouraging generalization in sequence-to-sequence models.

Jacob Andreas, ACL 2020. https://arxiv.org/abs/1904.09545

**Experiments**:

Look in the `exp` folder. Experiments labeled `retrieval` use GECA for data
augmentation.

**Data**:

- Semantic parsing dataset (Finnegan-Dolak et al.): 
  https://github.com/jkkummerfeld/text2sql-data

- SCAN dataset (Lake and Baroni): 
  https://github.com/brendenlake/SCAN

- Language modeling: Under `data/lm`. All data is from Wikipedia, except for the
  Na data, which is derived from "Cross-Lingual Word Embeddings for Low-Resource
  Language Modeling" (Adams et al. 2017). Note different train/test splits for
  Na.
