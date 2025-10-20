# Level Matching

A machine learning-based approach for matching energy levels between nuclear datasets using LambdaRank and optimal assignment. The system trains a LightGBM ranker on physical features (energy difference, angular momentum, parity, etc.) and applies the Hungarian algorithm to find globally optimal one-to-one correspondences. Outputs ranked match candidates with confidence scores for nuclear structure analysis.
