Detecting AI-Generated Text Using Probability Landscape Analysis (Ongoing Research) 
Investigating a robust approach for distinguishing AI-generated content from human-written text. 
Research Problem: Current AI text detectors often fail against modern language models and paraphrased outputs. 
Core Insight: AI-generated passages tend to lie near a peak in the model's probability landscape. Small semantic
preserving rewrites consistently reduce model probability, whereas genuine human-written text demonstrates greater 
stability. 
Methodology: - Generate semantic-preserving rewrites using T5. - Compare original and rewritten log probabilities. - 
Detect AI-generated content by thresholding probability differences. 
Future Work: - Evaluate on recent LLMs. - Ensemble multiple scoring models. - Chunk-based evaluation for long 
documents. - Study robustness against prompt engineering and adversarial rewriting.
