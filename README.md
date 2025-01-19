# TerjamaBench
Code used to run evaluation on [TerjamaBench](https://huggingface.co/datasets/atlasia/TerjamaBench).

__WARNING__: the dataset is gated, visit https://huggingface.co/datasets/atlasia/TerjamaBench to request access.

## Usage
```python
# Get the HF token from your profile
token = 'your_huggingface_token'

# Load the benchmark dataset
df = load_benchmark(token)

# Evaluate
references = df['Darija'].tolist()  # Target translations
predictions =   # Your model predictions

# Evaluate individual samples
scores = evaluate(references, predictions)

# Get average scores
avg_scores = evaluate_model(references, predictions)
print(f"Average scores: {avg_scores}")
```
