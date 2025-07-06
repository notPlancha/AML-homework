import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Sample corpus
corpus = "the quick brown fox jumps over the lazy dog".split()
vocab = list(set(corpus))
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

# Hyperparameters
embedding_dim = 10
context_window = 2


# Generate Skip-gram data
def generate_skipgram_data(corpus, context_window):
  data = []
  for i in range(context_window, len(corpus) - context_window):
    target = corpus[i]
    context = corpus[i - context_window : i] + corpus[i + 1 : i + context_window + 1]
    for ctx in context:
      data.append((target, ctx))
  return data


# Generate CBOW data
def generate_cbow_data(corpus, context_window):
  data = []
  for i in range(context_window, len(corpus) - context_window):
    context = corpus[i - context_window : i] + corpus[i + 1 : i + context_window + 1]
    target = corpus[i]
    data.append((context, target))
  return data


skipgram_data = generate_skipgram_data(corpus, context_window)
cbow_data = generate_cbow_data(corpus, context_window)


# Skip-gram model
class SkipGram(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(SkipGram, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.output = nn.Linear(embedding_dim, vocab_size)

  def forward(self, target):
    embed = self.embeddings(target)
    out = self.output(embed)
    return out


# CBOW model
class CBOW(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(CBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.output = nn.Linear(embedding_dim, vocab_size)

  def forward(self, context):
    embeds = self.embeddings(context)
    embeds = torch.mean(embeds, dim=0).view(1, -1)
    out = self.output(embeds)
    return out


# Training function
def train_model(model, data, is_skipgram=True, epochs=100):
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  for epoch in range(epochs):
    total_loss = 0
    for input_data, target in data:
      if is_skipgram:
        input_tensor = torch.tensor([word_to_ix[input_data]], dtype=torch.long)
        target_tensor = torch.tensor([word_to_ix[target]], dtype=torch.long)
      else:
        input_tensor = torch.tensor(
          [word_to_ix[w] for w in input_data], dtype=torch.long
        )
        target_tensor = torch.tensor([word_to_ix[target]], dtype=torch.long)

      optimizer.zero_grad()
      output = model(input_tensor)
      loss = loss_fn(output, target_tensor)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
      print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# Train both models
skipgram_model = SkipGram(vocab_size, embedding_dim)
cbow_model = CBOW(vocab_size, embedding_dim)

print("Training Skip-gram model:")
train_model(skipgram_model, skipgram_data, is_skipgram=True)

print("\nTraining CBOW model:")
train_model(cbow_model, cbow_data, is_skipgram=False)


# Compare embeddings
def compare_embeddings(model1, model2):
  emb1 = model1.embeddings.weight.data.numpy()
  emb2 = model2.embeddings.weight.data.numpy()
  print("\nCosine similarity between Skip-gram and CBOW embeddings:")
  for i, word in enumerate(vocab):
    sim = cosine_similarity([emb1[word_to_ix[word]]], [emb2[word_to_ix[word]]])[0][0]
    print(f"{word}: {sim:.4f}")


compare_embeddings(skipgram_model, cbow_model)

"""
Cosine similarity between Skip-gram and CBOW embeddings:
over: -0.0941
brown: -0.5166
jumps: 0.2075
quick: 0.3642
lazy: -0.7710
dog: -0.2590
fox: 0.3861
the: -0.1343

- The cosine similarities range from -0.7710 to 0.3861, indicating varying degrees of alignment
- Most similarities are relatively low, suggesting Skip-gram and CBOW learn different representations
- Positive similarities (jumps: 0.2075, quick: 0.3642, fox: 0.3861) show some agreement between models
- Negative similarities (especially lazy: -0.7710, brown: -0.5166) indicate opposing vector directions
- The word "the" shows weak negative similarity (-0.1343), which is interesting for such a common word
- Overall, the low similarity scores demonstrate that Skip-gram and CBOW capture different aspects
  of word relationships, with Skip-gram focusing on predicting context from target words and 
  CBOW predicting target words from context, leading to complementary but distinct embeddings
"""
