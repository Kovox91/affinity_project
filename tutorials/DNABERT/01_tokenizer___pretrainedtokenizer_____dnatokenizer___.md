# Chapter 1: Tokenizer (`PreTrainedTokenizer` & `DNATokenizer`)

Welcome to the DNABERT tutorial! In this first chapter, we'll explore a fundamental concept in making sense of DNA sequences with machine learning models: **Tokenization**.

Imagine you have a long scroll of DNA, like "AGCTAGCTAGCT..." and you want a computer program (our DNABERT model) to read and understand it. Computers don't understand "A", "G", "C", "T" directly in the way humans or biological systems do. They understand numbers! This is where tokenizers come in.

**What problem does a tokenizer solve?**
A tokenizer is like a translator and a sample preparer. It takes raw text (or in our case, DNA sequences) and converts it into a format that machine learning models can process, typically sequences of numbers called "IDs". For DNABERT, this involves breaking down the DNA into smaller, manageable, and meaningful chunks called "k-mers".

Think of a biologist preparing a DNA sample for a sequencing machine. The biologist doesn't just feed the entire chromosome into the machine. They first cut the DNA into smaller, readable fragments. Our `DNATokenizer` does something similar for our DNABERT model (the "sequencing machine" in this analogy).

## Key Concepts: `PreTrainedTokenizer` and `DNATokenizer`

### 1. `PreTrainedTokenizer`: The Foundation

`PreTrainedTokenizer` is a powerful base class provided by the Hugging Face Transformers library. It's a general-purpose tool that knows how to:
*   Load a "vocabulary" – a list of all unique pieces (tokens) it knows.
*   Convert text tokens into numerical IDs and vice-versa.
*   Add special tokens that models often need (like a "start" or "end" marker).

Many different tokenizers for various types of text (English, Chinese, code, etc.) are built on top of `PreTrainedTokenizer`.

### 2. `DNATokenizer`: Specialized for DNA

`DNATokenizer` is our special version, designed specifically for DNA sequences. It inherits all the good stuff from `PreTrainedTokenizer` but is tailored for DNA.
Its main job is to work with **k-mers**.

**What are k-mers?**
A "k-mer" is a short DNA sequence of a specific length "k".
For example, if k=3 (we call these 3-mers):
*   The DNA sequence "AGCT" would be broken into:
    *   "AGC" (starting at the 1st base)
    *   "GCT" (starting at the 2nd base)

These k-mers are overlapping. This way, the model can see patterns across small, connected regions of the DNA. DNABERT models are typically pre-trained using 3-mers, 4-mers, 5-mers, or 6-mers. The specific `DNATokenizer` you use will correspond to the k-mer size the DNABERT model was trained on.

## How to Use `DNATokenizer`

Let's walk through a simple example. Suppose we have the DNA sequence "GATTACA" and we want to prepare it for a DNABERT model that understands 3-mers.

### Step 1: Prepare the K-mer Sentence

The `DNATokenizer` expects our DNA to first be converted into a string of space-separated k-mers. We can write a simple helper function for this, similar to what's used in DNABERT's data processing scripts (like `seq2kmer` in `motif/motif_utils.py` or `get_kmer_sentence` in `examples/data_process_template/process_pretrain_data.py`).

```python
# A helper function to convert a DNA sequence to a string of k-mers
def dna_to_kmer_sentence(sequence, k):
    kmers = []
    for i in range(len(sequence) - k + 1): # Slide a window of size k
        kmers.append(sequence[i:i+k])
    return " ".join(kmers)

dna_sequence = "GATTACA"
k_value = 3 # We're using 3-mers
kmer_sentence = dna_to_kmer_sentence(dna_sequence, k_value)

print(f"Original DNA: {dna_sequence}")
print(f"K-mer sentence (k={k_value}): {kmer_sentence}")
```
**Output:**
```
Original DNA: GATTACA
K-mer sentence (k=3): GAT ATT TTA TAC ACA
```
This `kmer_sentence` is what we'll feed to the tokenizer.

### Step 2: Load the `DNATokenizer`

We need to load a `DNATokenizer` that matches our k-mer size. DNABERT provides pre-trained tokenizers. For 3-mers, the identifier might be "dna3". (You can find these in `src/transformers/tokenization_dna.py` under `PRETRAINED_VOCAB_FILES_MAP`).

```python
from transformers import DNATokenizer

# Load a tokenizer pre-configured for 3-mers.
# This will download the necessary vocabulary file if you don't have it.
# For DNABERT, you might use a path like "zhihan1996/DNABERT-2-117M"
# or a local path if you have downloaded the model files.
# For simplicity, let's assume "dna3" is a shortcut recognized by the library
# or a local path to the tokenizer files (vocab.txt).
# A common way is: tokenizer = DNATokenizer.from_pretrained('path/to/your/3-mer/model_or_vocab/')
# For this example, imagine 'dna3' points to the correct 3-mer vocabulary.
# Note: In a real scenario, you'd use the specific path to your downloaded model.
# For instance, if you downloaded "DNABERT-3", it would be:
# tokenizer = DNATokenizer.from_pretrained("./DNABERT-3_model_path/")
# For the purpose of this tutorial, we will use a conceptual name.
# Actual usage: tokenizer = DNATokenizer.from_pretrained("zhihan1996/DNABERT-BF100k-3mer")
# For now, let's assume `DNATokenizer.from_pretrained("dna3")` works by pointing to a config.
# The `tokenization_dna.py` file has PRETRAINED_VOCAB_FILES_MAP like:
# "dna3": "https://.../bert-config-3/vocab.txt"
# So, from_pretrained("dna3") would fetch this vocab for 3-mers.

tokenizer = DNATokenizer.from_pretrained("dna3")
```
This line initializes our tokenizer, loading its vocabulary (a list of all known 3-mers and special symbols).

### Step 3: Tokenize and Encode

Now, let's use the tokenizer to convert our `kmer_sentence` into numerical IDs. The `encode` method is perfect for this.

```python
# Our k-mer sentence from Step 1
kmer_sentence = "GAT ATT TTA TAC ACA"

# Tokenize the k-mer sentence and convert to input IDs
# add_special_tokens=True will add [CLS] at the beginning and [SEP] at the end.
input_ids = tokenizer.encode(kmer_sentence, add_special_tokens=True)

print(f"K-mer sentence: {kmer_sentence}")
print(f"Input IDs: {input_ids}")
```
**Example Output (IDs will vary based on the actual `vocab.txt`):**
```
K-mer sentence: GAT ATT TTA TAC ACA
Input IDs: [2, 6, 11, 20, 22, 14, 3]
```
*   `[CLS]` (ID `2`) is a special token often used to mark the beginning of a sequence.
*   `GAT`, `ATT`, etc., are converted to their respective IDs from the vocabulary.
*   `[SEP]` (ID `3`) is a special token often used to mark the end of a sequence or separate two sequences.

These `input_ids` are what the DNABERT model will actually consume!

### Step 4: Decode (Optional - for understanding)

We can also convert these IDs back to tokens or the original (k-mer) string to see what they represent.

```python
# Our input_ids from the previous step
# input_ids = [2, 6, 11, 20, 22, 14, 3] # Example IDs

decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(f"Decoded tokens: {decoded_tokens}")

decoded_string = tokenizer.decode(input_ids)
print(f"Decoded string: {decoded_string}")
```
**Example Output:**
```
Decoded tokens: ['[CLS]', 'GAT', 'ATT', 'TTA', 'TAC', 'ACA', '[SEP]']
Decoded string: [CLS] GAT ATT TTA TAC ACA [SEP]
```

## Under the Hood: How `DNATokenizer` Works

Let's peek behind the curtain.

1.  **Vocabulary (`vocab.txt`)**:
    When you load a `DNATokenizer` (e.g., `DNATokenizer.from_pretrained("dna3")`), it loads a vocabulary file. This file is typically named `vocab.txt`. For a 3-mer tokenizer, it looks something like this (a snippet from `src/transformers/dnabert-config/bert-config-3/vocab.txt`):

    ```text
    [PAD]
    [UNK]
    [CLS]
    [SEP]
    [MASK]
    AAA
    AAT
    AAC
    AAG
    ATA
    ...
    GGG
    ```
    Each line is a token (a k-mer or a special symbol like `[CLS]`), and its line number (0-indexed) becomes its ID.
    *   `[PAD]`: Used for padding sequences to the same length.
    *   `[UNK]`: Represents an "unknown" token not found in the vocabulary.
    *   `[CLS]`: Classification token, often put at the start.
    *   `[SEP]`: Separator token.
    *   `[MASK]`: Used in masked language modeling (a training technique).
    The `DNATokenizer` stores this mapping. When it sees "GAT", it looks it up and finds its ID.

2.  **The `encode` Process**:
    Let's trace what happens when you call `tokenizer.encode("GAT ATT TTA TAC ACA", add_special_tokens=True)`:

    ```mermaid
    sequenceDiagram
        participant User
        participant Encoder as "tokenizer.encode()"
        participant TokenizerMethod as "tokenizer._tokenize()"
        participant BasicTok as "BasicTokenizer"
        participant Vocab as "Vocabulary (vocab.txt)"

        User->>Encoder: "GAT ATT TTA TAC ACA", add_special_tokens=True
        Encoder->>TokenizerMethod: "GAT ATT TTA TAC ACA"
        TokenizerMethod->>BasicTok: tokenize("GAT ATT TTA TAC ACA")
        BasicTok-->>TokenizerMethod: Returns ["GAT", "ATT", "TTA", "TAC", "ACA"] (splits by space)
        TokenizerMethod-->>Encoder: List of k-mer tokens: ["GAT", "ATT", "TTA", "TAC", "ACA"]
        Note right of Encoder: If add_special_tokens=True
        Encoder->>Encoder: Prepend [CLS], Append [SEP] tokens <br/> Result: ["[CLS]", "GAT", ..., "ACA", "[SEP]"]
        loop For each token in list
            Encoder->>Vocab: Lookup token (e.g., "GAT")
            Vocab-->>Encoder: Return ID (e.g., 6)
        end
        Encoder-->>User: Numerical IDs (e.g., [2, 6, ..., 14, 3])
    ```

    *   **Input**: The k-mer sentence string: `"GAT ATT TTA TAC ACA"`.
    *   **Internal Tokenization (`_tokenize`)**: The `DNATokenizer`'s `_tokenize` method (from `src/transformers/tokenization_dna.py`) uses a `BasicTokenizer`. This `BasicTokenizer` simply splits the input string by whitespace. So, `"GAT ATT TTA TAC ACA"` becomes the list of strings: `["GAT", "ATT", "TTA", "TAC", "ACA"]`.
    *   **Adding Special Tokens**: If `add_special_tokens=True`, the `encode` method (from the base `PreTrainedTokenizer` class in `src/transformers/tokenization_utils.py`) adds `[CLS]` at the beginning and `[SEP]` at the end. The list becomes: `["[CLS]", "GAT", "ATT", "TTA", "TAC", "ACA", "[SEP]"]`.
    *   **Conversion to IDs**: Each token in this list is looked up in the loaded vocabulary. For instance, `[CLS]` maps to ID `2`, `GAT` maps to ID `6`, and so on.
    *   **Output**: The final list of numerical IDs, ready for the model.

3.  **Knowing the K-mer Size**:
    How does `DNATokenizer` know it's for 3-mers or 6-mers? It infers this from the size of the loaded vocabulary! The `src/transformers/tokenization_dna.py` file contains a dictionary `VOCAB_KMER`:
    ```python
    VOCAB_KMER = {
        "69": "3",  # A vocab with 69 entries (4^3 k-mers + 5 special tokens) is for 3-mers
        "261": "4", # 4^4 k-mers + 5 special tokens
        "1029": "5",# 4^5 k-mers + 5 special tokens
        "4101": "6",# 4^6 k-mers + 5 special tokens
    }
    ```
    When `DNATokenizer` loads a `vocab.txt`, it counts the number of lines (tokens) and uses this dictionary to set its `self.kmer` attribute.

## `PreTrainedTokenizer` vs `DNATokenizer` Summary

*   **`PreTrainedTokenizer`**:
    *   The general-purpose blueprint.
    *   Handles loading vocabularies, converting tokens to/from IDs, adding special tokens.
    *   Doesn't know anything specific about DNA or k-mers by itself.
*   **`DNATokenizer`**:
    *   A specialized version for DNA, built upon `PreTrainedTokenizer`.
    *   Its vocabulary (`vocab.txt`) specifically contains DNA k-mers (e.g., 3-mers, 6-mers) and a few special tokens.
    *   It expects the input DNA sequence to be pre-processed into a string of space-separated k-mers.
    *   It knows what 'k' (k-mer length) it's designed for based on its vocabulary size.

The critical first step of converting a raw DNA sequence like "GATTACA" into a k-mer sentence like "GAT ATT TTA TAC ACA" is often handled by preprocessing scripts. You can see an example of this logic in `get_kmer_sentence` within the `examples/data_process_template/process_pretrain_data.py` file, or `seq2kmer` in `motif/motif_utils.py`.

```python
# Simplified from examples/data_process_template/process_pretrain_data.py
# This shows how k-mer sentences are generated.
def get_kmer_sentence_from_raw_dna(original_string, kmer_length=3, stride=1):
    sentence = []
    for i in range(0, len(original_string) - kmer_length + 1, stride):
        sentence.append(original_string[i : i + kmer_length])
    return " ".join(sentence)

raw_dna = "GATTACA"
kmer_sent = get_kmer_sentence_from_raw_dna(raw_dna, kmer_length=3)
# kmer_sent would be "GAT ATT TTA TAC ACA"
```
This `kmer_sent` is then passed to `DNATokenizer`.

## Conclusion

Phew! We've covered a lot about tokenizers. You now understand that:
*   Tokenizers are essential for converting DNA sequences into numbers that models like DNABERT can understand.
*   `PreTrainedTokenizer` is the general base, and `DNATokenizer` is specialized for DNA, working with k-mers.
*   The process involves preparing a k-mer sentence from raw DNA, then using the tokenizer to convert these k-mers into numerical IDs, often adding special tokens like `[CLS]` and `[SEP]`.

By preparing our DNA data this way, we're setting the stage for the DNABERT model to perform its magic. Speaking of the model, in the next chapter, we'll dive into the DNABERT model itself!

Next up: [Pretrained Model (`PreTrainedModel` / `TFPreTrainedModel`)](02_pretrained_model___pretrainedmodel_____tfpretrainedmodel___.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)