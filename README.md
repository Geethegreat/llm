# llm: a language model from scratch, in C

A character-level neural language model written in plain C. No PyTorch, no autograd, no BLAS. Every matrix operation, the forward pass, and the backward pass (gradients derived by hand on paper) are implemented directly in this repo.

Trained on [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), it writes things like this:

```
$ ./main generate "SICINIUS"
SICINIUS:

VOLUMNIA:
Go, counsenthy.

LARTIUS:
O, RomU: Fow, le harm af goone though aplied sile of the thit
```

Not Shakespeare, but it has learned English-looking words, speaker names, colons, line breaks and verse-style formatting from raw characters alone.

**Video:** [Building an LLM from scratch in C](https://youtu.be/iVKCOAcnMDI)

## Highlights

- **Zero dependencies** beyond the C standard library (`stdio`, `stdlib`, `math`)
- **Hand-derived backprop**, verified against numerical gradient checking (analytical and numerical gradients agree to ~4 significant figures)
- **~26,000 parameters**, roughly 250,000x smaller than the smallest Llama 2
- **Temperature sampling** for controllable generation
- **Checkpointing**: weights and vocabulary are saved together, so generation never needs the training data

## How it works

The model is a small MLP (not a transformer) that sees the previous 8 characters and predicts a probability distribution over the next one.

```
8 previous characters
        |
        v
embedding lookup + concat      X  : (1, 8 x 16)
        |
        v
X . W1 + B1  ->  ReLU          H  : (1, 128)
        |
        v
H . W2 + B2  ->  softmax       p  : (1, vocab_size)
```

### Training

Each step samples a random position in the text, takes the 8 characters before it as input and the next character as the target, then:

```
delta_out = probs - target             # softmax + cross-entropy collapse to this
dW2 = H^T . delta_out      dB2 = delta_out
dH  = delta_out . W2^T
delta_Z = dH (elementwise) relu'(Z)
dW1 = X^T . delta_Z        dB1 = delta_Z
dX  = delta_Z . W1^T       # scattered back into the rows of the embedding table
```

then every parameter is updated with plain SGD: `param -= lr * grad`. The embedding table is trained the same way as the weights; only the rows for characters that appeared in the context receive gradient (summed if a character appears more than once).

### Generation

Seed with a prompt, run the forward pass, sample the next character from the softmax distribution (scaled by temperature), slide the context window, repeat.

Temperature divides the logits before softmax: below 1.0 gives safer, more repetitive text; above 1.0 gives more chaotic text; 1.0 is the model's raw distribution.

## Quick start

```bash
git clone https://github.com/Geethegreat/llm.git
cd llm
mkdir -p data models
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

cd src
gcc main.c matrix.c activations.c dataset.c model.c generate.c train.c -o main -lm
```

Train (writes the checkpoint to `models/model_shakespeare.txt`):

```bash
./main train
```

Generate (the prompt must be at least 8 characters, the context size):

```bash
./main generate "ROMEO and"
```

Run everything from inside `src/`, since the data and model paths in `main.c` are relative (`../data`, `../models`).

## Configuration

Hyperparameters are `#define`s, so changing them is a one-line edit and a rebuild. Note that `CONTEXT_SIZE` appears in `main.c`, `train.c` and `generate.c`, so keep the copies in sync.

| Setting | File | Default | Notes |
|---|---|---|---|
| `CONTEXT_SIZE` | `main.c`, `train.c`, `generate.c` | 8 | Characters of history the model sees |
| `EMBEDDING_DIM` | `main.c` | 16 | Size of each character's vector |
| `HIDDEN_DIM` | `main.c` | 128 | Hidden layer width |
| `TRAINING_STEPS` | `main.c` | 500,000 | One example per step |
| `LEARNING_RATE` | `train.c` | 0.1 | Lower it (0.01 to 0.05) if loss goes to NaN |
| `TEMPERATURE` | `generate.c` | 1.8 | Try 0.5, 1.0, 1.5 on the same prompt |

## Results

| Config | Final loss |
|---|---|
| hidden 32, context 8 | ~1.8 to 1.9 |
| hidden 128, context 8 | ~1.4 |

For reference, a bigram model bottoms out around 2.5 and a small transformer around 1.5 on this dataset. Loss starts near `ln(vocab_size)` = `ln(65)` = 4.17, which is a handy sanity check that the forward pass and softmax are wired correctly.

Going from hidden size 32 to 128 broke through the earlier plateau, which showed capacity (not training time) was the bottleneck.

## Project structure

```
llm/
├── data/
│   └── input.txt               training text (not committed, see Quick start)
├── models/
│   └── model_shakespeare.txt   checkpoint written by `./main train`
└── src/
    ├── matrix.c / .h           allocation, matmul, transpose, elementwise ops
    ├── activations.c / .h      ReLU and softmax
    ├── dataset.c / .h          load text, build char<->id vocab, sample (context, target)
    ├── model.c / .h            parameters, forward, backward, update, save/load, sampling
    ├── train.c / .h            training loop
    ├── generate.c / .h         autoregressive generation
    └── main.c                  CLI: `train` and `generate`
```

### Checkpoint format

A raw binary file: four `int`s (`vocab_size`, `embedding_dim`, `hidden_dim`, `context_size`), then the `float` arrays `C`, `W1`, `B1`, `W2`, `B2`, then the vocabulary characters. The vocabulary travels with the weights so that id 7 always means the same character it meant during training.

## Limitations

- Fixed 8-character window: no memory beyond it, no recurrence, no attention
- One hidden layer, so output is word-shaped rather than sentence-coherent
- Character-level only (no BPE or subword tokenization)
- Trains one example at a time (no batching), so per-step loss is noisy; the training log prints a running average
- Naive triple-loop matmul: fine at this scale, not meant for anything larger

## Ideas to extend it

- Add a second hidden layer (the backward pass would need generalizing into a loop)
- Mini-batching and a learning rate schedule
- Swap SGD for Adam
- Replace the MLP with a small attention block

## Acknowledgments

Built while studying Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) and [llm.c](https://github.com/karpathy/llm.c), which are excellent references for the ideas behind this project.

## License

MIT. See [LICENSE](LICENSE) for details.
