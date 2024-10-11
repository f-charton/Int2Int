# Int2Int

## Integer sequence to integer sequence translator

This is complete code for a sequence to sequence transformer, the goal is to train it (supervisedly) to translate short sequences of integers into short sequences of integers.

### My first experiment

One way to run it out of the box is to run (on an environment where you have pytorch and numpy installed):

```
python train.py --dump_path /some_path_on_your_computer/ --exp_name my_first_experiment --exp_id 1 --operation "gcd"
```

This should train a transformer to compute the GCD of integers, encoded in base 1000, generated on the fly (between 1 and a million).

You can try other operations with the parameter --operation : modular_add, modular_mul, fraction_compare, fraction_add, fraction_simplify, matrix_rank...

Modular operations use modulo 67 by default, you can change it with --modulo

Integers are encoded in base 1000 by default, you can change it with --base

These problem specific parameters are to be found in src/envs/arithmetic.py

If you don't have a NVIDIA GPU, you can train on CPU by stating --cpu true (it will be slow, you may want to set --epoch_size to something small, like 10000...)


### Running my first experiment

If everything goes fine, the log should appear on the screen. It will also be saved in

> /some_path_on_your_computer/my_first_experiment/1/train.log

(if you don't provide an exp_id, the program will create one for you)

After indicating the hyperparameters you use, and the model you train (and its number of parameters), the logs will describe the training, which is a series of **training epochs** (300,000 examples by default, can be changed via --epoch_size), separated by evaluations over a test set (generated on the fly in the case of the GCD). During epochs, the model will report the training loss (this should go down, but it can be bumpy, if it bumps too much your learning rate is probably too large).

```
INFO - 09/18/24 20:41:05 - 0:00:16 -     400 -  977.02 equations/s -  9768.07 words/s - ARITHMETIC:  0.5479 - LR: 1.0000e-04
INFO - 09/18/24 20:41:11 - 0:00:23 -     600 -  940.88 equations/s -  9408.46 words/s - ARITHMETIC:  0.4181 - LR: 1.0000e-04
INFO - 09/18/24 20:41:18 - 0:00:30 -     800 -  955.94 equations/s -  9558.79 words/s - ARITHMETIC:  0.3715 - LR: 1.0000e-04
```

Training losses are logged every 200 optimization steps (this is configurable, in trainer.py), here, you see it going down 0.55 to 0.42 to 0.37. Life is good!
The eq/s and words/s give an idea of the learning speed, eqs are examples: here with 950 examples / s you expect to complete a 300k example epoch in a little more than 6 minutes (we are on a GPU).

At the end of each epoch, the model runs test on a sample of size --eval_size (10k by default, you can make this smaller), examples are evaluated in batches of --batch_size_eval. During evaluation, the lines

```
INFO - 09/18/24 20:57:28 - 0:16:39 - (7168/10000) Found 102/128 valid top-1 predictions. Generating solutions ...
INFO - 09/18/24 20:57:28 - 0:16:39 -     Found 102/128 solutions in beam hypotheses.
```

indicate how many solutions were correct in each eval batch (here 102 our of 128). At the end, you should have a small report saying :

```
INFO - 09/18/24 20:57:29 - 0:16:40 - 8459/10000 (84.59%) equations were evaluated correctly.
INFO - 09/18/24 20:57:29 - 0:16:40 - 1: 6104 / 6107 (99.95%)
INFO - 09/18/24 20:57:29 - 0:16:40 - 2: 1581 / 1581 (100.00%)
INFO - 09/18/24 20:57:29 - 0:16:40 - 4: 356 / 356 (100.00%)
INFO - 09/18/24 20:57:29 - 0:16:40 - 5: 241 / 245 (98.37%)
INFO - 09/18/24 20:57:29 - 0:16:40 - 8: 86 / 86 (100.00%)
INFO - 09/18/24 20:57:29 - 0:16:40 - 10: 56 / 57 (98.25%)
INFO - 09/18/24 20:57:29 - 0:16:40 - 20: 22 / 22 (100.00%)
INFO - 09/18/24 20:57:29 - 0:16:40 - 25: 8 / 8 (100.00%)
INFO - 09/18/24 20:57:29 - 0:16:40 - 40: 2 / 2 (100.00%)
INFO - 09/18/24 20:57:29 - 0:16:40 - 50: 3 / 3 (100.00%)
```

84.6% of the test GCD were correctly calculated. Correct model predictions were GCD 1, 2, 4, 5, 8 .... products of powers of divisors of the base...

At the end of the peoch, the model exports a python dictionary containing detailed results. This is what you want to load (in a notebook) to draw learning curves, etc.

### Training from a data file

Training and test files can be provided with the parameters: `--train_data` and `--eval_data` (setting `--eval_size` to `-1` will cause the model to evaluate on all the eval data).

Training and test examples are written, one per line, as sequence of tokens, separated by whitespaces, the input and output being separated by a tab.


One specify the data type of the input and output, e.g.:  `--operation "data" --data_types '"int[5]:int"`

The supported data types at the moment are:
- `int` -- an integer

   encoded as `p ad ... a0` where `p` in `{+, -}` and `ai` are the digits of `a` in base `1000` (by default), e.g., `-3500` is represented as `- 3 500`


- `int[n]` -- an integer array of length

  represented as `Vn z1 ... zn` where `zi` are encoded as above

- `range(a, b)` -- an integer in the range `{a,...,b-1}`

encoded as a string in base 10, e.g., via `1101`.

For example, here are some python functions to encode the above data types, respectively:

```python3
def encode_integer(val, base=1000, digit_sep=" "):
    if val == 0:
        return '+ 0'
    sgn = '+' if val >= 0 else '-'
    val = abs(val)
    r = []
    while val > 0:
        r.append(str(val % base))
        val = val//base
    r.append(sgn)
    r.reverse()
    return digit_sep.join(r)

def encode_integer_array(x, base=1000):
    return f'V{len(x)} ' + " ".join(encode_integer(int(z), base) for z in x)

def encode_range(x):
    return str(int(x))
```

For example, for `GCD` we would use `int[2]:int` where

```
V2 + 1 24 + 16\t+ 16\n
```

represents `GCD (1024, 16) = 16`, in base `1000`. Note that here `V2`, `1`, `24`, `16` are words/tokens.

For an elliptic curve and if it has nontrivial rank, I would have something like `int[5]:range(2)`

```
V5 + 0 - 1 + 0 - 84 375 258 - 298 283 918 238\t1
```

The code is organised as follows:

train.py is the main file, you run python train.py with some parameters, you can train from generated data (using envs/generators), generate and export data (setting --export_data to true), or train and test from external data (using train_data and test_data)

src/envs contain the math-specific code

