# Noise Contrastive Priors

This project contains the source code for Noise Contrastive Priors (NCP) by
Danijar Hafner, Dustin Tran, Alex Irpan, Timothy Lillicrap, James Davidson.

## Running the code

Install dependencies:

```sh
pip3 install numpy tensorflow tensorflow_probability matplotlib ruamel.yaml
```

Active learning on the toy data set:

```sh
python3 -m ncp.scripts.toy_active --seeds 20 --logdir /path/to/logdir
```

Active learning on the flights delay data set:

```sh
python3 -m ncp.scripts.flights_active \
  --seeds 10 \
  --dataset /path/to/dataset \
  --logdir /path/to/logdir
```

The data set path should be a file prefix for four Numpy files named
`<dataset>-train-inputs.npy`, `<dataset>-train-targets.npy`,
`<dataset>-test-inputs.npy`, `<dataset>-test-targets.npy`.
