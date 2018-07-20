# Noise Contrastive Priors

This project contains the source code for Noise Contrastive Priors (NCP) by
Danijar Hafner, Dustin Tran, Alex Irpan, Timothy Lillicrap, James Davidson.

## Running the code

Active learning on the toy data set:

```sh
python3 -m ncp.scripts.toy_active --seeds 5 --logdir /path/to/logdir
```

Active learning on the flights delay data set:

```sh
python3 -m ncp.scripts.flights_active \
  --seeds 5 \
  --schedule schedule_10 \
  --dataset /path/to/dataset \
  --logdir /path/to/logdir
```

The data set path should be a file prefix for four Numpy files named
`<prefix>-train-inputs.npy`, `<prefix>-train-targets.npy`,
`<prefix>-test-inputs.npy`, `<prefix>-test-targets.npy`.

The schedule `schedule_10` selects batches of 10 labels at once. An alternative
`schedule_1` is available to select only one label at a time.
