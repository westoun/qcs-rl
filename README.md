# Reinforcement Learning for State Separation

This repository contains the source code to the paper
_"Learning State Separation for Quantum Circuit Synthesis"_
by Stein, Klikovits, and Wimmer from the [Institute of Business Informatics - Software Engineering](https://se.jku.at/) at the [Johannes Kepler University](https://www.jku.at/en), Linz.

## Experiments and Evaluation

The entrypoint for the experiments we reported in the paper lies
in the `main.py` file. If you wish to run your own experiments,
make sure you call the `run_experiment()`-function with the
experiment configuration you wish to evaluate.

The most important experiment parameters are

- `qubit_num`: number of qubits of each state.
- `gate_count`: number of randomly sampled gates used to create the
  states that the agent has to separate.
- `max_steps`: maximum number of gates the agent can propose before
  an episode is considered a failure.
- `total_timesteps`: amount of steps for which the agent is trained.

Once you have implemented the experiment configurations you wish to run,
execute

```
docker compose up
```

to build and run the corresponding docker container.
The results of each experiment run are saved in the
`logs/`-directory.

To generate figures similar to the ones reported in the paper,
open and run the `evaluation.ipynb` notebook. Make sure you
specify the correct log and target directories at the beginning
of the file.

Have fun!

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
