# Simulation of in-flight beam calibration

This repository contains the source code used to run the simulations for reconstructing the in-flight beam calibration of LiteBIRD and Planck detectors.

The code is based on the [litebird_sim](https://github.com/litebird/litebird_sim) framework and requires Python 3.10 or 3.11.

## Quick start with `uv`

This project uses [uv](https://docs.astral.sh/uv/) for Python development and dependency management.

### Installation

First, ensure you have `uv` installed. Then, clone the repository and synchronize the environment:

```sh
git clone https://github.com/ziotom78/litebird_inflight_beam_cal
cd litebird_inflight_beam_cal
uv python pin 3.11
uv sync
```

This will create a virtual environment in `.venv` and install all required dependencies (including LBS, AstroPy, and Healpy).

## Running the simulations

The simulation is divided into two steps. You should run them using `uv run SCRIPT` to ensure the correct environment is used.

You must run the code in two steps:

1. Simulate the scanning strategy (long!) using `uv run simulate_scanning_strategy.py PARAMETER_FILE1`;
2. Run the Monte Carlo simulation (quick) using `uv run symmetric_gaussian_beam_fit.py PARAMETER_FILE2`.

Both scripts need a TOML file as input. You can find several `.toml` examples in the folder.

**Important:** To run LiteBIRD-specific simulations, you must have the Instrument Model (IMo) set up. If you're a member of the LiteBIRD
collaboration, you can use the official IMo; otherwise, there is a public IMo available. See the [LBS User’s manual](https://litebird-sim.readthedocs.io/en/latest/imo.html#configuring-the-imo) for more information.

Both scripts produce a Markdown/HTML report, similar to this:

![](inflight-beam-sample-report.png)


# Development

To run linting or formatting with `ruff` (installed if you call `uv` with `--group dev`), use the command

```sh
uv run ruff check .
uv run ruff format .
```
