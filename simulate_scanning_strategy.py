#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
# Things to fix:
#
# sampling_rate_hz should be a field common to all detectors
# change "net_ukhz" into "NET_ukhz" in the Detector class

from dataclasses import dataclass
import warnings
import json
import logging as log
import sys
from typing import Dict, Any, List

import numpy as np
from tqdm import tqdm
import litebird_sim as lbs
from astropy.utils.exceptions import ErfaWarning
import healpy
from astropy.coordinates import (
    ICRS,
    get_body_barycentric,
    BarycentricMeanEcliptic,
    solar_system_ephemeris,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt


@dataclass
class Parameters:
    spin_boresight_angle_rad: float
    planet_name: str
    spin2ecl_delta_time_s: float
    detector_sampling_rate_hz: float
    radii_deg: List[float]
    output_nside: int = 1024
    output_map_file_name: str = "map.fits.gz"
    output_table_file_name: str = "observation_time_table.txt"


def load_parameters(sim: lbs.Simulation) -> Parameters:
    planet_params = sim.parameters["planet_scanning"]

    return Parameters(
        spin_boresight_angle_rad=sim.parameters["scanning_strategy"][
            "spin_boresight_angle_rad"
        ],
        planet_name=planet_params["planet_name"],
        spin2ecl_delta_time_s=planet_params["spin2ecl_delta_time_s"],
        detector_sampling_rate_hz=planet_params["sampling_rate_hz"],
        radii_deg=planet_params.get(
            "radii_deg", [0.1, 0.2, 0.5, 1, 5, 10, 20, 40, 60, 90, 135, 180]
        ),
        output_nside=planet_params["output_nside"],
        output_map_file_name=sim.base_path / "map.fits.gz",
        output_table_file_name=sim.base_path / "observation_time_table.txt",
    )


def time_per_radius(time_map_s, angular_radius_rad):
    # Given a map that associates the time spent observing with each
    # pixel in the reference frame of the detector's main beam, and
    # assuming that the main beam is aligned with the North Pole,
    # compute how much time was spent observing within an angular
    # radius equal to "ANGULAR_RADIUS_RAD".

    npix = len(time_map_s)
    nside = healpy.npix2nside(npix)
    theta, _ = healpy.pix2ang(nside, np.arange(npix))
    mask = theta <= angular_radius_rad
    return np.sum(time_map_s[mask])


def read_scanning_strategy(parameters: Dict[str, Any], imo: lbs.Imo, start_time):

    if "scanning_strategy_obj" in parameters:
        sstr = lbs.SpinningScanningStrategy.from_imo(
            imo, parameters["scanning_strategy_obj"]
        )
    else:
        sstr = lbs.SpinningScanningStrategy(
            spin_sun_angle_rad=0.0,
            precession_rate_hz=0.0,
            spin_rate_hz=1.0 / 60,
            start_time=start_time,
        )

    if "spin_sun_angle_rad" in parameters:
        sstr.spin_sun_angle_rad = np.deg2rad(parameters["spin_sun_angle_rad"])

    if "precession_period_min" in parameters:
        sstr.precession_rate_hz = 1.0 / (60.0 * parameters["precession_period_min"])

    if "spin_rate_rpm" in parameters:
        sstr.spin_rate_hz = parameters["spin_rate_rpm"] / 60.0

    return sstr


def main():
    warnings.filterwarnings("ignore", category=ErfaWarning)

    sim = lbs.Simulation(
        parameter_file=sys.argv[1],
        name="Observation of planets",
        description="""
This report contains the result of a simulation of the observation
of the sky, particularly with respect to the observation of planets.
""",
    )

    params = load_parameters(sim)

    if lbs.MPI_ENABLED:
        log.info("Using MPI with %d processes", lbs.MPI_COMM_WORLD.size)
    else:
        log.info("Not using MPI, serial execution")

    log.info("Generating the quaternions")
    scanning_strategy = read_scanning_strategy(
        sim.parameters["scanning_strategy"], sim.imo, sim.start_time
    )
    sim.generate_spin2ecl_quaternions(
        scanning_strategy=scanning_strategy,
        delta_time_s=params.spin2ecl_delta_time_s,
    )

    log.info("Creating the observations")
    instr = lbs.Instrument(
        name="instrum", spin_boresight_angle_rad=params.spin_boresight_angle_rad
    )
    detector = lbs.DetectorInfo(sampling_rate_hz=params.detector_sampling_rate_hz)
    sim.create_observations(
        detectors=[detector],
        num_of_obs_per_detector=sim.parameters["simulation"]["num_of_obs_per_detector"],
    )

    #################################################################
    # Here begins the juicy part

    log.info("The loop starts on %d processes", lbs.MPI_COMM_WORLD.size)
    sky_hitmap = np.zeros(healpy.nside2npix(params.output_nside), dtype=np.int32)
    detector_hitmap = np.zeros(healpy.nside2npix(params.output_nside), dtype=np.int32)
    dist_map_m2 = np.zeros(len(detector_hitmap))

    iterator = tqdm
    if lbs.MPI_ENABLED and lbs.MPI_COMM_WORLD.rank != 0:
        iterator = lambda x: x

    for obs in iterator(sim.observations):
        solar_system_ephemeris.set("builtin")

        times = obs.get_times(astropy_times=True)

        # We only compute the planet's position for the first sample in
        # the observation and then assume that it does not move
        # significantly. (In Ecliptic coordinates, Jupiter moves by
        # fractions of an arcmin over a time span of one hour)
        time0 = times[0]
        icrs_pos = get_body_barycentric(params.planet_name, time0)
        earth_pos = get_body_barycentric("earth", time0)

        # Compute the distance between the Earth and the planet
        distance_m = (earth_pos - icrs_pos).norm().to("m").value

        # Convert the ICRS r.f. into the barycentric mean Ecliptic r.f.,
        # which is the reference frame used by the LiteBIRD simulation
        # framework
        ecl_vec = (
            ICRS(icrs_pos)
            .transform_to(BarycentricMeanEcliptic)
            .cartesian.get_xyz()
            .value
        )

        # The variable ecl_vec is a 3-element vector. We normalize it so
        # that it has length one (using the L_2 norm, hence ord=2)
        ecl_vec /= np.linalg.norm(ecl_vec, axis=0, ord=2)

        # Convert the matrix to a N×3 shape by repeating the vector:
        # planets move slowly, so we assume that Jupiter stays fixed
        # during this observation.
        ecl_vec = np.repeat(ecl_vec.reshape(1, 3), len(times), axis=0)

        # Calculate the quaternions that convert the Ecliptic
        # reference system into the detector's reference system
        quats = obs.get_ecl2det_quaternions(
            sim.spin2ecliptic_quats,
            detector_quats=[detector.quat],
            bore2spin_quat=instr.bore2spin_quat,
        )

        # Make room for the xyz vectors in the detector's reference frame
        det_vec = np.empty_like(ecl_vec)

        # Do the rotation!
        lbs.all_rotate_vectors(det_vec, quats[0], ecl_vec)

        pixidx = healpy.vec2pix(
            params.output_nside, det_vec[:, 0], det_vec[:, 1], det_vec[:, 2]
        )
        bincount = np.bincount(pixidx, minlength=len(detector_hitmap))
        detector_hitmap += bincount
        dist_map_m2 += bincount / ((4 * np.pi * (distance_m ** 2)) ** 2)

        pointings = obs.get_pointings(
            sim.spin2ecliptic_quats, [detector.quat], instr.bore2spin_quat
        )[0]

        pixidx = healpy.ang2pix(params.output_nside, pointings[:, 0], pointings[:, 1])
        bincount = np.bincount(pixidx, minlength=len(sky_hitmap))
        sky_hitmap += bincount

    if lbs.MPI_ENABLED:
        sky_hitmap = lbs.MPI_COMM_WORLD.allreduce(sky_hitmap)
        detector_hitmap = lbs.MPI_COMM_WORLD.allreduce(detector_hitmap)
        dist_map_m2 = lbs.MPI_COMM_WORLD.allreduce(dist_map_m2)

    time_map_s = detector_hitmap / params.detector_sampling_rate_hz
    dist_map_m2[dist_map_m2 > 0] = np.power(dist_map_m2[dist_map_m2 > 0], -0.5)

    obs_time_per_radius_s = [
        time_per_radius(time_map_s, angular_radius_rad=np.deg2rad(radius_deg))
        for radius_deg in params.radii_deg
    ]

    if lbs.MPI_COMM_WORLD.rank == 0:
        # Create a plot of the observation time of the planet as a
        # function of the angular radius
        fig, ax = plt.subplots()
        ax.loglog(params.radii_deg, obs_time_per_radius_s)
        ax.set_xlabel("Radius [deg]")
        ax.set_ylabel("Observation time [s]")

        # Create a map showing how the observation time is distributed on
        # the sphere (in the reference frame of the detector)
        healpy.orthview(time_map_s, title="Time spent observing the source")

        sim.append_to_report(
            """

## Scanning strategy parameters

Parameter | Value
--------- | --------------
Angle between the spin axis and the Sun-Earth axis | {{ sun_earth_angle_deg }} deg
Angle between the spin axis and the boresight | {{ spin_boresight_angle_deg }} deg
Precession period | {{ precession_period_min }} min
Spin period | {{ spin_period_min }} min

## Observation of {{ params.planet_name | capitalize }}

![](detector_hitmap.png)

The overall time spent in the map is {{ overall_time_s }} seconds.

The time resolution of the simulation was {{ delta_time_s }} seconds.

Angular radius [deg] | Time spent [s]
-------------------- | ------------------------
{% for row in radius_vs_time_s -%}
{{ "%.1f"|format(row[0]) }} | {{ "%.1f"|format(row[1]) }}
{% endfor -%}

![](radius_vs_time.svg)

""",
            figures=[(plt.gcf(), "detector_hitmap.png"), (fig, "radius_vs_time.svg")],
            params=params,
            overall_time_s=np.sum(detector_hitmap) / params.detector_sampling_rate_hz,
            radius_vs_time_s=list(zip(params.radii_deg, obs_time_per_radius_s)),
            delta_time_s=1.0 / params.detector_sampling_rate_hz,
            sun_earth_angle_deg=np.rad2deg(scanning_strategy.spin_sun_angle_rad),
            spin_boresight_angle_deg=np.rad2deg(params.spin_boresight_angle_deg),
            precession_period_min=1.0 / (60.0 * scanning_strategy.precession_rate_hz),
            spin_period_min=1.0 / (60.0 * scanning_strategy.spin_rate_hz),
            det=detector,
        )

        healpy.write_map(
            params.output_map_file_name,
            (detector_hitmap, time_map_s, dist_map_m2, sky_hitmap),
            coord="DETECTOR",
            column_names=["HITS", "OBSTIME", "SQDIST", "SKYHITS"],
            column_units=["", "s", "m^2", ""],
            dtype=[np.int32, np.float32, np.float64, np.int32],
            overwrite=True,
        )

        np.savetxt(
            params.output_table_file_name,
            np.array(list(zip(params.radii_deg, obs_time_per_radius_s))),
            fmt=["%.2f", "%.5e"],
        )

        with (sim.base_path / "parameters.json").open("wt") as outf:
            time_value = scanning_strategy.start_time
            if not isinstance(time_value, (int, float)):
                time_value = str(time_value)
            json.dump(
                {
                    "scanning_strategy": {
                        "spin_sun_angle_rad": scanning_strategy.spin_sun_angle_rad,
                        "precession_rate_hz": scanning_strategy.precession_rate_hz,
                        "spin_rate_hz": scanning_strategy.spin_rate_hz,
                        "start_time": time_value,
                    },
                    "detector": {"sampling_rate_hz": params.detector_sampling_rate_hz},
                    "planet": {"name": params.planet_name},
                },
                outf,
                indent=2,
            )

    sim.flush()


if __name__ == "__main__":
    main()
