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
import astropy.units
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

EARTH_L2_DISTANCE_KM = 1_496_509.30522


@dataclass
class Parameters:
    spin_boresight_angle_rad: float
    planet_name: str
    spin2ecl_delta_time_s: float
    detector_sampling_rate_hz: float
    radii_deg: List[float]
    radius_au: List[float]
    L2_orbital_velocity_rad_s: List[float]
    phase_rad: float
    earth_L2_distance_km: float = EARTH_L2_DISTANCE_KM
    output_nside: int = 1024
    output_map_file_name: str = "map.fits.gz"
    output_table_file_name: str = "observation_time_table.txt"


def load_parameters(sim: lbs.Simulation) -> Parameters:
    planet_params = sim.parameters["planet_scanning"]
    scanning_params = sim.parameters["scanning_strategy"]

    return Parameters(
        spin_boresight_angle_rad=scanning_params["spin_boresight_angle_rad"],
        planet_name=planet_params["planet_name"],
        spin2ecl_delta_time_s=planet_params["spin2ecl_delta_time_s"],
        detector_sampling_rate_hz=planet_params["sampling_rate_hz"],
        radii_deg=planet_params.get(
            "radii_deg", [0.1, 0.2, 0.5, 1, 5, 10, 20, 40, 60, 90, 135, 180]
        ),
        earth_L2_distance_km=scanning_params.get(
            "earth_L2_distance_km", EARTH_L2_DISTANCE_KM
        ),
        output_nside=planet_params["output_nside"],
        output_map_file_name=sim.base_path / "map.fits.gz",
        output_table_file_name=sim.base_path / "observation_time_table.txt",
        radius_au=scanning_params.get("radius_au", [0.0, 0.0]),
        L2_orbital_velocity_rad_s=scanning_params.get(
            "L2_orbital_velocity_rad_s", [0.0, 0.0]
        ),
        phase_rad=scanning_params.get("phase", 0.0),
    )


def norm(vec):
    "Return the norm of a vector"
    return np.sqrt(vec.dot(vec))


def get_ecliptic_vec(vec):
    "Convert a coordinate in a XYZ vector expressed in the Ecliptic rest frame"
    return ICRS(vec).transform_to(BarycentricMeanEcliptic()).cartesian.get_xyz()


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
        sstr.spin_sun_angle_rad = parameters["spin_sun_angle_rad"]

    if "precession_period_min" in parameters:
        precession_period_min = parameters["precession_period_min"]
        if precession_period_min != 0:
            sstr.precession_rate_hz = 1.0 / (60.0 * parameters["precession_period_min"])
        else:
            sstr.precession_rate_hz = 0.0

    if "spin_rate_rpm" in parameters:
        sstr.spin_rate_hz = parameters["spin_rate_rpm"] / 60.0

    return sstr


def main():
    if len(sys.argv) != 2:
        print("Usage: {} PARAMETER_FILE".format(sys.argv[0]))
        sys.exit(1)

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
        scanning_strategy=scanning_strategy, delta_time_s=params.spin2ecl_delta_time_s
    )

    log.info("Creating the observations")
    instr = lbs.InstrumentInfo(
        name="instrum", spin_boresight_angle_rad=params.spin_boresight_angle_rad
    )
    detector = lbs.DetectorInfo(sampling_rate_hz=params.detector_sampling_rate_hz)

    conversions = [
        ("years", astropy.units.year),
        ("year", astropy.units.year),
        ("days", astropy.units.day),
        ("day", astropy.units.day),
        ("hours", astropy.units.hour),
        ("hour", astropy.units.hour),
        ("minutes", astropy.units.minute),
        ("min", astropy.units.minute),
        ("sec", astropy.units.second),
        ("s", astropy.units.second),
        ("km", astropy.units.kilometer),
        ("Km", astropy.units.kilometer),
        ("au", astropy.units.au),
        ("AU", astropy.units.au),
        ("deg", astropy.units.deg),
        ("rad", astropy.units.rad),
    ]

    def conversion(x, new_unit):
        if isinstance(x, str):
            for conv_str, conv_unit in conversions:
                if x.endswith(" " + conv_str):
                    value = float(x.replace(conv_str, ""))
                    return (value * conv_unit).to(new_unit).value
                    break
        else:
            return float(x)

    sim_params = sim.parameters["simulation"]
    durations = ["duration_s", "duration_of_obs_s"]
    for dur in durations:
        sim_params[dur] = conversion(sim_params[dur], "s")

    delta_t_s = sim_params["duration_of_obs_s"]
    sim.create_observations(
        detectors=[detector],
        num_of_obs_per_detector=int(
            sim_params["duration_s"] / sim_params["duration_of_obs_s"]
        ),
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

    # Time variable inizialized at the beginning of the simulation
    t = 0.0
    for obs in iterator(sim.observations):
        solar_system_ephemeris.set("builtin")

        times = obs.get_times(astropy_times=True)

        # We only compute the planet's position for the first sample in
        # the observation and then assume that it does not move
        # significantly. (In Ecliptic coordinates, Jupiter moves by
        # fractions of an arcmin over a time span of one hour)
        time0 = times[0]
        icrs_pos = get_ecliptic_vec(get_body_barycentric(params.planet_name, time0))
        earth_pos = get_ecliptic_vec(get_body_barycentric("earth", time0))

        # Move the spacecraft to L2
        L2_pos = earth_pos * (
            1.0 + params.earth_L2_distance_km / norm(earth_pos).to("km").value
        )
        # Creating a Lissajous orbit
        R1 = conversion(params.radius_au[0], "au")
        R2 = conversion(params.radius_au[1], "au")
        phi1_t = params.L2_orbital_velocity_rad_s[0] * t
        phi2_t = params.L2_orbital_velocity_rad_s[1] * t
        phase = conversion(params.phase_rad, "rad")
        orbit_pos = np.array(
            [
                -R1 * np.sin(np.arctan(L2_pos[1] / L2_pos[0])) * np.cos(phi1_t),
                R1 * np.cos(np.arctan(L2_pos[1] / L2_pos[0])) * np.cos(phi1_t),
                R2 * np.sin(phi2_t + phase),
            ]
        )
        orbit_pos = astropy.units.Quantity(orbit_pos, unit="AU")

        # Move the spacecraft to a Lissajous orbit around L2
        sat_pos = orbit_pos + L2_pos

        # Compute the distance between the spacecraft and the planet
        distance_m = norm(sat_pos - icrs_pos).to("m").value

        # This is the direction of the solar system body with respect
        # to the spacecraft, in Ecliptic coordinates
        ecl_vec = (icrs_pos - sat_pos).value

        # The variable ecl_vec is a 3-element vector. We normalize it so
        # that it has length one (using the L_2 norm, hence ord=2)
        ecl_vec /= np.linalg.norm(ecl_vec, axis=0, ord=2)

        # Convert the matrix to a N×3 shape by repeating the vector:
        # planets move slowly, so we assume that the planet stays fixed
        # during this observation.
        ecl_vec = np.repeat(ecl_vec.reshape(1, 3), len(times), axis=0)

        # Calculate the quaternions that convert the Ecliptic
        # reference system into the detector's reference system
        quats = lbs.get_ecl2det_quaternions(
            obs,
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
        dist_map_m2 += bincount / ((4 * np.pi * (distance_m**2)) ** 2)

        pointings = lbs.get_pointings(
            obs,
            sim.spin2ecliptic_quats,
            detector_quats=[detector.quat],
            bore2spin_quat=instr.bore2spin_quat,
        )[0]

        pixidx = healpy.ang2pix(params.output_nside, pointings[:, 0], pointings[:, 1])
        bincount = np.bincount(pixidx, minlength=len(sky_hitmap))
        sky_hitmap += bincount

        # updating the time variable
        t += delta_t_s

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
        healpy.orthview(time_map_s, title="Time spent observing the source", unit="s")

        if scanning_strategy.spin_rate_hz != 0:
            spin_period_min = 1.0 / (60.0 * scanning_strategy.spin_rate_hz)
        else:
            spin_period_min = 0.0

        if scanning_strategy.precession_rate_hz != 0:
            precession_period_min = 1.0 / (60.0 * scanning_strategy.precession_rate_hz)
        else:
            precession_period_min = 0.0

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
            spin_boresight_angle_deg=np.rad2deg(params.spin_boresight_angle_rad),
            precession_period_min=precession_period_min,
            spin_period_min=spin_period_min,
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
