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
from typing import Dict, Any, List, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt

from numba import njit
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


@dataclass
class DetectorInfo:
    name: str = ""
    wafer: Union[str, None] = None
    pixel: Union[int, None] = None
    pixtype: Union[str, None] = None
    channel: Union[str, None] = None
    sampling_rate_hz: float = 0.0
    fwhm_arcmin: float = 0.0
    ellipticity: float = 0.0
    net_ukrts: float = 0.0
    fknee_mhz: float = 0.0
    fmin_hz: float = 0.0
    alpha: float = 0.0
    pol: Union[str, None] = None
    orient: Union[str, None] = None
    quat: Any = np.array([0.0, 0.0, 0.0, 1.0])
    bandwidth_ghz: float = 0.0
    bandcenter_ghz: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, Any]):
        return DetectorInfo(**d)

    @staticmethod
    def from_imo(imo: lbs.Imo, objurl):
        return DetectorInfo.from_dict(imo.query(objurl).metadata)


@dataclass
class ChannelInfo:
    channel: str
    bandcenter_ghz: float
    bandwidth_ghz: float
    net_detector_ukrts: float
    net_channel_ukrts: float
    pol_sensitivity_channel_ukarcmin: float
    fwhm_arcmin: float
    fknee_mhz: float
    fmin_hz: float
    alpha: float
    number_of_detectors: int
    detector_names: List[str]

    @staticmethod
    def from_dict(d):
        return ChannelInfo(
            channel=d["channel"],
            bandcenter_ghz=d["bandcenter"],
            bandwidth_ghz=d["bandwidth"],
            net_detector_ukrts=d["net_detector_ukrts"],
            net_channel_ukrts=d["net_channel_ukrts"],
            pol_sensitivity_channel_ukarcmin=d["pol_sensitivity_channel_uKarcmin"],
            fwhm_arcmin=d["fwhm_arcmin"],
            fknee_mhz=d["fknee_mhz"],
            fmin_hz=d["fmin_hz"],
            alpha=d["alpha"],
            number_of_detectors=d["number_of_detectors"],
            detector_names=d["detector_names"],
        )

    @staticmethod
    def from_imo(imo, objref):
        obj = imo.query(objref)
        return ChannelInfo.from_dict(obj.metadata)

    def get_boresight_detector(self) -> DetectorInfo:
        return DetectorInfo(
            name="mock",
            channel=self.channel,
            fwhm_arcmin=self.fwhm_arcmin,
            net_ukrts=self.net_detector_ukrts,
            fknee_mhz=self.fknee_mhz,
            fmin_hz=self.fmin_hz,
            alpha=self.alpha,
            bandwidth_ghz=self.bandwidth_ghz,
            bandcenter_ghz=self.bandcenter_ghz,
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
            spin_sun_angle_deg=0.0,
            precession_period_min=0.0,
            spin_rate_rpm=1.0,
            start_time=start_time,
        )

    if "spin_sun_angle_deg" in parameters:
        sstr.spin_sun_angle_rad = np.deg2rad(parameters["spin_sun_angle_deg"])

    if "precession_period_min" in parameters:
        sstr.precession_rate_hz = 1.0 / (60.0 * parameters["precession_period_min"])

    if "spin_rate_rpm" in parameters:
        sstr.spin_rate_hz = parameters["spin_rate_rpm"] / 60.0

    return sstr


def read_detector(parameters: Dict[str, Any], imo: lbs.Imo):
    if "channel_obj" in parameters:
        detobj = ChannelInfo.from_imo(
            imo, parameters["channel_obj"]
        ).get_boresight_detector()

    elif "detector_obj" in parameters:
        detobj = DetectorInfo.from_imo(imo, parameters["detector_obj"])
    else:
        detobj = DetectorInfo()

    for param_name in (
        "name",
        "wafer",
        "pixel",
        "pixtype",
        "channel",
        "sampling_rate_hz",
        "fwhm_arcmin",
        "ellipticity",
        "net_ukrts",
        "fknee_mhz",
        "fmin_hz",
        "alpha",
        "pol",
        "orient",
        "bandwidth_ghz",
        "bandcenter_ghz",
    ):
        if param_name in parameters:
            setattr(detobj, param_name, parameters[param_name])

    return detobj


def main():
    warnings.filterwarnings("ignore", category=ErfaWarning)

    sim = lbs.Simulation(
        parameter_file=sys.argv[1],
        name="In-flight beam",
        description="""
    This is a simulation of in-flight beam reconstruction.
    """,
    )

    planet_params = sim.parameters["planet_scanning"]

    # Initialize variables with the values in the TOML file
    SPIN_BORESIGHT_ANGLE_DEG = sim.parameters["scanning_strategy"][
        "spin_boresight_angle_deg"
    ]

    PLANET_NAME = planet_params["planet_name"]
    SPIN2ECL_DELTA_TIME_S = planet_params["spin2ecl_delta_time_s"]
    DETECTOR_SAMPLING_RATE_HZ = planet_params["sampling_rate_hz"]
    OUTPUT_NSIDE = planet_params["output_nside"]
    OUTPUT_MAP_FILE_NAME = sim.base_path / planet_params["output_map_file_name"]
    OUTPUT_TABLE_FILE_NAME = sim.base_path / planet_params["output_table_file_name"]

    if lbs.MPI_ENABLED:
        log.info("Using MPI with %d processes", lbs.MPI_COMM_WORLD.size)
    else:
        log.info("Not using MPI, serial execution")

    log.info("Generating the quaternions")
    scanning_strategy = read_scanning_strategy(
        sim.parameters["scanning_strategy"], sim.imo, sim.start_time
    )
    sim.generate_spin2ecl_quaternions(
        scanning_strategy=scanning_strategy, delta_time_s=SPIN2ECL_DELTA_TIME_S,
    )

    log.info("Creating the observations")
    instr = lbs.Instrument(
        name="instrum", spin_boresight_angle_deg=SPIN_BORESIGHT_ANGLE_DEG
    )
    detectors = [
        read_detector(detdef, sim.imo) for detdef in sim.parameters["detectors"]
    ]
    assert len(detectors) == 1, "Only one detector is allowed in this simulation"

    detectors[0].sampling_rate_hz = DETECTOR_SAMPLING_RATE_HZ
    sim.create_observations(
        detectors=[detectors[0]],
        num_of_obs_per_detector=sim.parameters["simulation"]["num_of_obs_per_detector"],
    )

    #################################################################
    # Here begins the juicy part

    log.info("The loop starts on %d processes", lbs.MPI_COMM_WORLD.size)
    sky_hitmap = np.zeros(healpy.nside2npix(OUTPUT_NSIDE), dtype=np.int32)
    detector_hitmap = np.zeros(healpy.nside2npix(OUTPUT_NSIDE), dtype=np.int32)
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
        icrs_pos = get_body_barycentric(PLANET_NAME, time0)
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
            detector_quat=obs.detector.quat,
            bore2spin_quat=instr.bore2spin_quat,
        )

        # Make room for the xyz vectors in the detector's reference frame
        det_vec = np.empty_like(ecl_vec)

        # Do the rotation!
        lbs.all_rotate_vectors(det_vec, quats, ecl_vec)

        pixidx = healpy.vec2pix(
            OUTPUT_NSIDE, det_vec[:, 0], det_vec[:, 1], det_vec[:, 2]
        )
        bincount = np.bincount(pixidx, minlength=len(detector_hitmap))
        detector_hitmap += bincount
        dist_map_m2 += bincount / ((4 * np.pi * (distance_m ** 2)) ** 2)

        pointings = obs.get_pointings(
            sim.spin2ecliptic_quats, obs.detector.quat, instr.bore2spin_quat
        )

        pixidx = healpy.ang2pix(OUTPUT_NSIDE, pointings[:, 0], pointings[:, 1])
        bincount = np.bincount(pixidx, minlength=len(sky_hitmap))
        sky_hitmap += bincount

    if lbs.MPI_ENABLED:
        sky_hitmap = lbs.MPI_COMM_WORLD.allreduce(sky_hitmap)
        detector_hitmap = lbs.MPI_COMM_WORLD.allreduce(detector_hitmap)
        dist_map_m2 = lbs.MPI_COMM_WORLD.allreduce(dist_map_m2)

    time_map_s = detector_hitmap / DETECTOR_SAMPLING_RATE_HZ
    dist_map_m2[dist_map_m2 > 0] = np.power(dist_map_m2[dist_map_m2 > 0], -0.5)

    RADII_DEG = [0.1, 0.2, 0.5, 1, 5, 10, 20, 40, 60, 90, 135, 180]
    OBS_TIME_PER_RADIUS_S = [
        time_per_radius(time_map_s, angular_radius_rad=np.deg2rad(radius_deg))
        for radius_deg in RADII_DEG
    ]

    if lbs.MPI_COMM_WORLD.rank == 0:
        # Create a plot of the observation time of the planet as a
        # function of the angular radius
        fig, ax = plt.subplots()
        ax.loglog(RADII_DEG, OBS_TIME_PER_RADIUS_S)
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
    Angle between the spin axis and the boresight | {{ bore_spin_angle_deg }} deg
    Precession period | {{ precession_period_min }} min
    Spin period | {{ spin_period_min }} min

    ## Detector properties

    Parameter | Value
    --------- | -----------------
    Channel | {{det.channel}}
    FWHM | {{det.fwhm_arcmin}} arcmin
    NET | {{det.net_ukrts}} μK·√s
    Bandwidth | {{det.bandwidth_ghz}} GHz
    Band center | {{det.bandcenter_ghz}} GHz

    ## Observation of {{ planet_name | capitalize }}

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
            planet_name=PLANET_NAME,
            overall_time_s=np.sum(detector_hitmap) / DETECTOR_SAMPLING_RATE_HZ,
            radius_vs_time_s=list(zip(RADII_DEG, OBS_TIME_PER_RADIUS_S)),
            delta_time_s=1.0 / DETECTOR_SAMPLING_RATE_HZ,
            sun_earth_angle_deg=np.rad2deg(scanning_strategy.spin_sun_angle_rad),
            bore_spin_angle_deg=SPIN_BORESIGHT_ANGLE_DEG,
            precession_period_min=1.0 / (60.0 * scanning_strategy.precession_rate_hz),
            spin_period_min=1.0 / (60.0 * scanning_strategy.spin_rate_hz),
            det=detectors[0],
        )

        healpy.write_map(
            OUTPUT_MAP_FILE_NAME,
            (detector_hitmap, time_map_s, dist_map_m2, sky_hitmap),
            coord="DETECTOR",
            column_names=["HITS", "OBSTIME", "SQDIST", "SKYHITS"],
            column_units=["", "s", "m^2", ""],
            dtype=[np.int32, np.float32, np.float64, np.int32],
            overwrite=True,
        )

        np.savetxt(
            OUTPUT_TABLE_FILE_NAME,
            np.array(list(zip(RADII_DEG, OBS_TIME_PER_RADIUS_S))),
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
                    "detector": {
                        "sampling_rate_hz": DETECTOR_SAMPLING_RATE_HZ,
                        "fwhm_arcmin": detectors[0].fwhm_arcmin,
                        "net_ukrts": detectors[0].net_ukrts,
                        "bandwidth_ghz": detectors[0].bandwidth_ghz,
                        "bandcenter_ghz": detectors[0].bandcenter_ghz,
                    },
                    "planet": {"name": PLANET_NAME},
                },
                outf,
                indent=2,
            )

    sim.flush()


if __name__ == "__main__":
    main()
