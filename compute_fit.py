#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import json
from pathlib import Path
import sys

from tqdm import tqdm

import healpy
import numpy as np
from scipy import optimize, interpolate, integrate
import tomlkit

from simulate import DetectorInfo

SED_FILE = {
    "jupiter": "jupiter_sed_ghz_k.csv",
}

PLANET_RADII_M = {
    "jupiter": 7.1492e7,
    "mars": 3.389e6,
    "uranus": 2.5362e7,
    "neptune": 2.4622e7,
}


def beamfunc(pixel_theta, fwhm_arcmin, amplitude=1.0):
    return amplitude * np.exp(
        -4 * np.log(2) * (pixel_theta ** 2) / (np.deg2rad(fwhm_arcmin / 60.0) ** 2)
    )


def calc_beam_solid_angle(fwhm_arcmin):
    return integrate.quad(lambda θ: np.sin(θ) * beamfunc(θ, fwhm_arcmin), 0, np.pi)[0]


def main(data_path: Path):
    with next(data_path.glob("*.toml")).open("rt") as inpf:
        sim_parameters = tomlkit.loads("".join(inpf.readlines()))

    with (data_path / "parameters.json").open("rt") as inpf:
        d = json.load(inpf)["detector"]
        det = DetectorInfo(
            name="boresight",
            fwhm_arcmin=d["fwhm_arcmin"],
            net_ukrts=d["net_ukrts"],
            bandwidth_ghz=d["bandwidth_ghz"],
            bandcenter_ghz=d["bandcenter_ghz"],
            sampling_rate_hz=d["sampling_rate_hz"],
        )

    planet_name = sim_parameters["planet_scanning"]["planet_name"]

    # Calculate the brightness temperature of the planet over the band
    sed_data = np.loadtxt(SED_FILE[planet_name], delimiter=",")
    sed_fn = interpolate.interp1d(sed_data[:, 0], sed_data[:, 1])
    planet_temperature_k = (
        integrate.quad(
            sed_fn,
            det.bandcenter_ghz - det.bandwidth_ghz / 2,
            det.bandcenter_ghz + det.bandwidth_ghz / 2,
        )[0]
        / det.bandwidth_ghz
    )
    planet_radius_m = PLANET_RADII_M[planet_name]
    beam_solid_angle = calc_beam_solid_angle(fwhm_arcmin=det.fwhm_arcmin)
    sampling_time_s = 1.0 / det.sampling_rate_hz
    print(f"Planet temperature: {planet_temperature_k:.1f} K")
    print(f"Planet radius: {planet_radius_m:.2e} m")
    print(f"FWHM: {det.fwhm_arcmin:.2f} arcmin")
    print(f"Beam solid angle Ω: {beam_solid_angle:.2e} sterad")
    print(f"White noise level: {det.net_ukrts * 1e-6:.2e} K·√s")
    print(f"Sampling time: {sampling_time_s:.3f} s")

    input_map_file_name = (
        data_path / sim_parameters["planet_scanning"]["output_map_file_name"]
    )

    hit_map, time_map_s, dist_map_m2 = healpy.read_map(
        input_map_file_name, field=(0, 1, 2), verbose=False, dtype=np.float32
    )
    nside = healpy.npix2nside(len(dist_map_m2))
    pixel_theta, pixel_phi = healpy.pix2ang(nside, np.arange(len(hit_map)))

    gamma_map = beamfunc(pixel_theta, det.fwhm_arcmin)

    mask = (hit_map > 0.0) & (pixel_theta < np.deg2rad(3 * det.fwhm_arcmin / 60.0))
    print(f"Pixels used in the fit: {len(mask[mask])} / {len(mask)}")
    print(f"Overall integration time for γ: {np.sum(time_map_s[mask])} s")

    error_amplitude_map = (
        beam_solid_angle
        * (det.net_ukrts * 1e-6)
        / (
            np.pi
            * (planet_radius_m ** 2)
            * planet_temperature_k
            * np.sqrt(sampling_time_s)
        )
    ) * dist_map_m2

    fwhm_estimates_arcmin = np.empty(20)
    ampl_estimates = np.empty(len(fwhm_estimates_arcmin))
    for i in tqdm(range(len(fwhm_estimates_arcmin))):
        noise_gamma_map = gamma_map + error_amplitude_map * np.random.randn(
            len(dist_map_m2)
        )
        # Run the fit
        best_fit, pcov = optimize.curve_fit(
            beamfunc,
            pixel_theta[mask],
            noise_gamma_map[mask],
            p0=[det.fwhm_arcmin, 1.0],
        )

        fwhm_estimates_arcmin[i] = best_fit[0]
        ampl_estimates[i] = best_fit[1]

    print(
        f"FWHM: {np.mean(fwhm_estimates_arcmin):.3f} ± {np.std(fwhm_estimates_arcmin):.3f} arcmin"
    )
    print(f"Amplitude: {np.mean(ampl_estimates)} ± {np.std(ampl_estimates)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
