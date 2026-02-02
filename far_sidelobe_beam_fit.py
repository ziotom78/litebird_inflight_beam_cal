#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from dataclasses import dataclass
import json
from pathlib import Path
from shutil import copyfile
import sys
from tools import read_detector
from tqdm import tqdm
import litebird_sim as lbs
import healpy
import numpy as np
from scipy import optimize, interpolate, integrate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt


@dataclass
class Parameters:
    planet_name: str
    sed_file_name: str
    planet_radius_m: float
    scanning_simulation: Path
    detector: lbs.DetectorInfo
    num_of_mc_runs: int
    error_amplitude_map_file_name: str
    amplitude: float
    beam_deg: [float]
    fwhm_arcmin: float
    ecc: float
    incl_deg: float

    def __post_init__(self):
        self.sed_file_name = Path(self.sed_file_name)
        self.scanning_simulation = Path(self.scanning_simulation)


def load_parameters(sim: lbs.Simulation) -> Parameters:
    return Parameters(
        planet_name=sim.parameters["planet"]["planet_name"],
        sed_file_name=sim.parameters["planet"]["sed_file_name"],
        planet_radius_m=sim.parameters["planet"]["planet_radius_m"],
        scanning_simulation=sim.parameters["planet"]["scanning_simulation"],
        detector=read_detector(sim.parameters["detector"], sim.imo),
        num_of_mc_runs=sim.parameters["simulation"].get("num_of_mc_runs", 20),
        error_amplitude_map_file_name=sim.parameters["simulation"].get(
            "error_amplitude_map_file_name", "error_map.fits"
        ),
        amplitude=sim.parameters["detector"].get("amplitude", 1.0),
        beam_deg=sim.parameters["detector"].get("beam_deg", [0.0, 0.0]),
        fwhm_arcmin=sim.parameters["detector"].get("fwhm_arcmin", 1.0),
        ecc=sim.parameters["detector"].get("ecc", 0.0),
        incl_deg=sim.parameters["detector"].get("incl_deg", 0.0),
    )


def beamfunc(pixels, fwhm_arcmin, amplitude, ecc, incl_rad, theta_beam, phi_beam):
    pixel_theta, pixel_phi = pixels

    sigma_u = np.deg2rad(fwhm_arcmin / 60.0) / (2 * np.sqrt(2 * np.log(2)))
    sigma_v = sigma_u * (1 - ecc)

    vec = healpy.ang2vec(pixel_theta, pixel_phi)

    Ry = [
        [np.cos(theta_beam), 0, np.sin(theta_beam)],
        [0, 1, 0],
        [-np.sin(theta_beam), 0, np.cos(theta_beam)],
    ]
    Rz = [
        [np.cos(phi_beam), -np.sin(phi_beam), 0],
        [np.sin(phi_beam), np.cos(phi_beam), 0],
        [0, 0, 1],
    ]
    RotMat = np.dot(Rz, Ry)
    vecrot = np.dot(vec, RotMat)

    beam = np.zeros(len(vecrot))
    x = np.transpose(vecrot)[0]
    y = np.transpose(vecrot)[1]
    z = np.transpose(vecrot)[2]

    u = np.cos(incl_rad) * x + np.sin(incl_rad) * y
    v = -np.sin(incl_rad) * x + np.cos(incl_rad) * y
    beam = amplitude * np.exp(
        -((u / (np.sqrt(2) * sigma_u)) ** 2 + (v / (np.sqrt(2) * sigma_v)) ** 2)
    )
    beam[z < 0] = 0.0
    return beam


def calc_beam_solid_angle(fwhm_arcmin, ecc):
    sigma_x = np.deg2rad(fwhm_arcmin / 60.0) / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = sigma_x * (1 - ecc)
    beam = lambda y, x: np.exp(
        -((x / (np.sqrt(2) * sigma_x)) ** 2 + (y / (np.sqrt(2) * sigma_y)) ** 2)
    )

    return integrate.dblquad(
        beam,
        -10 * sigma_x,
        10 * sigma_x,
        lambda x: -10 * sigma_y,
        lambda x: 10 * sigma_y,
    )[0]


def project_map_beam(
    pixels, nside, width_deg, theta_beam, phi_beam, pixels_per_side=150
):
    gamma_map = np.zeros(len(pixels))
    vecrot = healpy.pix2vec(nside, np.arange(len(pixels)))
    pixrotate = healpy.vec2pix(nside, vecrot[0], vecrot[1], vecrot[2])
    Ry = [
        [np.cos(-theta_beam), 0, np.sin(-theta_beam)],
        [0, 1, 0],
        [-np.sin(-theta_beam), 0, np.cos(-theta_beam)],
    ]
    Rz = [
        [np.cos(-phi_beam), -np.sin(-phi_beam), 0],
        [np.sin(-phi_beam), np.cos(-phi_beam), 0],
        [0, 0, 1],
    ]
    RotMat = np.dot(Ry, Rz)
    vec = np.dot(np.linalg.inv(RotMat), vecrot)
    pix = healpy.vec2pix(nside, vec[0], vec[1], vec[2])
    index = 0
    for i in pix:
        gamma_map[index] = pixels[i]
        index += 1

    theta_max = np.deg2rad(width_deg)
    extr_inf = -np.sin(theta_max)
    extr_sup = np.sin(theta_max)
    u = np.linspace(extr_inf, extr_sup, pixels_per_side)
    v = np.linspace(extr_inf, extr_sup, pixels_per_side)
    u_grid, v_grid = np.meshgrid(u, v)
    theta_grid = np.arcsin(np.sqrt(u_grid**2 + v_grid**2))
    phi_grid = np.arctan2(v_grid, u_grid)
    return (
        u_grid,
        v_grid,
        healpy.get_interp_val(
            gamma_map, theta_grid.flatten(), phi_grid.flatten()
        ).reshape(pixels_per_side, -1),
    )


def create_uv_plot(
    fig,
    ax,
    pixels,
    nside,
    theta_beam,
    phi_beam,
    width_deg,
    contour_lines=True,
    smooth=False,
    fwhm_arcmin=None,
):
    from scipy.ndimage.filters import gaussian_filter

    u_grid, v_grid, grid = project_map_beam(
        pixels, nside, width_deg, theta_beam, phi_beam
    )

    if smooth:
        grid = gaussian_filter(grid, 0.7)

    cs = ax.contourf(u_grid, v_grid, grid, cmap=plt.cm.bone)

    if fwhm_arcmin:
        from matplotlib.patches import Circle

        ax.add_artist(
            Circle(
                (0, 0),
                np.sin(np.deg2rad(fwhm_arcmin / 120.0)),
                edgecolor="w",
                lw=5,
                facecolor="none",
                alpha=0.25,
            )
        )

    if contour_lines:
        cs2 = ax.contour(cs, levels=cs.levels[::2], colors="r")
        ax.clabel(cs2)

    ax.set_xlabel("U coordinate")
    ax.set_ylabel("V coordinate")
    ax.set_aspect("equal")

    cbar = fig.colorbar(cs)

    if contour_lines:
        cbar.add_lines(cs2)


def create_gamma_plots(
    gamma_map, gamma_error_map, nside, fwhm_arcmin, theta_beam, phi_beam
):
    plot_size_deg = fwhm_arcmin / 60.0

    gamma_fig, gamma_ax = plt.subplots()
    create_uv_plot(
        gamma_fig,
        gamma_ax,
        gamma_map,
        nside,
        theta_beam,
        phi_beam,
        width_deg=plot_size_deg,
        fwhm_arcmin=None,
    )

    gamma_error_fig, gamma_error_ax = plt.subplots()
    create_uv_plot(
        gamma_error_fig,
        gamma_error_ax,
        gamma_error_map,
        nside,
        theta_beam,
        phi_beam,
        width_deg=plot_size_deg,
        contour_lines=False,
        fwhm_arcmin=fwhm_arcmin,
    )

    gamma_over_error_fig, gamma_over_error_ax = plt.subplots()
    create_uv_plot(
        gamma_over_error_fig,
        gamma_over_error_ax,
        gamma_map / gamma_error_map,
        nside,
        theta_beam,
        phi_beam,
        width_deg=plot_size_deg,
        smooth=True,
        fwhm_arcmin=fwhm_arcmin,
    )

    return (plot_size_deg, gamma_fig, gamma_error_fig, gamma_over_error_fig)


def main(data_path: Path):
    sim = lbs.Simulation(
        parameter_file=sys.argv[1],
        name="In-flight estimation of the beam properties",
        description="""
This report contains the result of a simulation of the reconstruction
of in-flight beam parameters, assuming a scanning strategy and some
noise/optical properties of a detector.
""",
    )

    params = load_parameters(sim)

    det = read_detector(sim.parameters["detector"], sim.imo)

    # TODO: This should be done by the framework
    copyfile(src=params.sed_file_name, dst=sim.base_path / params.sed_file_name.name)

    # Calculate the brightness temperature of the planet over the band
    sed_data = np.loadtxt(params.sed_file_name, delimiter=",")
    sed_fn = interpolate.interp1d(sed_data[:, 0], sed_data[:, 1])
    planet_temperature_k = (
        integrate.quad(
            sed_fn,
            params.detector.bandcenter_ghz - params.detector.bandwidth_ghz / 2,
            params.detector.bandcenter_ghz + params.detector.bandwidth_ghz / 2,
        )[0]
        / params.detector.bandwidth_ghz
    )
    beam_solid_angle = calc_beam_solid_angle(
        fwhm_arcmin=params.fwhm_arcmin, ecc=params.ecc
    )
    sampling_time_s = 1.0 / params.detector.sampling_rate_hz

    input_map_file_name = params.scanning_simulation / "map.fits.gz"

    hit_map, time_map_s, dist_map_m2 = healpy.read_map(
        input_map_file_name, field=(0, 1, 2), verbose=False, dtype=np.float32
    )
    nside = healpy.npix2nside(len(dist_map_m2))
    pixel_theta, pixel_phi = healpy.pix2ang(nside, np.arange(len(hit_map)))

    gamma_map = beamfunc(
        healpy.pix2ang(nside, np.arange(len(hit_map))),
        params.fwhm_arcmin,
        params.amplitude,
        params.ecc,
        np.deg2rad(params.incl_deg),
        np.deg2rad(params.beam_deg[0]),
        np.deg2rad(params.beam_deg[1]),
    )
    mask = (hit_map > 0.0) & (gamma_map > params.amplitude * np.exp(-np.log(2) * 9 / 2))
    assert hit_map[mask].size > 0, "no data available for the fit"

    sim.append_to_report(
        r"""
## Detector properties
Parameter | Value
--------- | -----------------
Channel | {{det.channel}}
Sampling time | {{ "%.3f"|format(sampling_time_s) }} s
NET | {{det.net_ukrts}} μK·√s
Bandwidth | {{det.bandwidth_ghz}} GHz
Band center | {{det.bandcenter_ghz}} GHz
FWHM | {{det.fwhm_arcmin}} arcmin
Beam solid angle | {{ "%.3e"|format(beam_solid_angle)}} sterad
Amplitude | {{ "%.3e"|format(params.amplitude) }} 

## Properties of the planet
Parameter | Value
--------- | ----------------
Brightness temperature | {{ "%.1f"|format(planet_temperature_k) }} K
Effective radius | {{ "%.0e"|format(params.planet_radius_m) }} m

## Beam scanning
Parameter | Value
--------- | ----------------
Pixels used in the fit | {{ num_of_pixels_used }}
Integration time | {{ integration_time_s }} s
""",
        det=params.detector,
        params=params,
        beam_solid_angle=beam_solid_angle,
        sampling_time_s=sampling_time_s,
        planet_temperature_k=planet_temperature_k,
        num_of_pixels_used=len(mask[mask]),
        integration_time_s=np.sum(time_map_s[mask]),
    )

    error_amplitude_map = (
        beam_solid_angle
        * (params.detector.net_ukrts * 1e-6)
        / (
            np.pi
            * (params.planet_radius_m**2)
            * planet_temperature_k
            * np.sqrt(sampling_time_s)
        )
    ) * dist_map_m2

    (
        plot_size_deg,
        gamma_fig,
        gamma_error_fig,
        gamma_over_error_fig,
    ) = create_gamma_plots(
        gamma_map,
        error_amplitude_map,
        nside,
        fwhm_arcmin=params.detector.fwhm_arcmin,
        theta_beam=np.deg2rad(params.beam_deg[0]),
        phi_beam=np.deg2rad(params.beam_deg[1]),
    )

    sim.append_to_report(
        r"""
## Error on beam estimation
This is a representation of the model of the far sidelobe beam used in the
simulation, in $`(u,v)`$ coordinates, where
$`u = \sin\theta\,\cos\phi`$ and
$`v = \sin\theta\sin\phi`$:
![](gamma.svg)
Here is the result of the estimate of $`\delta\gamma`$, using
the following formula:
```math
\delta\gamma(\vec r) = 
    \frac{\Omega_b \cdot \text{WN}}{\pi r_\text{pl}^2\,T_\text{br,pl}\,\sqrt\tau} 
    \sqrt{\frac1{\sum_{i=1}^N \left(\frac1{4\pi d_\text{pl}^2(t_i)}\right)^2}},
```
where $`N`$ is the number of samples observed along direction $`\vec
r`$, WN is the white noise level expressed as
$`\text{K}\cdot\sqrt{s}`$, $`r_\text{pl}`$ is the planet's radius,
$`d_\text{pl}(t)`$ is the planet-spacecraft distance at time $`t`$,
and $`\tau`$ is the sample integration time (assumed equal for all the
samples).
And here is a plot of $`\delta\gamma`$, again in $`(u, v)`$
coordinates:
![](gamma_error.svg)
The ratio $`\gamma / \delta\gamma`$ represents the S/N ratio:
![](gamma_over_error.svg)
The size of each plot is {{ "%.1f"|format(plot_size_deg) }}°, and the
white shadow represents the size of the FWHM
({{ "%.1f"|format(fwhm_arcmin) }} arcmin).
""",
        fwhm_arcmin=params.detector.fwhm_arcmin,
        plot_size_deg=plot_size_deg,
        figures=[
            (gamma_fig, "gamma.svg"),
            (gamma_error_fig, "gamma_error.svg"),
            (gamma_over_error_fig, "gamma_over_error.svg"),
        ],
    )

    destfile = sim.base_path / params.error_amplitude_map_file_name
    healpy.write_map(
        destfile,
        [gamma_map, error_amplitude_map],
        coord="DETECTOR",
        column_names=["GAMMA", "ERR"],
        column_units=["", ""],
        dtype=[np.float32, np.float32, np.float32],
        overwrite=True,
    )
    fwhm_estimates_arcmin = np.empty(params.num_of_mc_runs)
    ampl_estimates = np.empty(len(fwhm_estimates_arcmin))
    ecc_estimates = np.empty(len(fwhm_estimates_arcmin))
    incl_estimates = np.empty(len(fwhm_estimates_arcmin))
    theta_beam_estimates = np.empty(len(fwhm_estimates_arcmin))
    phi_beam_estimates = np.empty(len(fwhm_estimates_arcmin))

    for i in tqdm(range(len(fwhm_estimates_arcmin))):
        noise_gamma_map = gamma_map + error_amplitude_map * np.random.randn(
            len(dist_map_m2)
        )
        # Run the fit
        best_fit, pcov = optimize.curve_fit(
            beamfunc,
            np.vstack((pixel_theta[mask], pixel_phi[mask])),
            noise_gamma_map[mask],
            p0=[
                params.fwhm_arcmin,
                params.amplitude,
                params.ecc,
                np.deg2rad(params.incl_deg),
                np.deg2rad(params.beam_deg[0]),
                np.deg2rad(params.beam_deg[1]),
            ],
            maxfev=5000,
        )

        fwhm_estimates_arcmin[i] = best_fit[0]
        ampl_estimates[i] = best_fit[1]
        ecc_estimates[i] = best_fit[2]
        incl_estimates[i] = best_fit[3]
        theta_beam_estimates[i] = best_fit[4]
        phi_beam_estimates[i] = best_fit[5]

    figs = []

    for dataset, name in [
        (fwhm_estimates_arcmin, "FWHM [arcmin]"),
        (ampl_estimates, "AMPL [arcmin]"),
        (ecc_estimates, "ECC"),
        (incl_estimates, "INCL [rad]"),
        (theta_beam_estimates, "THETA [rad]"),
        (phi_beam_estimates, "PHI [rad]"),
    ]:
        fig, ax = plt.subplots()
        ax.hist(dataset)
        ax.set_xlabel(name)
        ax.set_ylabel("Counts")
        figs.append(fig)

    sim.append_to_report(
        """
## Results of the Monte Carlo simulation
Parameter  | Value
---------- | -----------------
# of runs  | {{ num_of_runs }}
FWHM       | {{"%.3f"|format(fwhm_arcmin)}} ± {{"%.3e"|format(fwhm_err)}} arcmin
γ0         | {{"%.3f"|format(ampl)}} ± {{"%.3e"|format(ampl_err)}} arcmin
Ecc        | {{"%.3f"|format(ecc)}} ± {{"%.3e"|format(ecc_err)}} 
Inclination | {{"%.3f"|format(incl)}} ± {{"%.3e"|format(incl_err)}} rad
Theta_beam | {{"%.3f"|format(theta)}} ± {{"%.3e"|format(theta_err)}} rad
Phi_beam | {{"%.3f"|format(phi)}} ± {{"%.3e"|format(phi_err)}} rad
![](fwhm_distribution.svg)
![](ampl_distribution.svg)
![](ecc_distribution.svg)
![](incl_distribution.svg)
![](theta_distribution.svg)
![](phi_distribution.svg)
""",
        figures=[
            (figs[0], "fwhm_distribution.svg"),
            (figs[1], "ampl_distribution.svg"),
            (figs[2], "ecc_distribution.svg"),
            (figs[3], "incl_distribution.svg"),
            (figs[4], "theta_distribution.svg"),
            (figs[5], "phi_distribution.svg"),
        ],
        num_of_runs=len(fwhm_estimates_arcmin),
        fwhm_arcmin=np.mean(fwhm_estimates_arcmin),
        fwhm_err=np.std(fwhm_estimates_arcmin),
        ampl=np.mean(ampl_estimates),
        ampl_err=np.std(ampl_estimates),
        ecc=np.mean(ecc_estimates),
        ecc_err=np.std(ampl_estimates),
        incl=np.mean(incl_estimates),
        incl_err=np.std(incl_estimates),
        theta=np.mean(theta_beam_estimates),
        theta_err=np.std(theta_beam_estimates),
        phi=np.mean(phi_beam_estimates),
        phi_err=np.std(phi_beam_estimates),
    )

    json_file_name = sim.base_path / "results.json"
    with json_file_name.open("w") as f:
        json.dump(
            {
                "mc_fwhm_arcmin": fwhm_estimates_arcmin.tolist(),
                "mc_fwhm_arcmin_mean": np.mean(fwhm_estimates_arcmin),
                "mc_fwhm_err": np.std(fwhm_estimates_arcmin),
                "mc_amplitudes": ampl_estimates.tolist(),
                "mc_amplitudes_mean": np.mean(ampl_estimates),
                "mc_amplitudes_err": np.std(ampl_estimates),
                "mc_eccentricity": ecc_estimates.tolist(),
                "mc_eccentricity_mean": np.mean(ecc_estimates),
                "mc_eccentricity_err": np.std(ampl_estimates),
                "mc_inclinations": incl_estimates.tolist(),
                "mc_inclinations_mean": np.mean(incl_estimates),
                "mc_inclinations_err": np.std(incl_estimates),
                "mc_theta_beam": theta_beam_estimates.tolist(),
                "mc_theta_mean": np.mean(theta_beam_estimates),
                "mc_theta_err": np.std(theta_beam_estimates),
                "mc_phi_beam": phi_beam_estimates.tolist(),
                "mc_phi_mean": np.mean(phi_beam_estimates),
                "mc_phi_err": np.std(phi_beam_estimates),
            },
            f,
        )

    sim.flush()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
