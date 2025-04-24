import xrayutilities as xu

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

from skimage.feature import peak_local_max
from lmfit.models import PseudoVoigtModel, LinearModel

def custom_formatter(x, pos):
    if x == 0:
        return '0'
    exponent = int(np.floor(np.log10(np.abs(x))))
    base = x / 10**exponent
    return r'${:.1f} \times 10^{{{}}}$'.format(base, exponent)

def pseudo_voigt(filename, verbose = False, plot = False,
                 label_size = 20, xlim = None, ylim = None):
    """ 
    Fit a pseudo-voigt function to the data in the given file.

    Parameters
    ----------
    filename : str
        The path to the file containing the data to be fitted.
    verbose : bool, optional
        If True, print the fit results. Default is False.
    plot : bool, optional
        If True, plot the data and the fit. Default is False.
    label_size : int, optional
        The size of the labels in the plot. Default is 20.
    xlim : tuple, optional
        The x-axis limits for the plot. Default is None.
    ylim : tuple, optional
        The y-axis limits for the plot. Default is None.
    
    Returns
    -------
    fit : lmfit.ModelResult
        The result of the fit.
    """

    # Read the data from the file
    RAS_file = xu.io.rigaku_ras.RASFile(filename)
    RAS_file.Read()

    # TODO: Generalize so that it works with different scans (e.g. 2theta, omega, etc.)
    # Extract the Omega data
    df = pd.DataFrame(RAS_file.scan.data)

    # Setup the LMfit model
    model = PseudoVoigtModel()
    parameters = model.guess(df.int, x = df.Omega)
    output = model.fit(df.int, parameters, x = df.Omega)

    # Extract the fit results
    beta = output.best_values['sigma'] * 2
    omega = output.best_values['center']
    two_theta = RAS_file.scan.init_mopo['TwoTheta']

    if verbose:
        print(f"Omega: {omega:.2f}, Two Theta: {two_theta:.2f}, Beta: {beta:.2f}")
    
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

        # Plot the data points and the best fit line
        ax.plot(df.Omega, df.int, 'rx', label='Data', markersize=2)
        ax.plot(df.Omega, output.best_fit, 'b-', label='Best Fit', linewidth=2)

        # Custom formatter for the y-axis to use 10^x notation
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

        # Set the number and major ticks on both axes
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        # Add minor ticks with only 2 intervals between major ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        # Customize the tick parameters for better readability
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        ax.tick_params(axis='both', which='minor', labelsize=label_size - 2)
        ax.tick_params(axis='both', which='both', width=1.5)
        ax.tick_params(axis='both', which='both', length=6)
        ax.tick_params(axis='both', which='minor', length=4)
        ax.tick_params(axis='both', which='both', direction='in')
        ax.tick_params(axis='both', which='both', pad=10)
        ax.tick_params(axis='both', which='both', color='black')
        ax.tick_params(axis='both', which='both', labelcolor='black')
        ax.tick_params(axis='both', which='both', grid_color='gray')
        ax.tick_params(axis='both', which='both', grid_alpha=0.5)
        ax.tick_params(axis='both', which='both', grid_linewidth=0.5)
        ax.tick_params(axis='both', which='both', grid_linestyle='--')

        # Set labels for axes without italicizing LaTeX symbols
        ax.set_xlabel(r'$\mathrm{\Omega\ (degrees)}$', fontsize=label_size)
        ax.set_ylabel(r'Intensity (cps)', fontsize=label_size)

        # Thicken the borders of the plot
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        ax.set_xlim(xlim if xlim else (df.Omega.min(), df.Omega.max()))
        ax.set_ylim(ylim if ylim else (df.int.min(), df.int.max()))
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return output

def reciprocal_space_map(filename, verbose = False, threshold = 10,
        mode = 'TwoThetaOmega', label_size = 20, xlim = [-0.5, 0.5], ylim = None,
        title = 'Reciprocal Space Map'):
    """ 
    Plot a reciprocal space map from the given file.

    Parameters
    ----------
    filename : str
        The path to the file containing the data to be plotted.
    verbose : bool, optional
        If True, print the fit results. Default is False.
    threshold : int, optional
        The threshold value to filter the intensity values. Default is 10.
    label_size : int, optional
        The size of the labels in the plot. Default is 20.
    xlim : tuple, optional
        The x-axis limits for the plot. Default is None.
    ylim : tuple, optional
        The y-axis limits for the plot. Default is None.
    title : str, optional
        The title of the plot. Default is 'Reciprocal Space Map'.
    
    Returns
    -------
    None
    """

    # Read the data from the file
    RAS_file = xu.io.rigaku_ras.RASFile(filename)
    RAS_file.Read()

    # Calculate the offset angles and azimuth angles
    Rx = RAS_file.scan.init_mopo['Rx']
    Ry = RAS_file.scan.init_mopo['Ry']
    omega_x = np.deg2rad(Rx)
    omega_y = np.deg2rad(Ry)
    off_angle = np.rad2deg(np.arcsin(np.sqrt(np.sin(omega_x)**2 + np.sin(omega_y)**2)))
    azimuth_angle = np.rad2deg(np.arctan2(np.sin(omega_y), np.sin(omega_x)))

    if verbose:
        print(f"Offset Angle: {off_angle:.2f}, Azimuth Angle: {azimuth_angle:.2f}")
    
    # Extract the data
    list_series = []
    for scan_data in RAS_file.scans[len(RAS_file.scans) // 2:]:
        df = pd.DataFrame(scan_data.data).set_index(mode)
        list_series.append(df['int'].copy())
    df = pd.concat(list_series, axis = 1)

    # Threshold value to filter the intensity values
    masked_values = np.ma.masked_less(df.values, threshold)
    local_max_coords = peak_local_max(masked_values, min_distance = 2, threshold_abs = threshold)

    # Plot the reciprocal space map
    extent = (xlim[0], xlim[1], df.index.min(), df.index.max())
    log_norm = colors.LogNorm(vmin = threshold, vmax = df.values.max())
    im = plt.imshow(df.values, extent = extent, aspect = 'auto', origin = 'lower',
                   interpolation = 'nearest', cmap = 'jet', norm = log_norm)
    
    cbar = plt.colorbar(im, pad = 0.01, aspect = 10)
    cbar.set_label('Intensity (cps)', fontsize = label_size)
    cbar.ax.tick_params(labelsize = label_size)

    plt.axvline(x = 0, color = 'black', linestyle = '--', linewidth = 1)
    
    plt.xlabel(r'$\mathrm{\Delta\omega\ (degrees)}$', fontsize = label_size)
    plt.ylabel(r'$\mathrm{2\theta/\omega\ (degrees)}$', fontsize = label_size)
    plt.title(title, fontsize = label_size)

    plt.ylim(ylim if ylim else (df.index.min(), df.index.max()))
    plt.xticks(fontsize = label_size)

    for coord in local_max_coords:
        two_theta_idx, omega_idx = coord
        two_theta_value = df.index[two_theta_idx]
        omega_value = extent[0] + (extent[1] - extent[0]) * omega_idx / masked_values.shape[1]
        plt.plot(omega_value, two_theta_value, 'w+', markersize = 16, markeredgewidth = 2)


    # Customize the tick parameters for better readability
    plt.tick_params(axis = 'both', which = 'major', labelsize = label_size)
    plt.tick_params(axis = 'both', which = 'minor', labelsize = label_size - 2)
    plt.tick_params(axis = 'both', which = 'both', width = 1.5)
    plt.tick_params(axis = 'both', which = 'both', length = 6)
    plt.tick_params(axis = 'both', which = 'minor', length = 4)
    plt.tick_params(axis = 'both', which = 'both', direction = 'in')
    plt.tick_params(axis = 'both', which = 'both', pad = 10)
    plt.tick_params(axis = 'both', which = 'both', color = 'black')
    plt.tick_params(axis = 'both', which = 'both', labelcolor = 'black')
    plt.tick_params(axis = 'both', which = 'both', grid_color = 'gray')
    plt.tick_params(axis = 'both', which = 'both', grid_alpha = 0.5)
    plt.tick_params(axis = 'both', which = 'both', grid_linewidth = 0.5)
    plt.tick_params(axis = 'both', which = 'both', grid_linestyle = '--')

    # Thicken the borders of the plot
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.show()
    plt.close()


    return df