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
    """
    Custom formatter for axis ticks to display values in scientific notation.

    Parameters
    ----------
    x : float
        The tick value.
    pos : int
        The tick position (not used in this function).

    Returns
    -------
    str
        Formatted string for the tick value.
    """
    if x == 0:
        return '0'
    exponent = int(np.floor(np.log10(np.abs(x))))
    base = x / 10**exponent
    return r'${:.1f} \times 10^{{{}}}$'.format(base, exponent)

def pseudo_voigt(filename, normalized = False, verbose=False, plot=False,
                 label_size=20, xlim=None, ylim=None):
    """
    Fit a pseudo-Voigt function to the data in the given file.

    Parameters
    ----------
    filename : str
        The path to the file containing the data to be fitted.
    normalized : bool, optional
        If True, normalize the intensity data. Default is False.
    verbose : bool, optional
        If True, print the fit results. Default is False.
    plot : bool, optional
        If True, plot the data and the fit. Default is False.
    label_size : int, optional
        The font size for plot labels. Default is 20.
    xlim : tuple, optional
        The x-axis limits for the plot. Default is None.
    ylim : tuple, optional
        The y-axis limits for the plot. Default is None.

    Returns
    -------
    lmfit.ModelResult
        The result of the pseudo-Voigt fit.
    """
    # Read the data from the Rigaku RAS file
    RAS_file = xu.io.rigaku_ras.RASFile(filename)
    df = pd.DataFrame(RAS_file.scan.data)

    # Rescale the intensity so that the maximum value is 1
    if normalized:
        df['int'] = df['int'] / df['int'].max()

    # Set up the pseudo-Voigt model using lmfit
    model = PseudoVoigtModel()
    parameters = model.guess(df.int, x=df.Omega)
    output = model.fit(df.int, parameters, x=df.Omega)

    # Calculate offset and azimuth angles
    Rx = RAS_file.scan.init_mopo['Rx']
    Ry = RAS_file.scan.init_mopo['Ry']
    omega_x = np.deg2rad(Rx)
    omega_y = np.deg2rad(Ry)
    off_angle = np.rad2deg(np.arcsin(np.sqrt(np.sin(omega_x)**2 + np.sin(omega_y)**2)))
    azimuth_angle = np.rad2deg(np.arctan2(np.sin(omega_y), np.sin(omega_x)))

    if verbose:
        print(f"Rx: {Rx:.4f}, Ry: {Ry:.4f}")
        print(f"Offset Angle: {off_angle:.4f}, Azimuth Angle: {azimuth_angle:.4f}")

    # Extract fit results
    beta = output.best_values['sigma'] * 2
    omega = output.best_values['center']
    two_theta = RAS_file.scan.init_mopo['TwoTheta']

    if verbose:
        print(f"Omega: {omega:.2f}, Two Theta: {two_theta:.2f}, Beta: {beta:.2f}")
    
    if plot:
        # Plot the data and the fit
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    
        ax.plot(df.Omega, df.int, 'rx', label='Data', markersize=2)
        ax.plot(df.Omega, output.best_fit, 'b-', label='Best Fit', linewidth=2)

        if not normalized:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

        # Configure axis ticks and labels
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        ax.tick_params(axis='both', which='minor', labelsize=label_size - 2)

        # Thicken only x and y-axis lines and remove the rest of the border
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # Set axis labels
        ax.set_xlabel(r'$\mathrm{\omega\ (\degree)}$', fontsize=label_size)
        if normalized:
            ax.set_ylabel(r'Normalized intensity (a.u.)', fontsize=label_size)
            ax.set_ylim(0, 1)
        else:
            ax.set_ylabel(r'Intensity (cps)', fontsize=label_size)
            ax.set_ylim(ylim if ylim else (0, df.int.max()))

        # Set axis limits
        ax.set_xlim(xlim if xlim else (df.Omega.min(), df.Omega.max()))

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return output

def reciprocal_space_map(filename, verbose=False, threshold=10,
                         plot=False, mode='TwoThetaOmega', label_size=20, 
                         ylim=None, xlim=None, title='Reciprocal Space Map'):
    """
    Generate and optionally plot a reciprocal space map from the given file.

    Parameters
    ----------
    filename : str
        The path to the file containing the data to be plotted.
    verbose : bool, optional
        If True, print offset and azimuth angles. Default is False.
    threshold : int, optional
        Minimum intensity value to display. Default is 10.
    plot : bool or str, optional
        If 'contour', plot a contour map. If False, plot an image map. Default is False.
    mode : str, optional
        The scan mode to use for indexing (e.g., 'TwoThetaOmega'). Default is 'TwoThetaOmega'.
    label_size : int, optional
        The font size for plot labels. Default is 20.
    ylim : tuple, optional
        The y-axis limits for the plot. Default is None.
    xlim : tuple, optional
        The x-axis limits for the plot. Default is None.
    title : str, optional
        The title of the plot. Default is 'Reciprocal Space Map'.

    Returns
    -------
    None
    """
    # Read the data from the Rigaku RAS file
    RAS_file = xu.io.rigaku_ras.RASFile(filename)
    RAS_file.Read()

    # Calculate offset and azimuth angles
    Rx = RAS_file.scan.init_mopo['Rx']
    Ry = RAS_file.scan.init_mopo['Ry']
    omega_x = np.deg2rad(Rx)
    omega_y = np.deg2rad(Ry)
    off_angle = np.rad2deg(np.arcsin(np.sqrt(np.sin(omega_x)**2 + np.sin(omega_y)**2)))
    azimuth_angle = np.rad2deg(np.arctan2(np.sin(omega_y), np.sin(omega_x)))

    if verbose:
        print(f"Offset Angle: {off_angle:.2f}, Azimuth Angle: {azimuth_angle:.2f}")
    
    # Extract and process scan data
    list_series = []
    for scan_data in RAS_file.scans[len(RAS_file.scans) // 2:]:
        df = pd.DataFrame(scan_data.data).set_index(mode)
        temp = df[['int']].rename(
            columns={'int': float(scan_data.init_mopo['Omega'])}
        ).copy()
        list_series.append(temp)
    
    df = pd.concat(list_series, axis=1)
    df.columns = df.columns.astype(float)
    df.columns = (df.columns - np.array(df.columns).mean())
    df.index = pd.to_numeric(df.index, errors='coerce')

    # Mask intensity values below the threshold
    masked_values = np.ma.masked_less(df.values, threshold)

    # Plot the reciprocal space map
    extent = (df.columns.min(), df.columns.max(), df.index.min(), df.index.max())
    log_norm = colors.LogNorm(vmin=threshold, vmax=df.values.max())

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8)) 

    if plot == 'contour':
        # Plot a contour map
        X, Y = np.meshgrid(df.columns, df.index)
        masked_values = np.ma.masked_invalid(df.values)
        levels = np.logspace(np.log10(threshold), 
                             np.floor(np.log10(masked_values.max())), num=10)
        plt.contour(X, Y, masked_values, levels=levels, cmap='jet', norm=log_norm)
    else:
        # Plot an image map
        im = plt.imshow(df.values, extent=extent, aspect='auto', origin='lower',
                        interpolation='antialiased', cmap='jet', norm=log_norm)
        cbar = plt.colorbar(im, pad=0.01, aspect=10)
        cbar.set_label('Intensity (cps)', fontsize=label_size)
        cbar.ax.tick_params(labelsize=label_size)
        cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel(r'$\mathrm{\Delta\omega\ (\degree)}$', fontsize=label_size)
    plt.ylabel(r'$\mathrm{2\theta/\omega\ (\degree)}$', fontsize=label_size)
    plt.title(title, fontsize=label_size)

    # Configure axis ticks and limits
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    ax.tick_params(axis='both', which='minor', labelsize=label_size - 2)

    # Thicken only x and y-axis lines and remove the rest of the border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.xlim(xlim if xlim else extent[0], extent[1])
    plt.ylim(ylim if ylim else (df.index.min(), df.index.max()))

    plt.tight_layout()
    plt.show()
    plt.close()
