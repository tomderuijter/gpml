import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np


def add_background_map(ax, res='110m', alpha=0.5):
    ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'land', res,
            edgecolor='face',
            facecolor='white',
            # facecolor=cfeature.COLORS['land'],
            zorder=-1,
            alpha=alpha
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'ocean', res,
            edgecolor='face',
            facecolor='white',
            # facecolor=cfeature.COLORS['water'],
            zorder=2,
            alpha=alpha
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'coastline', res,
            edgecolor='black',
            facecolor='None',
            zorder=-1,
            alpha=alpha
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_boundary_lines_land', res,
            edgecolor='black',
            facecolor='None',
            linestyle=':',
            zorder=-1,
            alpha=alpha
    ))


def plot_area(dataset, request, area_prediction):
    bot_lat, top_lat, left_lon, right_lon = request['predict_area']

    plt.figure(figsize=(20, 15))

    # Plot stations
    sp = plt.scatter(
        dataset.longitude, dataset.latitude,
        s=10, c=dataset['TT2m_error'],
        edgecolor='face',
        vmin=-5, vmax=5,
    )

    # Contours
    contour_handle = plt.contour(
        area_prediction,
        np.arange(-5, 5, 1),
        antialiased=True,
        extent=(left_lon, right_lon, top_lat, bot_lat),
        zorder=999,
        alpha=0.5
    )
    plt.clabel(contour_handle, fontsize=11)

    # Color bar
    cb = plt.colorbar(sp)
    cb.set_label('Temperature error')
    plt.show()


def plt_matrix(mat, title=""):
    f = plt.figure(figsize=(12,12))
    cl = plt.imshow(mat)
    plt.colorbar(cl)
    plt.title(title)
    plt.show()


def plot_prediction_distribution(prediction):
    plt.figure()
    plt.title("Prediction distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Temperature error")
    prediction_vec = prediction.reshape(prediction.size, 1)
    prediction_vec = prediction_vec[~np.isnan(prediction_vec)]
    bins = np.arange(-5, 5, 0.1)
    counts, _, _ = plt.hist(prediction_vec, bins)

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(bin_centers, counts, '-b')
    plt.show()
