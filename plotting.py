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
    top_lat, bot_lat, left_lon, right_lon = request['predict_area']

    res = '50m'
    alpha = 1.0
    plt.figure(figsize=(20, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.axis('equal')
    add_background_map(ax, res, alpha)

    # Plot stations
    sp = plt.scatter(
        dataset.longitude, dataset.latitude,
        s=10, c=dataset['TT2m_error'],
        edgecolor='face',
        vmin=-5, vmax=5,
    )

    # Plot prediction contours
    cs = plt.contour(
        area_prediction,
        np.arange(-5, 5, 1),
        extent=(left_lon, right_lon, bot_lat, top_lat),
        antialiased=True,
        zorder=999
    )
    plt.clabel(cs, fontsize=11)

    cb = plt.colorbar()
    cb.set_label('Temperature error')
    plt.xlim([left_lon, right_lon])
    plt.ylim([bot_lat, top_lat])
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
    plt.clf()
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(bin_centers, counts, '-b')
    plt.show()
