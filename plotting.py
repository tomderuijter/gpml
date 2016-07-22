import cartopy.feature as cfeature


def add_background_map(ax, res='110m', alpha=0.5):
    ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'land', res,
            edgecolor='face',
            facecolor=cfeature.COLORS['land'],
            zorder=-1,
            alpha=alpha
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'ocean', res,
            edgecolor='face',
            facecolor=cfeature.COLORS['water'],
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
    ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'lakes', res,
            edgecolor='face',
            facecolor=cfeature.COLORS['water'],
            zorder=-1,
            alpha=alpha
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
            'physical', 'rivers_lake_centerlines', res,
            edgecolor=cfeature.COLORS['water'],
            facecolor='none',
            zorder=-1,
            alpha=alpha
    ))