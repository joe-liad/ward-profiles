# Display settings
# Quick snippet to adjust luminance of a given color
# (for manual gradients etc)
def adjust_luma(col, luma):
    import colorsys
    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(col))
    cl = colorsys.hls_to_rgb(c[0], c[1] * luma, c[2])
    return matplotlib.colors.to_hex(cl)


# Shared variables to control visual appearance.
dpi = 100
colors = [
    "#1f77b4",  # blue
    "#ff7f03",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#7f7f7f",  # gray
]
# Standard line/dot color
plot_color = colors[0]  # blue
# Colors for common reference points
reference_color = "#bbb"  #'darkgray' # colors[5] # gray
# Colors when highlighting one among many
foreground_color = colors[0]
# background_color = adjust_luma(foreground_color, 1.8)
background_color = "lightgray"
# Custom palettes: don't start at white
min_cmap_idx = 0.3
blues = matplotlib.colors.ListedColormap(
    matplotlib.cm.get_cmap("Blues")(np.linspace(min_cmap_idx, 1, 256))
)
blues_r = matplotlib.colors.ListedColormap(
    matplotlib.cm.get_cmap("Blues_r")(np.linspace(0, 1 - min_cmap_idx, 256))
)
reds = matplotlib.colors.ListedColormap(
    matplotlib.cm.get_cmap("Reds")(np.linspace(min_cmap_idx, 1, 256))
)

# Plot types
# Generic plot functions.
def scatter_plot(
    d,
    ward,
    d_refs=None,
    ref_names=["London", "England"],
    index_col="Name",
    ref_index_col="Name",
    flip_axes=False,
):
    """Just the data, tick labels & legend, no title or axis labels

    Args:
        d (_type_): _description_
        ward (_type_): _description_
        d_refs (_type_, optional): _description_. Defaults to None.
        ref_names (list, optional): _description_. Defaults to ['London', 'England'].
        index_col (str, optional): _description_. Defaults to 'Name'.
        ref_index_col (str, optional): _description_. Defaults to 'Name'.
        flip_axes (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: set of legend handles
    """

    # Lewisham
    data = d.set_index(index_col)
    names = list(reversed(data.index))
    segments = data.columns

    # Plot
    for name in names:
        values = data.loc[name][segments].values
        locations = range(len(values))
        if flip_axes:  # Flip our use of x and y axes?
            locations, values = values, locations
        if name == ward:
            plt.scatter(
                locations, values, color=foreground_color, s=10 * 10, zorder=100
            )
        else:
            plt.scatter(locations, values, color=background_color, zorder=0)

    # Custom legend
    legend_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            label=ward,
            marker="o",
            color="w",
            markerfacecolor=foreground_color,
            markersize=10,
        ),
        matplotlib.lines.Line2D(
            [0],
            [0],
            label="Other Lewisham wards",
            marker="o",
            color="w",
            markerfacecolor=background_color,
        ),
    ]

    # Reference geographies
    if d_refs is not None:
        refs = d_refs.set_index(index_col)

        for idx, name in enumerate(ref_names):
            # Plot
            values = refs.loc[name][segments].values
            locations = range(len(values))
            if flip_axes:  # Flip our use of x and y axes?
                locations, values = values, locations
            color = colors[1 + idx]
            plt.scatter(locations, values, color=color, marker="x", zorder=50)
            # Legend
            legend_handles += [
                matplotlib.lines.Line2D(
                    [0], [0], label=name, marker="x", color="w", markeredgecolor=color
                ),
            ]
    if flip_axes:
        plt.yticks(range(len(segments)), segments)
    else:
        plt.xticks(range(len(segments)), segments)

    return legend_handles


def barh_plot(
    d,
    ward,
    d_refs,
    ref_names=["London", "England"],
    index_col="Name",
    ref_index_col="Name",
):
    """ Just the data, tick labels & legend, no title or axis labels

    Args:
        d (_type_): _description_
        ward (_type_): _description_
        d_refs (_type_): _description_
        ref_names (list, optional): _description_. Defaults to ['London', 'England'].
        index_col (str, optional): _description_. Defaults to 'Name'.
        ref_index_col (str, optional): _description_. Defaults to 'Name'.

    Returns:
        _type_: set of legend handles.
    """

    # Lewisham
    data = d.set_index(index_col)
    segments = data.columns
    refs = d_refs.set_index(index_col)
    total_h = 0.8
    h = total_h / (1 + len(ref_names))  # height per bar, to fill the available space
    h0 = -total_h / 2 + h / 2.0  # initial offset for the first bar

    for idx, segment in enumerate(segments):
        value = data[segment].loc[ward]
        plt.barh(idx + h0, value, color=colors[0], height=h)

        # Reference geographies
        for ridx, name in enumerate(ref_names):
            value = refs[segment].loc[name]
            color = colors[1 + ridx]
            plt.barh(idx + h0 + h + ridx * h, value, color=color, height=h)

    # Custom legend
    legend_handles = [
        matplotlib.lines.Line2D([0], [0], label=ward, color=colors[0], lw=8),
    ]
    for ridx, name in enumerate(ref_names):
        legend_handles += [
            matplotlib.lines.Line2D([0], [0], label=name, color=colors[1 + ridx], lw=8),
        ]

    plt.yticks(range(len(segments)), segments)
    plt.gca().invert_yaxis()

    return legend_handles


# Just the data, tick labels & legend, no title or axis labels
# Returns a set of legend handles.
# TODO: refactor this code: use barh_plot as a basis, and add whiskers as decoration
def bar_plot_group_whiskers(
    d,
    ward,
    d_refs,
    ref_names=["London", "England"],
    index_col="Name",
    ref_index_col="Name",
):
    """ Just the data, tick labels & legend, no title or axis labels"""

    # Lewisham
    data = d.set_index(index_col)
    segments = data.columns
    refs = d_refs.set_index(index_col)

    for idx, segment in enumerate(segments):
        main = data[segment].loc[ward]
        all = data[segment]
        others = data[segment].drop(ward)
        # locations = np.arange(len(values)) + 0.0

        plt.barh(idx - 0.3, main, color=colors[0], height=0.2)
        plt.barh(idx - 0.1, np.mean(all), color=colors[1], height=0.2)
        plt.plot(
            [np.min(all), np.max(all)],
            [idx - 0.1, idx - 0.1],
            "-",
            marker="|",
            color=colors[0],  #'#333333',
            lw=1,
        )
        # Reference geographies
        for ridx, name in enumerate(ref_names):
            value = refs[segment].loc[name]
            color = colors[2 + ridx]
            plt.barh(idx + 0.1 + ridx * 0.2, value, color=color, height=0.2)

    # Custom legend
    legend_handles = [
        matplotlib.lines.Line2D([0], [0], label=ward, color=colors[0], lw=8),
        matplotlib.lines.Line2D(
            [0], [0], label="Lewisham (average and range)", color=colors[1], lw=8
        ),
    ]
    for ridx, name in enumerate(ref_names):
        legend_handles += [
            matplotlib.lines.Line2D([0], [0], label=name, color=colors[2 + ridx], lw=8),
        ]

    plt.yticks(range(len(segments)), segments)
    plt.gca().invert_yaxis()

    return legend_handles


# Data summaries
def summary_population(totals, female, ward):
    total = totals.loc[ward].Total
    female_total = female.loc[ward].sum()
    female_share = female_total * 100.0 / total
    return IPython.display.Markdown(
        f"""
  {ward} has an estimated population of **{total:,} residents**.

  Of these, **{female_total:,} ({female_share:.1f}%) are female** residents, 
  and **{total-female_total:,} ({100-female_share:.1f}%) male**.
  """
    )


def summary_english_proficiency(d, pop, ward):
    data = d.set_index("Name").loc[ward]
    en = data["English is main language"]
    high = data["Can speak English well or very well"]
    low = data["Cannot speak English well or at all"]
    low_ref = d["Cannot speak English well or at all"].mean()
    low_pop = int(low * pop.loc[ward] / 100)
    return IPython.display.Markdown(
        f"""
  English is the main language for {en:.1f}% residents in {ward}.

  Of the remaining residents, {high:.1f}% can speak English well or very well.
  
  On the other hand, **{low:.1f}% of residents cannot speak English well or at all**,
  compared to a Lewisham average of {low_ref:.1f}%.
  
  This means an estimated {low_pop:,} people in {ward} may require everyday language support.
  """
    )


def summary_econ_activity(d, ward):
    data = d.set_index("Name").loc[ward]
    unemployed_rel = d["Unemployed"].mean()
    unemployed = data["Unemployed"]
    inactive = data["Inactive"]
    return IPython.display.Markdown(
        f"""
  **The unemployment rate in {ward} is {unemployed:.1f}%** among residents aged 16-74, 
  compared to a Lewisham average of {unemployed_rel:.1f}%.
  
  {inactive:.1f}% of residents are not economically active, 
  for example because they are in education, looking after home or family, 
  long-term sick or disabled, or in retirement.
  """
    )


def summary_universal_credit(d, ward, date):
    data = d.set_index("Name").loc[ward]
    households_with_uc = data["Households on Universal Credit"].astype(int)
    return IPython.display.Markdown(
        f"""
  In {ward} as of {date}, {households_with_uc:,} households were receiving Universal Credit.
  """
    )


def summary_imd(imd, ward):
    d = imd.set_index("LSOA11NM")
    deciles = d[d.Name == ward].IMD_decile
    num_lsoas = len(deciles)
    num_1_2 = len(deciles[deciles.isin([1, 2])])

    return IPython.display.Markdown(
        f"""
  Of the {num_lsoas} LSOAs in {ward}, **{'all ' if num_lsoas==num_1_2 else ''}{num_1_2} rank in the bottom 20% of the country** (decile 1 or 2.)

  They are: {", ".join(deciles.keys().values)}.
  """
    )


def summary_educational_attainment(d, ward):
    data = d.set_index("Name").loc[ward]
    no_quals = data["No qualifications"]
    l1 = data["Level 1 qualifications"]
    l2 = data["Level 2 qualifications"]
    apprentice = data["Apprenticeship"]
    l3 = data["Level 3 qualifications"]
    l4 = data["Level 4 qualifications and above"]
    other = data["Other qualifications"]

    return IPython.display.Markdown(
        f"""
  There are [nine qualification levels](https://www.gov.uk/what-different-qualification-levels-mean/list-of-qualification-levels) in England: Entry level and then levels 1-9. 
  
  Levels 1-2 cover GCSE and equivalents, Level 3 covers A level and equivalents, and Levels 4-8 cover higher education equivalents.

  In {ward}, {no_quals:0.0f}% of residents have no qualifications; {l1:0.0f}% have Level 1 qualifications; {l2:0.0f}% have Level 2 qualifications; {apprentice:0.0f}% have Apprenticeship; {l3:0.0f}% have Level 3 qualifications; and {l4:0.0f}% have Level 4 qualifications and above.
  """
    )


# DATA PLOTS
# Bespoke plot functions catering to specific data sets.
#

# Population scatter plot
def plot_population(
    title, pop, ward, pop_refs, ref_names=["LONDON", "ENGLAND"], dpi=dpi
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = scatter_plot(pop, ward, pop_refs, ref_names)
    plt.legend(handles=legend_handles, loc="upper right")

    plt.xlabel("Age bracket")
    plt.ylabel("Population (%)")


# Eng. prof. bar & whiskers plot
def plot_english_proficiency(
    title, d, ward, d_refs, ref_names=["London", "England"], index_col="Name", dpi=dpi
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = bar_plot_group_whiskers(
        d,
        ward,
        d_refs,
        ref_names=ref_names,
        colors=[
            "#1f77b4",  # darkblue
            "#90c6ec",  # lightblue
            "#bbb",
            "#ddd",
        ],
    )

    plt.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(1, 0.5))

    plt.xlabel("Residents aged 16-74 (%)")
    plt.yticks(rotation=0, ha="right")


# Low english proficiency map
def map_low_english_proficiency(title, d, ward, cmap=blues, dpi=dpi):
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=dpi)
    plt.title(f"{title} in {ward}")
    d.plot(
        ax=ax,
        column="Cannot speak English well or at all",
        legend=True,
        cmap=cmap,
        # mapclassify parameters for breaks
        # See https://pysal.org/mapclassify/api.html
        scheme="NaturalBreaks",
        classification_kwds={
            "k": 4,
        },
        legend_kwds={
            # 'labels': bin_labels,
            "title": "Population speaking little\nor no English (%)"
        },
        # legend_title='Population (%)',
        # legend_labels=lsoa_imd.IMD_decile
    )
    lsoa11.plot(ax=ax, facecolor="none", linewidth=0.3, edgecolor="white")
    wd22.plot(ax=ax, facecolor="none", linewidth=1, edgecolor="white")
    wd22[wd22.WD22NM_proposed == ward].plot(
        ax=ax, facecolor="none", linewidth=3, edgecolor="black"
    )
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    # plt.legend()


# Main Language bar plots
def plot_languages(
    title,
    d,
    ward,
    num_entries=15,
    skip_first=True,
    #  d_refs, ref_names=['Lewisham', 'England'],
    index_col="Name",
    dpi=dpi,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    plt.title(f"{title} in {ward}")

    # Select the subset of top n languages in Lewisham
    languages_ranked = list(
        main_language.drop(columns=index_col).mean().sort_values(ascending=False).keys()
    )
    if skip_first:  # Exclude English?
        columns = languages_ranked[1 : num_entries + 1]
    else:
        columns = languages_ranked[:num_entries]
    data = d[[index_col] + columns]

    # Quick hacks to get this plot type to work without proper reference data...
    # TODO: refactor the bar plots once we have a better sense of what we need.

    # Construct a Lewisham reference: ward averages
    d_ref = main_language.drop(columns="Name").mean().to_frame().transpose()
    d_ref[index_col] = "Lewisham"

    # Plot these
    legend_handles = barh_plot(
        data,
        ward,
        d_ref,
        ["Lewisham"],  # d_refs, ref_names=ref_names,
        colors=[
            plot_color,  #'#1f77b4', # darkblue
            # reference_color, #'#90c6ec', # lightblue
            "#ccc",
        ],
    )

    plt.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(1, 0.5))

    plt.xlabel("Residents older than 3 (%)")
    plt.yticks(rotation=0, ha="right")


# Educational attainment scatter plot
def plot_educational_attainment(
    title, d, ward, d_refs, ref_names=["London", "England"]
):

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
    plt.title(f"{title} in {ward}")

    legend_handles = scatter_plot(
        d,
        ward,
        d_refs,
        ref_names,
        index_col="Name",
        ref_index_col="geography",
        flip_axes=True,
    )
    plt.legend(handles=legend_handles, loc="lower right")
    plt.xlabel("Residents older than 16 (%)")
    plt.ylabel("Qualification")


def plot_econ_activity(
    title, d, ward, d_refs, ref_names=["London", "England"], index_col="Name", dpi=dpi
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = bar_plot_group_whiskers(
        d,
        ward,
        d_refs,
        ref_names=ref_names,
        colors=[
            "#1f77b4",  # darkblue
            "#90c6ec",  # lightblue
            # plot_color,
            # '#888', # darkest gray
            "#bbb",
            "#ddd",
        ],
    )

    plt.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(1, 0.5))

    plt.xlabel("Residents aged 16-74 (%)")
    plt.yticks(rotation=0, ha="right")


# universal credit scatter plot
def plot_universal_credit(title, d, ward):

    fig, ax = plt.subplots(1, 1, figsize=(8, 2), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = scatter_plot(
        d, ward, index_col="Name", ref_index_col="geography", flip_axes=True
    )
    plt.legend(handles=legend_handles, loc="lower right")
    plt.xlabel("Percentage of households receiving Universal Credit")
    plt.yticks([])


# UC map
def map_universal_credit(title, d_oa, ward, cmap=blues, dpi=dpi):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=dpi)
    plt.title(f"{title} in {ward}")
    d_oa.plot(
        ax=ax,
        column="Households on Universal Credit",
        legend=True,
        cmap=cmap,
        # mapclassify parameters for breaks
        # See https://pysal.org/mapclassify/api.html
        scheme="NaturalBreaks",
        classification_kwds={"k": 4},
        legend_kwds={
            "title": "Number of households per\nOutput Area receiving\nUniversal Credit"
        },
    )
    wd22.plot(ax=ax, facecolor="none", linewidth=1, edgecolor="white")
    wd22[wd22.WD22NM_proposed == ward].plot(
        ax=ax, facecolor="none", linewidth=3, edgecolor="black"
    )
    plt.box(False)
    plt.xticks([])
    plt.yticks([])


# Accommodation type scatter plot
def plot_accommodation_type_scatter(
    title,
    accommodation_type,
    ward,
    accommodation_type_refs,
    ref_names=["London", "England"],
    flip_axes=True,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = scatter_plot(
        accommodation_type,
        ward,
        accommodation_type_refs,
        ref_names=ref_names,
        flip_axes=flip_axes,
    )
    plt.legend(handles=legend_handles, loc="best")

    plt.xlabel("Households (%)")
    plt.yticks(rotation=0, ha="right")
    plt.gca().margins(y=0.2)
    plt.gca().invert_yaxis()


# Accommodation type bar plot with whiskers for other wards
def plot_accommodation_type_bar(
    title, d, ward, d_refs, ref_names=["London", "England"], index_col="Name", dpi=dpi
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = bar_plot_group_whiskers(
        d,
        ward,
        d_refs,
        ref_names=ref_names,
        colors=[
            "#1f77b4",  # darkblue
            "#90c6ec",  # lightblue
            # plot_color,
            # '#888', # darkest gray
            "#bbb",
            "#ddd",
        ],
    )

    plt.legend(handles=legend_handles, loc="lower right")
    plt.xlabel("Households (%)")
    plt.yticks(rotation=0, ha="right")


# Tenure households scatter plot
def plot_tenure_households_scatter(
    title,
    tenure_households,
    ward,
    tenure_households_refs,
    ref_names=["London", "England"],
    flip_axes=True,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=150)
    plt.title(f"{title} in {ward}")

    legend_handles = scatter_plot(
        tenure_households,
        ward,
        tenure_households_refs,
        ref_names=ref_names,
        flip_axes=flip_axes,
    )
    plt.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(1, 0.5))

    plt.xlabel("Households (%)")
    plt.yticks(rotation=0, ha="right")
    plt.gca().margins(y=0.2)
    plt.gca().invert_yaxis()


# Tenure households bar plot with whiskers for other wards
def plot_tenure_households_bar(
    title, d, ward, d_refs, ref_names=["London", "England"], index_col="Name", dpi=dpi
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = bar_plot_group_whiskers(
        d,
        ward,
        d_refs,
        ref_names=ref_names,
        colors=[
            "#1f77b4",  # darkblue
            "#90c6ec",  # lightblue
            # plot_color,
            # '#888', # darkest gray
            "#bbb",
            "#ddd",
        ],
    )

    plt.legend(handles=legend_handles, loc="lower right")
    plt.xlabel("Households (%)")
    plt.yticks(rotation=0, ha="right")


# Central heating scatter plot
def plot_central_heating_scatter(
    title,
    central_heating,
    ward,
    central_heating_refs,
    ref_names=["London", "England"],
    flip_axes=True,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = scatter_plot(
        central_heating,
        ward,
        central_heating_refs,
        ref_names=ref_names,
        flip_axes=flip_axes,
    )
    plt.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(1, 0.5))
    # plt.legend(handles=legend_handles, loc="best")

    plt.xlabel("Households (%)")
    plt.yticks(rotation=0, ha="right")
    plt.gca().margins(y=0.2)
    plt.gca().invert_yaxis()


# Central heating bar plot with whiskers for other wards
def plot_central_heating_bar(
    title, d, ward, d_refs, ref_names=["London", "England"], index_col="Name", dpi=dpi
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = bar_plot_group_whiskers(
        d,
        ward,
        d_refs,
        ref_names=ref_names,
        colors=[
            "#1f77b4",  # darkblue
            "#90c6ec",  # lightblue
            # plot_color,
            # '#888', # darkest gray
            "#bbb",
            "#ddd",
        ],
    )

    plt.legend(handles=legend_handles, loc="upper right")
    plt.xlabel("Households (%)")
    plt.yticks(rotation=0, ha="right")


# Car van availability scatter plot
def plot_car_van_availability_scatter(
    title,
    car_van_availability,
    ward,
    car_van_availability_refs,
    ref_names=["London", "England"],
    flip_axes=True,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = scatter_plot(
        car_van_availability,
        ward,
        car_van_availability_refs,
        ref_names=ref_names,
        flip_axes=flip_axes,
    )
    plt.legend(handles=legend_handles, loc="lower right")

    plt.xlabel("Households (%)")
    plt.yticks(rotation=0, ha="right")
    plt.gca().margins(y=0.2)
    plt.gca().invert_yaxis()


# Car and van availability bar plot with whiskers for other wards
def plot_car_van_availability_bar(
    title, d, ward, d_refs, ref_names=["London", "England"], index_col="Name", dpi=dpi
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    plt.title(f"{title} in {ward}")

    legend_handles = bar_plot_group_whiskers(
        d,
        ward,
        d_refs,
        ref_names=ref_names,
        colors=[
            "#1f77b4",  # darkblue
            "#90c6ec",  # lightblue
            "#bbb",
            "#ddd",
        ],
    )

    plt.legend(handles=legend_handles, loc="best")
    plt.xlabel("Households (%)")
    plt.yticks(rotation=0, ha="right")


# Deprivation map
def map_imd(title, imd_lsoa, ward, cmap=blues_r, dpi=dpi):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=dpi)
    plt.title(f"{title} in {ward}")
    bins = range(2, 12, 2)  # 2..10 in steps of 2: upper bounds per bin
    bin_labels = [f"{v-1}-{v}" for v in bins]
    bin_labels[0] = f"{bin_labels[0]} (most deprived)"
    bin_labels[-1] = f"{bin_labels[-1]} (least deprived)"
    imd_lsoa.plot(
        ax=ax,
        column="IMD_decile",
        legend=True,
        cmap=cmap,
        # mapclassify parameters for breaks
        # See https://pysal.org/mapclassify/api.html
        scheme="UserDefined",
        classification_kwds={"bins": bins},
        legend_kwds={"labels": bin_labels, "title": "IMD Decile"}
    )
    lsoa11.plot(ax=ax, facecolor="none", linewidth=0.3, edgecolor="white")
    wd22.plot(ax=ax, facecolor="none", linewidth=1, edgecolor="white")
    wd22[wd22.WD22NM_proposed == ward].plot(
        ax=ax, facecolor="none", linewidth=3, edgecolor="black"
    )
    plt.box(False)
    plt.xticks([])
    plt.yticks([])


# Deprivation histogram: proportions of deciles, overlaid with Lewisham
def plot_imd_share(title, d, ward, index_col="Name", dpi=dpi):
    plt.figure(figsize=(8, 3), dpi=dpi)
    plt.title(f"{title} in {ward}")

    with sns.axes_style("ticks"):
        dw = d[d.Name == ward]
        dw.IMD_decile.hist(
            range=(1, 11),  # zorder=0,
            weights=np.ones(len(dw)) / len(dw) * 100,
            color=plot_color,
            width=0.6,
            rwidth=0.5,  # alpha=0.8,
            align="mid",
            grid=False,
            label=ward,
        )
        d.IMD_decile.hist(
            range=(1, 11),
            zorder=-1,
            weights=np.ones(len(d)) / len(d) * 100,
            color=reference_color,
            width=0.6,
            rwidth=1,  # alpha=0.8,
            # linewidth=0,
            align="mid",
            grid=False,
            label="Lewisham",
        )
        bins = range(1, 11)
        bin_labels = [f"{v}" for v in bins]
        bin_labels[0] = f"{bin_labels[0]}\n(most deprived)"
        bin_labels[-1] = f"{bin_labels[-1]}\n(least deprived)"
        plt.xticks(np.arange(1, len(bin_labels) + 1) + 0.5, bin_labels)

    plt.legend(loc="upper right")

    plt.xlabel("IMD Decile")
    plt.ylabel("Share of LSOAs (%)")

