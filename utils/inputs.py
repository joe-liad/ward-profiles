import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import IPython
from liad import colab

def join_gdf_df(left, right, index_left, index_right):
    """join a geodataframe and a dataframe
    Args:
        left (GeoDataFrame): geodataframe
        right (DataFrame): dataframe
        index_left (int): gdf linked column
        index_right (int): df linked column

    Returns:
        GeoDataFrame: geodataframe including imported dataframe records
    """
    if left.index.name != index_left:
        left.reset_index(inplace=True)
        left.set_index(index_left, inplace=True)
    if right.index.name != index_right:
        right.set_index(index_right, inplace=True)        
    ret = left.join(right)
    ret = ret.reset_index().drop(columns=['index'])
    return ret


def explore_metric(
    left,
    right,
    index_left="Name",
    index_right="Name",
    column="Total",
    legend=False,
    tooltip=["Name", "Total", "Forecast_e", "No_of_coun"],
):
    """use the geopandas explore function
    e.g. explore_metric(left=lewisham, right=utils.pop_age_groups, index_left='Name', index_right='Name', column='10-14', legend=True, tooltip=['Name', '10-14', 'Forecast_e','No_of_coun'])

    Args:
        left (GeoDataFrame): a spatial dataframe
        right ((Geo)DataFrame): a (spatial) dataframe
        index_left (str): the name of the index column in left
        index_right (str): the name of the index column in right
        column (str, optional): which column to symbolise. Defaults to 'Total'.
        legend (bool, optional): include legend. Defaults to False.
        tooltip (list, optional): fields for tooltip content. Defaults to ['Name', 'Total', 'Forecast_e','No_of_coun'].
    """
    gdf = join_gdf_df(left, right, index_left, index_right)

    return gdf.explore(
        column=column,
        scheme="BoxPlot",
        legend=legend,
        style_kwds={"stroke": True, "color": "#00b7eb", "weight": 1, "fillOpacity": 1},
        legend_kwds={"caption": column},
        attr='Lewisham Insight & Delivery | Map tiles by <a href="http://stamen.com">Stamen</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> | Map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        highlight=True,
        tooltip=tooltip,
        tiles="Stamen Toner",
    )

# Guess if we are in a colab notebook
import socket
if colab.in_colab():
    from google.colab import drive
    drive.mount("/content/gdrive")
    project_dir = "/content/gdrive/MyDrive/WardProfiles"
else:
    if socket.gethostname() == 'L0159L':
        project_dir = "c:\\Users\\jleach\\code\\ward-profiles"
    else:
        project_dir = "/Volumes/GoogleDrive/My Drive/WardProfiles"
        
# OA geometry
oa11 = gpd.read_file(f"{project_dir}/boundaries/oa/lbl_oa11_20m.gpkg")
lsoa11 = gpd.read_file(f"{project_dir}/boundaries/oa/lbl_lsoa11_20m.gpkg")

# Ward geometry
wd22 = gpd.read_file(f"{project_dir}/boundaries/2022_wards/lbl_wd22_proposed.gpkg")

# Ward names and ward Code->Name lookup
ward_codes_names = pd.read_csv(
    f"{project_dir}/lookups/2022_wards/lbl_wd22_proposed.csv"
)
wards = ward_codes_names.WD22NM_proposed.unique().tolist()
ward_name_lookup = ward_codes_names.set_index(
    "WD22CD_proposed"
).WD22NM_proposed.to_dict()


def add_column_from_lookup(lookup, src_col, dst_col, drop_src_col=False):
    """Generic function to derive a new dataframe column from an existing one with
        the help of a lookup.
        This is a closure (it binds some variables to a generic function.)
        This way we can configure different versions for different types of source data.

    Args:
        lookup (_type_): _description_
        src_col (_type_): _description_
        dst_col (_type_): _description_
        drop_src_col (bool, optional): _description_. Defaults to False.
    """

    def add_column(d):
        t = d.copy()
        # Add derived column
        t[dst_col] = t[src_col].map(lambda v: lookup[v])
        # Drop source column?
        if drop_src_col:
            t = t.drop(columns=[src_col])
        return t

    return add_column


# Utility function to add a 'Name' column to dataframes where we have ward codes
add_ward_name = add_column_from_lookup(
    ward_name_lookup, "WD22CD_proposed", "Name", drop_src_col=True
)

# Populations -- note we add a 'Name' column:
# Absolute totals:
pop_totals = add_ward_name(
    pd.read_csv(f"{project_dir}/population/lbl_pop_est_2020_5ybins_all_wd22.csv")
)
pop_totals = pop_totals.set_index("Name")[["Total"]]
pop_female_totals = add_ward_name(
    pd.read_csv(f"{project_dir}/population/lbl_pop_est_2020_5ybins_female_wd22.csv")
)
pop_female_totals = pop_female_totals.set_index("Name")[["Total"]]
pop_male_totals = add_ward_name(
    pd.read_csv(f"{project_dir}/population/lbl_pop_est_2020_5ybins_male_wd22.csv")
)
pop_male_totals = pop_male_totals.set_index("Name")[["Total"]]

# Relative share by age group and gender:
pop_age_groups = add_ward_name(
    pd.read_csv(f"{project_dir}/population/lbl_pop_est_2020_5ybins_all_share_wd22.csv")
)
pop_female = add_ward_name(
    pd.read_csv(
        f"{project_dir}/population/lbl_pop_est_2020_5ybins_female_share_wd22.csv"
    )
)
pop_male = add_ward_name(
    pd.read_csv(f"{project_dir}/population/lbl_pop_est_2020_5ybins_male_share_wd22.csv")
)

# These already have a 'Name' column
pop_all_refs = pd.read_csv(
    f"{project_dir}/population/references_pop_est_2020_5ybins_all_share.csv"
)
pop_female_refs = pd.read_csv(
    f"{project_dir}/population/references_pop_est_2020_5ybins_female_share.csv"
)
pop_male_refs = pd.read_csv(
    f"{project_dir}/population/references_pop_est_2020_5ybins_male_share.csv"
)

# Languages
english_proficiency = add_ward_name(
    pd.read_csv(
        f"{project_dir}/languages/lbl_english_proficiency_share_coarse_wd22.csv"
    )
)
english_proficiency_refs = pd.read_csv(
    f"{project_dir}/languages/references_english_proficiency_share_coarse.csv"
)
english_proficiency_oa = oa11.merge(
    pd.read_csv(
        f"{project_dir}/languages/lbl_english_proficiency_share_coarse_oa11.csv"
    )
)
main_language = add_ward_name(
    pd.read_csv(f"{project_dir}/languages/lbl_main_language_detailed_share_wd22.csv")
)

main_language_absolute = add_ward_name(
    pd.read_csv(f"{project_dir}/languages/lbl_main_language_detailed_wd22.csv")
)

# SUMMARY STATS
def summary_population(totals, female, ward):
  total = totals.loc[ward].Total
  female_total = female.loc[ward].sum()
  female_share = female_total * 100. / total
  return IPython.display.Markdown(f"""
  {ward} has an estimated population of **{total:,} residents**.

  Of these, **{female_total:,} ({female_share:.1f}%) are female** residents, 
  and **{total-female_total:,} ({100-female_share:.1f}%) male**.
  """)

def summary_english_proficiency(d, pop, ward):
  data = d.set_index('Name').loc[ward]
  en = data['English is main language']  
  high = data['Can speak English well or very well']
  low = data['Cannot speak English well or at all']
  low_ref = d['Cannot speak English well or at all'].mean()
  low_pop = int(low * pop.loc[ward] / 100)
  return IPython.display.Markdown(f"""
  English is the main language for {en:.1f}% residents in {ward}.

  Of the remaining residents, {high:.1f}% can speak English well or very well.
  
  On the other hand, **{low:.1f}% of residents cannot speak English well or at all**,
  compared to a Lewisham average of {low_ref:.1f}%.
  
  This means an estimated {low_pop:,} people in {ward} may require everyday language support.
  """)

def summary_econ_activity(d, ward):
  data = d.set_index('Name').loc[ward]
  unemployed_rel = d['Unemployed'].mean()
  unemployed = data['Unemployed']
  inactive = data['Inactive']  
  return IPython.display.Markdown(f"""
  **The unemployment rate in {ward} is {unemployed:.1f}%** among residents aged 16-74, 
  compared to a Lewisham average of {unemployed_rel:.1f}%.
  
  {inactive:.1f}% of residents are not economically active, 
  for example because they are in education, looking after home or family, 
  long-term sick or disabled, or in retirement.
  """)

def summary_universal_credit(d, ward, date):
  data = d.set_index('Name').loc[ward]
  households_with_uc = data['Households on Universal Credit'].astype(int)
  return IPython.display.Markdown(f"""
  In {ward} as of {date}, {households_with_uc:,} households were receiving Universal Credit.
  """)

def summary_imd(imd, ward):
  d = imd.set_index('LSOA11NM')
  deciles = d[d.Name==ward].IMD_decile
  num_lsoas = len(deciles)
  num_1_2 = len(deciles[deciles.isin([1,2])])
  
  return IPython.display.Markdown(f"""
  Of the {num_lsoas} LSOAs in {ward}, **{'all ' if num_lsoas==num_1_2 else ''}{num_1_2} rank in the bottom 20% of the country** (decile 1 or 2.)

  They are: {", ".join(deciles.keys().values)}.
  """)

def summary_educational_attainment(d, ward):
  data = d.set_index('Name').loc[ward]
  no_quals = data['No qualifications']
  l1 = data['Level 1 qualifications']
  l2 = data['Level 2 qualifications']
  apprentice = data['Apprenticeship']
  l3 = data['Level 3 qualifications']
  l4 = data['Level 4 qualifications and above']
  other = data['Other qualifications']
  
  return IPython.display.Markdown(f"""
  There are [nine qualification levels](https://www.gov.uk/what-different-qualification-levels-mean/list-of-qualification-levels) in England: Entry level and then levels 1-9. 
  
  Levels 1-2 cover GCSE and equivalents, Level 3 covers A level and equivalents, and Levels 4-8 cover higher education equivalents.

  In {ward}, {no_quals:0.0f}% of residents have no qualifications; {l1:0.0f}% have Level 1 qualifications; {l2:0.0f}% have Level 2 qualifications; {apprentice:0.0f}% have Apprenticeship; {l3:0.0f}% have Level 3 qualifications; and {l4:0.0f}% have Level 4 qualifications and above.
  """)

  

# Deprivation
imd = add_ward_name(pd.read_csv(f"{project_dir}/deprivation/lbl_imd_lsoa11.csv"))
imd_lsoa = lsoa11.merge(imd, on="LSOA11CD")

# benefits
benefits = add_ward_name(pd.read_csv(f"{project_dir}/benefits/lbl_benefits_claimants_total_wd22.csv"))
benefits_female = add_ward_name(pd.read_csv(f"{project_dir}/benefits/lbl_benefits_claimants_female_wd22.csv"))
benefits_male   = add_ward_name(pd.read_csv(f"{project_dir}/benefits/lbl_benefits_claimants_male_wd22.csv"))

# country of birth
country_of_birth = add_ward_name(pd.read_csv(f"{project_dir}/country_of_birth/lbl_country_of_birth_wd22.csv"))

# educational_attainment
educational_attainment = add_ward_name(pd.read_csv(f"{project_dir}/educational_attainment/lbl_educational_attainment_wd22.csv"))

# employment
economic_activity = add_ward_name(pd.read_csv(f"{project_dir}/employment/lbl_economic_activity_groups_wd22.csv"))
hours_worked = add_ward_name(pd.read_csv(f"{project_dir}/employment/lbl_hours_worked_wd22.csv"))
occupation_share = add_ward_name(pd.read_csv(f"{project_dir}/employment/lbl_occupation_share_wd22.csv"))
occupation_minor_groups = add_ward_name(pd.read_csv(f"{project_dir}/employment/lbl_occupation_minor_groups_wd22.csv"))

# ethnicity
ethnicity = add_ward_name(pd.read_csv(f"{project_dir}/ethnicity/lbl_ethnicity_groups_wd22.csv"))

# fuel_poverty
fuel_poverty = add_ward_name(pd.read_csv(f"{project_dir}/fuel_poverty/lbl_fp_lsoa11.csv"))

health_unpaid_care = add_ward_name(pd.read_csv(f"{project_dir}/health/lbl_health_and_provision_of_unpaid_care_groups_wd22.csv"))

accommodation_type_groups = add_ward_name(pd.read_csv(f"{project_dir}/housing/lbl_accommodation_type_groups_wd22.csv"))

car_van_availability = add_ward_name(pd.read_csv(f"{project_dir}/housing/lbl_car_van_availability_wd22.csv"))

household_compositions_groups = add_ward_name(pd.read_csv(f"{project_dir}/housing/lbl_household_compositions_groups_wd22.csv"))
occupancy_rating = add_ward_name(pd.read_csv(f"{project_dir}/housing/lbl_occupancy_rating_wd22.csv"))
rooms_bedrooms_central_heating = add_ward_name(pd.read_csv(f"{project_dir}/housing/lbl_rooms_bedrooms_central_heating_wd22.csv"))
# rooms_bedrooms_central_heating_groups = add_ward_name(pd.read_csv(f"{project_dir}/housing/lbl_rooms_bedrooms_central_heating_groups_wd22.csv"))
tenure_households = add_ward_name(pd.read_csv(f"{project_dir}/housing/lbl_tenure_households_wd22.csv"))

religion_groups = add_ward_name(pd.read_csv(f"{project_dir}/religion/lbl_religion_groups_wd22.csv"))