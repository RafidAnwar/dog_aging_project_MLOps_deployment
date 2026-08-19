import numpy as np
import pandas as pd
import statsmodels.api as sm


DISEASE_COL_MAP = {
    "cancer": "hs_health_conditions_cancer",
    "gastrointestinal": "hs_health_conditions_gastrointestinal",
    "skin": "hs_health_conditions_skin",
    "oral": "hs_health_conditions_oral",
    "neurological": "hs_health_conditions_neurological",
    "kidney": "hs_health_conditions_kidney",
    "liver": "hs_health_conditions_liver",
    "cardiac": "hs_health_conditions_cardiac",
    "orthopedic": "hs_health_conditions_orthopedic",
}

DISEASE_LABEL_MAP = {
    "cancer": "Cancer",
    "gastrointestinal": "Gastrointestinal disease",
    "skin": "Skin disease",
    "oral": "Oral disease",
    "neurological": "Neurological disease",
    "kidney": "Kidney disease",
    "liver": "Liver disease",
    "cardiac": "Cardiac disease",
    "orthopedic": "Orthopedic disease",
}

def get_most_common_disease_for_breed(
    data: pd.DataFrame,
    breed_name: str,
    close_threshold: float = 5.0,
) -> str:
    if breed_name in (None, "", "— select a breed —"):
        return "Select a breed to view the disease summary."

    breed_data = data[data["Breed"] == breed_name].copy()

    if breed_data.empty:
        return "No disease summary is available for this breed."

    disease_percentages: dict[str, float] = {}

    for disease_key, disease_column in DISEASE_COL_MAP.items():
        if disease_column not in breed_data.columns:
            continue

        values = pd.to_numeric(
            breed_data[disease_column],
            errors="coerce",
        ).fillna(0)

        valid_values = values[~values.isin([1, 3])]

        if valid_values.empty:
            continue

        binary_values = valid_values.map(lambda value: 0 if value == 0 else 1)
        disease_percentages[disease_key] = binary_values.mean() * 100

    if not disease_percentages:
        return "No disease summary is available for this breed."

    ranked = sorted(
        disease_percentages.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    first_key, first_percent = ranked[0]

    if len(ranked) == 1:
        return (
            f"Most common recorded disease for **{breed_name}**: "
            f"**{DISEASE_LABEL_MAP[first_key]}** ({first_percent:.1f}%)."
        )

    second_key, second_percent = ranked[1]

    if abs(first_percent - second_percent) <= close_threshold:
        return (
            f"Most common recorded diseases for **{breed_name}**: "
            f"**{DISEASE_LABEL_MAP[first_key]}** ({first_percent:.1f}%) and "
            f"**{DISEASE_LABEL_MAP[second_key]}** ({second_percent:.1f}%)."
        )

    return (
        f"Most common recorded disease for **{breed_name}**: "
        f"**{DISEASE_LABEL_MAP[first_key]}** ({first_percent:.1f}%)."
    )

def _run_logit(
    disease_column: str,
    variable_column: str,
    data: pd.DataFrame,
    disease_name: str,
    hypothesis: str,
    positive_suggestion: str,
    negative_suggestion: str,
) -> tuple[str | None, str | None]:
    try:
        regression_data = pd.DataFrame(
            {
                "exposure_group": data[variable_column],
                "outcome": data[disease_column],
            }
        )

        regression_data = (
            regression_data
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if len(regression_data) < 10:
            return None, None

        if regression_data["outcome"].nunique() < 2:
            return None, None

        if regression_data["exposure_group"].nunique() < 2:
            return None, None

        design_matrix = sm.add_constant(
            regression_data["exposure_group"],
            has_constant="add",
        )

        result = sm.Logit(
            regression_data["outcome"],
            design_matrix,
        ).fit(
            disp=0,
            warn_convergence=False,
            maxiter=200,
        )

        if result.pvalues["exposure_group"] >= 0.05:
            return None, None

        odds_ratio = float(np.exp(result.params["exposure_group"]))

        if not np.isfinite(odds_ratio):
            return None, None

        if odds_ratio > 1:
            finding = (
                f"If {hypothesis}, odds of {disease_name} disease are "
                f"**{(odds_ratio - 1) * 100:.2f}% MORE**."
            )
            return finding, negative_suggestion

        finding = (
            f"If {hypothesis}, odds of {disease_name} disease are "
            f"**{(1 - odds_ratio) * 100:.2f}% LESS**."
        )
        return finding, positive_suggestion

    except Exception:
        return None, None

def run_regression(disease_key, variable_group, data):
    disease_col  = DISEASE_COL_MAP[disease_key]
    disease_name = disease_key
    mask   = (data[disease_col] != 1) & (data[disease_col] != 3)
    df_dis = data[mask].copy()
    df_dis[disease_col] = df_dis[disease_col].map(lambda x: 0 if x == 0 else 1)
    suggestions, findings = [], []

    if variable_group == "diet":
        specs = {
            "df_diet_consistency":                  (lambda s: s.map(lambda x: 0 if x >= 3 else 1),       "dog has a very consistent diet",                                       "Make sure your dog follows a very consistent diet!",                                           "Make sure dog's diet is not consistent!"),
            "df_appetite":                          (lambda s: s.map(lambda x: 0 if x == 1 else 1),       "dog shows good appetite",                                              "Keep an eye on your dog's appetite and make sure it is good.",                                 "Keep an eye on your dog's appetite; being always hungry is bad."),
            "df_primary_diet_component_organic":    (lambda s: s.map(lambda x: 0 if x == False else 1),   "dog's primary diet is organic",                                        "Try to feed organic foods to your dog!",                                                       "Try not to feed organic foods to your dog!"),
            "df_primary_diet_component_grain_free": (lambda s: s.map(lambda x: 0 if x == False else 1),   "dog's diet is grain free",                                             "Try to feed grain free food to your dog!",                                                     "Try not to feed grain free food to your dog!"),
            "df_primary_diet_component_change_recent":(lambda s: s.map(lambda x: 1 if x == True else 0),  "recent changes have been made to the dog's diet",                      "Try to change dog's primary diet component frequently!",                                       "Try not to change your dog's primary diet component frequently!"),
            "df_weight_change_last_year":           (lambda s: s.map(lambda x: 0 if x == 0 else 1),       "dog weight varied from last year",                                     "Keep an eye on your dog's weight! Go to vet if it changes over the year!",                     "Keep an eye on your dog's weight! Go to vet if it does not change over the year!"),
            "df_treats_frequency":                  (lambda s: s.map(lambda x: 0 if x == 0 or x == 4 else 1),"owner does not give treats at least once a day",                   "Try to give your dog treats moderately!",                                                      "Try not to give your dog treats moderately!"),
            "df_infrequent_supplements":            (lambda s: s.map(lambda x: 0 if x == False else 1),   "supplements are given less often than every day",                       "Try to give your dog supplements less often than every day!",                                   "Try to give your dog supplements every day!"),
        }
    elif variable_group == "physical_activity":
        specs = {
            "pa_activity_level":                       (lambda s: s.map(lambda x: 0 if x == 1 else 1),    "dog's lifestyle over the past year has been active",                   "Try to make sure your dog has an active lifestyle!",                                           "Try to make sure your dog does not have an active lifestyle."),
            "pa_physical_games_frequency":             (lambda s: s.map(lambda x: 1 if x <= 3 else 0),    "dog fetches items or plays games involving physical activity",          "Play games such as Frisbee with your dog or ask your dog to fetch items more.",                 "Don't play games such as Frisbee with your dog that often."),
            "pa_avg_activity_intensity":               (lambda s: s.map(lambda x: 0 if x == 1 else 1),    "average activity intensity included jogging and sprinting",             "Try to make sure that your dog does a good amount of jogging and sprinting!",                   "Try to make sure that your dog does not do too much jogging and sprinting!"),
            "pa_swim":                                 (lambda s: s.map(lambda x: 1 if x == True else 0), "dog goes swimming",                                                    "Take your dog swimming more often!",                                                           "Don't take your dog swimming too much!"),
            "pa_moderate_weather_sun_exposure_level":  (lambda s: s.map(lambda x: 1 if x <= 2 else 0),    "dog has good sun exposure on moderate weather days",                   "On moderate days (40–85°F) make sure your dog has exposure to sun.",                           "On moderate days (40–85°F) make sure your dog is in the shade."),
            "pa_moderate_weather_daily_hours_outside": (lambda s: s.map(lambda x: 0 if x in [1, 5] else 1),"dog spends less than 3 hours outdoors on moderate weather days",      "On moderate days make sure your dog spends less than 3 hours outdoors.",                       "On moderate days make sure your dog spends more than 3 hours outdoors."),
            "pa_other_aerobic_activity_frequency":     (lambda s: s.map(lambda x: 1 if x >= 3 else 0),    "dog gets other aerobic activity more than once a week",                "Do more aerobic activities that elevate heart rate more than once a week.",                     "Do aerobic activities that elevate heart rate less than once a week."),
            "pa_on_leash_walk_frequency":              (lambda s: s.map(lambda x: 1 if x >= 3 else 0),    "dog is active on a lead/leash more than once a week",                  "Your dog should be active on a lead/leash more than once a week.",                             "Your dog should be active on a lead/leash less than once a week."),
        }
    elif variable_group == "behavior":
        specs = {
            "db_aggression_level_food_taken_away":          (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog shows No/Rarely aggression when food is taken away by a family member", "Your dog has a lesser chance of disease if it is rarely aggressive when food is taken away.", "Your dog may have a higher chance of disease if it is very aggressive when food is taken away."),
            "db_fear_level_bathed_at_home":                 (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog shows no fear while bathed at home",                                    "Be friendly and gentle while bathing your dog so it does not become afraid.",                 "Try not to be overly friendly while bathing so the dog develops some caution."),
            "db_fear_level_nails_clipped_at_home":          (lambda s: s.map(lambda x: 0 if x >= 3 else 1), "dog shows No/Rare fear/anxiety while getting nails clipped at home",        "Be gentle while clipping nails so your dog shows no fear/anxiety.",                          "It is not so important to be extra gentle while clipping nails."),
            "db_left_alone_restlessness_frequency":         (lambda s: s.map(lambda x: 0 if x >= 3 else 1), "dog shows less/zero restlessness or agitation when left alone",             "Get help and take care of your dog if it shows fear or agitation when left alone.",           "There is nothing to fear if your dog shows agitation when left alone."),
            "db_urinates_alone_frequency":                  (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog does not urinate when left alone",                                      "Get help and take care of your dog if it urinates when left alone.",                          "There is nothing to fear if your dog urinates while left alone."),
            "db_urinates_in_home_frequency":                (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog does not urinate in home against objects",                              "Consult a vet if your dog is urinating in home against objects.",                             "There is nothing to worry about if your dog urinates in home against objects."),
            "db_aggression_level_unknown_aggressive_dog":   (lambda s: s.map(lambda x: 0 if x >= 2 else 1), "dog shows less/no aggression when approached by an unfamiliar dog",         "Teach your dog to be calm when approached by an unfamiliar dog.",                             "There is no risk if your dog shows aggression toward unfamiliar dogs."),
            "db_hyperactive_frequency":                     (lambda s: s.map(lambda x: 1 if x <= 2 else 0), "dog is less hyperactive/restless",                                          "Make sure your dog is not hyperactive, restless, or having trouble settling down.",           "Make sure your dog is hyperactive and restless."),
        }
    elif variable_group == "environment":
        specs = {
            "de_lifetime_residence_count":                    (lambda s: s.map(lambda x: 1 if x <= 3 else 0),    "dog's owner has had fewer than three residences in the dog's lifetime",    "Try not to change residence very frequently during your dog's lifetime.",                      "Try to move to more than three residences."),
            "de_room_or_window_air_conditioning_present":     (lambda s: s.map(lambda x: 0 if x == 0 else 1),    "owner's residence has sufficient air conditioning",                         "Ensure your residence where you live with your dog has good air conditioning.",                "Ensure your residence does not have air conditioning."),
            "de_drinking_water_is_filtered":                  (lambda s: s.map(lambda x: 0 if x == 0 else 1),    "drinking water in owner's residence is not filtered",                       "Filtered water is not a must to ensure your dog's health.",                                   "Ensure your dog has access to filtered drinking water."),
            "de_asbestos_present":                            (lambda s: s.map(lambda x: 0 if x == 0 else 1),    "asbestos is present in owner's residence floor",                            "Try to make sure that your residence floor is asbestos-free.",                                "Try to make sure there is asbestos in your residence floor."),
            "de_floor_types_wood":                            (lambda s: s.map(lambda x: 1 if x == True else 0), "dog owner's residence has a wooden floor",                                  "Try not to have wooden floors in your residence.",                                            "Try to have wooden floors in your residence."),
            "de_routine_toys":                                (lambda s: s.map(lambda x: 1 if x == True else 0), "dog regularly licks, chews, or plays with toys",                            "Ensure your dog regularly licks, chews, and plays with toys.",                                "Try not to let your pet play with chewing/licking toys regularly."),
            "de_neighborhood_has_sidewalks":                  (lambda s: s.map(lambda x: 0 if x > 0 else 1),     "neighbourhood does not have many sidewalks",                                "Try to make sure your neighbourhood does not have many sidewalks.",                           "Having too many sidewalks in a neighbourhood is not a problem."),
            "de_neighborhood_has_parks":                      (lambda s: s.map(lambda x: 1 if x == True else 0), "there are parks or green spaces within half a mile of the owner's home",   "Try to live in a neighbourhood with parks or green spaces within half a mile.",               "It is not important to have parks or green spaces nearby."),
            "de_dogpark":                                     (lambda s: s.map(lambda x: 1 if x == True else 0), "neighbourhood has dog parks",                                               "Try to live in a neighbourhood that has dog parks.",                                          "It is not necessary to have dog parks in the neighbourhood."),
            "de_recreational_spaces":                         (lambda s: s.map(lambda x: 0 if x == False else 1),"there are recreational spaces in the neighbourhood",                        "Try to choose a neighbourhood that has recreational spaces.",                                 "Try not to choose a neighbourhood that has recreational spaces."),
            "de_sitter_or_daycare":                           (lambda s: s.map(lambda x: 0 if x == False else 1),"dog has been taken to a daycare centre",                                    "Try to take your pet to a daycare centre.",                                                   "It is not necessary to take your pet to a daycare centre."),
            "de_traffic_noise_in_home_frequency":             (lambda s: s.map(lambda x: 1 if x > 1 else 0),     "traffic noise in the home is more frequent",                                "It is not an issue if there is frequent traffic noise in the neighbourhood.",                  "Try to avoid neighbourhoods where traffic noise is more frequent."),
        }
    else:
        specs = {}

    for col, (transform, hypo, psugg, nsugg) in specs.items():
        if col not in df_dis.columns:
            continue
        df_dis[col] = transform(df_dis[col])
        finding, sugg = _run_logit(disease_col, col, df_dis, disease_name, hypo, psugg, nsugg)
        if finding:
            findings.append(finding)
            suggestions.append(sugg)

    return suggestions, findings