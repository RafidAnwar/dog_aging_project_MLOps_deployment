
WITH cslb AS (
    SELECT *
    FROM DAP_2021_CSLB_v1
    WHERE rowid IN (
        SELECT MAX(rowid)
        FROM DAP_2021_CSLB_v1
        GROUP BY dog_id
    )
),
breeds AS (
    SELECT dd_breed_pure
    FROM DAP_2021_HLES_dog_owner_v1
    WHERE dd_breed_pure IS NOT NULL
    GROUP BY dd_breed_pure
    HAVING COUNT(*) >= 15
)
SELECT
    o.dog_id,
    d.Breed,
    o.dd_age_years,
    o.dd_spayed_or_neutered,
    o.dd_breed_pure,

    o.df_diet_consistency,
    o.df_appetite,
    o.df_primary_diet_component_organic,
    o.df_primary_diet_component_grain_free,
    o.df_primary_diet_component_change_recent,
    o.df_weight_change_last_year,
    o.df_treats_frequency,
    o.df_infrequent_supplements,

    o.pa_activity_level,
    o.pa_physical_games_frequency,
    o.pa_avg_activity_intensity,
    o.pa_swim,
    o.pa_moderate_weather_sun_exposure_level,
    o.pa_moderate_weather_daily_hours_outside,
    o.pa_other_aerobic_activity_frequency,
    o.pa_on_leash_walk_frequency,

    o.db_aggression_level_food_taken_away,
    o.db_fear_level_bathed_at_home,
    o.db_fear_level_nails_clipped_at_home,
    o.db_left_alone_restlessness_frequency,
    o.db_urinates_alone_frequency,
    o.db_urinates_in_home_frequency,
    o.db_aggression_level_unknown_aggressive_dog,
    o.db_hyperactive_frequency,

    o.de_lifetime_residence_count,
    o.de_room_or_window_air_conditioning_present,
    o.de_drinking_water_is_filtered,
    o.de_asbestos_present,
    o.de_floor_types_wood,
    o.de_routine_toys,
    o.de_neighborhood_has_sidewalks,
    o.de_neighborhood_has_parks,
    o.de_dogpark,
    o.de_recreational_spaces,
    o.de_sitter_or_daycare,
    o.de_traffic_noise_in_home_frequency,

    o.hs_health_conditions_cancer,
    o.hs_health_conditions_gastrointestinal,
    o.hs_health_conditions_skin,
    o.hs_health_conditions_oral,
    o.hs_health_conditions_neurological,
    o.hs_health_conditions_kidney,
    o.hs_health_conditions_liver,
    o.hs_health_conditions_cardiac,
    o.hs_health_conditions_orthopedic,

    o.dd_weight_lbs,
    o.hs_health_conditions_eye,
    o.hs_health_conditions_ear,

    c.cslb_pace,
    c.cslb_stare,
    c.cslb_stuck,
    c.cslb_recognize,
    c.cslb_walk_walls,
    c.cslb_avoid,
    c.cslb_find_food,

    c.cslb_pace_6mo,
    c.cslb_stare_6mo,
    c.cslb_defecate_6mo,
    c.cslb_food_6mo,
    c.cslb_recognize_6mo,
    c.cslb_active_6mo,
    c.cslb_score

FROM DAP_2021_HLES_dog_owner_v1 AS o

JOIN DAP_2021_DogOverview_v1 AS d
    ON o.dog_id = d.dog_id

LEFT JOIN cslb AS c
    ON o.dog_id = c.dog_id

/*
Data pre-processing :
Dog age < 15 years
Dog reproductive status >> Only neutered/spayed
Dog breed minimum 15 samples
*/

WHERE o.dd_age_years < 15
    AND d.Breed <> 'Unknown/Unknown'
    AND o.dd_spayed_or_neutered = 'True'
    AND o.dd_breed_pure IN (
        SELECT dd_breed_pure
        FROM breeds
    );