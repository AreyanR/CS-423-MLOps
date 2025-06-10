![Pokemon Pipeline Screenshot](ss.png)

## Pipeline Description

The pipeline is called `pokemon_transformer`. It performs a series of transformations to prepare the Pokémon dataset for modeling. These steps include mapping, target encoding, outlier handling, scaling, and imputation.

### Pipeline Steps

1. **map_type_1 (CustomMappingTransformer):**  
   Maps the `Type_1` column to integers.

2. **target_egg1 (CustomTargetTransformer):**  
   Applies target encoding to the `Egg_Group_1` column using the `isLegendary` label.

3. **target_egg2 (CustomTargetTransformer):**  
   Applies target encoding to the `Egg_Group_2` column using the `isLegendary` label.

4. **tukey_total, tukey_attack, tukey_defense, tukey_speed, tukey_hp (CustomTukeyTransformer):**  
   Applies outlier treatment to the numeric features.

5. **scale_total, scale_attack, scale_defense, scale_speed, scale_hp (CustomRobustTransformer):**  
   Applies robust scaling to the same numeric features.

6. **impute (CustomKNNTransformer):**  
   Handles any missing values in the data using K-Nearest Neighbors imputation.

### Random State

rs = 44
