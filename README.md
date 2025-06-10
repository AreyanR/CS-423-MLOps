![Pokemon Pipeline Screenshot](ss.png)

## Pipeline Description

The pipeline is called `pokemon_transformer`. It performs a series of transformations to prepare the Pokémon dataset for modeling. These steps include mapping, target encoding, outlier handling, scaling, and imputation.

### Pipeline Steps

1. **target_type1 (CustomTargetTransformer):**  
   Applies target encoding to the `Type_1` column based on the `isLegendary` label.

2. **target_type2 (CustomTargetTransformer):**  
   Applies target encoding to the `Type_2` column based on the `isLegendary` label.

3. **target_egg1 (CustomTargetTransformer):**  
   Applies target encoding to the `Egg_Group_1` column based on the `isLegendary` label.

4. **target_egg2 (CustomTargetTransformer):**  
   Applies target encoding to the `Egg_Group_2` column based on the `isLegendary` label.

5. **tukey_total, tukey_attack, tukey_defense, tukey_speed, tukey_hp (CustomTukeyTransformer):**  
   Handles outliers in numeric features by applying inner Tukey fences.

6. **scale_total, scale_attack, scale_defense, scale_speed, scale_hp (CustomRobustTransformer):**  
   Applies robust scaling to numeric features for normalization.

7. **impute (CustomKNNTransformer):**  
   Imputes any remaining missing values using K-Nearest Neighbors.

### Feature Columns Used

- `Total`, `Type_1`, `Type_2`, `Attack`, `Defense`, `Speed`, `Egg_Group_1`, `Egg_Group_2`, `HP`

### Random State

`rs = 178`