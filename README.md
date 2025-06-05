# Pokemon Data Wrangling Pipeline

## Pipeline Overview

My pipeline runs a bunch of custom transformers in order. Most of what it does is convert non-numeric columns to numeric ones, and it also takes care of any missing data.

Here’s a quick visual of the pipeline:

![Pokemon Pipeline Screenshot](screenshot.png)


## Pipeline Steps

The pipeline is called `pokemon_transformer` and has these steps:

1. **`map_type_1` (CustomMappingTransformer):** Converts the main Pokemon type (`Type_1`) into numbers by mapping each type to its own integer.
2. **`map_type_2` (CustomMappingTransformer):** Same idea for `Type_2`, including ‘None’ for any missing values.
3. **`map_egg1` (CustomMappingTransformer):** Turns the primary egg group (`Egg_Group_1`) into integers.
4. **`map_egg2` (CustomMappingTransformer):** Same as above but for `Egg_Group_2`, again with ‘None’ for missing.
5. **`map_mega` (CustomMappingTransformer):** Changes the `hasMegaEvolution` column from True/False to 1/0.


## Random State
121