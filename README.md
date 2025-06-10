# pokemon_transformer Pipeline

![Pokemon Pipeline Screenshot](ss.png)

## What this pipeline does

- **Target encode:**  
  Encodes `Type_1`, `Type_2`, `Egg_Group_1`, and `Egg_Group_2` using target encoding.

- **Handle outliers (Tukey):**  
  Clips outliers in `Total`, `Attack`, `Defense`, `Speed`, and `HP` using the inner Tukey method.

- **Scale numeric features:**  
  Robustly scales `Total`, `Attack`, `Defense`, `Speed`, and `HP`.

- **Impute missing values:**  
  Fills in any remaining missing values using KNN (k=5).

### Features used

- `Total`
- `Type_1`
- `Type_2`
- `Attack`
- `Defense`
- `Speed`
- `Egg_Group_1`
- `Egg_Group_2`
- `HP`

**Random state:** `rs = 178`
