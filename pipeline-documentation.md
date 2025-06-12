# pokemon_transformer Pipeline

![Pokemon Pipeline Screenshot](ss.png)

## What this pipeline does

- **Target encode:**
  Encodes `Type_1` and `Type_2` using target encoding.

- **Handle outliers (Tukey):**
  Clips outliers in  `Attack`, `Defense`, `Speed`, `HP`, `Sp_Atk`, and `Sp_Def` using the inner Tukey method.

- **Scale numeric features:**
  Robustly scales `Attack`, `Defense`, `Speed`, `HP`, `Sp_Atk`, and `Sp_Def`.

- **Impute missing values:**
  Fills in any remaining missing values using KNN (k=5).

### Features used

- `Type_1`
- `Type_2`
- `Attack`
- `Defense`
- `Speed`
- `Sp_Atk`
- `Sp_Def`
- `HP`

**Random state:** `rs = 45`