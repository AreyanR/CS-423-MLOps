# pokemon_transformer Pipeline

![Pokemon Pipeline Screenshot](ss.png)

## What this pipeline does

## 1. Target Encoding: Type 1  
**Transformer:** `CustomTargetTransformer('Type_1')`  
**Design Choice:** Target encoding for the primary Pokémon type  
**Rationale:**  
- Converts `Type_1` into a numeric value based on the average target (e.g., legendary status)
- Handles many unique types and encodes missing values as the overall mean

---

## 2. Target Encoding: Type 2  
**Transformer:** `CustomTargetTransformer('Type_2')`  
**Design Choice:** Target encoding for the secondary Pokémon type  
**Rationale:**  
- Same method for dual-typed Pokémon
- Encodes missing or None as the global target mean

---

## 3. Outlier Treatment: Attack  
**Transformer:** `CustomTukeyTransformer('Attack', 'inner')`  
**Design Choice:** Tukey outlier clipping with inner fence (1.5×IQR)  
**Rationale:**  
- Reduces moderate outliers in Attack stat  
- Maintains realistic range of values

---

## 4. Outlier Treatment: Defense  
**Transformer:** `CustomTukeyTransformer('Defense', 'inner')`  
**Design Choice:** Tukey outlier clipping with inner fence (1.5×IQR)  
**Rationale:**  
- Prevents rare, extreme Defense values from skewing the feature  
- Preserves the overall defensive distribution

---

## 5. Outlier Treatment: Speed  
**Transformer:** `CustomTukeyTransformer('Speed', 'inner')`  
**Design Choice:** Tukey outlier clipping with inner fence (1.5×IQR)  
**Rationale:**  
- Limits impact of extremely fast/slow Pokémon  
- Keeps distribution fair for modeling

---

## 6. Outlier Treatment: HP  
**Transformer:** `CustomTukeyTransformer('HP', 'inner')`  
**Design Choice:** Tukey outlier clipping with inner fence (1.5×IQR)  
**Rationale:**  
- Controls rare high/low HP stats  
- Retains the real HP range for most Pokémon

---

## 7. Outlier Treatment: Sp_Atk  
**Transformer:** `CustomTukeyTransformer('Sp_Atk', 'inner')`  
**Design Choice:** Tukey outlier clipping with inner fence (1.5×IQR)  
**Rationale:**  
- Prevents rare, extreme Sp_Atk from distorting scaling  
- Preserves useful variance in special attack

---

## 8. Outlier Treatment: Sp_Def  
**Transformer:** `CustomTukeyTransformer('Sp_Def', 'inner')`  
**Design Choice:** Tukey outlier clipping with inner fence (1.5×IQR)  
**Rationale:**  
- Clips moderate outliers in Sp_Def  
- Keeps feature distribution appropriate for modeling

---

## 9. Robust Scaling: Attack  
**Transformer:** `CustomRobustTransformer('Attack')`  
**Design Choice:** Robust scaling using median and IQR  
**Rationale:**  
- Makes Attack less sensitive to outliers after clipping  
- Appropriate for stat distributions that may be skewed

---

## 10. Robust Scaling: Defense  
**Transformer:** `CustomRobustTransformer('Defense')`  
**Design Choice:** Robust scaling using median and IQR  
**Rationale:**  
- Reduces influence of rare high/low Defense values

---

## 11. Robust Scaling: Speed  
**Transformer:** `CustomRobustTransformer('Speed')`  
**Design Choice:** Robust scaling using median and IQR  
**Rationale:**  
- Ensures Speed is on a comparable scale  
- Minimizes impact from remaining outliers

---

## 12. Robust Scaling: HP  
**Transformer:** `CustomRobustTransformer('HP')`  
**Design Choice:** Robust scaling using median and IQR  
**Rationale:**  
- Normalizes HP after outlier treatment  
- Reduces effect of rare extreme HP

---

## 13. Robust Scaling: Sp_Atk  
**Transformer:** `CustomRobustTransformer('Sp_Atk')`  
**Design Choice:** Robust scaling using median and IQR  
**Rationale:**  
- Scales Sp_Atk fairly across Pokémon

---

## 14. Robust Scaling: Sp_Def  
**Transformer:** `CustomRobustTransformer('Sp_Def')`  
**Design Choice:** Robust scaling using median and IQR  
**Rationale:**  
- Handles remaining outliers in Sp_Def  
- Normalizes distribution for modeling

---

## 15. Imputation  
**Transformer:** `CustomKNNTransformer(n_neighbors=5)`  
**Design Choice:** KNN imputation with 5 neighbors  
**Rationale:**  
- Fills missing values using similarities between Pokémon  
- k=5 is a balanced choice for accurate imputation

---

## Pipeline Execution Order Rationale

- Encode categorical features first so stats are numeric
- Treat outliers before scaling to prevent them affecting scaling
- Scale before imputation so KNN distances are meaningful
- Impute last so missing values are filled using all processed features

---

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