# House-prices-Advanced-regression-techniques

👩‍💻 Author: Barbara Sánchez (@bsanchezv)

## 📖 Data Dictionary

This dataset contains detailed information about residential properties in Ames, Iowa. Below is a brief version of what you'll find in the data description file:

- **SalePrice** - the property's sale price in dollars. This is the target variable that you're trying to predict.
- **MSSubClass** - The building class
- **MSZoning** - The general zoning classification
- **LotFrontage** - Linear feet of street connected to property
- **LotArea** - Lot size in square feet
- **Street** - Type of road access
- **Alley** - Type of alley access
- **LotShape** - General shape of property
- **LandContour** - Flatness of the property
- **Utilities** - Type of utilities available
- **LotConfig** - Lot configuration
- **LandSlope** - Slope of property
- **Neighborhood** - Physical locations within Ames city limits
- **Condition1** - Proximity to main road or railroad
- **Condition2** - Proximity to main road or railroad (if a second is present)
- **BldgType** - Type of dwelling
- **HouseStyle** - Style of dwelling
- **OverallQual** - Overall material and finish quality
- **OverallCond** - Overall condition rating
- **YearBuilt** - Original construction date
- **YearRemodAdd** - Remodel date
- **RoofStyle** - Type of roof
- **RoofMatl** - Roof material
- **Exterior1st** - Exterior covering on house
- **Exterior2nd** - Exterior covering on house (if more than one material)
- **MasVnrType** - Masonry veneer type
- **MasVnrArea** - Masonry veneer area in square feet
- **ExterQual** - Exterior material quality
- **ExterCond** - Present condition of the material on the exterior
- **Foundation** - Type of foundation
- **BsmtQual** - Height of the basement
- **BsmtCond** - General condition of the basement
- **BsmtExposure** - Walkout or garden level basement walls
- **BsmtFinType1** - Quality of basement finished area
- **BsmtFinSF1** - Type 1 finished square feet
- **BsmtFinType2** - Quality of second finished area (if present)
- **BsmtFinSF2** - Type 2 finished square feet
- **BsmtUnfSF** - Unfinished square feet of basement area
- **TotalBsmtSF** - Total square feet of basement area
- **Heating** - Type of heating
- **HeatingQC** - Heating quality and condition
- **CentralAir** - Central air conditioning
- **Electrical** - Electrical system
- **1stFlrSF** - First Floor square feet
- **2ndFlrSF** - Second floor square feet
- **LowQualFinSF** - Low quality finished square feet (all floors)
- **GrLivArea** - Above grade (ground) living area square feet
- **BsmtFullBath** - Basement full bathrooms
- **BsmtHalfBath** - Basement half bathrooms
- **FullBath** - Full bathrooms above grade
- **HalfBath** - Half baths above grade
- **Bedroom** - Number of bedrooms above basement level
- **Kitchen** - Number of kitchens
- **KitchenQual** - Kitchen quality
- **TotRmsAbvGrd** - Total rooms above grade (does not include bathrooms)
- **Functional** - Home functionality rating
- **Fireplaces** - Number of fireplaces
- **FireplaceQu** - Fireplace quality
- **GarageType** - Garage location
- **GarageYrBlt** - Year garage was built
- **GarageFinish** - Interior finish of the garage
- **GarageCars** - Size of garage in car capacity
- **GarageArea** - Size of garage in square feet
- **GarageQual** - Garage quality
- **GarageCond** - Garage condition
- **PavedDrive** - Paved driveway
- **WoodDeckSF** - Wood deck area in square feet
- **OpenPorchSF** - Open porch area in square feet
- **EnclosedPorch** - Enclosed porch area in square feet
- **3SsnPorch** - Three season porch area in square feet
- **ScreenPorch** - Screen porch area in square feet
- **PoolArea** - Pool area in square feet
- **PoolQC** - Pool quality
- **Fence** - Fence quality
- **MiscFeature** - Miscellaneous feature not covered in other categories
- **MiscVal** - $Value of miscellaneous feature
- **MoSold** - Month Sold
- **YrSold** - Year Sold
- **SaleType** - Type of sale
- **SaleCondition** - Condition of sale

## ✂️ Dataset split

Kaggle provides two CSV files: `train.csv` and `test.csv`. This means that the split of the dataset was already done.

| Data       | Observations | Percentage |
|------------|-------------:|-----------:|
| `train.csv` | 1460 | 50.02% |
| `test.csv` | 1459| 49.98% |

However, in this project, `train.csv` is divided into traning (80%) and validation subsets (20%). So, the final proportion is as follows:

| Data       | Observations | Percentage |
|------------|-------------:|-----------:|
| Training data | 1168 | 40.01% |
| Validation data | 292 | 10.00% |
| Testing data | 1459 | 49.98% |


Why this approach?
- Kaggle already reserves ~50% of the data for testing.
- We use 80/20 split on `train.csv` to ensure enough data for training while keeping a validation set for hyperparameter tuning.
- For better robustness, **K-Fold Cross-Validation** can be applied instead of a single split.

