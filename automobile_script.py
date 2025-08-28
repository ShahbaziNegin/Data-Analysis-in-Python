# وارد کردن کتابخانه‌های مورد نیاز
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# تعریف نام ستون‌های دیتاست
columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
           'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
           'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
           'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
           'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

# خواندن دیتاست از URL
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
    header=None,
    names=columns,
    na_values="?"
)

# نمایش اطلاعات اولیه دیتاست
print("Display first 5 rows of the dataset:")
print(df.head())

print(df)

print("dataset information:")
print(df.info())

print("\n Statistical Summary:")
print(df.describe(include="all"))

print("\n Missing Values per Column:")
print(df.isnull().sum())

print("dataset size:", df.shape)


# حذف سطرهایی که ستون قیمت مقدار گمشده دارند
df = df[df['price'].notna()]

# مدیریت مقادیر گمشده در ستون‌های عددی و رشته‌ای
num_cols = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm']

for col in num_cols:
    df[col] = df[col].astype(float)
    median_val = df[col].median()
    df[col] = df[col].replace(np.nan, median_val)


df['num-of-doors'] = df['num-of-doors'].replace(np.nan, df['num-of-doors'].mode()[0])

# بررسی مجدد مقادیر گمشده پس از پاک‌سازی
print("Missing Values per Column after Cleaning:")
print(df.isnull().sum())

# خلاصه آماری برای ستون‌های عددی
print("Statistical Summary for Numeric Columns:")
print(df.describe())

# خلاصه آماری برای ستون‌های رشته‌ای
print("\nStatistical Summary for Categorical Columns:")
print(df.describe(include=['object']))



# پرتکرارترین شرکت سازنده در دیتاست
Favorite_make = df['make'].mode()[0]
Favorite_make_perc = (df['make'].value_counts().iloc[0] / df['make'].count()) * 100
print(f"Favorite make: {Favorite_make} ({Favorite_make_perc}%)")


# پرتکرارترین نوع بدنه در دیتاست
Favorite_bodystyle = df.describe(include='object').loc['top','body-style']
print("Favorite_bodystyle=",Favorite_bodystyle)
#
Favorite_bodystyle_Percentage=(df.describe(include='object').loc['freq','body-style']/
                               df.describe(include='object').loc['count','body-style'])*100
print("Favorite_bodystyle_Percentage=",Favorite_bodystyle_Percentage)



# ایجاد متغیرهای ساختگی برای شرکت‌های سازنده
company_dummies = pd.get_dummies(df["make"])
company_dummies = company_dummies.astype(int)
sums = company_dummies.sum()
print("Companies:")
print(sums)

# شمارش تعداد خودروها بر اساس شرکت
print("Number of cars by company:")
print(df['make'].value_counts())

# شمارش تعداد خودروها بر اساس نوع محور محرک
print("\nNumber of cars by drive-wheels type:")
print(df['drive-wheels'].value_counts())




# تحلیل قیمت بر اساس شرکت و نوع بدنه
# ایجاد زیرمجموعه داده با ستون‌های مورد نیاز
df1 = df[["make", "body-style", "price"]].copy()
# اطمینان از نوع داده float برای ستون price (در صورت نیاز)
df1["price"] = df1["price"].astype(float)
# نمایش زیرمجموعه داده
print("Subset of data (make, body-style, price):")
print(df1)
# محاسبه میانگین قیمت بر اساس شرکت و نوع بدنه
grouped_df = df1.groupby(["make", "body-style"], as_index=False).agg(mean_price=("price", "mean"))
# نمایش خروجی گروه‌بندی
print("Mean price by make and body style:")
print(grouped_df)




# رسم هیستوگرام برای ستون‌های عددی

numeric_cols = ['price', 'horsepower', 'engine-size', 'curb-weight', 'city-mpg', 'highway-mpg']
fig, axes = plt.subplots(2, 3, figsize=(10,6))  # smaller figure
axes = axes.flatten()  # flatten 2D array to 1D for easy iteration

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], bins=20, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

# انتخاب ستون‌های عددی برای ماتریس همبستگی
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# نمایش
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap for Numeric Columns")
plt.show()


# رسم پراکندگی: قدرت موتور در مقابل قیمت
plt.figure(figsize=(8,5))
sns.scatterplot(x='horsepower', y='price', data=df)
plt.title("Horsepower vs Price")
plt.show()


# پیش‌بینی قیمت با استفاده از مدل رگرسیون خطی
X = df[['engine-size', 'horsepower', 'curb-weight']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ارزیابی مدل
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression RMSE: {rmse:.2f}")
print(f"Linear Regression R²: {r2:.2f}")

# رسم پراکندگی: قیمت واقعی در مقابل قیمت پیش‌بینی‌شده
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

