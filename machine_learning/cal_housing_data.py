# # housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
# # plt.subplots_adjust(hspace=0.7, wspace=0.4)
# # plt.show()


# rng = np.random.RandomState(0)
# indices = rng.choice(
# np.arange(housing.frame.shape[0]), size=500, replace=False
# )

# # Drop the unwanted columns
# columns_drop = ["Longitude", "Latitude"]
# subset = housing.frame.iloc[indices].drop(columns=columns_drop)
# # Quantize the target and keep the midpoint for each interval
# subset["MedHouseVal"] = pd.qcut(subset["MedHouseVal"], 6, retbins=False)
# subset["MedHouseVal"] = subset["MedHouseVal"].apply(lambda x: x.mid)

# # Create a pair plot
# sns.pairplot(data=subset,
#         hue="MedHouseVal",
#         kind='scatter',
#         diag_kind='kde',
#         palette="viridis"
# )

# plt.suptitle('Pair Plot of California Housing Data', y=1.02)  # Adjust title position
# plt.show()

# exit()