import pandas as pd
import matplotlib.pyplot as plt

# Charger les 3 jours
dfs = []
for day in [-2, -1, 0]:
    df = pd.read_csv(f'ROUND1/prices_round_1_day_{day}.csv', sep=';')
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# Filtrer OSMIUM seulement, enlever les lignes sans quotes
osm = df_all[(df_all['product'] == 'ASH_COATED_OSMIUM') & (df_all['mid_price'] > 1000)].copy()

# === 1. SPREAD ===
osm['spread'] = osm['ask_price_1'] - osm['bid_price_1']

print("=== SPREAD ASH_COATED_OSMIUM ===")
print(osm['spread'].describe())
print(f"\nSpread value counts (top 10):")
print(osm['spread'].value_counts().head(10))

# === 2. BID/ASK DISTRIBUTION ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Spread over time (day 0)
osm_d0 = osm[osm['day'] == 0]
axes[0][0].plot(osm_d0['timestamp'], osm_d0['spread'], linewidth=0.5)
axes[0][0].set_title('Spread over time (Day 0)')
axes[0][0].set_ylabel('Spread')

# Histogram des mid_prices
axes[0][1].hist(osm['mid_price'], bins=50, edgecolor='black')
axes[0][1].set_title('Distribution of Mid Price (all days)')
axes[0][1].axvline(x=10000, color='red', linestyle='--', label='10000')
axes[0][1].legend()

# Bid_price_1 distribution
axes[1][0].hist(osm['bid_price_1'].dropna(), bins=50, edgecolor='black', color='blue', alpha=0.7)
axes[1][0].set_title('Distribution of Best Bid')
axes[1][0].axvline(x=10000, color='red', linestyle='--')

# Ask_price_1 distribution
axes[1][1].hist(osm['ask_price_1'].dropna(), bins=50, edgecolor='black', color='orange', alpha=0.7)
axes[1][1].set_title('Distribution of Best Ask')
axes[1][1].axvline(x=10000, color='red', linestyle='--')

plt.tight_layout()
plt.savefig('osmium_spread_analysis.png', dpi=150)
plt.show()

# === 3. WALL MID (prix avec le plus gros volume) ===
print("\n=== VOLUME ANALYSIS ===")
print("Bid volumes (level 1) stats:")
print(osm['bid_volume_1'].describe())
print("\nAsk volumes (level 1) stats:")
print(osm['ask_volume_1'].describe())

# Vérifier les niveaux 2 et 3
print("\n=== LEVEL 2 & 3 FILL RATE ===")
print(f"Bid level 2 present: {osm['bid_price_2'].notna().sum()} / {len(osm)}")
print(f"Bid level 3 present: {osm['bid_price_3'].notna().sum()} / {len(osm)}")
print(f"Ask level 2 present: {osm['ask_price_2'].notna().sum()} / {len(osm)}")
print(f"Ask level 3 present: {osm['ask_price_3'].notna().sum()} / {len(osm)}")