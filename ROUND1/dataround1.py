import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Charger les 3 jours
dfs = []
for day in [-2, -1, 0]:
    df = pd.read_csv(f'ROUND1/prices_round_1_day_{day}.csv', sep=';')
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# Filtrer les mid_price à 0 (pas de quotes)
df_all = df_all[df_all['mid_price'] > 1000]

# Créer un timestamp global pour visualiser les 3 jours bout à bout
df_all['global_ts'] = (df_all['day'] + 2) * 1_000_000 + df_all['timestamp']

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

for i, product in enumerate(['ASH_COATED_OSMIUM', 'INTARIAN_PEPPER_ROOT']):
    sub = df_all[df_all['product'] == product].sort_values('global_ts')
    ax = axes[i]
    
    # Colorer par jour
    for day in [-2, -1, 0]:
        d = sub[sub['day'] == day]
        ax.plot(d['global_ts'], d['mid_price'], label=f'Day {day}', linewidth=0.8)
    
    ax.set_title(f'{product}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mid Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[1].set_xlabel('Global Timestamp')
plt.tight_layout()
plt.savefig('mid_prices_round1.png', dpi=150)
plt.show()

print("Graphique sauvegardé: mid_prices_round1.png")