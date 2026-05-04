"""
Chicago Crime - Analisis 5W+1H Multi-Feature (Tanpa Algoritma ML)
Part 2: WHEN, WHERE
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

OUT = r'c:\Python\Data_Mining\output_5w1h'
df = pd.read_pickle(f'{OUT}/_temp_df.pkl')
print(f"Loaded {len(df):,} rows")

# ==================================================================
# OPSI 3: WHEN - Kapan Kejahatan Terjadi?
# ==================================================================
print("\n" + "="*60)
print(" OPSI 3: WHEN - Kapan Kejahatan Terjadi?")
print("="*60)

# 3.1 Tren Tahunan
print("\n[3.1] Tren Tahunan:")
yearly = df['Year'].value_counts().sort_index()
for y, n in yearly.items():
    print(f"  {y}: {n:>6,}")

fig, ax = plt.subplots(figsize=(14, 5))
yearly.plot(kind='line', ax=ax, marker='o', color='steelblue', linewidth=2)
ax.fill_between(yearly.index, yearly.values, alpha=0.2)
ax.set_title('WHEN: Tren Jumlah Kejahatan per Tahun', fontsize=13, fontweight='bold')
ax.set_xlabel('Tahun'); ax.set_ylabel('Jumlah Kasus')
plt.tight_layout()
plt.savefig(f'{OUT}/when_01_trend.png', bbox_inches='tight')
plt.close()

# 3.2 Distribusi per Jam
print("\n[3.2] Distribusi per Jam:")
hour_dist = df['Hour'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(14, 5))
colors = ['#e74c3c' if (h >= 22 or h < 6) else '#3498db' for h in range(24)]
ax.bar(hour_dist.index, hour_dist.values, color=colors)
ax.set_title('WHEN: Distribusi Kejahatan per Jam (Merah=Malam)', fontsize=13, fontweight='bold')
ax.set_xlabel('Jam (0-23)'); ax.set_ylabel('Jumlah Kasus')
ax.set_xticks(range(24))
plt.tight_layout()
plt.savefig(f'{OUT}/when_02_jam.png', bbox_inches='tight')
plt.close()

# 3.3 Heatmap Jam x Hari
print("\n[3.3] Heatmap Jam x Hari:")
dow_labels = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
pivot = df.pivot_table(index='Hour', columns='DayOfWeek', values='Primary Type', aggfunc='count')
pivot.columns = dow_labels

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pivot, cmap='YlOrRd', annot=False, fmt=',', ax=ax, linewidths=0.5)
ax.set_title('WHEN: Heatmap Kejahatan - Jam x Hari', fontsize=13, fontweight='bold')
ax.set_ylabel('Jam'); ax.set_xlabel('Hari')
plt.tight_layout()
plt.savefig(f'{OUT}/when_03_heatmap.png', bbox_inches='tight')
plt.close()

# 3.4 Crime Type per Jam (Top 5)
print("\n[3.4] Pola Jam per Top 5 Crime Type:")
top5 = df['Primary Type'].value_counts().head(5).index.tolist()
fig, ax = plt.subplots(figsize=(14, 6))
for crime in top5:
    hourly = df[df['Primary Type']==crime]['Hour'].value_counts().sort_index()
    ax.plot(hourly.index, hourly.values, marker='.', linewidth=2, label=crime)
ax.set_title('WHEN: Pola Jam per Top 5 Jenis Kejahatan', fontsize=13, fontweight='bold')
ax.set_xlabel('Jam (0-23)'); ax.set_ylabel('Jumlah Kasus')
ax.set_xticks(range(24))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/when_04_crime_per_jam.png', bbox_inches='tight')
plt.close()

# 3.5 Seasonal Analysis
print("\n[3.5] Analisis Musiman:")
season_order = ['Winter','Spring','Summer','Fall']
season_total = df['Season'].value_counts().reindex(season_order)
for s, n in season_total.items():
    print(f"  {s}: {n:>6,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
season_total.plot(kind='bar', ax=axes[0], color=['#3498db','#2ecc71','#e74c3c','#e67e22'])
axes[0].set_title('Total Kejahatan per Musim')
axes[0].set_xlabel('Musim'); axes[0].set_xticklabels(season_order, rotation=0)

season_crime = df.groupby(['Season','Primary Type']).size().unstack(fill_value=0)
season_crime = season_crime.reindex(season_order)[top5]
season_crime.plot(kind='bar', ax=axes[1], width=0.8)
axes[1].set_title('Top 5 Crime Types per Musim')
axes[1].set_xticklabels(season_order, rotation=0)
axes[1].legend(fontsize=7, loc='upper right')
plt.suptitle('WHEN: Analisis Musiman', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/when_05_seasonal.png', bbox_inches='tight')
plt.close()

# 3.6 Weekend vs Weekday
print("\n[3.6] Weekend vs Weekday:")
wk_label = {0: 'Weekday', 1: 'Weekend'}
wk_crime = df.groupby(['IsWeekend','Primary Type']).size().unstack(fill_value=0)
wk_pct = wk_crime.div(wk_crime.sum(axis=1), axis=0) * 100
print("  Top crime proportions:")
for c in top5:
    wd = wk_pct.loc[0, c] if c in wk_pct.columns else 0
    we = wk_pct.loc[1, c] if c in wk_pct.columns else 0
    print(f"    {c:<30s} Weekday: {wd:.1f}% | Weekend: {we:.1f}%")
print("  [Saved] when_01 ~ when_05")

# ==================================================================
# OPSI 4: WHERE - Di Mana Kejahatan Terjadi?
# ==================================================================
print("\n" + "="*60)
print(" OPSI 4: WHERE - Di Mana Kejahatan Terjadi?")
print("="*60)

# 4.1 Top 20 Location Description
print("\n[4.1] Top 20 Lokasi Kejadian:")
loc_dist = df['Location Description'].value_counts().head(20)
for i, (l, n) in enumerate(loc_dist.items(), 1):
    print(f"  {i:2d}. {str(l):<45s} {n:>6,}")

fig, ax = plt.subplots(figsize=(12, 8))
loc_dist.plot(kind='barh', ax=ax, color=sns.color_palette('coolwarm', 20))
ax.set_title('WHERE: Top 20 Lokasi Kejadian (Location Description)', fontsize=13, fontweight='bold')
ax.set_xlabel('Jumlah Kasus')
plt.tight_layout()
plt.savefig(f'{OUT}/where_01_top_locations.png', bbox_inches='tight')
plt.close()

# 4.2 Crime per District
print("\n[4.2] Crime per Distrik:")
dist_crime = df['District'].value_counts().sort_index().dropna()
fig, ax = plt.subplots(figsize=(14, 5))
dist_crime.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('WHERE: Jumlah Kejahatan per Distrik Kepolisian', fontsize=13, fontweight='bold')
ax.set_xlabel('Distrik'); ax.set_ylabel('Jumlah Kasus')
plt.tight_layout()
plt.savefig(f'{OUT}/where_02_district.png', bbox_inches='tight')
plt.close()

# 4.3 Crime Type x Location Heatmap
print("\n[4.3] Heatmap: Crime Type x Location:")
top10_loc = df['Location Description'].value_counts().head(10).index
top8_crime = df['Primary Type'].value_counts().head(8).index
ct_loc = pd.crosstab(df[df['Primary Type'].isin(top8_crime)]['Primary Type'],
                      df[df['Location Description'].isin(top10_loc)]['Location Description'])
ct_loc = ct_loc.reindex(index=top8_crime, columns=top10_loc).fillna(0)

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(ct_loc, annot=True, fmt=',.0f', cmap='YlGnBu', ax=ax, linewidths=0.5)
ax.set_title('WHERE: Heatmap Jenis Kejahatan x Lokasi', fontsize=13, fontweight='bold')
ax.set_ylabel('Jenis Kejahatan'); ax.set_xlabel('Lokasi')
plt.xticks(rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/where_03_crime_location_heatmap.png', bbox_inches='tight')
plt.close()

# 4.4 Geospatial Scatter
print("\n[4.4] Scatter Plot Geospasial:")
df_geo = df.dropna(subset=['Latitude','Longitude'])
df_geo = df_geo[(df_geo['Latitude'].between(41.6, 42.1)) & 
                (df_geo['Longitude'].between(-87.95, -87.5))]

fig, ax = plt.subplots(figsize=(10, 12))
for crime in top5:
    sub = df_geo[df_geo['Primary Type']==crime].sample(min(5000, len(df_geo[df_geo['Primary Type']==crime])), random_state=42)
    ax.scatter(sub['Longitude'], sub['Latitude'], s=0.5, alpha=0.3, label=crime)
ax.set_title('WHERE: Distribusi Geospasial - Top 5 Crime Types', fontsize=13, fontweight='bold')
ax.set_xlabel('Longitude (Bujur)'); ax.set_ylabel('Latitude (Lintang)')
ax.legend(markerscale=10, fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/where_04_geospatial.png', bbox_inches='tight')
plt.close()

# 4.5 Arrest Rate per District
print("\n[4.5] Arrest Rate per Distrik:")
dist_ar = df.groupby('District').agg(Total=('Arrest','count'), Arrested=('Arrest','sum'))
dist_ar['Rate'] = dist_ar['Arrested'] / dist_ar['Total'] * 100
dist_ar = dist_ar.sort_values('Rate')

fig, ax = plt.subplots(figsize=(14, 5))
colors = ['#e74c3c' if r < 20 else '#2ecc71' if r > 35 else '#f39c12' for r in dist_ar['Rate']]
ax.bar(dist_ar.index.astype(str), dist_ar['Rate'], color=colors)
ax.set_title('WHERE: Arrest Rate per Distrik (Merah<20%, Hijau>35%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Distrik'); ax.set_ylabel('Arrest Rate (%)')
ax.axhline(y=dist_ar['Rate'].mean(), color='gray', linestyle='--', label=f'Mean: {dist_ar["Rate"].mean():.1f}%')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/where_05_arrest_district.png', bbox_inches='tight')
plt.close()
print("  [Saved] where_01 ~ where_05")

print("\n[PART 2 COMPLETE]")
