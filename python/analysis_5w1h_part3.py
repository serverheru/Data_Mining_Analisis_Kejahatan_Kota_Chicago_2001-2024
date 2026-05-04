"""
Chicago Crime - Analisis 5W+1H Multi-Feature (Tanpa Algoritma ML)
Part 3: WHY, HOW
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

OUT = r'c:\Python\Data_Mining\output_5w1h'
df = pd.read_pickle(f'{OUT}/_temp_df.pkl')
print(f"Loaded {len(df):,} rows")

# ==================================================================
# OPSI 5: WHY - Mengapa Kejahatan Terjadi?
# ==================================================================
print("\n" + "="*60)
print(" OPSI 5: WHY - Mengapa Kejahatan Terjadi? (Inferensi)")
print("="*60)

# 5.1 Conditional Probability
print("\n[5.1] Probabilitas Bersyarat P(Arrest | Kondisi):")
conditions = [
    ('Malam Hari (IsNight=1)', df['IsNight']==1),
    ('Siang Hari (IsNight=0)', df['IsNight']==0),
    ('Weekend', df['IsWeekend']==1),
    ('Weekday', df['IsWeekend']==0),
    ('Domestic=True', df['Domestic']==True),
    ('Domestic=False', df['Domestic']==False),
    ('Lokasi: STREET', df['Location Description']=='STREET'),
    ('Lokasi: APARTMENT', df['Location Description']=='APARTMENT'),
    ('Lokasi: RESIDENCE', df['Location Description']=='RESIDENCE'),
    ('Lokasi: SIDEWALK', df['Location Description']=='SIDEWALK'),
]
print(f"  {'Kondisi':<35s} {'P(Arrest)':>10s}  {'N':>8s}")
print("  " + "-"*55)
cond_data = []
for name, mask in conditions:
    sub = df[mask]
    rate = sub['Arrest'].mean() * 100
    cond_data.append((name, rate, len(sub)))
    print(f"  {name:<35s} {rate:>9.2f}%  {len(sub):>8,}")

fig, ax = plt.subplots(figsize=(12, 6))
names = [c[0] for c in cond_data]
rates = [c[1] for c in cond_data]
colors = ['#e74c3c' if r > 30 else '#3498db' for r in rates]
ax.barh(names, rates, color=colors)
ax.set_title('WHY: Probabilitas Arrest berdasarkan Kondisi', fontsize=13, fontweight='bold')
ax.set_xlabel('P(Arrest) %')
ax.axvline(x=df['Arrest'].mean()*100, color='gray', linestyle='--', label=f'Rata-rata: {df["Arrest"].mean()*100:.1f}%')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/why_01_conditional_prob.png', bbox_inches='tight')
plt.close()

# 5.2 Multi-Feature Profile (Radar) per Crime Type
print("\n[5.2] Profil Multi-Feature per Top 5 Crime Type:")
top5 = df['Primary Type'].value_counts().head(5).index.tolist()
metrics = ['Arrest Rate','Domestic Rate','Night Rate','Weekend Rate']

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
profile_data = []
for idx, crime in enumerate(top5):
    sub = df[df['Primary Type']==crime]
    vals = [sub['Arrest'].mean()*100, sub['Domestic'].mean()*100,
            sub['IsNight'].mean()*100, sub['IsWeekend'].mean()*100]
    profile_data.append(vals)
    print(f"  {crime}:")
    for m, v in zip(metrics, vals):
        print(f"    {m}: {v:.1f}%")
    
    axes[idx].barh(metrics, vals, color=sns.color_palette('Set2', 4))
    axes[idx].set_title(crime, fontsize=9, fontweight='bold')
    axes[idx].set_xlim(0, 100)
    for i, v in enumerate(vals):
        axes[idx].text(v+1, i, f'{v:.0f}%', va='center', fontsize=8)

plt.suptitle('WHY: Profil Multi-Feature per Top 5 Crime Type', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/why_02_crime_profiles.png', bbox_inches='tight')
plt.close()

# 5.3 Domestic vs Non-Domestic
print("\n[5.3] Domestic vs Non-Domestic Comparison:")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (label, mask) in enumerate([('Domestic', df['Domestic']==True), ('Non-Domestic', df['Domestic']==False)]):
    sub = df[mask]
    sub['Primary Type'].value_counts().head(7).plot(kind='barh', ax=axes[0], alpha=0.6, label=label)
    sub['Hour'].value_counts().sort_index().plot(kind='line', ax=axes[1], alpha=0.8, label=label, linewidth=2)

axes[0].set_title('Crime Type Distribution'); axes[0].legend()
axes[1].set_title('Distribusi Jam'); axes[1].legend(); axes[1].set_xlabel('Jam')

dom_loc = df[df['Domestic']==True]['Location Description'].value_counts().head(5)
nondom_loc = df[df['Domestic']==False]['Location Description'].value_counts().head(5)
comp = pd.DataFrame({'Domestic': dom_loc, 'Non-Domestic': nondom_loc}).fillna(0).head(7)
comp.plot(kind='barh', ax=axes[2])
axes[2].set_title('Top Lokasi')

plt.suptitle('WHY: Domestic vs Non-Domestic', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/why_03_domestic_comparison.png', bbox_inches='tight')
plt.close()

# 5.4 Night vs Day Crime Profile
print("\n[5.4] Night vs Day - Perubahan Crime Type:")
night = df[df['IsNight']==1]['Primary Type'].value_counts(normalize=True).head(10) * 100
day = df[df['IsNight']==0]['Primary Type'].value_counts(normalize=True).head(10) * 100
comp_nd = pd.DataFrame({'Night(%)': night, 'Day(%)': day}).fillna(0)
comp_nd['Diff'] = comp_nd['Night(%)'] - comp_nd['Day(%)']
comp_nd = comp_nd.sort_values('Diff')
print(comp_nd.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#e74c3c' if d > 0 else '#3498db' for d in comp_nd['Diff']]
ax.barh(comp_nd.index, comp_nd['Diff'], color=colors)
ax.set_title('WHY: Perubahan Proporsi Kejahatan Malam vs Siang (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Selisih Proporsi (Malam - Siang)')
ax.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'{OUT}/why_04_night_vs_day.png', bbox_inches='tight')
plt.close()

# 5.5 Correlation Heatmap
print("\n[5.5] Correlation Heatmap:")
corr_cols = ['Hour','DayOfWeek','Month','IsNight','IsWeekend','Arrest','Domestic','District','Year']
df_num = df[corr_cols].copy()
df_num['Arrest'] = df_num['Arrest'].astype(int)
df_num['Domestic'] = df_num['Domestic'].astype(int)
corr = df_num.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('WHY: Correlation Heatmap (Multi-Feature)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/why_05_correlation.png', bbox_inches='tight')
plt.close()

# 5.6 Chi-Square Tests
print("\n[5.6] Chi-Square Independence Tests:")
chi_tests = [
    ('Arrest x Domestic', 'Arrest', 'Domestic'),
    ('Arrest x IsNight', 'Arrest', 'IsNight'),
    ('Arrest x IsWeekend', 'Arrest', 'IsWeekend'),
]
for name, c1, c2 in chi_tests:
    ct = pd.crosstab(df[c1], df[c2])
    chi2, p, dof, _ = chi2_contingency(ct)
    cramers_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
    sig = 'SIGNIFIKAN' if p < 0.05 else 'TIDAK SIGNIFIKAN'
    print(f"  {name:<25s} Chi2={chi2:>12.1f}  p={p:.2e}  Cramer's V={cramers_v:.4f}  [{sig}]")
print("  [Saved] why_01 ~ why_05")

# ==================================================================
# OPSI 6: HOW - Bagaimana Kejahatan Dilakukan?
# ==================================================================
print("\n" + "="*60)
print(" OPSI 6: HOW - Bagaimana Kejahatan Dilakukan?")
print("="*60)

# 6.1 Modus Operandi - Sub-kategori
print("\n[6.1] Modus Operandi (Description) per Top 5 Crime:")
top5 = df['Primary Type'].value_counts().head(5).index.tolist()
fig, axes = plt.subplots(5, 1, figsize=(14, 18))
for idx, crime in enumerate(top5):
    sub = df[df['Primary Type']==crime]['Description'].value_counts().head(8)
    print(f"\n  {crime}:")
    for d, n in sub.items():
        print(f"    {d}: {n:,}")
    sub.plot(kind='barh', ax=axes[idx], color=sns.color_palette('Set2', 8))
    axes[idx].set_title(f'{crime} - Sub-kategori', fontsize=10, fontweight='bold')
    axes[idx].tick_params(labelsize=8)
plt.suptitle('HOW: Modus Operandi per Top 5 Crime Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/how_01_modus.png', bbox_inches='tight')
plt.close()

# 6.2 Weapon Involvement
print("\n[6.2] Weapon Involvement Analysis:")
weapon_keywords = {
    'HANDGUN': df['Description'].str.contains('HANDGUN', na=False).sum(),
    'KNIFE/CUTTING': df['Description'].str.contains('KNIFE|CUTTING', na=False).sum(),
    'OTHER FIREARM': df['Description'].str.contains('FIREARM|OTHER FIRE', na=False).sum(),
    'OTHER WEAPON': df['Description'].str.contains('DANGEROUS WEAPON', na=False).sum(),
    'NO WEAPON': df['Description'].str.contains('NO WEAPON|STRONG ARM', na=False).sum(),
}
weapon_df = pd.Series(weapon_keywords).sort_values(ascending=True)
for w, n in weapon_df.items():
    print(f"  {w}: {n:,}")

fig, ax = plt.subplots(figsize=(10, 5))
weapon_df.plot(kind='barh', ax=ax, color=['#2ecc71','#f39c12','#e67e22','#e74c3c','#c0392b'])
ax.set_title('HOW: Keterlibatan Senjata dalam Kejahatan', fontsize=13, fontweight='bold')
ax.set_xlabel('Jumlah Kasus')
plt.tight_layout()
plt.savefig(f'{OUT}/how_02_weapon.png', bbox_inches='tight')
plt.close()

# 6.3 Severity Distribution
print("\n[6.3] Distribusi Tingkat Keparahan (Severity):")
sev_order = ['Rendah','Sedang','Tinggi','Kritis']
sev_dist = df['CrimeSeverity'].value_counts().reindex(sev_order).fillna(0)
for s, n in sev_dist.items():
    print(f"  {s}: {int(n):>7,} ({n/len(df)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sev_colors = ['#2ecc71','#f39c12','#e74c3c','#8e44ad']
axes[0].pie(sev_dist.values, labels=sev_dist.index, autopct='%1.1f%%',
            colors=sev_colors, startangle=90)
axes[0].set_title('Distribusi Severity')

# Severity trend per year
sev_year = df.groupby(['Year','CrimeSeverity']).size().unstack(fill_value=0)
sev_year = sev_year.reindex(columns=sev_order)
sev_year_pct = sev_year.div(sev_year.sum(axis=1), axis=0) * 100
sev_year_pct.plot(kind='area', ax=axes[1], stacked=True, color=sev_colors, alpha=0.7)
axes[1].set_title('Tren Severity per Tahun (%)')
axes[1].set_ylabel('Proporsi (%)')
axes[1].legend(fontsize=8)

plt.suptitle('HOW: Distribusi & Tren Tingkat Keparahan', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/how_03_severity.png', bbox_inches='tight')
plt.close()

# 6.4 Severity vs Arrest Rate
print("\n[6.4] Severity vs Arrest Rate:")
sev_arrest = df.groupby('CrimeSeverity').agg(
    Total=('Arrest','count'), Arrested=('Arrest','sum')
)
sev_arrest['Rate'] = sev_arrest['Arrested'] / sev_arrest['Total'] * 100
sev_arrest = sev_arrest.reindex(sev_order)
for s, row in sev_arrest.iterrows():
    print(f"  {s:<10s} Arrest Rate: {row['Rate']:.1f}% ({int(row['Arrested']):,}/{int(row['Total']):,})")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(sev_arrest.index, sev_arrest['Rate'], color=sev_colors)
ax.set_title('HOW: Arrest Rate per Tingkat Keparahan', fontsize=13, fontweight='bold')
ax.set_ylabel('Arrest Rate (%)')
for i, (s, row) in enumerate(sev_arrest.iterrows()):
    ax.text(i, row['Rate']+1, f'{row["Rate"]:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/how_04_severity_arrest.png', bbox_inches='tight')
plt.close()
print("  [Saved] how_01 ~ how_04")

# Cleanup
import os
try:
    os.remove(f'{OUT}/_temp_df.pkl')
except:
    pass

print("\n" + "="*60)
print(" SEMUA ANALISIS 5W+1H SELESAI!")
print("="*60)
print(f"\n Output: {OUT}/")
print(f" Total: 25 visualisasi PNG")
