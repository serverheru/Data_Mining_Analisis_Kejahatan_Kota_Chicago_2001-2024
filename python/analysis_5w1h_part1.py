"""
Chicago Crime - Analisis 5W+1H Multi-Feature (Tanpa Algoritma ML)
Part 1: Setup, Load, WHAT, WHO
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings, os
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

OUT = r'c:\Python\Data_Mining\output_5w1h'
os.makedirs(OUT, exist_ok=True)

# ==================================================================
# LOAD & PREPROCESS
# ==================================================================
print("="*60)
print(" ANALISIS 5W+1H - CHICAGO CRIME DATASET")
print("="*60)
print("\n[SETUP] Loading data...")

cols = ['Date','Primary Type','Description','Location Description',
        'Arrest','Domestic','District','Ward','Community Area',
        'Year','Latitude','Longitude','Beat','IUCR','FBI Code','Block']

df = pd.read_csv(r'c:\Python\Data_Mining\Mining.csv', usecols=cols,
                  nrows=500000, low_memory=False)
df = df.sample(n=300000, random_state=42).reset_index(drop=True)

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
df['Hour'] = df['Date'].dt.hour
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month

season_map = {12:'Winter',1:'Winter',2:'Winter',
              3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',
              9:'Fall',10:'Fall',11:'Fall'}
df['Season'] = df['Month'].map(season_map)
df['IsNight'] = df['Hour'].apply(lambda x: 1 if (x >= 22 or x < 6) else 0)
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Severity mapping
severity_map = {'01A':'Kritis','01B':'Kritis',
    '02':'Tinggi','04A':'Tinggi','04B':'Tinggi','03':'Tinggi','15':'Tinggi',
    '05':'Sedang','06':'Sedang','07':'Sedang','09':'Sedang',
    '08A':'Sedang','08B':'Sedang',
    '10':'Rendah','11':'Rendah','14':'Rendah','16':'Rendah','17':'Rendah',
    '18':'Rendah','19':'Rendah','20':'Rendah','22':'Rendah','24':'Rendah','26':'Rendah'}
df['CrimeSeverity'] = df['FBI Code'].map(severity_map).fillna('Rendah')

print(f"  Sample: {len(df):,} rows | Year: {df['Year'].min()}-{df['Year'].max()}")

# ==================================================================
# OPSI 1: WHAT - Apa yang Terjadi?
# ==================================================================
print("\n" + "="*60)
print(" OPSI 1: WHAT - Apa yang Terjadi?")
print("="*60)

# 1.1 Distribusi Frekuensi Top 20
crime_dist = df['Primary Type'].value_counts().head(20)
print("\n[1.1] Top 20 Jenis Kejahatan (Primary Type):")
for i, (c, n) in enumerate(crime_dist.items(), 1):
    print(f"  {i:2d}. {c:<40s} {n:>7,} ({n/len(df)*100:5.2f}%)")

fig, ax = plt.subplots(figsize=(12, 8))
crime_dist.plot(kind='barh', ax=ax, color=sns.color_palette('viridis', 20))
ax.set_title('WHAT: Top 20 Jenis Kejahatan (Primary Type)', fontsize=14, fontweight='bold')
ax.set_xlabel('Jumlah Kasus')
for i, v in enumerate(crime_dist.values):
    ax.text(v + 200, i, f'{v:,}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/what_01_distribusi_crime.png', bbox_inches='tight')
plt.close()

# 1.2 Sub-kategori per Top 5 Crime Type
print("\n[1.2] Sub-kategori (Description) untuk Top 5 Crime Types:")
top5 = crime_dist.head(5).index.tolist()
fig, axes = plt.subplots(1, 5, figsize=(24, 6))
for idx, crime in enumerate(top5):
    sub = df[df['Primary Type']==crime]['Description'].value_counts().head(7)
    print(f"\n  {crime}:")
    for s, n in sub.items():
        print(f"    - {s}: {n:,}")
    sub.plot(kind='barh', ax=axes[idx], color=sns.color_palette('Set2', 7))
    axes[idx].set_title(crime, fontsize=9, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].tick_params(labelsize=7)
plt.suptitle('WHAT: Sub-kategori (Description) per Top 5 Crime Types', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/what_02_subcategory.png', bbox_inches='tight')
plt.close()

# 1.3 Pareto Analysis
print("\n[1.3] Pareto Analysis (80/20):")
all_crimes = df['Primary Type'].value_counts()
cum_pct = all_crimes.cumsum() / all_crimes.sum() * 100
n_80 = (cum_pct <= 80).sum() + 1
print(f"  {n_80} jenis kejahatan menyumbang 80% dari total kasus")

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.bar(range(len(all_crimes.head(15))), all_crimes.head(15).values, color='steelblue', alpha=0.8)
ax1.set_ylabel('Jumlah Kasus', color='steelblue')
ax1.set_xticks(range(len(all_crimes.head(15))))
ax1.set_xticklabels(all_crimes.head(15).index, rotation=45, ha='right', fontsize=8)
ax2 = ax1.twinx()
ax2.plot(range(len(cum_pct.head(15))), cum_pct.head(15).values, 'ro-', linewidth=2)
ax2.set_ylabel('Kumulatif (%)', color='red')
ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80%')
ax2.legend()
plt.title('WHAT: Pareto Analysis - Distribusi Kejahatan', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/what_03_pareto.png', bbox_inches='tight')
plt.close()
print("  [Saved] what_01, what_02, what_03")

# ==================================================================
# OPSI 2: WHO - Siapa yang Terlibat?
# ==================================================================
print("\n" + "="*60)
print(" OPSI 2: WHO - Siapa yang Terlibat?")
print("="*60)

# 2.1 Arrest Rate Global
arrest_pct = df['Arrest'].mean() * 100
domestic_pct = df['Domestic'].mean() * 100
print(f"\n[2.1] Arrest Rate Global: {arrest_pct:.2f}%")
print(f"      Domestic Rate      : {domestic_pct:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].pie([arrest_pct, 100-arrest_pct], labels=['Arrest','No Arrest'],
            autopct='%1.1f%%', colors=['#e74c3c','#3498db'], startangle=90,
            wedgeprops=dict(width=0.4))
axes[0].set_title('Arrest Rate', fontsize=12, fontweight='bold')
axes[1].pie([domestic_pct, 100-domestic_pct], labels=['Domestic','Non-Domestic'],
            autopct='%1.1f%%', colors=['#e67e22','#2ecc71'], startangle=90,
            wedgeprops=dict(width=0.4))
axes[1].set_title('Domestic Violence Rate', fontsize=12, fontweight='bold')
plt.suptitle('WHO: Profil Arrest & Domestic', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/who_01_arrest_rate.png', bbox_inches='tight')
plt.close()

# 2.2 Arrest Rate per Crime Type
print("\n[2.2] Arrest Rate per Crime Type:")
ar_by_crime = df.groupby('Primary Type').agg(
    Total=('Arrest','count'), Arrested=('Arrest','sum')
).assign(Rate=lambda x: x['Arrested']/x['Total']*100)
ar_by_crime = ar_by_crime.sort_values('Rate', ascending=True)
print(ar_by_crime.tail(15).to_string())

fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#e74c3c' if r > 50 else '#3498db' for r in ar_by_crime['Rate']]
ar_by_crime['Rate'].plot(kind='barh', ax=ax, color=colors)
ax.set_title('WHO: Arrest Rate per Jenis Kejahatan', fontsize=13, fontweight='bold')
ax.set_xlabel('Arrest Rate (%)')
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{OUT}/who_02_arrest_by_crime.png', bbox_inches='tight')
plt.close()

# 2.3 Tren Arrest Rate Tahunan
print("\n[2.3] Tren Arrest Rate Tahunan:")
yearly = df.groupby('Year').agg(Total=('Arrest','count'), Arrested=('Arrest','sum'))
yearly['Rate'] = yearly['Arrested'] / yearly['Total'] * 100
for y, row in yearly.iterrows():
    print(f"  {y}: {row['Rate']:.1f}% ({int(row['Arrested']):,}/{int(row['Total']):,})")

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.bar(yearly.index, yearly['Total'], color='lightsteelblue', alpha=0.7, label='Total Kasus')
ax1.set_ylabel('Total Kasus', color='steelblue')
ax2 = ax1.twinx()
ax2.plot(yearly.index, yearly['Rate'], 'ro-', linewidth=2, markersize=6, label='Arrest Rate')
ax2.set_ylabel('Arrest Rate (%)', color='red')
ax2.set_ylim(0, 50)
plt.title('WHO: Tren Tahunan - Total Kasus vs Arrest Rate', fontsize=13, fontweight='bold')
fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
plt.tight_layout()
plt.savefig(f'{OUT}/who_03_arrest_trend.png', bbox_inches='tight')
plt.close()

# 2.4 Domestic Violence Profile
print("\n[2.4] Domestic Violence Profile:")
dom = df[df['Domestic']==True]
print(f"  Total Domestic Cases: {len(dom):,}")
print(f"  Top Crime Types (Domestic):")
dom_crimes = dom['Primary Type'].value_counts().head(5)
for c, n in dom_crimes.items():
    print(f"    {c}: {n:,}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
dom_crimes.head(7).plot(kind='barh', ax=axes[0], color='coral')
axes[0].set_title('Domestic: Top Crime Types')
dom['Location Description'].value_counts().head(7).plot(kind='barh', ax=axes[1], color='salmon')
axes[1].set_title('Domestic: Top Locations')
dom['Hour'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='indianred')
axes[2].set_title('Domestic: Distribusi Jam')
axes[2].set_xlabel('Jam')
plt.suptitle('WHO: Profil Kekerasan Domestik (Domestic=True)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/who_04_domestic_profile.png', bbox_inches='tight')
plt.close()

# 2.5 Chi-Square: Arrest x Domestic
ct = pd.crosstab(df['Arrest'], df['Domestic'])
chi2, p, dof, expected = chi2_contingency(ct)
print(f"\n[2.5] Chi-Square Test: Arrest x Domestic")
print(f"  Chi2 = {chi2:.2f}, p-value = {p:.2e}, dof = {dof}")
print(f"  Kesimpulan: {'SIGNIFIKAN' if p < 0.05 else 'TIDAK SIGNIFIKAN'} (alpha=0.05)")
print(f"  Contingency Table:\n{ct}")
print("  [Saved] who_01 ~ who_04")

# Save df for part 2
df.to_pickle(f'{OUT}/_temp_df.pkl')
print("\n[PART 1 COMPLETE] Saved temp data for Part 2")
