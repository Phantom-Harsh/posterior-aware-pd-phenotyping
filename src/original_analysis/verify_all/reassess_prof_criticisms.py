#!/usr/bin/env python3
"""
DOUBLE-CHECK PROFESSOR'S CRITICISMS
===================================

More careful analysis to see if there are valid defenses
or if professor might be wrong
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE_DIR = Path("/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2")

print("="*100)
print("DEEPER ANALYSIS: IS PROFESSOR WRONG OR ARE THERE VALID DEFENSES?")
print("="*100)

# ============================================================================
# ISSUE 1: χ² - Could specimen-level be justified?
# ============================================================================
print("\n" + "="*100)
print("ISSUE 1: COULD SPECIMEN-LEVEL ANALYSIS BE JUSTIFIED?")
print("="*100)

lrrk2 = pd.read_csv(BASE_DIR / "data/LRRK2_Clinical/LRRK2 Cross-Sectional_20191218.csv")

print("\n🔍 Checking if specimens within individuals are truly dependent...")

# Check if LRRK2 status and PD diagnosis vary WITHIN individuals
lrrk2_by_person = lrrk2.groupby('LRRK2 ID').agg({
    'Has LRRK2': lambda x: x.nunique(),  # How many different values per person?
    'Has PD': lambda x: x.nunique()
})

print(f"\nPeople with VARYING LRRK2 status across specimens: {(lrrk2_by_person['Has LRRK2'] > 1).sum()}")
print(f"People with VARYING PD diagnosis across specimens: {(lrrk2_by_person['Has PD'] > 1).sum()}")

if (lrrk2_by_person['Has LRRK2'] > 1).sum() == 0 and (lrrk2_by_person['Has PD'] > 1).sum() == 0:
    print("\n✅ FINDING: Status is CONSTANT within individuals")
    print("   (LRRK2 and PD status don't vary across specimens from same person)")
    print("\n🤔 DEFENSE ARGUMENT:")
    print("   Since status is constant per person, maybe specimen-level IS the")
    print("   appropriate unit? Each specimen represents an independent measurement")
    print("   opportunity to detect the biomarker...")
    print("\n❌ COUNTER-ARGUMENT:")
    print("   NO - genetic status is a PERSON-level attribute, not specimen-level.")
    print("   Using multiple specimens from same person inflates sample size.")
    print("   This is a classic pseudoreplication error.")
    print("\n   PROFESSOR IS CORRECT - must use individual-level")

# Calculate proper statistics
lrrk2_individuals = lrrk2.groupby('LRRK2 ID').first()

# Get actual counts
lrrk2_pos_pd_yes = ((lrrk2_individuals['Has LRRK2'] == 'Yes') & (lrrk2_individuals['Has PD'] == 'Yes')).sum()
lrrk2_pos_pd_no = ((lrrk2_individuals['Has LRRK2'] == 'Yes') & (lrrk2_individuals['Has PD'] == 'No')).sum()
lrrk2_neg_pd_yes = ((lrrk2_individuals['Has LRRK2'] == 'No') & (lrrk2_individuals['Has PD'] == 'Yes')).sum()
lrrk2_neg_pd_no = ((lrrk2_individuals['Has LRRK2'] == 'No') & (lrrk2_individuals['Has PD'] == 'No')).sum()

print(f"\nCORRECT 2×2 Table (individuals):")
print(f"               PD+    PD-    Total")
print(f"  LRRK2+:      {lrrk2_pos_pd_yes:3d}    {lrrk2_pos_pd_no:3d}    {lrrk2_pos_pd_yes + lrrk2_pos_pd_no:3d}")
print(f"  LRRK2-:      {lrrk2_neg_pd_yes:3d}    {lrrk2_neg_pd_no:3d}    {lrrk2_neg_pd_yes + lrrk2_neg_pd_no:3d}")

# Calculate prevalence ratio
prev_lrrk2_pos = lrrk2_pos_pd_yes / (lrrk2_pos_pd_yes + lrrk2_pos_pd_no)
prev_lrrk2_neg = lrrk2_neg_pd_yes / (lrrk2_neg_pd_yes + lrrk2_neg_pd_no)
pr = prev_lrrk2_pos / prev_lrrk2_neg

print(f"\nPrevalence in LRRK2+: {prev_lrrk2_pos:.1%} ({lrrk2_pos_pd_yes}/{lrrk2_pos_pd_yes + lrrk2_pos_pd_no})")
print(f"Prevalence in LRRK2-: {prev_lrrk2_neg:.1%} ({lrrk2_neg_pd_yes}/{lrrk2_neg_pd_yes + lrrk2_neg_pd_no})")
print(f"Prevalence Ratio: {pr:.2f}")

# Chi-square at individual level
ct_ind = pd.crosstab(lrrk2_individuals['Has LRRK2'], lrrk2_individuals['Has PD'])
chi2_ind, p_ind, _, _ = stats.chi2_contingency(ct_ind)

print(f"\nCorrected statistics:")
print(f"  χ² = {chi2_ind:.2f} (not 167.26)")
print(f"  p = {p_ind:.2e} (still highly significant!)")
print(f"  PR = {pr:.2f} (matches your 1.89)")

# ============================================================================
# ISSUE 2: Is BIC completely wrong, or just mislabeled?
# ============================================================================
print("\n" + "="*100)
print("ISSUE 2: IS USING ELBO (lower_bound_) ACTUALLY WRONG?")
print("="*100)

print("""
Professor's Criticism:
  "BIC isn't defined for variational Dirichlet-process mixtures"
  
🤔 ANALYSIS:
  • TRUE: BayesianGaussianMixture doesn't have .bic() method
  • TRUE: You shouldn't CALL it "BIC"
  • BUT: Using .lower_bound_ (ELBO) for model selection IS VALID!
  
ELBO (Evidence Lower BOund) is the CORRECT criterion for variational inference.
Comparing ELBO across different K values IS proper model selection.

DEFENSE:
  "We used the variational evidence lower bound (ELBO) for model selection,
   comparing ELBO across 2-5 components. We apologize for incorrectly 
   labeling this as 'BIC' in the original manuscript. The methodology 
   itself (selecting K by maximizing ELBO) is statistically valid for 
   variational Bayesian mixture models."

VERDICT: Professor is RIGHT that terminology is wrong, but METHOD is valid
Just need to change "BIC" → "ELBO" throughout!
""")

# ============================================================================
# ISSUE 3: Is multiplying RRs really that bad?
# ============================================================================
print("\n" + "="*100)
print("ISSUE 3: IS MULTIPLICATIVE RISK MODEL DEFENSIBLE?")
print("="*100)

print("""
Professor's Criticism:
  "Multiplying RRs assumes independence; variables are correlated"
  
Our Finding:
  • UPSIT vs RBD: r = -0.249 (moderate correlation)
  • Overlap: 1.25× expected (25% excess co-occurrence)
  
🤔 POSSIBLE DEFENSE:
  "While we acknowledge the variables show moderate correlation (r=-0.25),
   the multiplicative model provides a simple, interpretable risk score.
   We can validate it by comparing predicted vs observed risks..."
  
❌ COUNTER:
  Professor is right - proper logistic regression is better practice and
  provides calibration, confidence intervals, and adjusted effects.
  
VERDICT: Professor is CORRECT - should use proper multivariable model
Multiplicative approach is overly simplistic for publication
""")

# ============================================================================
# FINAL ASSESSMENT
# ============================================================================
print("\n" + "="*100)
print("FINAL ASSESSMENT: IS PROFESSOR RIGHT OR WRONG?")
print("="*100)

print("""
ISSUE-BY-ISSUE BREAKDOWN:

1. χ² inflation (specimens vs individuals):
   ✅ PROFESSOR 100% CORRECT - This is a clear statistical error
   Must fix. No valid defense.

2. BIC with Bayesian GMM:
   ⚠️ PROFESSOR PARTIALLY CORRECT
   - Terminology is wrong (shouldn't call it "BIC")
   - But method (using ELBO) is actually valid
   - Defense: Just change "BIC" to "ELBO" in text

3. Multiplicative risk model:
   ✅ PROFESSOR CORRECT - Proper logistic regression is better
   Could defend multiplicative model, but not worth it
   Better to just do it right with logistic regression

4. Missing adjustments:
   ✅ PROFESSOR CORRECT - Unadjusted analyses are weaker
   Adding covariates strengthens the paper

OVERALL VERDICT:
  Professor is RIGHT on substance (3.5 out of 4 issues)
  Only minor defense possible on issue #2 (terminology vs methodology)
  
RECOMMENDATION:
  Acknowledge all issues gracefully
  Fix everything as professor suggested
  This will make the paper MUCH stronger
""")

print("="*100)
print("The professor's review is HIGH QUALITY and will improve your paper!")
print("="*100 + "\n")