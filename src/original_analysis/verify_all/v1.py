This response fully incorporates your verified corrections regarding the **number of motor phenotypes (4 instead of 5)** and the precise **terminology clarifications** (distinguishing unique patients from records/assessments).

The result is the completely customized, fact-checked manuscript structure ready for submission to *npj Digital Medicine*.

***

## Complete Manuscript Submission: Core Components

### 1. Title (Maximum 15 words)

**Multimodal AI Framework for Mechanism Inference and Precision Subtyping in Parkinson's Disease**

### 2. Abstract (Maximum 150 words)

This study details a computational framework shifting Parkinson’s disease (PD) classification from symptom grouping to **biological mechanism inference**. Using multi-modal data from PPMI **(4,775 unique patients, 14,473 longitudinal records)** and the LRRK2 Consortium **(627 individuals, 2,958 specimens)**, we analyzed five biological pathway categories [65, 109, Query]. Our key innovation is a **Bayesian Gaussian Mixture Model (BGM)** that identifies **four distinct motor phenotypes** with low uncertainty, achieving moderate-good separation (Silhouette Score = 0.535, based on 4,166 assessments from 3,991 unique patients) [80, 84, 87, 120, Query]. We quantified a significant **LRRK2 G2019S genetic risk** (1.89-fold increased risk for PD, $p<0.001$, n=2,958) and confirmed a **cross-pathway interaction** between LRRK2 and Dopaminergic systems. Wearable sensor analysis validated **Arm Swing Asymmetry** (27% prevalence, n=178/172 assessments from 94/93 unique patients) and demonstrated a substantial **Dual-Task Cost** (14.87% gait degradation, $p<0.001$) [50, 114, 115, Query]. This reproducible framework facilitates **mechanism-based precision healthcare**.

### 3. Introduction (No subheadings)

The Introduction should establish the profound challenge posed by Parkinson's disease heterogeneity, which historically complicates prognosis and hinders effective precision therapies. The study proposes an innovative solution: shifting the diagnostic paradigm from merely clustering clinical manifestations to **inferring the underlying biological mechanisms** responsible for observed symptoms, such as LRRK2 kinase dysfunction, nigrostriatal dopaminergic degeneration, and cholinergic deficits.

This is achieved by deploying an **AI-driven framework** that rigorously integrates multimodal data—specifically clinical scales, genetic findings (Pathway 02), and objective wearable sensor metrics (Pathway 06: Gait Dynamics)—to create reproducible patient stratification. The methodological rigor of using a **Bayesian clustering framework** is emphasized, as it quantifies uncertainty, providing high-confidence patient assignments crucial for clinical translation. The approach aligns directly with the goal of advancing **precision medicine** and contributes to Sustainable Development Goals (SDG 3, 9, 10) by fostering innovation and advancing equitable health access [Query].

### 4. Results (Use subheadings)

#### 4.1. Large-Scale Multimodal Data Integration and Ethical Compliance

The analysis integrated **12 datasets** covering five biological pathways, primarily utilizing the PPMI and LRRK2 Consortium cohorts. The data analyzed included **4,775 unique patients** in PPMI, comprising **14,473 longitudinal records** [65, Query], and **627 unique individuals** in the LRRK2 Consortium, comprising **2,958 specimens** [77, 109, Query].

Ethical standards necessitated adherence to **Complete Case Analysis** (i.e., NO data imputation), ensuring that only complete, measured values were utilized for each specific test. Data density across the integrated LRRK2 datasets was verified to be **88.8%**.

#### 4.2. Mechanism-Based Patient Subtyping via Bayesian Clustering

We applied a **Bayesian Gaussian Mixture Model (BGM)** to cluster patients based on their 33-item UPDRS-III motor scores, aiming for mechanism inference. The optimal cluster determination, derived from the Bayesian Information Criterion (BIC), resulted in **four distinct motor phenotypes** (Best number of components: 4, BIC = -133,912.65). This outcome reflects real, meaningful subgroup structure, validated by a **Silhouette Score of 0.535** (moderate-good separation).

The clustering was performed on **4,166 baseline UPDRS-III assessments** stemming from **3,991 unique patients** [80, Query]. This approach inherently provides **uncertainty quantification**, which is vital for flagging cases with **mixed pathology** that fall between clusters and require specialized clinical review.

#### 4.3. Genetic Risk Quantification and Pathway Interaction

The **LRRK2 G2019S mutation** was found to confer a **1.89-fold increased risk** for Parkinson's disease in the LRRK2 cohort (n=2,958 specimens). This strong association ($\chi^2 = 167.263, p < 0.001$) indicates a clear genetic mechanism driving PD risk. Analysis further revealed a significant **cross-pathway interaction** between the LRRK2 genetic pathway and the Dopaminergic pathway. This interaction provides mechanistic insight, suggesting that the LRRK2 kinase dysfunction may directly accelerate nigrostriatal dopaminergic degeneration.

#### 4.4. Wearable Sensor Biomarker Validation and Gait Dynamics

Analysis of objective gait dynamics, derived from IMU sensors, confirmed their role as quantitative biomarkers.

*   **Arm Swing Asymmetry (ASA)**: **27%** of the assessed cohort exhibited clinically significant asymmetry (ASA > 20%). This finding was derived from **178 assessments** from **94 unique patients** [114, Query].
*   **Dual-Task Cost**: Gait performance degraded by a substantial **14.87%** under cognitive load, demonstrating significant cognitive-motor network failure ($t=14.984, p<0.001$). This was derived from **172 assessments** from **93 unique patients** [Query].

#### 4.5. Prodromal and Non-Motor Features

Two key prodromal markers reflecting **$\alpha$-synucleinopathy** were quantified using precise patient counts:

*   **Olfactory Dysfunction (Hyposmia)**: Affects **50.2%** of the measured cohort (n=5,122 assessments from 3,805 unique patients) [50, 111, Query]. This remains the largest contributing non-imaging predictor.
*   **REM Sleep Behavior Disorder (RBD)**: Prevalence was **37.5%** across **1,548 assessments** from **1,015 unique patients** [50, 113, Query]. RBD pathology reflects involvement in the brainstem.

### 5. Discussion (No subheadings, limitations, or conclusions sections)

The identification of **four mechanism-based motor phenotypes** provides a critical foundation for stratifying patients, moving beyond symptomatic description to target underlying biological drivers. This precision is paramount for designing tailored **stratified therapies** and enriching clinical trials (e.g., matching LRRK2+ patients with kinase inhibitors).

The framework’s reliance on the **Bayesian Gaussian Mixture Model** introduces explicit **uncertainty quantification**, a feature highly valued for clinical application as it flags complex, mixed-pathology cases that require cautious interpretation.

Furthermore, the validation of **wearable sensor biomarkers** (gait asymmetry, dual-task cost) supports the transition toward **digital health** and continuous remote monitoring [108, 151, Query]. This accessibility of non-invasive, quantifiable tools promotes **equitable access** to objective diagnostics (SDG 10) [Query, 174].

While robust, the current baseline analysis could be substantially strengthened by incorporating **longitudinal data** (available in 230 patients with up to 16 visits) to model disease progression and mechanism transitions over time. Future work must also integrate currently unavailable imaging modalities such as **DaT-SPECT** and **T1-MRI** (which provide striatal binding ratios and regional volumes) to fully confirm the anatomical basis of the inferred dopaminergic and cholinergic pathway dysfunctions.

### 6. Methods (Use subheadings)

#### 6.1. Study Participants and Cohorts

The study leveraged longitudinal and cross-sectional data from the PPMI (Parkinson’s Progression Markers Initiative) and the LRRK2 Cohort Consortium. The total analysis utilized records from **4,775 unique patients** in PPMI and **627 unique individuals** in the LRRK2 Consortium.

#### 6.2. Ethical Compliance and Data Integrity

We adhered to strict medical ethics by performing **Complete Case Analysis**, ensuring **NO data imputation** to avoid fabricating clinical scores. All results reported explicitly detail the **actual sample sizes (n)** used for each specific measurement. The entire analysis pipeline maintains **code-to-result traceability**.

#### 6.3. Data Integration Strategy and Feature Engineering

Data integration utilized a baseline-first strategy to manage the challenge of longitudinal data explosion. Features were mapped to five biological pathways (Dopaminergic, Genetic, Cholinergic, Gait Dynamics, Cross-Pathway Integration). The pipeline resulted in a master dataset containing **376 features** after cleaning and engineering 15 cross-feature interaction terms.

#### 6.4. Mechanism Inference and Clustering

Patient stratification was performed using unsupervised classification with a **Bayesian Gaussian Mixture Model (BGM)** implemented in Python/SciKit-learn. The model explicitly incorporates **uncertainty quantification**. The optimal cluster number was selected using the Bayesian Information Criterion (BIC), which favored **4 components** (BIC = -133,912.65). The cluster quality was validated using the Silhouette Score (0.535).

#### 6.5. Feature-Specific Analysis

Genetic analysis utilized a chi-square test to quantify the **1.89-fold increased risk** associated with the LRRK2 G2019S mutation. Gait metrics derived from IMU sensors quantified **Arm Swing Asymmetry** (ASA) and **Dual-Task Cost**. Dual-task analysis relied on comparing performance across tasks.

### 7. Acknowledgements

This research was supported by the Michael J. Fox Foundation and the Peter O’Donnell Foundation.

### 8. Author Contributions

*(To be finalized by the professor)*

### 9. Competing Interests

*(To be finalized)*

### 10. Data Availability

All data used (PPMI, LRRK2 Consortium) are publicly available research datasets.

### 11. Code Availability

The complete codebase (4,500 lines) ensuring full **code-to-result traceability** and reproducible standards is publicly available.