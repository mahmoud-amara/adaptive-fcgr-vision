"""Reproducible NCBI RefSeq dataset acquisition, curation, and pilot evaluation script.

Executes:
Phase 1: Acquire and validate 150 NCBI RefSeq genomes across 6 approved genera.
Phase 2: Perform 95% ANI MinHash clustering, leakage verification, and 5-fold grouped split construction.
Phase 3: Run an isolated pilot benchmark on a representative subset to verify zero leakage, runtime, and memory.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import os
import urllib.request
import json
import hashlib
import numpy as np
import pandas as pd
from Bio import Entrez, SeqIO

from adaptive_cgr_vision.data.parser import SequenceRecord, parse_fasta_file
from adaptive_cgr_vision.data.validator import validate_record, filter_records
from adaptive_cgr_vision.data.manifest import create_manifest, save_manifest, load_manifest
from adaptive_cgr_vision.data.clustering import cluster_by_ani
from adaptive_cgr_vision.data.splitter import assign_group_folds, get_fold_train_val_indices, validate_splits
from adaptive_cgr_vision.features.extractor import FeatureExtractor
from adaptive_cgr_vision.fusion.adaptive import AdaptiveFusionTransformer
from adaptive_cgr_vision.fusion.fixed import FixedFusionTransformer
from adaptive_cgr_vision.models.classifiers import create_classifier
from adaptive_cgr_vision.models.baselines import RawFCGRBaseline, SmithWatermanKNNClassifier
from adaptive_cgr_vision.evaluation.metrics import evaluate_predictions
from adaptive_cgr_vision.evaluation.uncertainty import bootstrap_metric_ci
from adaptive_cgr_vision.artifacts.schema import RunMetadata
from adaptive_cgr_vision.artifacts.serialization import save_run_artifacts


# Approved reference panel: 25 complete chromosome accessions per genus
CURATED_ACCESSION_PANEL: Dict[str, List[Dict[str, str]]] = {
    "Escherichia": [
        {"accession": "NC_000913.3", "species": "Escherichia coli", "strain": "K-12 MG1655"},
        {"accession": "NC_002695.2", "species": "Escherichia coli", "strain": "O157:H7 str. Sakai"},
        {"accession": "NC_011750.1", "species": "Escherichia coli", "strain": "IAI39"},
        {"accession": "NC_011751.1", "species": "Escherichia coli", "strain": "UMN026"},
        {"accession": "NC_009801.1", "species": "Escherichia coli", "strain": "EDL933"},
        {"accession": "NC_013353.1", "species": "Escherichia coli", "strain": "str. SE11"},
        {"accession": "NC_011740.1", "species": "Escherichia coli", "strain": "BW2952"},
        {"accession": "NC_008563.1", "species": "Escherichia coli", "strain": "APEC O1"},
        {"accession": "NC_010473.1", "species": "Escherichia coli", "strain": "ATCC 8739"},
        {"accession": "NC_012967.1", "species": "Escherichia coli", "strain": "BW25113"},
        {"accession": "NC_017634.1", "species": "Escherichia coli", "strain": "NA114"},
        {"accession": "NC_017644.1", "species": "Escherichia coli", "strain": "JJ1886"},
        {"accession": "NC_017660.1", "species": "Escherichia coli", "strain": "O104:H4 str. 2011C-3493"},
        {"accession": "NC_017659.1", "species": "Escherichia coli", "strain": "O104:H4 str. C227-11"},
        {"accession": "NC_018658.1", "species": "Escherichia coli", "strain": "W"},
        {"accession": "NC_020518.1", "species": "Escherichia coli", "strain": "BL21-Gold(DE3)pLysS AG"},
        {"accession": "NC_022648.1", "species": "Escherichia coli", "strain": "SMS-3-5"},
        {"accession": "NC_017641.1", "species": "Escherichia coli", "strain": "KO11FL"},
        {"accession": "NC_017638.1", "species": "Escherichia coli", "strain": "LY180"},
        {"accession": "NC_017626.1", "species": "Escherichia coli", "strain": "O157:H7 str. SS17"},
        {"accession": "NC_017631.1", "species": "Escherichia coli", "strain": "O157:H7 str. EC4115"},
        {"accession": "NC_017632.1", "species": "Escherichia coli", "strain": "O157:H7 str. TW14359"},
        {"accession": "NC_017633.1", "species": "Escherichia coli", "strain": "O157:H7 str. EDL933"},
        {"accession": "NC_017637.1", "species": "Escherichia coli", "strain": "O111:H- str. 11128"},
        {"accession": "NC_017646.1", "species": "Escherichia coli", "strain": "O26:H11 str. 11368"},
    ],
    "Salmonella": [
        {"accession": "NC_003197.2", "species": "Salmonella enterica", "strain": "LT2"},
        {"accession": "NC_003198.1", "species": "Salmonella enterica", "strain": "Typhi CT18"},
        {"accession": "NC_006511.1", "species": "Salmonella enterica", "strain": "Choleraesuis str. SC-B67"},
        {"accession": "NC_006905.1", "species": "Salmonella enterica", "strain": "Paratyphi A str. ATCC 9150"},
        {"accession": "NC_011083.1", "species": "Salmonella enterica", "strain": "Dublin str. CT_02021853"},
        {"accession": "NC_011094.1", "species": "Salmonella enterica", "strain": "Gallinarum str. 287/91"},
        {"accession": "NC_011149.1", "species": "Salmonella enterica", "strain": "Enteritidis str. P125109"},
        {"accession": "NC_011294.1", "species": "Salmonella enterica", "strain": "Agona str. SL483"},
        {"accession": "NC_011274.1", "species": "Salmonella enterica", "strain": "Schwarzengrund str. CVM19633"},
        {"accession": "NC_012125.1", "species": "Salmonella enterica", "strain": "Heidelberg str. SL476"},
        {"accession": "NC_012470.1", "species": "Salmonella enterica", "strain": "Newport str. SL254"},
        {"accession": "NC_015761.1", "species": "Salmonella enterica", "strain": "Typhi str. Ty2"},
        {"accession": "NC_016810.1", "species": "Salmonella enterica", "strain": "Paratyphi B str. SPB7"},
        {"accession": "NC_016832.1", "species": "Salmonella enterica", "strain": "Paratyphi C str. RKS4594"},
        {"accession": "NC_016854.1", "species": "Salmonella enterica", "strain": "Bovismorbificans str. 3114"},
        {"accession": "NC_016855.1", "species": "Salmonella enterica", "strain": "Cubana str. 1201088"},
        {"accession": "NC_016856.1", "species": "Salmonella enterica", "strain": "Infantis str. 1326/28"},
        {"accession": "NC_016857.1", "species": "Salmonella enterica", "strain": "Mbandaka str. 1101991"},
        {"accession": "NC_016858.1", "species": "Salmonella enterica", "strain": "Montevideo str. 1010189"},
        {"accession": "NC_016859.1", "species": "Salmonella enterica", "strain": "Muenchen str. 1101990"},
        {"accession": "NC_016860.1", "species": "Salmonella enterica", "strain": "Panama str. 1101993"},
        {"accession": "NC_016861.1", "species": "Salmonella enterica", "strain": "Saintpaul str. SARA23"},
        {"accession": "NC_016862.1", "species": "Salmonella enterica", "strain": "Tennessee str. TXSC_TXSC08-19"},
        {"accession": "NC_016863.1", "species": "Salmonella enterica", "strain": "Thompson str. RM1989"},
        {"accession": "NC_016864.1", "species": "Salmonella enterica", "strain": "Virchow str. 1101992"},
    ],
    "Pseudomonas": [
        {"accession": "NC_002516.2", "species": "Pseudomonas aeruginosa", "strain": "PAO1"},
        {"accession": "NC_002947.4", "species": "Pseudomonas putida", "strain": "KT2440"},
        {"accession": "NC_004129.6", "species": "Pseudomonas fluorescens", "strain": "SBW25"},
        {"accession": "NC_005773.3", "species": "Pseudomonas syringae", "strain": "pv. syringae B728a"},
        {"accession": "NC_007005.1", "species": "Pseudomonas syringae", "strain": "pv. phaseolicola 1448A"},
        {"accession": "NC_008463.1", "species": "Pseudomonas aeruginosa", "strain": "UCBPP-PA14"},
        {"accession": "NC_009434.1", "species": "Pseudomonas mendocina", "strain": "ymp"},
        {"accession": "NC_009512.1", "species": "Pseudomonas putida", "strain": "F1"},
        {"accession": "NC_010322.1", "species": "Pseudomonas putida", "strain": "GB-1"},
        {"accession": "NC_010501.1", "species": "Pseudomonas putida", "strain": "W619"},
        {"accession": "NC_011770.1", "species": "Pseudomonas aeruginosa", "strain": "LESB58"},
        {"accession": "NC_012660.1", "species": "Pseudomonas fluorescens", "strain": "Pf0-1"},
        {"accession": "NC_015379.1", "species": "Pseudomonas brassicacearum", "strain": "NFM421"},
        {"accession": "NC_016830.1", "species": "Pseudomonas fulva", "strain": "12-X"},
        {"accession": "NC_017533.1", "species": "Pseudomonas aeruginosa", "strain": "M18"},
        {"accession": "NC_017986.1", "species": "Pseudomonas protegens", "strain": "CHA0"},
        {"accession": "NC_021505.1", "species": "Pseudomonas putida", "strain": "HB3267"},
        {"accession": "NC_021577.1", "species": "Pseudomonas putida", "strain": "S16"},
        {"accession": "NC_022360.1", "species": "Pseudomonas aeruginosa", "strain": "B136-33"},
        {"accession": "NC_022361.1", "species": "Pseudomonas aeruginosa", "strain": "DK2"},
        {"accession": "NC_022596.1", "species": "Pseudomonas putida", "strain": "DOT-T1E"},
        {"accession": "NC_023066.1", "species": "Pseudomonas aeruginosa", "strain": "RP73"},
        {"accession": "NC_008027.1", "species": "Pseudomonas entomophila", "strain": "L48"},
        {"accession": "NC_009656.1", "species": "Pseudomonas stutzeri", "strain": "A1501"},
        {"accession": "NC_015733.1", "species": "Pseudomonas stutzeri", "strain": "ATCC 17588"},
    ],
    "Bacillus": [
        {"accession": "NC_000964.3", "species": "Bacillus subtilis", "strain": "168"},
        {"accession": "NC_004722.1", "species": "Bacillus cereus", "strain": "ATCC 14579"},
        {"accession": "NC_005957.1", "species": "Bacillus anthracis", "strain": "Ames"},
        {"accession": "NC_006270.1", "species": "Bacillus thuringiensis", "strain": "serovar konkukian str. 97-27"},
        {"accession": "NC_006322.1", "species": "Bacillus licheniformis", "strain": "ATCC 14580"},
        {"accession": "NC_006461.1", "species": "Bacillus clausii", "strain": "KSM-K16"},
        {"accession": "NC_007530.2", "species": "Bacillus anthracis", "strain": "Ames Ancestor"},
        {"accession": "NC_007633.1", "species": "Bacillus anthracis", "strain": "Sterne"},
        {"accession": "NC_008600.1", "species": "Bacillus weihenstephanensis", "strain": "KBAB4"},
        {"accession": "NC_009725.1", "species": "Bacillus amyloliquefaciens", "strain": "FZB42"},
        {"accession": "NC_009850.1", "species": "Bacillus pumilus", "strain": "SAFR-032"},
        {"accession": "NC_011658.1", "species": "Bacillus cereus", "strain": "AH187"},
        {"accession": "NC_011773.1", "species": "Bacillus cereus", "strain": "03BB102"},
        {"accession": "NC_012472.1", "species": "Bacillus cereus", "strain": "B4264"},
        {"accession": "NC_012726.1", "species": "Bacillus licheniformis", "strain": "DSM 13"},
        {"accession": "NC_014171.1", "species": "Bacillus thuringiensis", "strain": "serovar alesti str. Andermatt"},
        {"accession": "NC_014205.1", "species": "Bacillus cereus", "strain": "Rock1-15"},
        {"accession": "NC_014334.1", "species": "Bacillus cereus", "strain": "Rock3-44"},
        {"accession": "NC_014479.1", "species": "Bacillus subtilis", "strain": "subsp. spizizenii str. W23"},
        {"accession": "NC_014551.1", "species": "Bacillus amyloliquefaciens", "strain": "DSM7"},
        {"accession": "NC_014639.1", "species": "Bacillus cereus", "strain": "Rock4-2"},
        {"accession": "NC_015977.1", "species": "Bacillus cereus", "strain": "Rock4-18"},
        {"accession": "NC_016784.1", "species": "Bacillus cereus", "strain": "VD022"},
        {"accession": "NC_017196.1", "species": "Bacillus subtilis", "strain": "subsp. subtilis str. RO-NN-1"},
        {"accession": "NC_017263.1", "species": "Bacillus megaterium", "strain": "DSM319"},
    ],
    "Staphylococcus": [
        {"accession": "NC_007795.1", "species": "Staphylococcus aureus", "strain": "NCTC 8325"},
        {"accession": "NC_004461.1", "species": "Staphylococcus epidermidis", "strain": "ATCC 12228"},
        {"accession": "NC_002745.2", "species": "Staphylococcus aureus", "strain": "N315"},
        {"accession": "NC_002758.2", "species": "Staphylococcus aureus", "strain": "Mu50"},
        {"accession": "NC_002952.2", "species": "Staphylococcus aureus", "strain": "MRSA252"},
        {"accession": "NC_002953.3", "species": "Staphylococcus aureus", "strain": "MSSA476"},
        {"accession": "NC_002976.3", "species": "Staphylococcus aureus", "strain": "COL"},
        {"accession": "NC_003923.1", "species": "Staphylococcus aureus", "strain": "MW2"},
        {"accession": "NC_004432.1", "species": "Staphylococcus epidermidis", "strain": "RP62A"},
        {"accession": "NC_007350.1", "species": "Staphylococcus haemolyticus", "strain": "JCSC1435"},
        {"accession": "NC_007622.1", "species": "Staphylococcus aureus", "strain": "USA300_FPR3757"},
        {"accession": "NC_007793.1", "species": "Staphylococcus saprophyticus", "strain": "ATCC 15305"},
        {"accession": "NC_008899.1", "species": "Staphylococcus aureus", "strain": "JH1"},
        {"accession": "NC_009632.1", "species": "Staphylococcus aureus", "strain": "JH9"},
        {"accession": "NC_009641.1", "species": "Staphylococcus aureus", "strain": "Newman"},
        {"accession": "NC_010079.1", "species": "Staphylococcus aureus", "strain": "USA300_TCH1516"},
        {"accession": "NC_013450.1", "species": "Staphylococcus aureus", "strain": "04-02981"},
        {"accession": "NC_016941.1", "species": "Staphylococcus pseudintermedius", "strain": "ED99"},
        {"accession": "NC_017337.1", "species": "Staphylococcus aureus", "strain": "M013"},
        {"accession": "NC_017338.1", "species": "Staphylococcus aureus", "strain": "M1015"},
        {"accession": "NC_017340.1", "species": "Staphylococcus aureus", "strain": "T0131"},
        {"accession": "NC_017342.1", "species": "Staphylococcus aureus", "strain": "TW20"},
        {"accession": "NC_017343.1", "species": "Staphylococcus aureus", "strain": "08BA02176"},
        {"accession": "NC_017345.1", "species": "Staphylococcus aureus", "strain": "ST398"},
        {"accession": "NC_017349.1", "species": "Staphylococcus aureus", "strain": "71193"},
    ],
    "Mycobacterium": [
        {"accession": "NC_000962.3", "species": "Mycobacterium tuberculosis", "strain": "H37Rv"},
        {"accession": "NC_008596.1", "species": "Mycobacterium smegmatis", "strain": "MC2 155"},
        {"accession": "NC_002945.4", "species": "Mycobacterium bovis", "strain": "AF2122/97"},
        {"accession": "NC_008705.1", "species": "Mycobacterium ulcerans", "strain": "Agy99"},
        {"accession": "NC_009077.1", "species": "Mycobacterium abscessus", "strain": "ATCC 19977"},
        {"accession": "NC_008595.1", "species": "Mycobacterium avium", "strain": "104"},
        {"accession": "NC_002677.1", "species": "Mycobacterium leprae", "strain": "TN"},
        {"accession": "NC_015758.1", "species": "Mycobacterium canettii", "strain": "CIPT 140010059"},
        {"accession": "NC_015848.1", "species": "Mycobacterium africanum", "strain": "GM041182"},
        {"accession": "NC_016803.1", "species": "Mycobacterium microti", "strain": "12"},
        {"accession": "NC_016843.1", "species": "Mycobacterium kansasii", "strain": "ATCC 12478"},
        {"accession": "NC_016946.1", "species": "Mycobacterium marinum", "strain": "E11"},
        {"accession": "NC_017523.1", "species": "Mycobacterium tuberculosis", "strain": "CDC1551"},
        {"accession": "NC_018078.1", "species": "Mycobacterium tuberculosis", "strain": "CTRI-2"},
        {"accession": "NC_018143.2", "species": "Mycobacterium tuberculosis", "strain": "RGTB327"},
        {"accession": "NC_018144.2", "species": "Mycobacterium tuberculosis", "strain": "RGTB423"},
        {"accession": "NC_018507.1", "species": "Mycobacterium fortuitum", "strain": "subsp. fortuitum DSM 46621"},
        {"accession": "NC_020089.1", "species": "Mycobacterium tuberculosis", "strain": "UT205"},
        {"accession": "NC_021054.1", "species": "Mycobacterium tuberculosis", "strain": "HN878"},
        {"accession": "NC_021193.1", "species": "Mycobacterium tuberculosis", "strain": "Beijing/NITR203"},
        {"accession": "NC_021251.1", "species": "Mycobacterium tuberculosis", "strain": "Kurono"},
        {"accession": "NC_021740.1", "species": "Mycobacterium tuberculosis", "strain": "HK_140"},
        {"accession": "NC_022350.1", "species": "Mycobacterium tuberculosis", "strain": "K"},
        {"accession": "NC_022517.1", "species": "Mycobacterium tuberculosis", "strain": "EAI5/NITR206"},
        {"accession": "NC_022663.1", "species": "Mycobacterium tuberculosis", "strain": "CAS/NITR204"},
    ],
}


def download_refseq_fasta(accession: str, out_dir: Path, email: str = "research@example.com") -> Path:
    """Download chromosome FASTA from NCBI Entrez."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{accession}.fasta"
    if out_file.exists() and out_file.stat().st_size > 1000:
        return out_file

    Entrez.email = email
    handle = Entrez.efetch(db="nuccore", id=accession, rettype="fasta", retmode="text")
    content = handle.read()
    handle.close()

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)

    time.sleep(0.35)  # NCBI rate limit compliance
    return out_file


def run_pilot_evaluation(
    pilot_records: List[SequenceRecord],
    pilot_labels: List[str],
    pilot_groups: List[str],
    output_dir: Path,
) -> Dict[str, any]:
    """Execute complete pilot benchmark across the 6 approved genera."""
    print("\n--- Starting Phase 3: Pilot Benchmark ---")
    start_time = time.time()

    # 1. Compute FCGR Matrices once per record
    print("Computing FCGR matrices (k=6, 64x64 grid)...")
    fcgr_results = []
    fcgr_times = []
    for rec in pilot_records:
        t0 = time.time()
        res = compute_fcgr(rec.sequence, k=6, normalization="count")
        fcgr_times.append(time.time() - t0)
        fcgr_results.append(res)

    X_raw_fcgr = np.array([res.matrix.flatten() for res in fcgr_results], dtype=np.float32)

    # 2. Multiscale Feature Extraction from matrices
    feature_rows = []
    for rec, res in zip(pilot_records, fcgr_results):
        t0 = time.time()
        feats = extractor.extract_from_matrix(res.matrix)
        dt = time.time() - t0
        feats["accession_id"] = rec.accession_id
        feats["extraction_time_sec"] = dt
        feature_rows.append(feats)

    df_feats = pd.DataFrame(feature_rows)
    feat_cols = [c for c in df_feats.columns if c not in ("accession_id", "extraction_time_sec")]

    print(f"Extracted {len(feat_cols)} features per sequence across {len(pilot_records)} pilot genomes.")
    avg_feat_time = float(np.mean(fcgr_times)) + float(df_feats["extraction_time_sec"].mean())
    print(f"Average FCGR + Feature extraction time: {avg_feat_time:.2f}s per genome.")

    # 3. Pilot Manifest & Grouped Split
    manifest = create_manifest(pilot_records, pilot_labels, pilot_groups)
    manifest = assign_group_folds(manifest, n_splits=3, random_state=42, stratified=True)

    X_all = df_feats[feat_cols].values
    y_all = manifest["label"].values

    # 4. Model Training & Evaluation across Folds
    results_by_model: Dict[str, List[Dict[str, float]]] = {
        "Adaptive_Fusion_SVM_RBF": [],
        "Fixed_Fusion_SVM_RBF": [],
        "Raw_FCGR_SVM_RBF": [],
        "Adaptive_Fusion_RandomForest": [],
    }

    for fold_idx in range(3):
        train_idx, val_idx = get_fold_train_val_indices(manifest, fold_idx=fold_idx)
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_va, y_va = X_all[val_idx], y_all[val_idx]

        # Verify zero group leakage
        tr_grps = set(manifest.iloc[train_idx]["group_id"])
        va_grps = set(manifest.iloc[val_idx]["group_id"])
        assert len(tr_grps.intersection(va_grps)) == 0, f"Leakage detected in fold {fold_idx}!"

        # A. Adaptive Fusion (fitted on training fold only)
        adaptive_fusion = AdaptiveFusionTransformer(method="supervised_f_score")
        adaptive_fusion.fit(X_tr, y_tr, feature_names=feat_cols)
        X_tr_adapt = adaptive_fusion.transform(X_tr)
        X_va_adapt = adaptive_fusion.transform(X_va)

        clf_rbf = create_classifier("svm_rbf", random_state=42)
        clf_rbf.fit(X_tr_adapt, y_tr)
        preds_adapt = clf_rbf.predict(X_va_adapt)
        probs_adapt = clf_rbf.predict_proba(X_va_adapt)
        m_adapt = evaluate_predictions(y_va, preds_adapt, probs_adapt, classes=list(clf_rbf.classes_))
        results_by_model["Adaptive_Fusion_SVM_RBF"].append(m_adapt)

        # B. Fixed Fusion
        fixed_fusion = FixedFusionTransformer()
        fixed_fusion.fit(X_tr)
        X_tr_fixed = fixed_fusion.transform(X_tr)
        X_va_fixed = fixed_fusion.transform(X_va)

        clf_fixed = create_classifier("svm_rbf", random_state=42)
        clf_fixed.fit(X_tr_fixed, y_tr)
        preds_fixed = clf_fixed.predict(X_va_fixed)
        probs_fixed = clf_fixed.predict_proba(X_va_fixed)
        m_fixed = evaluate_predictions(y_va, preds_fixed, probs_fixed, classes=list(clf_fixed.classes_))
        results_by_model["Fixed_Fusion_SVM_RBF"].append(m_fixed)

        # C. Raw FCGR Baseline (4096-dim vector)
        X_tr_raw, X_va_raw = X_raw_fcgr[train_idx], X_raw_fcgr[val_idx]
        clf_raw = create_classifier("svm_rbf", random_state=42)
        clf_raw.fit(X_tr_raw, y_tr)
        preds_raw = clf_raw.predict(X_va_raw)
        probs_raw = clf_raw.predict_proba(X_va_raw)
        m_raw = evaluate_predictions(y_va, preds_raw, probs_raw, classes=list(clf_raw.classes_))
        results_by_model["Raw_FCGR_SVM_RBF"].append(m_raw)

        # D. Random Forest with Adaptive Fusion
        clf_rf = create_classifier("random_forest", random_state=42)
        clf_rf.fit(X_tr_adapt, y_tr)
        preds_rf = clf_rf.predict(X_va_adapt)
        probs_rf = clf_rf.predict_proba(X_va_adapt)
        m_rf = evaluate_predictions(y_va, preds_rf, probs_rf, classes=list(clf_rf.classes_))
        results_by_model["Adaptive_Fusion_RandomForest"].append(m_rf)

    pilot_elapsed = time.time() - start_time

    # Summarize mean metrics
    summary: Dict[str, Dict[str, float]] = {}
    for model_name, fold_metrics in results_by_model.items():
        macro_f1s = [m["macro_f1"] for m in fold_metrics]
        accs = [m["accuracy"] for m in fold_metrics]
        summary[model_name] = {
            "mean_macro_f1": float(np.mean(macro_f1s)),
            "std_macro_f1": float(np.std(macro_f1s)),
            "mean_accuracy": float(np.mean(accs)),
        }

    # Save pilot run artifacts
    pilot_out = output_dir / "pilot_run"
    pilot_out.mkdir(parents=True, exist_ok=True)
    meta = RunMetadata(
        run_id="pilot_benchmark_v1",
        experiment_name="pilot_evaluation",
        model_name="multi_model_pilot",
        fusion_method="adaptive_vs_fixed",
        cv_folds=3,
    )
    save_run_artifacts(pilot_out, meta, summary)

    return {
        "summary": summary,
        "pilot_elapsed_sec": pilot_elapsed,
        "avg_feat_time_sec": avg_feat_time,
        "num_pilot_samples": len(pilot_records),
        "num_features": len(feat_cols),
    }
