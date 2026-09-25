# Public lower-extremity sources

OsteoVigil tests use open images only. Do not add private clinical DICOMs, CPT patient films, or HSS studies. Large downloads stay under `data/downloaded/` and `data/external/`, which are gitignored.

## Used in the automated rigor pass

| Source | URL | License | What it tests |
| --- | --- | --- | --- |
| TCIA Soft-tissue-Sarcoma, patient STS_028 | Series `1.3.6.1.4.1.14519.5.2.1.5168.1900.913590972108839969190922752912` via NBIA `getImage`. Collection DOI [10.7937/K9/TCIA.2015.7GO2GSKS](https://doi.org/10.7937/K9/TCIA.2015.7GO2GSKS) | CC BY 3.0 | Adult calf CT. Image-left shaft is the near-normal cortex control. Image-right has more soft tissue (documented left-calf extraskeletal osteogenic sarcoma). Sum projections of the shaft are the synthetic radiographs. A cortical notch cut from this same public volume is the known-weak case. |
| In-repo reference tib/fib phantom | `src/cpt_predictor/reference_leg.py` | Generated in this repository (MIT) | Digitally reconstructed radiographs of a normal shaft. Clean AP and lateral projections must stay matched, and doubling PixelSpacing must double the measured diameter. |

STS_028 is the series the local pytest target actually solves. Download and run:

```bash
python3 scripts/download_public_abnormal_cts.py --case sts028
python3 -m pytest tests/test_public_lower_extremity.py -m public_data
```

Without that cache, the `public_data` tests skip. The radiograph regressions in the same file still run.

## Catalogued, not required for the default rigor run

`scripts/download_public_abnormal_cts.py` can also fetch these TCIA series. They are the same two collections and the same license. They were not needed once STS_028 covered a normal-range shaft, a soft-tissue-mass side, and a derived cortical defect.

| Case | Patient | Documented site | License | Collection |
| --- | --- | --- | --- | --- |
| sts010 | STS_010 | left calf myxofibrosarcoma | CC BY 3.0 | Soft-tissue-Sarcoma |
| sts025 | STS_025 | right calf malignant fibrous histiocytoma | CC BY 3.0 | Soft-tissue-Sarcoma |
| sts032 | STS_032 | right calf pleomorphic liposarcoma | CC BY 3.0 | Soft-tissue-Sarcoma |
| sts042 | STS_042 | left knee synovial sarcoma | CC BY 3.0 | Soft-tissue-Sarcoma |
| sts049 | STS_049 | right calf round-cell liposarcoma | CC BY 3.0 | Soft-tissue-Sarcoma |
| sts051 | STS_051 | left popliteal synovial sarcoma | CC BY 3.0 | Soft-tissue-Sarcoma |
| tcga_a8vf | TCGA-QQ-A8VF | lower limb, BodyPartExamined LOWERLIMB | CC BY 3.0 | [TCGA-SARC](https://doi.org/10.7937/K9/TCIA.2016.CX6YLSUX) |

NBIA endpoint: `https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=<uid>`

## Already in the demo tree, not the shaft-matched case

| Source | URL | License | What it is |
| --- | --- | --- | --- |
| Weight-bearing talocrural CT sample | [10.5281/zenodo.4274217](https://doi.org/10.5281/zenodo.4274217) | CC0, per the article data-availability statement for [Lenz et al. 2021](https://doi.org/10.1038/s41598-021-86567-7) | Distal tibia, fibula, and ankle. `data/demo/normal_real_talocrural` when present. The page should call the field distal, not a full tibial shaft. It is not the normal midshaft match. |

## Not used

Visible Human, NIH Open-i teaching radiographs, Orthanc demo series, and the VSD full-body CTs on Zenodo (CC BY-NC-SA, multi-hundred-megabyte NIfTI) were considered. STS_028 already supplies a quantitative calf CT and projections with known PixelSpacing, so those downloads were not added.
