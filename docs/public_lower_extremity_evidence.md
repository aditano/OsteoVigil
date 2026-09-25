# Public lower-extremity evidence

Research biomechanical estimates only. This is not a diagnosis, not a medical device, and not a clearance to walk.

Numbers below are from `pytest tests/test_public_lower_extremity.py` on the cached TCIA STS_028 calf CT (CC BY 3.0). Tests assert ranges, not these exact decimals. Axial failure for a short cortical tibial segment is expected in the tens of kilonewtons when cortical area is a few hundred square millimetres and yield is about 120 MPa. Hundreds of kilonewtons, or a sub-kilonewton score for an intact shaft, would be absurd.

| Case | Modality | Percent vs normal | Failure load | Reliable | Expectation |
| --- | --- | --- | --- | --- | --- |
| STS_028 image-left shaft (less soft tissue) | CT | 10.7% weaker | 41,422 N | yes | Pass. Within 25% of a size-matched normal shaft. Tens of kN. |
| STS_028 image-right shaft (more soft tissue, sarcoma side) | CT | 21.9% weaker | 37,781 N | yes | Pass. Weaker. Not hundreds of percent stronger. The documented abnormality is extraskeletal, so the cortex can stay near the other leg. |
| Same image-left volume with a midshaft cortical notch | CT | 51.3% weaker | 23,556 N | yes | Pass. Known-weak derivative. Weaker than the intact public shaft. |
| Sum projections of a public shaft crop (known spacing) | Radiograph | 18.8% weaker | 29,571 N | yes | Pass. Canal resolved. Outer and inner diameters within 4 mm of the CT cortical widths on that crop (about 22 mm AP, 27 mm lateral, inner about 13 mm). |
| Reference phantom DRR, AP and lateral | Radiograph | matched | same order as the patient film | yes | Pass. Existing calibration test. Not ~90% weaker. |
| Flat adult shaft, no medullary valley | Radiograph | not scored | not scored | no | Pass. Unresolved canal. The headline is not a precise percent weaker. |
| Missing PixelSpacing | Radiograph | not scored | not scored | no | Pass. No walking clearance. |
| Modality the loader does not support | n/a | not scored | not scored | no | Pass. The loader raises. No walking text is produced. |
| Body mass absent from the DICOM header | either | unchanged by the label | unchanged |  | The report says the entered kilograms are not a measured patient weight. Header weight, when present, is labeled as coming from the DICOM header. |

The old 90.8% weaker headline (5,443 N versus 59,164 N, inner diameters estimated) came from a 2 mm cortex fallback while the reference canal was resolved. An unresolved canal is now `unreliable`, and the page does not present that fallback as a precise percent.

## Browser check

The built page was served by the local API and driven in headless Chrome.

| Upload | What the page showed |
| --- | --- |
| STS_028 image-left shaft, DICOM, header weight 58 kg | 10.7% weaker, failure load 41,422 N, body mass "58.0 kg (from the DICOM header)". |
| Sum projections of a public shaft crop, written as CR | 28.5% weaker, 26,035 N, reliable. Outer diameters AP 23.4 mm and lateral 27.3 mm, inners 13.7 mm and 13.9 mm. The 16-bit DICOM export clips the brightest projection sums, so this is a few kilonewtons below the float pytest row (18.8%, 29,571 N). Diameters still match the CT. |
| Flat AP bar near 22 mm with no medullary valley | Headline: "The medullary canal was not resolved, so a percent versus a normal leg is not reported." Failure load, cortical area, and reference load say "not scored". Body mass says entered, not from the DICOM header. Walking is not estimated. |
