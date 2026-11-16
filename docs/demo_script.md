# OCT Retina Screening Assistant — Demo Script

**Target duration:** Approximately 3 minutes  
**Rehearsals:** Twice before the presentation

| Time | Show | Say |
|---|---|---|
| 0:00 | Laptop with Wi-Fi visibly switched off | "No internet, no GPU, an ordinary laptop." |
| 0:15 | Upload a CNV sample | "The system provides a screening result, confidence, triage level and model explanation." |
| 0:45 | Switch the interface to हिन्दी | "The same interface and result are available in Hindi for operators who prefer it." |
| 1:05 | Upload a NORMAL sample | "The system identifies scans classified as normal." |
| 1:25 | Upload a selfie | "The input safety checks reject images that do not meet the OCT screening requirements." |
| 1:40 | Show a borderline or low-confidence case, selected beforehand using the CLI | "When the model is uncertain, the case is flagged for human review." |
| 2:00 | About panel and benchmark slide | "Measured model size: [X MB], inference time: [Y ms], and macro-F1: [Z%] on the patient-level test split." |
| 2:30 | Closing slide | "Same intelligence, a fraction of the size: Advanced Frugal Innovation." |

## Recording plan

- Record a backup video on Saturday using OBS Studio or Windows Game Bar (`Win + Alt + R`).
- Record the final submission video on Sunday.
- Show Wi-Fi switched off during the final recording.

## Before the demo

- [ ] Confirm the app runs with Wi-Fi disabled.
- [ ] Prepare one CNV sample and one NORMAL sample.
- [ ] Verify the English and Hindi language switch.
- [ ] Prepare a non-OCT image for the safety demonstration.
- [ ] Identify a low-confidence sample using the CLI.
- [ ] Measure model size and inference time on the demo laptop.
- [ ] Obtain the actual macro-F1 score from the patient-level test split.
- [ ] Replace all benchmark placeholders with measured values.
- [ ] Rehearse the complete presentation twice.

**Important:** Do not present placeholder benchmark values as measured results. Complete the benchmark section only after the real model and test results are available.
