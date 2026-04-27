# OCCAMY — Artifact Repository

This repository contains the input files and placement plan outputs
for the scheduling experiments presented in the paper, along with
the Python implementations of OCCAMY and the MVSP baseline.

> **Note:** This is an anonymous artifact submission for review purposes.
> The repository demonstrates the core scheduling components and
> reproduces the placement plans reported in the paper.
> Full documentation and a complete reproduction guide will be
> provided upon acceptance.

---

## Repository Structure

```
.
├── model_occamy.py        # OCCAMY scheduler implementation
├── model_MVSP.py          # MVSP baseline scheduler implementation
│
├── input/
│   ├── mono_1/            # Single-site scenario without degradation (Section 5.3)
│   ├── mono_2/            # Single-site scenario with controlled degradation (Section 5.4)
│   └── multi/             # Multi-site, multi-application scenario (Section 5.5)
│
└── output/
    ├── mono_1/            # Placement plans for mono_1
    ├── mono_2/            # Placement plans for mono_2
    └── multi/             # Placement plans for multi
```

---

## Scenarios

### Single-site without degradation (`input/mono_1/`)
Corresponds to Section 5.3 (RQ1). Input files encode the single-site
deployment scenario used to evaluate OCCAMY's latency prediction
accuracy and scaling decisions for the OCR application, compared
against INVAR and HPA baselines.

### Single-site with controlled degradation (`input/mono_2/`)
Corresponds to Section 5.4 (RQ2). Input files encode the two-application
(OCR at high priority, YOLO at low priority) single-site scenario used
to evaluate OCCAMY's controlled degradation mechanism under increasing
request rates and resource scarcity.

### Multi-site, multi-application (`input/multi/`)
Corresponds to Section 5.5 (RQ3/RQ4). Input files encode the
five-site, three-application scenario (OCR, ResNet, YOLO) used to
evaluate OCCAMY's global placement decisions under both
`min_slots` and `min_cost` objectives, compared against MVSP.

---

## Running the Schedulers

**Requirements:**
- Python 3.8+
- [Gurobi Optimizer](https://www.gurobi.com) (with a valid licence)
- `scipy`, `numpy`

**Generate a placement plan with OCCAMY:**
```bash
python model_occamy.py --input input/mono_1/<config>.json \
                       --round 1
```

**Generate a placement plan with MVSP:**
```bash
python model_MVSP.py --input input/multi/<config>.json \
                     --round 1
```

The output JSON files contain the placement plans as reported
in the paper (number of instances per application per site,
selected model variants, and predicted P99 latencies).

---

## Input File Format

Each input JSON file specifies:
- Edge site capacities (in resource slots)
- Application definitions (model variants, resource demands,
  service rates, SLO thresholds, priority levels)
- User group definitions (request rates, network latencies
  to each site)
- Optimization objective (`min_slots` or `min_cost`)

---

## Output File Format

Each output JSON file contains the placement plan produced
by the scheduler:
- Number of instances allocated per application per site
- Assigned Local User Groups per application to each site 
- Request rate from each assigned Local User Groups

---

## Relation to Paper

| Experiment | Input folder | Paper section |
|---|---|---|
| Single-site, no degradation | `input/mono_1/` | Section 5.3 |
| Single-site, with degradation | `input/mono_2/` | Section 5.4 |
| Multi-site, multi-application | `input/multi/` | Section 5.5 |