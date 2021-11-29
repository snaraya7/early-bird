# [Paper: Simplifying Software Defect Prediction (via the ``early bird'' Heuristic)](paper.pdf)

<img src="https://upload.wikimedia.org/wikipedia/commons/c/c5/The_Early_Bird..._%28165702619%29.jpg" width="250">

## Prerequisites

* Linux Terminal
* python 2.7.5 and python 3.6.7
* Git support with lfs

### On your Linux terminal

1. $ `git lfs install`
1. $ `git lfs clone https://github.com/snaraya7/simplifying-software-analytics.git`
1. $ `cd simplifying-software-analytics`
1. $ `pip3 install -r requirements.txt`

## Steps to replicate results (table) of RQ's 1 to 4

### For example, let us reproduce RQ1 results

1. `Manually change RQ = 1 in Constants.py`
1. `python3 rq_run.py` (Estimated time ~24hr on a multi-core machine. The script write many csv files (one per project) having results of classifiers tested on project releases)
1. `python3 rq_export_scores.py` (Estimated time few min. Moves the current RQ results to 'results' folder and then this script aggregates all the evaluation scores from all the projects and writes it to eight separate text file (policy + learner pair)) in to the results/sk folder. 
1. `sk_all.bat` (Estimated time 10 to 15 min. Ranks all the evaluation scores using the Scott-Knott test to produce multiple csv files under results/sk folder.
1. `rq_write_table.py` (Estimated time few sec. Writes the final results table as per the RQ in the paper into a csv under results/sk folder).

The above steps can be repeated for remaining RQ's 2 to 4.

## Dataset

#### Project csv's are available [here](https://github.com/snaraya7/simplifying-software-analytics/tree/master/data) and its associated release csv's are available [here](https://github.com/snaraya7/simplifying-software-analytics/tree/master/data/release_info)

### Other details
1. Portions of code and data transferred from prior [study](https://ieeexplore.ieee.org/abstract/document/9401968) and [SAIL](https://sailhome.cs.queensu.ca/replication/featred-vs-featsel-defectpred/) for TCA+ related code


