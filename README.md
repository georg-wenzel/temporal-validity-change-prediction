# Temporal Validity Change Prediction (TVCP)
## !!DISCLAIMERS!!
- GitHub Copilot was used for minor coding assistance (code completion) in this repository. The produced code was always manually evaluated by the developer to ensure it is sensible and works as intended.
- This is a research project. The publicized code and data and its derivatives should only be used for research contexts. Please refer to our article "Temporal Validity Change Prediction" (Section: Conclusion) for further details on the intended use.


## !!PUBLIC REPOSITORY NOTICE!!
This repository is a public version of the original repository. Due to Twitter's developer terms, we may not share 
the original dataset publicly. Thus, the following files/directories are not available in this repository:

- `data/source_data.csv`
- `data/candidates_processed.pkl`
- `data/candidates_processed.csv`
- `data/dataset.csv`
- `data/dataset_splits/`
- `data/dataset_processing/`
- `data/model_eval/LLM/`
- `data/model_eval/semantic_eval/output.csv`
- `data/qualitative_analysis.xlsx`
- `mturk_results/`

A suite of tests related to dataset functions has also been removed due to not being compatible with the 
public version of the repository.

If you wish to replicate the results of the paper, you can find a dehydrated version of the dataset [here](https://doi.org/10.5281/zenodo.8340858).
This will allow you to reconstruct the dataset and the corresponding splits directly. The resulting dataset will be 
compatible with the model training process. For access to other files or the dataset containing the processed tweet texts, 
please contact the authors directly.

**The remainder of this README is a copy of the original README, and may contain references to files that are not 
available in this repository.**

## Dataset Creation
To create the dataset, we first collect tweets into a .csv file containing (Tweet ID, Text) for each row.  
This source file is stored at `data/source_data.csv`. We extract statements from these candidate sentences via the pipeline 
in `MAIN_TweetPreprocessing.ipynb`. The extracted statements are stored in pickle form at `data/candidates_processed.pkl` 
and in .csv form at `data/candidates_processed.csv`.  

Additionally, we create subsets of the candidate statements, sorted by model score (predicted temporal validity duration), 
to act as batches for crowdsourcing. We create `data/dataset_processing/batches_p1/batch{i}.csv` (1<=i<=10), but only 
post the first 3 batches as crowdsourcing tasks. The crowdsourcing results are stored in 
`data/dataset_processing/annotated_p1/p1_batch{i}_annotated.csv` (1<=i<=3). The first crowdsourcing task is to predict 
the expected temporal validity duration of each statement. In cases where there is no agreement between the two crowdworkers, 
a third vote is supplied to generate a majority vote.

We use the majority vote (denoted as `final_vote` in the annotated P1 batches) to create batches for the second 
crowdsourcing task, which are stored in `data/dataset_processing/batches_p2/batch{i}_p2.csv` (1<=i<=3).
In this crowdsourcing task, we present crowdworkers the text and the previously estimated temporal validity duration. 
We then ask them to provide follow-up statements that respectively decrease, increase, and maintain the temporal 
validity duration estimate. Additionally, we ask them to provide a new estimate for the temporal validity duration when 
their follow-up is considered as a context. After concatenation of the batch results and after splitting each instance 
into three pairs, respectively consisting of the original statement and one of the follow-up statements, we obtain the 
final dataset, which is stored at `data/dataset.csv`.

Additionally, we provide two different types of evaluation splits. We provide an 80/10/10 split of the dataset, 
stored at `data/dataset_splits/{train,dev,test}.csv`. We also provide a 5-fold cross-validation split, stored at 
`data/dataset_splits/holdout_training/{test,train}{i}.csv` (0<=i<=4). 

To preserve the crowdsourced value in its original form, we provide the extracted items from crowdsourcing in their 
original form at `mturk_results/`. This also includes raw results from several pilot tests (in subdirectories 
`pilot1`, `pilot2` and `softlaunch`). Note that these results are not part of the final dataset, and may contain 
subpar annotations, differing data structures, supplementary files, etc., due to the ongoing development process.

Finally, our crowdsourcing tasks included mechanical turk qualification tests to ensure that crowdworkers were
familiar with the task and capable of performing it. These qualification tests are stored at `mturk_qual_tests/`. 
Further details on the qualification tests can be found in the paper.

**_Note: Contrary to our publications, in this repository, "target statement" is interchangable with "context statement".
This is an artefact of the crowdsourcing process, where the phrase "target statement" was deemed confusing to participants.
In the paper, "context statement" is interchangable with "follow-up statement" instead._**

## Dataset Analysis
We provide two jupyter notebooks to analyze the dataset creation process. In `MAIN_AnalyzeCrowdsourcing.ipynb`, we 
analyze crowdsourcing statistics (i.e., the number of participants, work times, agreement of temporal validity 
annotations, and statements which are annotated inconsistently). In `MAIN_AnalyzeDataset.ipynb`, we evaluate the 
dataset in terms of keyness of words with respect to the different sets, text length, class distribution, and temporal 
expressions.

We further perform qualitative analysis of a subset of the data. For 100 randomly selected statements, we analyze: 
- The temporal validity duration annotations of the target statement. If the statement is annotated inconsistently, we identify possible reasons.
- The type of temporal information in the target statement.
- The type of temporal validity change (e.g., whether the expected duration of an event or its occurrence time changes).

These randomly selected statements are taken from the first test fold (`data/dataset_splits/holdout_training/test0.csv`),
and the results are stored in an Excel sheet at `data/qualitative_analysis.xlsx`.

## Models and Training
We test several model structures, which are explained in detail in the written paper. We store the definition and training 
process of these models in the `Models_Native` directory. This directory contains a set of notebooks for the different 
types of models, as well as a utils package with shared code such as model structures, evaluation loops, and other utility 
functions. More detail on the models and training process can be found in the paper. 

The directory `data/model_eval` contains .pkl files with data on the training process for any fine-tuned models (e.g., training and 
validation loss over epochs and final evaluation metrics on the test set). For LLMs, we only store 
the generated results in .csv files, as there is no specific training process, and we prompt in a few-shot setting.

We further analyze the best-performing model (SelfExplain) with- and without the multitask loss, and compare the results with GPT-3.5-turbo.
We choose GPT-3.5-turbo both due to its general current presence in NLP, and because it achieved the highest EM score out of all prompted LLMs. 
In this notebook, we also test statistical significance of the results produced by the multitask loss.
This notebook is stored at `MAIN_SemanticEvaluation.ipynb`.