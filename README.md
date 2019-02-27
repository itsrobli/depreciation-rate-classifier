# Automatic tax depreciation rate tagger/classifier

This program uses machine learning to predict the tax depreciation rate and category when given an arbitrary text description of an asset/capital expenditure (capex).

## Goal

The goal is to achieve over 90% prediction accuracy in all scenarios.

[Currently](#validation-and-accuracy), it can achieve:
- 80%+ accuracy when text is similar in style to the training data regardless of small spelling errors, ordering of words, language variations (e.g. -ed, -ing, -s, etc).
- Only around 50% or less accuracy when the text is of a very different style to the training data.

Please feel free to contribute suggests, corrections, and training data.


# Problem to be solved - a brief background

In the Real Estate Investment Trust (REIT / Property Funds) industry, real estate assets are held for investment purposes to provide income distributions to investors. A feature of Australian REITs for investors are non-taxable cash distributions which are primarily driven by tax deductible/depreciable capex.

Capex can result in various tax outcomes depending on the jurisdiction. In Australia, generally, depending on the nature of the cost, can be categorised and treated as: 

1. Immediately tax deductible; 
2. Depreciated as a tax asset over its effective life at [various](https://www.ato.gov.au/law/view/document?LocID=%22TXR%2FTR20184%2FNAT%2FATO%2FatTABLEB%22&PiT=99991231235958#TABLEB) depreciation rates; 
3. Be eligible for a building allowance deduction over 25 or 40 years; or
4. Added to the cost base of the asset to reduce the capital gain upon sale of the property.

An expert can be engaged to create a tax depreciation schedule (aka QS report) that breaks down costs into these categories. Industry practice is that this is usually done for high-value capex e.g. a property acquisition or major development project. However, a lot of the day-to-day capex is high-volume and low-value. This does not make it cost effective to outsource to an expert. As such, most businesses are classifying these items manually based on accounting system descriptions or invoices (or not at all and missing out on tax deductions). 

These manual processes, by nature of being high-volume and based on arbitrary descriptions, are time consuming; and by nature of being low-value, do not result in material errors if an item is misclassified. These attributes make this problem ideal for machine learning to solve.


# How does this program work?

While doing some machine learning exercises, particularly [this one](https://joaorafaelm.github.io/blog/text-classification-with-python), it occurred to me that this problem is essentially a text classification problem. Done manually, a person would read an arbitrary cost description from a cost report, like "chillers for precinct A", and then [look it up in the tables](https://www.ato.gov.au/law/view/document?DocID=TXR%2FTR20184%2FNAT%2FATO%2F00023) to find it's either 25 years or 20 years effective life depending on the type of chiller.

The machine learning model is built using training data (arbitrary text descriptions of capex) that has previously been reliably classified (tagged with a depreciation type/rate). Once the model is trained, text descriptions of capex can be fed in, and it will predict the tax attributes.

![workflow](img/ml_workflow.png)

The text is pre-processed to help the machine learning model find "meaning" in the text; rather than doing something more basic like string (word) matching, word length counting, etc. This means that the model can overcome small spelling mistakes, variances in ordering, and different forms of the same word (e.g. 'carpet' and 'carpeting' "mean" the same thing). This is vital to the usefulness of the model as it's designed to accept arbitrary text data to provide maximum flexibility in its usage. Further machine learning details [below](#some-machine-learning-details).

## Validation and accuracy

The trained model is tested by predicting results using other data with a known classification (testing data). The predicted classification is then matched against the known classifications in the testing data giving an overall accuracy %.

Using proprietary training data not included in this repo, a model was developed using 18,899 rows of training data which produced:

- 89.6% accuracy when using a random subset of the training data as testing data.
- 83.8% accuracy using K-Fold Cross Validation using 10 subsets (more thorough version of above).
- 92.3% accuracy when entirely tested against itself (artificially high).

This repo includes [generic training data](#training-data). It is developed from the Australian Tax Office [effective life tables](https://www.ato.gov.au/law/view/document?DocID=TXR/TR20184/NAT/ATO/00001) (run the `demo.py` included, and follow the instructions to run an accuracy report). It works ok for demo purposes.

I've found that using capex descriptions of a language style which is very different from the training data causes the accuracy to deteriorate rapidly. More on this [here](#further-improvements-in-the-works). As such, a model created using the included sample training data has limited flexibility in production.

## Future applications

The trained model can easily be used programmatically as part of a broader automated system. The goal is to have a workflow like:

1. Arbitrarily structured text can be fed directly into the model from the accounting system transaction descriptions, or other system/process that itemises the costs. This flexibility is the key reason for using machine learning vs another method that is more basic and reliable (but would require structured text inputs).

2. The text is classified using the machine learning model. 

3. The classification is combined with the original accounting system transaction data to generate the accounting/ERP system input file (e.g. journal entry) which would then be ingested into the tax depreciation module. 

## Further improvements in the works...

The training dataset has high reliability of having the correct classification, however, the text descriptions are of a single language style. By training the model with real world text descriptions (different styles of language) and offering feedback and corrections, the model's accuracy may be improved. 

Another thing to point out is that the model appears to bias towards classifications where there is more training data of a certain category. However, this is not necessarily a weakness. If the training data set is reflective of a business's actual depreciation register, the natural distribution of assets/capex purchased would be reflected in the predictions (e.g. it wouldn't be normal to have 1,000s of air conditioners for every 1 chair).

These issues can be seen in the accuracy reports when using K-Fold Cross Validation (validation/testing data is not used for training) resulting in low accuracy vs testing the data against itself (validation/testing data is the same as the training data) resulting in almost perfect accuracy.

I've found that using new training data (different language style, different descriptions) results in higher prediction accuracies. 

### Next iteration

My next experiment with this is to generate additional training data by finding synonym combinations and alternative expressions for the existing training data. This should overcome some of the language style issues. E.g. 'stamp duty' vs 'duty' vs 'duties' vs 'stamp' vs 'SD' vs 'stmp dty'.

Also, crowd-sourced training data can be gathered to further refine the model for real world text description. By making available a testing environment where a user can feed in their live examples from their existing, manual process, the user can manually flag and correct wrong predictions. This feedback data can be combined with the other training data for further refinement of the model. 

Ultimately, a lot of experimentation is required to see what works to achieve the [goal](#goal).

## Some machine learning details

In order to run machine learning algorithms on text based data like this, we need to transform the text into numerical values. Pre-processing this text is vital to the accuracy of the model.
 
Bag-of-words is one of the most used models for text. It essentially assigns a numerical value to words, creating a list of numbers.

Converting letters or words into numbers by e.g. counting frequencies of letters might not be enough. For example, long words or descriptions can bias the model because it is focusing on the assigned numerical value which really denotes length and not meaning.

Other problems relate to trying to extract "meaning" from words. Take for example the words ‘carpet’ and ‘carpeting’. They would be considered different words under a strict numerical assignment. This problem can be solved with various methods e.g. we can group together the inflected forms of a word. For example, the words ‘walked’, ‘walks’ and ‘walking’, can be grouped into their base form, the verb ‘walk’. Another issue worth noting is that some words despite the fact that they appear frequently, they do not really make any difference for classification because they don't change the meaning, and could even help mis-classify a text description. Words like ‘a’, ‘an’, ‘the’, ‘to’, ‘or’ etc, are known as stop-words. These words can be ignored during the machine learning process.

These issues and others are dealt with during the training of the model.


# Code usage info

The code is in `src/text_classifier_deprn_rates.py`. Once the `DeprnPredictor` Class has been instantiated, call the `predict_description(user_description)` method by feeding it some text to classify. Use a loop and call this method over and over again for each text description. 

The method returns two objects being:
1. A `pandas` Series object containing the various details of the tax category predicted.
2. An "account" description string matching how the training data labels its categories. This is basically a made up code for each possible tax category. See `src/account_meanings.csv` for a table showing the meanings. The reason for this design is that most accounting/ERP systems use something like this behind the scenes to drive the tax depreciation module. It can be anything so long as it matches the training data.

The `DeprnPredictor` Class is just a simple wrapper around some common machine learning libraries and techniques for classifying text data. It abstracts some of the fiddlier steps like pre-processing and building the pipeline. As a trade-off some bugs can occur due to how `pickle` is handled by the underlying ML libraries.

## Training data

Included in the repo is sample training data (`src/data_training/sample_training_data.csv`) based on the Australian Tax Office [effective life tables](https://www.ato.gov.au/law/view/document?DocID=TXR/TR20184/NAT/ATO/00001). 

I would note from experience that the accuracy of the predictions is highly dependent on good training data. This sample data creates decent results if the language used is very similar to it.

The best data to use would be actual descriptions of items from a REIT's fixed asset/depreciation register and underlying accounting transaction descriptions. Unfortunately, this data is proprietary to organisations unless donated to this repo.

However, I have some [ideas](#next-iteration) to generate some data.

## Demo

See `demo.py` for an example usage of this wrapper. Sample interaction below:

```
Evaluate using user input.
"QQ" to quit.
"CR" to see classification report.
Otherwise...
Enter a depreciable asset description: 
landscaping
Input from user:
	 landscaping
Result:
	account: 			CGT_Cost_Base_0
	deprn rate: 		0.0% prime cost
	effective life: 	0 years effective life
	tax category: 		CGT_Cost_Base
END of Result

Enter a depreciable asset description: 
demolition
Input from user:
	 demolition
Result:
	account: 			CGT_Cost_Base_0
	deprn rate: 		0.0% prime cost
	effective life: 	0 years effective life
	tax category: 		CGT_Cost_Base
END of Result

Enter a depreciable asset description: 
alarms
Input from user:
	 alarms
Result:
	account: 			Div40_Plant_6
	deprn rate: 		16.6% prime cost
	effective life: 	6 years effective life
	tax category: 		Div40_Plant
END of Result

Enter a depreciable asset description: 
fence
Input from user:
	 fence
Result:
	account: 			Div40_Plant_20
	deprn rate: 		5.0% prime cost
	effective life: 	20 years effective life
	tax category: 		Div40_Plant
END of Result

Enter a depreciable asset description: 
chiller
Input from user:
	 chiller
Result:
	account: 			Div40_Plant_25
	deprn rate: 		4.0% prime cost
	effective life: 	25 years effective life
	tax category: 		Div40_Plant
END of Result

Enter a depreciable asset description: 
project to install chillers
Input from user:
	 project to install chillers
Result:
	account: 			Div40_Plant_25
	deprn rate: 		4.0% prime cost
	effective life: 	25 years effective life
	tax category: 		Div40_Plant
END of Result
```


# Installation

Feel free to install the components yourself. Probably easiest to install [Anaconda](https://www.anaconda.com) as it comes with everything.

#### 1. Python 3.6+

[Download](https://www.python.org).

#### 2. Download all data that NLTK uses

[NLTK](https://www.nltk.org) is a key component of pre-processing the text data.

```bash
python -m nltk.downloader all;
```

#### 3. Python libraries

This program relies on the [`pandas`](http://pandas.pydata.org) and [`scikit-learn`](https://scikit-learn.org/) libraries.
