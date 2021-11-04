# Mitigating Language-Dependent Ethnic Bias in BERT
This repository contains the code and data for the paper "Mitigating Language-Dependent Ethnic Bias in BERT" (EMNLP 2021)

## Setup

Python 3.7

For externel libraries,
```python
pip install -r requirements.txt
```

## Evaluate

### Evaluation Templates

Template-based evaluation is employed in this paper. The attributes and templates used in the experiments are available in `templates/`.

* Templates don't give any significant clue that can guess one's ethnicity. 
* Attributes are mostly occupations and include social positions (e.g., immigrant, refugee)

### Evaluation Configuration

Before the evaluation, check `configuration.py` and set the model and evaluation conditions. In addition, we can change the target ethnicities in here.


### Evaluation Script

If you have any custom model trained for this task, you should the path to the trained model as argument (endswith `.pt`). Unless, the code will load pretrained model from the transformers hub.

```python
export LANG=de

python score.py --language $LANG --custom_model_path /path/to/model/(optional)
```