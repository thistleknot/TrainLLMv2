import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, EvalPrediction, TrainingArguments, TrainerControl, TrainerState
import math
from peft import LoraConfig, get_peft_model, LoraModel
from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig
import transformers
from transformers import pipeline
from torch.utils.data import Dataset
from datasets import load_dataset#, Dataset
import datasets
import numpy as np
from transformers.trainer_callback import TrainerCallback
from typing import List, Optional  # Add the import statement at the beginning of your file
from transformers import logging
from typing import Dict, Optional, Any
from tqdm import tqdm
from transformers import TrainerState
from datetime import datetime
import copy
from transformers import TrainerControl, TrainerState
import tempfile
from sklearn.model_selection import train_test_split, KFold
import pickle
import json
import random
#from random import sample
