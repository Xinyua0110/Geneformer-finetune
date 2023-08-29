#Geneformer Fine-Tuning for Cell Annotation Application
import os
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"


from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns; sns.set()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments

from geneformer import DataCollatorForCellClassification

#Prepare training and evaluation datasets
# load cell type dataset (includes all tissues)
train_dataset=load_from_disk("/home/zjlab/Genecorpus-30M/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset")

dataset_list = []
evalset_list = []
organ_list = []
target_dict_list = []

for organ in Counter(train_dataset["organ_major"]).keys():
    # collect list of tissues for fine-tuning (immune and bone marrow are included together)
    if organ in ["bone_marrow"]:  
        continue
    elif organ=="immune":
        organ_ids = ["immune","bone_marrow"]
        organ_list += ["immune"]
    else:
        organ_ids = [organ]
        organ_list += [organ]
    
    print(organ)
    
    # filter datasets for given organ
    def if_organ(example):
        return example["organ_major"] in organ_ids
    trainset_organ = train_dataset.filter(if_organ, num_proc=16)
    
    # per scDeepsort published method, drop cell types representing <0.5% of cells
    celltype_counter = Counter(trainset_organ["cell_type"])
    total_cells = sum(celltype_counter.values())
    cells_to_keep = [k for k,v in celltype_counter.items() if v>(0.005*total_cells)]
    def if_not_rare_celltype(example):
        return example["cell_type"] in cells_to_keep
    trainset_organ_subset = trainset_organ.filter(if_not_rare_celltype, num_proc=16)
      
    # shuffle datasets and rename columns
    trainset_organ_shuffled = trainset_organ_subset.shuffle(seed=42)
    trainset_organ_shuffled = trainset_organ_shuffled.rename_column("cell_type","label")
    trainset_organ_shuffled = trainset_organ_shuffled.remove_columns("organ_major")
    
    # create dictionary of cell types : label ids
    target_names = list(Counter(trainset_organ_shuffled["label"]).keys())
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    target_dict_list += [target_name_id_dict]
    
    # change labels to numerical ids
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    labeled_trainset = trainset_organ_shuffled.map(classes_to_ids, num_proc=16)
    
    # create 80/20 train/eval splits
    labeled_train_split = labeled_trainset.select([i for i in range(0,round(len(labeled_trainset)*0.8))])
    labeled_eval_split = labeled_trainset.select([i for i in range(round(len(labeled_trainset)*0.8),len(labeled_trainset))])
    
    # filter dataset for cell types in corresponding training set
    trained_labels = list(Counter(labeled_train_split["label"]).keys())
    def if_trained_label(example):
        return example["label"] in trained_labels
    labeled_eval_split_subset = labeled_eval_split.filter(if_trained_label, num_proc=16)

    dataset_list += [labeled_train_split]
    evalset_list += [labeled_eval_split_subset]

trainset_dict = dict(zip(organ_list,dataset_list))
traintargetdict_dict = dict(zip(organ_list,target_dict_list))

evalset_dict = dict(zip(organ_list,evalset_list))



#Fine-Tune with cell classification learning objective and Quantify predictive performance
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1
    }


# set model parameters
# max input size
max_input_size = 2 ** 11  # 2048

# set training hyperparameters
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and eval
geneformer_batch_size = 12
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# number of epochs
epochs = 10
# optimizer
optimizer = "adamw"


for organ in organ_list:
    print(organ)
    organ_trainset = trainset_dict[organ]
    organ_evalset = evalset_dict[organ]
    organ_label_dict = traintargetdict_dict[organ]
    
    # set logging steps
    logging_steps = round(len(organ_trainset)/geneformer_batch_size/10)
    
    
    
    # reload pretrained model
    model = BertForSequenceClassification.from_pretrained("/home/zjlab/geneformer-6L-30M/", 
                                                      
                                                      num_labels=len(organ_label_dict.keys()),
                                                      output_attentions = False,
                                                      output_hidden_states = False).to("cuda")
    
    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    output_dir = f"/path/to/models/{datestamp}_geneformer_CellClassifier_{organ}_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"
    
    # ensure not overwriting previously saved model
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")

    # make output directory
    subprocess.call(f'mkdir {output_dir}', shell=True)
    
    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
    }
    
    training_args_init = TrainingArguments(**training_args)

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=organ_trainset,
        eval_dataset=organ_evalset,
        compute_metrics=compute_metrics
    )
    # train the cell type classifier
    trainer.train()
    predictions = trainer.predict(organ_evalset)
    with open(f"{output_dir}predictions.pickle", "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval",predictions.metrics)
    trainer.save_model(output_dir)
    

