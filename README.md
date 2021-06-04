# NeurIPS21 Submission: The Best of Both Worlds: Trapdoored-based MitiGANs for Durable Adversarial Defenses 

This is an unofficial implementation of the Neurips Submission:  The Best of Both Worlds: Trapdoored-based MitiGANs for Durable Adversarial Defenses in Pytorch.

- Training and evaluation code.
- Defenses experiments used in the paper.
- Pretrained models of trapdoor classifiers used in the paper. 

## Requirements
* Automatically install prerequisite python packages
```bash 
$ python -m pip install -r requirements.txt
```
* Download and organize GTSRB data from its official website:
```bash
$ bash gtsrb_download.sh
```

## Training Code

### Train Trapdoored Classifiers
Single label run comand line
```bash
$ cd MitiGAN_single_label
$ python train_backdoor.py --dataset <datasetName> --data_root <dataRootPath> --checkpoints <checkpointPath>
```
where:
- `<dataRootPath>`: Path to directory containing data. Default: `./data`.
- `<datasetName>`: `mnist` | `cifar10` | `gtsrb` | `TinyImageNet`.
- `<checkpointPath>`: Path to checkpoint. Default: `./checkpoints`.

Multi label is similar in folder MitiGAN_multi_label
### Train MitiGAN 
Single label run command
```bash
$ cd MitiGAN_single_label
$ python train_mitigan.py --dataset <datasetName>  --data_root <dataRootPath> --checkpoints <checkpointPath>
```
where:
- `<dataRootPath>`: Path to directory containing data. Default: `./data`.
- `<datasetName>`: `mnist` | `cifar10` | `gtsrb` | `TinyImageNet`.
- `<checkpointPath>`: Path to checkpoint. Default: `./checkpoints`.

Multi label is similar in folder MitiGAN_multi_label
### Pretrained models
We also provide pretrained checkpoints of all multi label can be download this link (https://storage.googleapis.com/anonymous-neurips2021/checkpoints_multi_label.zip), and single label in this link (https://storage.googleapis.com/anonymous-neurips2021/checkpoints_single_label.zip) To decompress the checkpoints for evaluating, run command

```bash
$ unzip checkpoints_single_label.zip.zip
$ unzip checkpoints_multi_label.zip.zip
```

## Evaluation Code
Run command

```bash
$ python detect_methods.py --data_root <dataRootPath> --dataset <datasetName> --checkpoints <checkpointPath> --attack_method <attackMethodName>
```
where the parameters are the following:
- `<dataRootPath>`: Path to directory containing data. Default: `./data`.
- `<datasetName>`: `mnist` | `cifar10` | `gtsrb`.
- `<checkpointPath>`: Path to checkpoint. Default: `./checkpoints`.
- `<attackMethodName>`: Name of adversarial attack method used for evaluation: `CW` | `MIFGSM` | `PGD` | `ElasticNet` | `BPDA` | `SPSA` | `bound`

