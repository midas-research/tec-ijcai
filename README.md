#TEC: A Time Evolving Contextual Graph Model for Speaker State Analysis in Political Debates

This codebase contains the python scripts for STHAN-SR, the model for the IJCAI 2021 paper "TEC: A Time Evolving Contextual Graph Model for Speaker State Analysis in Political Debates".

## Environment & Installation Steps
Python 3.6, Pytorch, Pytorch-Geometric and networkx.


```python
pip install -r requirements.txt
```

## Dataset and Preprocessing 

Download the dataset and follow preprocessing steps from [here](https://github.com/midas-research/gpols-coling). 


## Run

Execute the following python command to train STHAN-SR: 
```python
python train_model.py
```

## Cite
Consider citing our work if you use our codebase

```c
@inproceedings{sawhney2021stock,
  title={TEC: A Time Evolving Contextual Graph Model for Speaker State Analysis in Political Debates},
  author={Sawhney, Ramit and Agarwal, Shivam and Wadhwa, Arnav and Derr, Tyler and Shah, Rajiv Ratn},
  booktitle={Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  year={2021}
}
```

