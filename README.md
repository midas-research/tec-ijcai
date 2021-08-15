#TEC: A Time Evolving Contextual Graph Model for Speaker State Analysis in Political Debates

This codebase contains the python scripts for TEC, the model for the IJCAI 2021 paper [TEC: A Time Evolving Contextual Graph Model for Speaker State Analysis in Political Debates](https://www.ijcai.org/proceedings/2021/0489)

## Environment & Installation Steps
Python 3.6, Pytorch, Pytorch-Geometric and networkx.


```python
pip install -r requirements.txt
```

## Dataset and Preprocessing 

Download the dataset and follow preprocessing steps from [here](https://github.com/midas-research/gpols-coling). 


## Run

Execute the following python command to train TEC: 
```python
python train_model.py
```

## Cite
Consider citing our work if you use our codebase

```c
@inproceedings{ijcai2021-0489,
  title     = {TEC: A Time Evolving Contextual Graph Model for Speaker State Analysis in Political Debates},
  author    = {Sawhney, Ramit and Agarwal, Shivam and Wadhwa, Arnav and Shah, Rajiv},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {3552--3558},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/489},
  url       = {https://doi.org/10.24963/ijcai.2021/489},
}

```

