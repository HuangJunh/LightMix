# LightMix
J. Huang, B. Xue, Y. Sun, M. Zhang, and G. G. Yen, “Lightmix: Multi-objective search for lightweight mixed-scale convolutional neural
networks,” IEEE Transactions on Emerging Topics in Computational Intelligence (Early Access), pp. 1–15, 2025. DOI: 10.1109/TETCI.2025.3572041.

📑 [Read the Paper](https://ieeexplore.ieee.org/abstract/document/11023228)


## Architecture Search
Specify the hyperparameters for multi-objective evolutionary search and the target dataset in `global.ini`.  
```shell
python main.py --gpu 0
```
## Train and Test the Searched Model
Specify the hyperparameters for training in `main.py` and the target script name under `./scripts` after architecture search.  
```shell
python main.py --test --gpu 0 --script_name 'indi-1_00'
```


## Citation
If you use this code in your research, please cite the following paper:
```bibtex
@ARTICLE{LightMix,
  author={Huang, Junhao and Xue, Bing and Sun, Yanan and Zhang, Mengjie and Yen, Gary G.},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence},
  title={LightMix: Multi-Objective Search for Lightweight Mixed-Scale Convolutional Neural Networks},
  year={2025},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TETCI.2025.3572041}}
```