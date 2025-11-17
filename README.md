<div align="center">
<center>
<a href="https://arxiv.org/abs/2506.15635" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-FindingDory-red?logo=arxiv" height="20" />
</a>
<a href="https://findingdory-benchmark.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-FindingDory-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/yali30/findingdory-qwen2.5-VL-3B-finetuned" target="_blank"">
    <img alt="Huggingface Model" src="https://img.shields.io/badge/Model-FindingDory-yellow?logo=huggingface" />
</a>
<a href="https://huggingface.co/datasets/yali30/findingdory/" target="_blank"">
    <img alt="Huggingface Dataset" src="https://img.shields.io/badge/Dataset-FindingDory-yellow?logo=huggingface" />
</a>
</center>
</div>

<div align="center">
<h1>FindingDory: A Benchmark to Evaluate Memory in Embodied Agents</h1>
<p>
  <a href="https://www.karmeshyadav.com/">Karmesh Yadav*</a>,
  <a href="https://yusufali98.github.io/">Yusuf Ali*</a>,
  <a href="https://gunshigupta.netlify.app/">Gunshi Gupta</a>,
  <a href="https://www.cs.ox.ac.uk/people/yarin.gal/website/">Yarin Gal</a>,
  <a href="https://faculty.cc.gatech.edu/~zk15/">Zsolt Kira</a>
</p>
</div>

# FindingDory Habitat Codebase
Repository for evaluating different agents on the [FindingDory](https://findingdory-benchmark.github.io/) task.

## Installation Instructions
After cloning the repository, simply run the setup script:

```bash
./setup.sh
```

## Datasets
To run the FindingDory evaluations, you need to download a bunch of datasets. Use the following script from FindingDory root:
```bash
./findingdory/scripts/download_data.sh
```

## Evaluation
Try running Qwen2.5VL on findingdory using the following command:
```
bash findingdory/scripts/findingdory_eval/run_qwen.sh
```


## Citation

If you find our paper and code useful in your research, please consider giving us a star ⭐ and citing our work 📝 :)

```bibtex
@article{yadav2025findingdory,
  title={FindingDory: A Benchmark to Evaluate Memory in Embodied Agents},
  author={Yadav, Karmesh and Ali, Yusuf and Gupta, Gunshi and Gal, Yarin and Kira, Zsolt},
  journal={arXiv preprint arXiv:2506.15635},
  year={2025}
}
