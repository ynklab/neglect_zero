# Analysis of the Neglect_Zero Effect in Large Language Models
This repository contains codes and data of the paper "[Analysis of the Neglect-Zero Effect in Large Language Models](https://aclanthology.org/2026.acl-srw.91/)", accepted to ACL2026SRW.


## How to run

We prepare toml file, so you can setup by running this:
```bash
$ uv sync
```

After completing the setup, you can run the following code to perform inference with LLMs.
Please note that you must ensure that the open-weight models and necessary APIs are accessible beforehand.
Additionally, make sure to replace the placeholders {model} and {path_of_model} according to the specific model you wish to use.
```bash
$ uv run python main_{model}.py --model_name_or_path {path_of_model} --batch_id 1-16
$ uv run python results_integrater.py
```

After that, if you wish to conduct the statistical analysis, open and run `./results/analysis.Rmd` using RStudio or a similar environment.
Note that the `./results` already contains the pre-saved inference data from our experiment, allowing you to consuct the statistical analysis directly.


> [!NOTE]
> Our dataset is made by the dataset of [Klochowicz et al. (2025)](https://escholarship.org/uc/item/36w6x7z9). 


## Citation

If you use this code in any published research, please cite the following:

- Jin Tanaka, Daiki Matsuoka, Ryoma Kumon, and Hitomi Yanaka. 2026. Analysis of the Neglect-Zero Effect in Large Language Models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pages 1052–1064, San Diego, California, United States. Association for Computational Linguistics.

```bibtex
@inproceedings{tanaka-etal-2026-analysis,
    title = "Analysis of the Neglect-Zero Effect in Large Language Models",
    author = "Tanaka, Jin  and
      Matsuoka, Daiki  and
      Kumon, Ryoma  and
      Yanaka, Hitomi",
    editor = "T.Y.S.S., Santosh  and
      Rodriguez, Juan Diego  and
      de Gibert, Ona",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 4: Student Research Workshop)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-srw.91/",
    doi = "10.18653/v1/2026.acl-srw.91",
    pages = "1052--1064",
    ISBN = "979-8-89176-393-7"
}
```


## Contact

For any questions, please contact jt141214@g.ecc.u-tokyo.ac.jp.


## License

- Apache-2.0 license