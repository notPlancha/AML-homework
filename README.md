# Advanced Machine Learning (DA633E) Labs resolution

You report should have following instructions: (you can upload a pdf file (contains explanations) + your code in a zip file or a colab notebook with explanation including the following topics)

- How to run your code (especially, if not using notebooks and using other libraries)
- Describe your dataset and explain whether it's a classification or regression problem.
- Any library you are using that we did not cover, so I can download the right one (with the right version!). Appropriate pip commands in your notebooks would be the best.
- Custom folder structures. If I need to download some specific folder from drive or something. Please make sure that how to connect it in the code is clear on how I can change it to assess.
- Anything you think that must be written to complement the code and the report

# Setup

For these labs, [pixi](pixi.sh) was used to manage dependencies and their versions, leveraging the conda ecosystem. To install:

```bash
# Widnows
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex" 
# Linux & macos
curl -fsSL https://pixi.sh/install.sh | sh
# You might need to restart your terminal or source your shell for the changes to take effect.
```

To run the code:

```sh
pixi install # installs the environment with the dependencies in pixi.toml
pixi run jupyter lab # vscode detects the environment automatically 
```

Alternatively, you can load it conda:

```sh
conda env create --name envname --file=environment.yml
```

Finally, you can just use pip. The versions are specificed in `pixi.toml`.

The labs are inside `src/`. The repository (which includes everyhting besides the data) is available at github.com/notPlancha/AML-homework

# Lab 1

The dataset used is [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist). It is composed of grayscaled picutres of handsigned letters (in csv), with the objective of classifying the letters.