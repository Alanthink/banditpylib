```
 _                     _ _ _               _ _ _     
| |                   | (_) |             | (_) |    
| |__   __ _ _ __   __| |_| |_ _ __  _   _| |_| |__  
| '_ \ / _` | '_ \ / _` | | __| '_ \| | | | | | '_ \ 
| |_) | (_| | | | | (_| | | |_| |_) | |_| | | | |_) |
|_.__/ \__,_|_| |_|\__,_|_|\__| .__/ \__, |_|_|_.__/ 
                              | |     __/ |          
                              |_|    |___/                
```

A lightweight python library for bandit algorithms

![Unit Test](https://github.com/Alanthink/banditpylib/workflows/Unit%20Test/badge.svg?branch=dev) ![Style Check](https://github.com/Alanthink/banditpylib/workflows/Style%20Check/badge.svg?branch=dev)

## Features

* object-oriented design
* multiprocesses support
* friendly runtime info

## Getting Started

### Installing

```shell
# run under `banditpylib` root directory
pip install .
```

### Example

![output example](example.jpg)

Please check this [notebook](examples/ordinary_bandit.ipynb) to figure out how to generate this figure.

### Running the Tests

```shell
# run all tests
pytest
```

## Implemented Policies

### Single Player Protocol

#### Regret Minimization

| Bandit Type | Policies |
|     :---      |      :--- |
| Ordinary Bandit   | `EpsGreedy`, `UCB` , `ThompsonSampling` |
| Ordinary MNL Bandit   | `EpsGreedy`, `UCB`, `ThompsonSampling` |

For a detailed description, please check the [documentation](https://alanthink.github.io/banditpylib-doc/).

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

* This project is inspired by [libbandit](https://github.com/tor/libbandit) and [banditlib](https://github.com/jkomiyama/banditlib) which are both c++ libraries for bandit algorithms.
* This readme file is following the style of [README-Template.md](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2).
* The title is generated by [TAAG](http://patorjk.com/software/taag/#p=display&f=Graffiti&t=Type%20Something%20).
