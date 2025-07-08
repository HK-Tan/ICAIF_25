<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Python][Python.js]][Python-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a>
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/9d/University_of_California%2C_Los_Angeles_logo.png" alt="Logo" width="300" height="100">
  </a>

  <h3 align="center">Cluster-driven Hierarchical Representation of Large Asset Universes for Optimal Portfolio Construction</h3>

  <p align="center">
    Hong Kiat Tan - James Chen - Haoyang Lyu - Isaac-Neil Zanoria - Mihai Cucuringu
    <br />
    <br />
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#Repository Structure">Repository structure</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#Future Enhancements">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Welcome to our research project repository! This endeavor is a collaborative effort by... We are honored to be supervised by Professor Mihai Cucuringu from UCLA.

The central theme of our research revolves around the application of clustering in portfolio construction and forecasting realized covariance. Our goal is to shed light on how clustering can enhance traditional portfolio construction methods and potentially provide a more optimal risk-return tradeoff.

Project Objectives:

* Examine the impact of various clustering techniques on portfolio construction/optimization.
* Evaluate the ability of clustering to improve the forecasting of realized covariance.
* Explore the implications of using clustering in a portfolio management context.

Use the `README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With


[![Python][Python.js]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

This project is built on two major external library:
* [SigNet](https://github.com/alan-turing-institute/SigNet): library that implements specific graph clustering algorithms.
  ```sh
  pip install git+https://github.com/alan-turing-institute/SigNet.git
  ```

* [PyPortfolioOpt](): library that implements portfolio optimization methods.
  ```sh
  pip install PyPortfolioOpt
  ```

### Repository Structure

Everyones of those following files are in the folder **Code**.

1. **Code**: This folder contains two types of files :
* ```process.py ``` and ```module1.py ```  contain all the foundational functions used to build the portfolio, evaluate and plot the portfolio performances
* ```overall_perf.ipynb ``` and ```january.ipynb ``` are two notebooks were we call functions defined in the two previous .py files in the right order.


2. **Data**: csv file containing the daily return of the assets that compose our portfolio.

3. **SigNet** (temporary): file copy of the relevant part of [Signet](https://github.com/alan-turing-institute/SigNet) for our coding implementation. It is temporary as we have not found a way to use externally the code in the SigNet repository.

<!-- ### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- USAGE EXAMPLES -->
## Usage



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply reach out to one of us by email!
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact



[Project Link](https://github.com/jamesdchen/ICAIF_25/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Future Enhancements

Stay tuned as we continue to refine our methodologies and possibly expand our horizons to other financial markets or clustering techniques.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: www.linkedin.com/in/naïl-khelifa-581665220
[ENSAE-logo]: images/screenshot.png
[Python.js]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/

