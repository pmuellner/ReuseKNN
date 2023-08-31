# ReuseKNN: Neighborhood Reuse for Differentially-Private KNN-Based Recommendations

Python-based source-code for reproducing our work published in ACM Transactions of Intelligent Systems and Technology [1].
We use five public datasets: MovieLens 1M [2], Douban [3], LastFM [4], Ciao [5], and Goodreads [6, 7].

## Usage
For reproducing our experiments with DP on the MovieLens 1M dataset, please run
````
python rating_prediction.py --dataset_name ml-1m --use_dp True
````

For combining neighborhood reuse with NeuCF [8], run
````
python NeuReuse.py --dataset_name ml-1m --generate_embeddings True --generate_recommendations True 
````

The analysis of the recommendations (evaluation of recommendation accuracy and user privacy) can be found in 
<i>Rating Prediction Visualizations.ipynb</i> and <i>results/NeuReuse/Visualization.ipynb</i> 
respectively.



## Requirements
* python 3
* numpy
* pandas
* sklearn
* tensorflow
* matplotlib
* cython
* suprise
* pickle

(for detail see requirements.txt)

## Contributors
* Peter Müllner, Know-Center GmbH, pmuellner [AT] know [minus] center [DOT] at (Contact)
* Elisabeth Lex, Graz University of Technology
* Markus Schedl, Johannes Kepler University Linz and Linz Institute of Technology
* Dominik Kowald, Know-Center GmbH and Graz University of Technology

## References
[1] Peter Müllner, Elisabeth Lex, Markus Schedl, and Dominik Kowald. 2023. 
<i>ReuseKNN: Neighborhood Reuse for Differentially Private KNN-Based Recommendations</i>.
ACM Trans. Intell. Syst. Technol. 14, 5, Article 80 (October 2023), 29 Pages. 
https://doi.org/10.1145/3608481

[2] F. Maxwell Harper and Joseph A. Konstan. 2015. <i>The MovieLens datasets: History and context</i>. ACM Transactions on
Interactive Intelligent Systems 5, 4 (2015), 1–19.

[3] Longke Hu, Aixin Sun, and Yong Liu. 2014. <i>Your neighbors affect your ratings: On geographical neighborhood influence to rating prediction</i>. In Proc. of SIGIR’14

[4] Dominik Kowald, Markus Schedl, and Elisabeth Lex. 2020. <i>The unfairness of popularity bias in music recommendation:
A reproducibility study</i>. In Proc. of ECIR'20

[5] Guibing Guo, Jie Zhang, Daniel Thalmann, and Neil Yorke-Smith. 2014. <i>ETAF: An extended trust antecedents framework for trust prediction</i>. In Proc. of ASONAM’14.

[6] Mengting Wan and Julian J. McAuley. 2018. <i>Item recommendation on monotonic behavior chains</i>. In Proc. of ACM
RecSys’18. 86–94.

[7] Mengting Wan, Rishabh Misra, Ndapa Nakashole, and Julian J. McAuley. 2019. <i>Fine-grained spoiler detection from
large-scale review corpora</i>. In Proc. of ACL’19. Association for Computational Linguistics, 2605–2610.

[8] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. <i>Neural collaborative filtering</i>.
In Proc. of WWW’17. 173–182.