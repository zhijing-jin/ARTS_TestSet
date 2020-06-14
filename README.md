
This is the repository for our 2020 paper 
"[Tasty Burgers, Soggy Fries: Probing Aspect Robustness in Aspect-Based Sentiment Analysis](http://zhijing-jin.com/files/papers/absa2020.pdf)".

### Data
We provide a **Aspect Robustness Probing** test set for [SemEval 2014](http://alt.qcri.org/semeval2014/task4/) Aspect-Based Sentiment Analysis (ABSA).
- Our new enriched test sets are at [data/arts_testset](data/arts_testset/)

#### Data Generation Process

- We generate our new probing test set by **automatic strategies**: 

<object data="data/img/overview.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="data/img/overview.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="data/img/overview.pdf">Download PDF</a>.</p>
    </embed>
</object>

### How to Use Our Code
If you have a **new** ABSA dataset, you can run our code to generate you own **aspect robustness probing** test set.
```
python code/main.py -dataset_name laptop
``` 

If you have more questions, please feel free to submit a [GitHub issue](https://github.com/XINGXIAOYU/ARTS_testset_for_ABSA/issues).


 