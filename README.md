LABR: A Large-SCale Arabic Book Reviews Dataset
===============================================

This dataset contains over 63,000 book reviews in Arabic. It is the largest sentiment analysis dataset for Arabic to-date. The book reviews were harvested from the website [Goodreads](http://www.goodreads.com) during the month or March 2013. Each book review comes with the goodreads review id, the user id, the book id, the rating (1 to 5) and the text of the review.

Contents:
---------

- README.txt: this file

- data/
                      
  - reviews.tsv: a tab separated file containing the "cleaned up" reviews. It contains over 63,000 reviews. The format is:
                     
                     rating<TAB>review id<TAB>user id<TAB>book id<TAB>review
                     
    where:
                     rating: the user rating on a scale of 1 to 5
                     review id: the goodreads.com review id
                     user id: the goodreads.com user id
                     book id: the goodreads.com book id
                     review: the text of the review
  
  - 2class-balanced-train/test.txt: text file containing indices of reviews 
                     (from the reviews.tsv file) that are in the training/test
                     sets. Balanced means the number of reviews in the 
                     positive/negative classes are equal. The ratings are 
                     converted into positive (rating 4 & 5) and negative 
                     (rating 1 & 2) and rating 3 is ignored.
                     
  - 2class-unbalanced-train/test.txt: the same, but the sizes of the calsses 
                     are not equal.
                     
  - 5class-balanced/unbalanced-train/test.txt: the same, but for 5 classes 
                     instead of just 2.

- python/
  
   - labr.py: the main interface to the dataset. Contains functions that can
              read/write training and test sets.
              
   - experiments_acl2013.py: a Python script containing the code used to 
              generate the experiments in the reference ACL 2013 paper.
              
   - demo.py: a simple demo file showing the usage of the dataset and class.
   
   
Reference
---------
Please cite this paper for any usage of the dataset:

Mohamed Aly and Amir Atiya. LABR: Large-scale Arabic Book Reviews Dataset.
Association of Computational Linguistics (ACL), Bulgaria, August 2013.
