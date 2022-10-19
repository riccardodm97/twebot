<h1 align="center">TweBot </h1>

In recent years, online social media platforms have become an indispensable part of our daily routines. These online public spaces are getting largely occupied
by suspicious and manipulative social media bots, governed by software but disguising as human users. 
The problem of detecting bots has strong implications. For example, bots have been used to sway political elections by distorting online discourse or to manipulate the stock market. 

In this project we tackle this problem, by focusing on data. Our goal was to understand what kind of information is needed to accurately discriminate bot from humans on the Twitter social media platform. To do this we decided to feed our models more and more data from multiple sources to discover what piece of information works best for this task.

We used (nearly) the same model at all steps, with increasingly more data to learn from: 
- the text from only one tweet per user 
- the text from one tweet + custom metadata features on that tweet 
- the text from multiple tweets + custom statistical features on the aggregated tweets
- all the above + custom features from the user account information 

We decided to start from simple text of a single tweet; in fact, given the same pool of users, a tweet-based bot detection approach would have significantly more labeled examples to exploit at training time, and would be much faster and flexible when actually deployed. However, prior results in bot detection suggested that tweet text alone is not highly predictive of bot accounts and many works on account-level classification have found that user metadata tends to be the best predictor for bot detection. For this reason we tackle both account-level bot detection and tweet-level bot detection, with a custom DL model based on LSTM. Moreover, given the tabular nature of the metadata features, and with a view to achieve better model efficiency and interpretability, we also decided to test the performance of a Random Forest classifier on the detection task.

All the experiments are conducted on the [TwiBot-20](https://arxiv.org/abs/2106.13088) dataset. 

We demonstrate that, from just one single tweet, is still possible to discriminate bots from humans, with a acceptable level of accuracy. At the same time, the results clearly show that larger amount of information is needed to achieve the best performance. 

## **Results**

| Model | Accuracy | Recall | F1-score |
| --- | --- | --- | --- |
| SingleTweet | 0.655 | 0.682 | 0.674 |
| SingleTweetAndMetadata | 0.656 | 0.682 | 0.676 |
| MultiTweetAndMetadata | 0.702 | 0.708 | 0.705 |
| TweetAndAccount | 0.733 | 0.729 | 0.730 |
| Random Forest | 0.762 | 0.886 | 0.801 |
