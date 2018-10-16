Extra Credit:
We built positive and negative sentiment models on the Democrat and Republican labeled data, trained it, and created respective tables and plots for them. We accomplished this task by training CrossValidators on the labeled Democrat and Republican data and transforming the unlabeled data using the models we generated. We then manipulated the probabilities to classify the comments as positive/not positive and negative/not negative, to do this we used the same thresholds for positive and negative as we did for the Donald Trump data. After classifying the unlabeled data, we ran queries on this data to extract information about the positive/negative sentiment on Democrats and Republicans across, time, states, and submission/comment scores.




Sources in addition to those referenced in the spec:
https://ragrawal.wordpress.com/2015/10/02/spark-custom-udf-example/
https://spark.apache.org/docs/2.2.0/ml-features.html#countvectorizer
https://docs.databricks.com/spark/latest/spark-sql/index.html#spark-sql-examples
https://docs.databricks.com/spark/latest/spark-sql/index.html#spark-sql-examples