
from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.types import ArrayType, StringType, DoubleType, FloatType, IntegerType

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.linalg import Vectors

import cleantext

states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
# IMPORT OTHER MODULES HERE
def clean(body):
    parsed_text, unigrams, bigrams, trigrams = cleantext.sanitize(body)
    retMe = unigrams.split(" ")
    retMe.extend(bigrams.split(" "))
    retMe.extend(trigrams.split(" "))
    return retMe

def cutoff(link_id):
    return link_id[3:]
  
def getProbability(prob):
    return float(prob[1])
def validstate(state):
    if(state in states):
        return 1
    else:
        return 0

def main(context):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED
    
    # TASK 1
    try:
        labeled = sqlContext.read.parquet("labeled_parquet")
    except:
        labeled = sqlContext.read.format("csv").option("header", "true").load("labeled_data.csv")
        labeled.write.parquet("labeled_parquet")
    try:
        comments =sqlContext.read.parquet("comments-minimal_parquet")
    except:
        comments = sqlContext.read.json("comments-minimal.json.bz2")
        comments.write.parquet("comments-minimal_parquet")
    try:
        submissions = sqlContext.read.parquet("submissions_parquet")
    except:
        submissions = sqlContext.read.json("submissions.json.bz2")
        submissions.write.parquet("submissions_parquet")
    
    # TASK 2
    comments.createOrReplaceTempView("comments")
    labeled.createOrReplaceTempView("labeled")
    rel_comments = sqlContext.sql("select \
        labeled.Input_id, labeled.labeldem, labeled.labelgop, labeled.labeldjt, body \
        from comments join labeled on id=Input_id")
    # TASK 3
    
    # TASK 4
    #REMOVE
    """
    try:
        rel_comments = sqlContext.read.parquet("rel_comments_parquet")
    except:
        rel_comments.write.parquet("rel_comments_parquet")
    """

    rel_comments.createOrReplaceTempView("rel_comments")
    sqlContext.registerFunction("sanitize", clean, ArrayType(StringType()))


    # TASK 5
    withGrams = sqlContext.sql("select Input_id, labeldem, labelgop, labeldjt,\
     sanitize(body) as sanitized from rel_comments")
    
    
    #REMOVE
    
    
    
    # TASK 6A
    #REMOVE
    """
    try:
        withGrams = sqlContext.read.parquet("withGrams_parquet")
    except:
        withGrams.write.parquet("withGrams_parquet")
    """
    withGrams.printSchema()

    
    cv = CountVectorizer(inputCol="sanitized", outputCol="vectorized", minDF=5.0, binary=True)
    model = cv.fit(withGrams)
    withVectors = model.transform(withGrams)
    withVectors.show()
    
    # TASK 6B
    withVectors.createOrReplaceTempView("withVectors")    
    pos_neg = sqlContext.sql("select *, \
        case when labeldjt = 1 then 1 else 0 end as positive, \
        case when labeldjt = -1 then 1 else 0 end as negative \
        from withVectors")
    pos_neg.show()

    pos_neg.createOrReplaceTempView("pos_neg")
    pos = sqlContext.sql("select positive as label, vectorized from pos_neg")
    #pos = posq.createDataFrame()
    pos.show()
    

    # TASK 7
    # Initialize two logistic regression models.
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="label", featuresCol="vectorized", maxIter=10)

    neg = sqlContext.sql("select negative as label, vectorized from pos_neg")
    #neg = negq.createDataFrame()
    neg.show()
    
    neglr = LogisticRegression(labelCol="label", featuresCol="vectorized", maxIter=10)
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator()
    negEvaluator = BinaryClassificationEvaluator()
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=5)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=5)
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50

    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)
     
    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.save("www/pos.model")
    negModel.save("www/neg.model")
    
    
    posModel = CrossValidatorModel.load("www/pos.model")
    negModel = CrossValidatorModel.load("www/neg.model")
    
    # TASK 8  
    submissions.createOrReplaceTempView("submissions")
    sqlContext.registerFunction("cutoff", cutoff, StringType())
    """joinstuff = sqlContext.sql("select comments.created_utc as timestamp, submissions.id as link_id, comments.author_flair_text as state, comments.id as comment_id, comments.body as body \
        from comments inner join submissions on cutoff(comments.link_id) = submissions.id")"""
    joinstuff = sqlContext.sql("select comments.created_utc as timestamp, comments.score as c_score, submissions.score as s_score, submissions.id as link_id, comments.author_flair_text as state, comments.id as comment_id, comments.body as body \
        from comments inner join submissions on cutoff(comments.link_id) = submissions.id")
    joinstuff.show()
    
    # TASK 9
    
    try:
        tempUnseen = sqlContext.read.parquet("tempUnseen_parquet")
    except:
        tempUnseen = sqlContext.sql("select id, body from comments where body not RLIKE '^&gt' AND body not LIKE '%/s%'")
        tempUnseen = tempUnseen.sample(False, 0.2, None)
        tempUnseen.write.parquet("tempUnseen_parquet")
  
    tempUnseen.createOrReplaceTempView("tempUnseen")
    tempUnseen.show()
      
    try:
        unseenWithGrams = sqlContext.read.parquet("unseenWithGrams_parquet")
    except:
        unseenWithGrams = sqlContext.sql("select id, sanitize(body) as sanitized from tempUnseen")
        unseenWithGrams.write.parquet("unseenWithGrams_parquet")
      
    
    unseenWithGrams.createOrReplaceTempView("unseenWithGrams")
    unseenWithGrams.printSchema()
    unseenWithGrams.show()
    
    #do the count vectorizer again
    unseenWithVectors = model.transform(unseenWithGrams)
    unseenWithVectors.show()
    unseenWithVectors.createOrReplaceTempView("unseenWithVectors")
    
    posResult = posModel.transform(unseenWithVectors)
    negResult = negModel.transform(unseenWithVectors)
    
    posResult.createOrReplaceTempView("posResult")
    negResult.createOrReplaceTempView("negResult")
    posResult.show()
    negResult.show()
    
    sqlContext.registerFunction("getProbability", getProbability, FloatType())
    try:
        unseenComments = sqlContext.read.parquet("unseenComments_parquet")
    except:
        unseenComments = sqlContext.sql("select posResult.id as id, posResult.vectorized as vectorized, \
            posResult.probability as pos_prob, negResult.probability as neg_prob, \
            case when getProbability(posResult.probability) > 0.2 then 1 else 0 end as pos, \
            case when getProbability(negResult.probability) > 0.25 then 1 else 0 end as neg \
            from posResult inner join negResult on posResult.id = negResult.id")
        unseenComments.write.parquet("unseenComments_parquet")

    unseenComments.createOrReplaceTempView("unseenComments")
    unseenComments.show()

    joinstuff.createOrReplaceTempView("joinstuff")

    try: 
        djtSentiment = sqlContext.read.parquet("djtSentiment_parquet")
    except:
        djtSentiment = sqlContext.sql("select joinstuff.timestamp as timestamp, joinstuff.link_id as link_id,  \
        joinstuff.comment_id as comment_id, joinstuff.state as state, joinstuff.body as body, unseenComments.pos as pos, unseenComments.neg as neg, joinstuff.c_score as c_score, joinstuff.s_score as s_score \
        from joinstuff inner join unseenComments on joinstuff.comment_id = unseenComments.id")
        djtSentiment.write.parquet("djtSentiment_parquet")

    djtSentiment.createOrReplaceTempView("djtSentiment")
    djtSentiment.show()
    

    # TASK 10
    sqlContext.registerFunction("validstate", validstate, IntegerType())
    
    djtPosNegPerc = sqlContext.sql("select SUM(pos)/COUNT(*) as pos_avg, SUM(neg)/COUNT(*) as neg_avg from djtSentiment group by link_id")
    djtDayPerc = sqlContext.sql("select DATE(FROM_UNIXTIME(timestamp)) as date, SUM(pos)/COUNT(*) as pos_avg_day, SUM(neg)/COUNT(*) as neg_avg_day \
        from djtSentiment group by date")
    djtStatePerc = sqlContext.sql("select state, SUM(pos)/COUNT(*) as Positive, SUM(neg)/COUNT(*) as Negative from djtSentiment \
        where validstate(state) = 1 group by state")
    djtC_score = sqlContext.sql("select c_score as comment_score, SUM(pos)/COUNT(*) as Positive, SUM(neg)/COUNT(*) as Negative \
        from djtSentiment where validstate(state) = 1 group by c_score")
    djtS_score = sqlContext.sql("select s_score as submission_score, SUM(pos)/COUNT(*) as Positive, SUM(neg)/COUNT(*) as Negative \
        from djtSentiment where validstate(state) = 1 group by s_score")

    
    
    import shutil
    import os
    import glob
    
    try:
        shutil.rmtree("time_data.csvfolder")
        shutil.rmtree("state_data.csvfolder")
        shutil.rmtree("submission_score.csvfolder")
        shutil.rmtree("comment_score.csvfolder")
    except:
        pass

    try:
        djtDayPerc.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("time_data.csvfolder")
        djtStatePerc.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("state_data.csvfolder")
        djtC_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("comment_score.csvfolder")
        djtS_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("submission_score.csvfolder")
        for file in glob.glob("time_data.csvfolder/*.csv"):
            os.rename(file, "time_data.csv")
        for file in glob.glob("state_data.csvfolder/*.csv"):
            os.rename(file, "state_data.csv")
        for file in glob.glob("comment_score.csvfolder/*.csv"):
            os.rename(file, "comment_score.csv")
        for file in glob.glob("submission_score.csvfolder/*.csv"):
            os.rename(file, "submission_score.csv")
    except:
        pass


    try:
        shutil.rmtree("time_data.csvfolder")
        shutil.rmtree("state_data.csvfolder")
        shutil.rmtree("submission_score.csvfolder")
        shutil.rmtree("comment_score.csvfolder")
    except:
        pass

  
        #DEMOCRATS - EXTRA CREDIT
    dem_pos_neg = sqlContext.sql("select *, \
        case when labeldem = 1 then 1 else 0 end as positive, \
        case when labeldem = -1 then 1 else 0 end as negative \
        from withVectors")
    dem_pos_neg.show()

    dem_pos_neg.createOrReplaceTempView("dem_pos_neg")

    dem_pos = sqlContext.sql("select positive as label, vectorized from dem_pos_neg")
    dem_pos.show()
    dem_neg = sqlContext.sql("select negative as label, vectorized from dem_pos_neg")
    dem_neg.show()
    

    dem_posTrain, dem_posTest = dem_pos.randomSplit([0.5, 0.5])
    dem_negTrain, dem_negTest = dem_neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    dem_posModel = posCrossval.fit(dem_posTrain)
    print("Training negative classifier...")
    dem_negModel = negCrossval.fit(dem_negTrain)
     
    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    dem_posModel.save("www/dem_pos.model")
    dem_negModel.save("www/dem_neg.model")


    dem_posModel = CrossValidatorModel.load("www/dem_pos.model")
    dem_negModel = CrossValidatorModel.load("www/dem_neg.model")


    dem_posResult = dem_posModel.transform(unseenWithVectors)
    dem_negResult = dem_negModel.transform(unseenWithVectors)
    

    dem_posResult.createOrReplaceTempView("dem_posResult")
    dem_negResult.createOrReplaceTempView("dem_negResult")
    dem_posResult.show()
    dem_negResult.show()


    try:
        dem_unseenComments = sqlContext.read.parquet("dem_unseenComments_parquet")
    except:
        dem_unseenComments = sqlContext.sql("select dem_posResult.id as id, dem_posResult.vectorized as vectorized, \
            dem_posResult.probability as pos_prob, dem_negResult.probability as neg_prob, \
            case when getProbability(dem_posResult.probability) > 0.2 then 1 else 0 end as pos, \
            case when getProbability(dem_negResult.probability) > 0.25 then 1 else 0 end as neg \
            from dem_posResult inner join dem_negResult on dem_posResult.id = dem_negResult.id")
        dem_unseenComments.write.parquet("dem_unseenComments_parquet")

    dem_unseenComments.createOrReplaceTempView("dem_unseenComments")
    dem_unseenComments.show()

    

    try: 
        demSentiment = sqlContext.read.parquet("dem_Sentiment_parquet")
    except:
        demSentiment = sqlContext.sql("select joinstuff.timestamp as timestamp, joinstuff.link_id as link_id,  \
        joinstuff.comment_id as comment_id, joinstuff.state as state, joinstuff.body as body, dem_unseenComments.pos as pos, \
        dem_unseenComments.neg as neg, joinstuff.c_score as c_score, joinstuff.s_score as s_score \
        from joinstuff inner join dem_unseenComments on joinstuff.comment_id = dem_unseenComments.id")
        demSentiment.write.parquet("dem_Sentiment_parquet")

    demSentiment.createOrReplaceTempView("demSentiment")
    demSentiment.show()
  
    demPosNegPerc = sqlContext.sql("select SUM(pos)/COUNT(*) as pos_avg, SUM(neg)/COUNT(*) as neg_avg from demSentiment group by link_id")
    demDayPerc = sqlContext.sql("select DATE(FROM_UNIXTIME(timestamp)) as date, SUM(pos)/COUNT(*) as pos_avg_day, SUM(neg)/COUNT(*) as neg_avg_day \
        from demSentiment group by date")
    demStatePerc = sqlContext.sql("select state, SUM(pos)/COUNT(*) as Positive, SUM(neg)/COUNT(*) as Negative from demSentiment \
        where validstate(state) = 1 group by state")
    demC_score = sqlContext.sql("select c_score as comment_score, SUM(pos)/COUNT(*) as Positive, SUM(neg)/COUNT(*) as Negative \
        from demSentiment where validstate(state) = 1 group by c_score")
    demS_score = sqlContext.sql("select s_score as submission_score, SUM(pos)/COUNT(*) as Positive, SUM(neg)/COUNT(*) as Negative \
        from demSentiment where validstate(state) = 1 group by s_score")
        
        
    try:
        shutil.rmtree("dem_time_data.csvfolder")
        shutil.rmtree("dem_state_data.csvfolder")
        shutil.rmtree("dem_submission_score.csvfolder")
        shutil.rmtree("dem_comment_score.csvfolder")
    except:
        pass

    try:
        demDayPerc.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("dem_time_data.csvfolder")
        demStatePerc.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("dem_state_data.csvfolder")
        demC_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("dem_comment_score.csvfolder")
        demS_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("dem_submission_score.csvfolder")
        for file in glob.glob("dem_time_data.csvfolder/*.csv"):
            os.rename(file, "dem_time_data.csv")
        for file in glob.glob("dem_state_data.csvfolder/*.csv"):
            os.rename(file, "dem_state_data.csv")
        for file in glob.glob("dem_comment_score.csvfolder/*.csv"):
            os.rename(file, "dem_comment_score.csv")
        for file in glob.glob("dem_submission_score.csvfolder/*.csv"):
            os.rename(file, "dem_submission_score.csv")
    except:
        pass


    try:
        shutil.rmtree("dem_time_data.csvfolder")
        shutil.rmtree("dem_state_data.csvfolder")
        shutil.rmtree("dem_submission_score.csvfolder")
        shutil.rmtree("dem_comment_score.csvfolder")
    except:
        pass



    #GOP
    gop_pos_neg = sqlContext.sql("select *, \
        case when labelgop = 1 then 1 else 0 end as positive, \
        case when labelgop = -1 then 1 else 0 end as negative \
        from withVectors")
    gop_pos_neg.show()

    gop_pos_neg.createOrReplaceTempView("gop_pos_neg")

    gop_pos = sqlContext.sql("select positive as label, vectorized from gop_pos_neg")
    gop_pos.show()
    gop_neg = sqlContext.sql("select negative as label, vectorized from gop_pos_neg")
    gop_neg.show()
    

    gop_posTrain, gop_posTest = gop_pos.randomSplit([0.5, 0.5])
    gop_negTrain, gop_negTest = gop_neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    gop_posModel = posCrossval.fit(gop_posTrain)
    print("Training negative classifier...")
    gop_negModel = negCrossval.fit(gop_negTrain)
     
    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    gop_posModel.save("www/gop_pos.model")
    gop_negModel.save("www/gop_neg.model")


    gop_posModel = CrossValidatorModel.load("www/gop_pos.model")
    gop_negModel = CrossValidatorModel.load("www/gop_neg.model")

    gop_posResult = gop_posModel.transform(unseenWithVectors)
    gop_negResult = gop_negModel.transform(unseenWithVectors)

    gop_posResult.createOrReplaceTempView("gop_posResult")
    gop_negResult.createOrReplaceTempView("gop_negResult")
    gop_posResult.show()
    gop_negResult.show()


    try:
        gop_unseenComments = sqlContext.read.parquet("gop_unseenComments_parquet")
    except:
        gop_unseenComments = sqlContext.sql("select gop_posResult.id as id, gop_posResult.vectorized as vectorized, \
            gop_posResult.probability as pos_prob, gop_negResult.probability as neg_prob, \
            case when getProbability(gop_posResult.probability) > 0.2 then 1 else 0 end as pos, \
            case when getProbability(gop_negResult.probability) > 0.25 then 1 else 0 end as neg \
            from gop_posResult inner join gop_negResult on gop_posResult.id = gop_negResult.id")
        gop_unseenComments.write.parquet("gop_unseenComments_parquet")

    gop_unseenComments.createOrReplaceTempView("gop_unseenComments")
    gop_unseenComments.show()

    

    try: 
        gopSentiment = sqlContext.read.parquet("gop_Sentiment_parquet")
    except:
        gopSentiment = sqlContext.sql("select joinstuff.timestamp as timestamp, joinstuff.link_id as link_id,  \
        joinstuff.comment_id as comment_id, joinstuff.state as state, joinstuff.body as body, gop_unseenComments.pos as pos, \
        gop_unseenComments.neg as neg, joinstuff.c_score as c_score, joinstuff.s_score as s_score \
        from joinstuff inner join gop_unseenComments on joinstuff.comment_id = gop_unseenComments.id")
        gopSentiment.write.parquet("gop_Sentiment_parquet")

    gopSentiment.createOrReplaceTempView("gopSentiment")
    gopSentiment.show()


    gopPosNegPerc = sqlContext.sql("select SUM(pos)/COUNT(*) as pos_avg, SUM(neg)/COUNT(*) as neg_avg from gopSentiment group by link_id")
    gopDayPerc = sqlContext.sql("select DATE(FROM_UNIXTIME(timestamp)) as date, SUM(pos)/COUNT(*) as pos_avg_day, SUM(neg)/COUNT(*) as neg_avg_day \
        from gopSentiment group by date")
    gopStatePerc = sqlContext.sql("select state, SUM(pos)/COUNT(*) as Positive, SUM(neg)/COUNT(*) as Negative from gopSentiment \
        where validstate(state) = 1 group by state")
    gopC_score = sqlContext.sql("select c_score as comment_score, SUM(pos)/COUNT(*) as Positive, SUM(neg)/COUNT(*) as Negative \
        from gopSentiment where validstate(state) = 1 group by c_score")
    gopS_score = sqlContext.sql("select s_score as submission_score, SUM(pos)/COUNT(*) as Positive, SUM(neg)/COUNT(*) as Negative \
        from gopSentiment where validstate(state) = 1 group by s_score")
        
    try:
        shutil.rmtree("gop_time_data.csvfolder")
        shutil.rmtree("gop_state_data.csvfolder")
        shutil.rmtree("gop_submission_score.csvfolder")
        shutil.rmtree("gop_comment_score.csvfolder")
    except:
        pass

    try:
        gopDayPerc.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("gop_time_data.csvfolder")
        gopStatePerc.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("gop_state_data.csvfolder")
        gopC_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("gop_comment_score.csvfolder")
        gopS_score.repartition(1).write.format("com.databricks.spark.csv").option("header","true").save("gop_submission_score.csvfolder")
        for file in glob.glob("gop_time_data.csvfolder/*.csv"):
            os.rename(file, "gop_time_data.csv")
        for file in glob.glob("gop_state_data.csvfolder/*.csv"):
            os.rename(file, "gop_state_data.csv")
        for file in glob.glob("gop_comment_score.csvfolder/*.csv"):
            os.rename(file, "gop_comment_score.csv")
        for file in glob.glob("gop_submission_score.csvfolder/*.csv"):
            os.rename(file, "gop_submission_score.csv")
    except:
        pass


    try:
        shutil.rmtree("gop_time_data.csvfolder")
        shutil.rmtree("gop_state_data.csvfolder")
        shutil.rmtree("gop_submission_score.csvfolder")
        shutil.rmtree("gop_comment_score.csvfolder")
    except:
        pass

    

if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)