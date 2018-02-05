package com.ml.pipe

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql._

object BikeSharingPrediction {

  def configLogging() = {
    import org.apache.log4j.{Level, Logger}
    Logger.getRootLogger().setLevel(Level.FATAL)
  }

  def main(args:Array[String]) = {

    // Create a new SparkSession interface
    val spark = SparkSession
      .builder.appName("BikeSharingPrediction")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "file:///c:/tmp")
      .getOrCreate()

    configLogging()

    val hour_df = spark.read.format("csv")
      .option("header", "true")
      .load("hour.csv")

    val hour_df_drop_cols = hour_df.drop("instant").drop("registered")
      .drop("dteday").drop("yr").drop("casual")

    // Convert all the colums from String to double for fitting VectorAssembler
    val hour_df_cast_double = hour_df_drop_cols.schema.foldLeft(hour_df_drop_cols) {
      case (acc, col) => acc.withColumn(col.name, hour_df_drop_cols(col.name).cast("double"))
    }

    //val hour_df_cast_double = hour_df_drop_cols.map(_.toString())

   /// println(hour_df_cast_double)
    // Randomly split the dataframe 7:3 -> train_dataframe: test_dataframe
    val Array(train_df, test_df) = hour_df_cast_double.randomSplit(Array(0.7, 0.3))
    train_df.cache()
    test_df.cache()

    //hour_df_cast_double.printSchema()

    val featuresCol = hour_df_cast_double.drop("cnt").columns

   featuresCol.foreach(println)

    // build the pipeline
    val assembler = new VectorAssembler()
      .setInputCols(featuresCol)
      .setOutputCol("aFeatures")

    // Automatically identify categorical features, and index them.
    val vectorIndexer = new VectorIndexer()
      .setInputCol("aFeatures")
      .setOutputCol("features")
      .setMaxCategories(24)

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("cnt")
      .setFeaturesCol("features")

    // Chain indexer and tree in a Pipeline
    val dt_pipeline = new Pipeline()
      .setStages(Array(assembler, vectorIndexer, dt))

    // Train model.  This also runs the indexer.
    val dt_pipelineModel = dt_pipeline.fit(train_df)

    // Use pipelineModel.transform to predict
    val predicted_df = dt_pipelineModel.transform(test_df)

    predicted_df.select("season",
      "mnth",
      "hr",
      "cnt",
      "prediction").show(10)

    // Evaluate the accuracy
    // Select (prediction, true label: cnt) and compute test error
    val evaluator = new RegressionEvaluator()
      .setLabelCol("cnt")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    // Calculate RMSE
    val rmse = evaluator.evaluate(predicted_df)

    println("Root Mean Squared Error (RMSE) on test data = " + rmse)


    /**
      *  Implement TrainValidation to find the best model selection
      */

    // Use ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.
    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(5, 10, 15, 20))
      .addGrid(dt.maxBins, Array(25, 35, 45, 50))
      .build()

    // Build trainValidationSplit
    val tvs = new TrainValidationSplit()
      .setEstimator(dt)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)

    // Create tvs_pipeline to include tvs we just built
    val tvs_pipeline = new Pipeline()
      .setStages(Array(assembler, vectorIndexer, dt, tvs))

    // Use tvs.pipeline.fit to train and valid, and it will return tvs_pipelineModel
    // Run train validation split, and choose the best set of parameters.
   // val tvs_pipelineModel = tvs_pipeline.fit(train_df)

 //   val bestModel = tvs_pipelineModel.stages

  //  tvs_pipelineModel.transform(test_df)
  //    .select("features", "label", "prediction")
   //   .show()

  //  bestModel.foreach(println)







  }

}
