package com.ml.pipe

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Citation: https://archive.ics.uci.edu/ml/datasets/covertype
  * Predicting forest cover type from cartographic variables.
  */

object DecisionTreeClassification {

  def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
    val countsByCategory = data.map(_.label).countByValue()
    val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
    counts.map(_.toDouble / counts.sum)
  }

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.FATAL)

    val conf = new SparkConf().setMaster("local[*]").setAppName("forest")

    val sc = new SparkContext(conf)

    val rawData = sc.textFile("covtype.data")

    val data = rawData.map{ line =>

      val values = line.split(',').map(_.toDouble)

      /**
        * 1 -- Rawah Wilderness Area
        * 2 -- Neota Wilderness Area
        * 3 -- Comanche Peak Wilderness Area
        * 4 -- Cache la Poudre Wilderness Area
        */
      val wilderness_area = values.slice(10, 14).indexOf(1.0).toDouble

      /**
        * Soil Types by OneHotEncoding
        */
      val soil_type = values.slice(14, 54).indexOf(1.0).toDouble

      val featureVector = Vectors.dense(values.slice(0, 10) :+ wilderness_area :+ soil_type)

      // Retrieve label column, which is the last one
      val label = values.last - 1

      LabeledPoint(label, featureVector)
    }

    /**
      * randomly split the trainData, validationData, testData
      */
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()

    //  val forest = RandomForest.trainClassifier(
    //   trainData, 7, Map(10 -> 4, 11 -> 40), 20,
    //   "auto", "entropy", 30, 300
    // )

    val model = DecisionTree.trainClassifier(
      trainData.union(cvData), 7, Map[Int,Int](), "entropy", 30, 40
      // trainData,             7, Map[Int,Int](), "gini",     4, 100
    )

    val metrics = getMetrics(model, trainData.union(cvData))

    val evaluations = for (impurity <- Array("gini", "entropy");
                           depth    <- Array(10, 20, 30);
                           bins     <- Array(10, 40, 70, 100, 130, 160, 190, 220, 250, 275, 300))
      yield {
        val model = DecisionTree.trainClassifier(
          trainData, 7, Map[Int, Int](), //Map(10 -> 4, 11 -> 40),
          impurity, depth, bins)

        val trainAccuracy = getMetrics(model, trainData).accuracy
        val cvAccuracy = getMetrics(model, cvData).accuracy

        ((impurity, depth, bins), (trainAccuracy, cvAccuracy))
      }

    evaluations.sortBy(_._2).reverse.foreach(println)

    // model.save(sc, "Model")
    //val sameModel = DecisionTreeModel.load(sc, "testModel")

    println("Print classification tree model:\n" + model.toDebugString)

    sc.stop()

  }

}


/**
val data = rawData.map { line =>
      val values = line.split(',').map(_.toDouble)
      val featureVector = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }
  **/


// println(metrics.confusionMatrix)
// println(metrics.accuracy)

//(0 until 7).map(
//   cat => (metrics.precision(cat), metrics.recall(cat))
//  ).foreach(println)

/**
val trainPriorPorbabilities = classProbabilities(trainData)

    val cvPriorProbabilities = classProbabilities(cvData)

    trainPriorPorbabilities.zip(cvPriorProbabilities).map {
      case (trainProb, cvProb) => trainProb * cvProb
    }.sum
  **/


// val predictionsAndLabels = cvData.map(example => (model.predict(example.features), example.label))
// val accuracy = new MulticlassMetrics(predictionsAndLabels).accuracy
// ((impurity, depth, bins), accuracy)