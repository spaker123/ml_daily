package com.ml.pipe

import org.apache.log4j._
import org.apache.spark.mllib.recommendation._
import org.apache.spark.{SparkConf, SparkContext}

object MusicRecommend {

  def main(args:Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.FATAL)

    val conf = new SparkConf().setMaster("local").setAppName("runrecommender")

    val sc = new SparkContext(conf)

    val rawUserArtistData = sc.textFile("user_artist_data.txt")

    val rawArtistData = sc.textFile("artist_data.txt")
    val artistByID = rawArtistData.map{ line =>
      val (id, name) = line.span(_ != '\t')
      if (name.isEmpty) {
        None
      } else {
        try {
          Some((id.toInt, name.trim))
        } catch {
          case e: NumberFormatException => None
        }
      }
    }

    val rawArtistAlias = sc.textFile("artist_alias.txt")
    val artistAlias = rawArtistAlias.flatMap{ line =>
      val tokens = line.split('\t')
      if (tokens(0).isEmpty) {
        None
      } else {
        Some((tokens(0).toInt, tokens(1).toInt))
        //(tokens(0).toInt, tokens(1).toInt)
      }
    }.collectAsMap()

    val bArtistAlias = sc.broadcast(artistAlias)

    val trainData = rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      Rating(userID, finalArtistID, count)
    }.cache()

    val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

    val rawArtistsForUser = rawUserArtistData.map(_.split(' ')).filter {
      case Array(user,_,_) => user.toInt == 2093760
    }

    val existingProducts = rawArtistsForUser.map {
      case Array(_,artist,_) => artist.toInt
    }.collect().toSet
    existingProducts.foreach(println)

    artistByID.filter {
      case Some((id, name)) => existingProducts.contains(id)
      case None => false
    }.collect().foreach(println)

    val recommendations = model.recommendProducts(2093760, 5)
    recommendations.foreach(println)

    val recommendationProduct = recommendations.map(_.product).toSet

    artistByID.filter{
      case Some((id, name)) =>
        recommendationProduct.contains(id)
      case None => false
    }.collect().foreach(println)


  }
}
