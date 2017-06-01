package proxy

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{Row, _}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.types._
import utilities._
import org.apache.spark.sql.functions.{sum,when}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession,SQLContext}

import supervisedAlgorithms._


/**
  * Created by galujanm on 2/27/17.
  */
object supervisedMain {

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder
      .appName("anomaly1")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val normal = spark.read.parquet("/home/galujanm/proxy/000054_0")
    import org.apache.spark.sql.functions._
    val normalLabel = normal.withColumn("label", lit(0))
    val attackA = spark.read.parquet("/home/galujanm/proxy/proxy_attacks_A")
    val attackALabel = attackA.withColumn("label", lit(1))

    val DF1 = normalLabel.union(attackALabel)

    val DF2 = supervisedFeatureCreation.run(spark,DF1)

    val DF3 = supervisedDataPrepTrain.run(spark, DF2)

    println("This is the training DF")
    DF3.show()

    val (labelIndexer,featureIndexer) = supervisedIndexer.run(spark, DF3)

    val model = RandomForest.run(spark,DF3,labelIndexer,featureIndexer)

    model.write.overwrite().save("/home/galujanm/proxy/RF_proxy")

    val predictions = model.transform(DF3.sample(false,.05))

    predictions.select("predictedLabel", "label", "features").show(5)


    def getProb: (DenseVector => Double) = { vector => vector(0)}
    val udfgetProb = udf(getProb)

    val predictions2 = predictions.withColumn("positiveProb", udfgetProb(col("probability")))

    val predictions3 = predictions2.select("predictedLabel","positiveProb","label")

    def addRank (df: DataFrame) = spark.createDataFrame(
      df.rdd.zipWithIndex.map{ case (row, rank) =>  Row.fromSeq(row.toSeq :+ rank) },
      StructType(df.schema.fields :+ StructField("rank", LongType, false))
    )

    //val predictions5 = addRank2(predictions3.sort("positiveProb"))
    val predictions4 = predictions3.sort("positiveProb")
    val predictions5 = addRank(predictions4)
    //val predictions5 = dfZipWithIndex.run(predictions3.sort("positiveProb"))

    val predictions6 = predictions5.filter($"label" === 1.0)

    predictions6.printSchema()
    predictions6.show()

    spark.stop()
  }

}
