package proxy

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, _}
import utilities._
import org.apache.spark.sql.{SparkSession,SQLContext}
import org.apache.spark.sql.functions.{sum,when}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
  * Created by galujanm on 2/27/17.
  */
object supervisedTesting {

  case class Record (predictedLabel: Option[String], positiveProb: Option[Double],label : Option[Double])

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder
      .appName("anomaly1")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val normal = spark.read.parquet("/home/galujanm/proxy/000007_0") //.sample(false,0.70) //.limit(40000)    ///000054_0") //000007_0
    import org.apache.spark.sql.functions._
    val normalLabel = normal.withColumn("label", lit(0))
    val attacks = spark.read.parquet("/home/galujanm/proxy/proxy_attacks_B")
    val attacksLabel = attacks.withColumn("label", lit(1))

    val DF0 = normalLabel.union(attacksLabel)

    println(DF0.count())

    val resconttypeLabelsSaved = spark.read.parquet("/home/galujanm/proxy/resconttypeLabels").rdd.flatMap(_.toSeq).collect
    val resconttypeFilterUDF = udf {label:String => resconttypeLabelsSaved.contains(label)}
    //val DF00 = DF0.filter( resconttypeFilterUDF(DF0("resconttype")) )

    //println(DF00.count())

    val respcodeLabelsSaved = spark.read.parquet("/home/galujanm/proxy/respcodeLabels").rdd.flatMap(_.toSeq).collect
    val respcodeFilterUDF = udf {label:String => respcodeLabelsSaved.contains(label)}
    //val DF000 = DF00.filter( respcodeFilterUDF(DF00("respcode")) )

    //println(DF000.count())

    val reqmethodLabelsSaved = spark.read.parquet("/home/galujanm/proxy/reqmethodLabels").rdd.flatMap(_.toSeq).collect
    val reqmethodFilterUDF = udf {label:String => reqmethodLabelsSaved.contains(label)}
    //val DF1 = DF000.filter( reqmethodFilterUDF(DF000("reqmethod")) )

    //println(DF1.count())


    //We will filter certain columns to make sure we have the same labels/levels for categorical variables in train and test
    //This applies to

    val DF2 = supervisedFeatureCreation.run(spark,DF0)

    val DF3 = supervisedDataPrepTest.run(spark, DF2)

    println("This is the test DF")
    DF3.show()

    val model = PipelineModel.load("/home/galujanm/proxy/RF_proxy")

    val predictions = model.transform(DF3)

    //predictions.filter("label IS NULL").show()

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

    predictions6.show()

    spark.stop()
  }


}
