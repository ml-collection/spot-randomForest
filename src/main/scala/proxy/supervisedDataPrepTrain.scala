package proxy

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

/**
  * Created by galujanm on 3/1/17.
  */
object supervisedDataPrepTrain {

  def run(spark: SparkSession, DF5: DataFrame ): DataFrame = {

    /** The next lines save all the distinct levels for categorical variables */
    import spark.implicits._
    val resconttypeLabels = DF5.select("resconttype").distinct.map(_.getString(0)).collect
    val resconttypeLabelsDF = spark.sparkContext.parallelize(resconttypeLabels).toDF("resconttype")
    resconttypeLabelsDF.write.mode(SaveMode.Overwrite).save("/home/galujanm/proxy/resconttypeLabels")
    //val resconttypeLabelsSaved = spark.read.parquet("/home/galujanm/proxy/resconttypeLabels")

    val respcodeLabels = DF5.select("respcode").distinct.map(_.getString(0)).collect
    val respcodeLabelsDF = spark.sparkContext.parallelize(respcodeLabels).toDF("respcode")
    respcodeLabelsDF.write.mode(SaveMode.Overwrite).save("/home/galujanm/proxy/respcodeLabels")
    //val respcodeLabelsSaved = spark.read.parquet("/home/galujanm/proxy/respcodeLabels")

    val reqmethodLabels = DF5.select("reqmethod").distinct.map(_.getString(0)).collect
    val reqmethodLabelsDF = spark.sparkContext.parallelize(reqmethodLabels).toDF("reqmethod")
    reqmethodLabelsDF.write.mode(SaveMode.Overwrite).save("/home/galujanm/proxy/reqmethodLabels")
    //val reqmethodLabelsSaved = spark.read.parquet("/home/galujanm/proxy/reqmethodLabels")

    //Indexing columns
    val stringColumns = Array("label", "resconttype", "respcode", "reqmethod")
    val index_transformers: Array[org.apache.spark.ml.PipelineStage] = stringColumns.map(
      cname => new StringIndexer()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_index").setHandleInvalid("skip")
    )


    val index_pipeline = new Pipeline().setStages(index_transformers)
    index_pipeline.write.overwrite().save("/home/galujanm/proxy/index_pipeline")

    val index_model = index_pipeline.fit(DF5)
    index_model.write.overwrite().save("/home/galujanm/proxy/index_model")

    val df_indexed = index_model.transform(DF5)

    val indexColumns = df_indexed.columns.filter(x => x contains "index")

    //If you don't do the following you will get an error Exception in thread "main" java.lang.IllegalArgumentException: requirement failed: Cannot have an empty string for name.
    val one_hot_encoders: Array[org.apache.spark.ml.PipelineStage] = indexColumns.map(
      cname => new OneHotEncoder()
        .setInputCol(cname)
        .setOutputCol(s"${cname}_vec")
    )

    val one_hot_pipeline = new Pipeline().setStages(one_hot_encoders)
    one_hot_pipeline.write.overwrite().save("/home/galujanm/proxy/one_hot_pipeline")

    val one_hot_fitted = one_hot_pipeline.fit(df_indexed)
    one_hot_fitted.write.overwrite().save("/home/galujanm/proxy/one_hot_fitted")

    val df_encoded = one_hot_fitted.transform(df_indexed)

      //from Stackoverflow
    val featInd = Array("alexa", "entropy", "useragent", "uriLen", "resconttype_index_vec", "respcode_index_vec", "reqmethod_index_vec")
    val targetInd = df_encoded.columns.indexOf("label_index")


    val assembler = new VectorAssembler()
      .setInputCols(featInd)
      .setOutputCol("features")

    assembler.write.overwrite().save("/home/galujanm/proxy/assembler")

    val DF6 = assembler.transform(df_encoded)

    println("End of supervisedDataPrepTrain")

    return DF6
  }

}
