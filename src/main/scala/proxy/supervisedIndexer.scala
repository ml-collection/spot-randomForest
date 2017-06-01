package proxy

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorIndexer, VectorIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by galujanm on 2/28/17.
  */
object supervisedIndexer {

  def run(spark: SparkSession, DF: DataFrame): (StringIndexerModel,VectorIndexerModel) = {


    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label_index")
      .setOutputCol("indexedLabel").setHandleInvalid("skip")
      .fit(DF)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(10)
      .fit(DF)

    return(labelIndexer,featureIndexer)

  }

}
