package proxy
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{SparkSession, _}


/**
  * Created by galujanm on 3/1/17.
  */
object supervisedDataPrepTest {

  def run(spark: SparkSession, DF5: DataFrame ): DataFrame = {


    //val index_pipeline = Pipeline.load("/home/galujanm/proxy/index_pipeline")
    //val index_model = index_pipeline.fit(DF5)

    val index_model = PipelineModel.load("/home/galujanm/proxy/index_model")

    val df_indexed = index_model.transform(DF5)

    //val one_hot_pipeline = Pipeline.load("/home/galujanm/proxy/one_hot_pipeline")
    //val df_encoded = one_hot_pipeline.fit(df_indexed).transform(df_indexed)

    val one_hot_fitted = PipelineModel.load("/home/galujanm/proxy/one_hot_fitted")
    val df_encoded = one_hot_fitted.transform(df_indexed)

    val assembler = VectorAssembler.load("/home/galujanm/proxy/assembler")

    val DF6 = assembler.transform(df_encoded)

    return DF6

  }

}
