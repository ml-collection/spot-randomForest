package proxy

import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import utilities._

/**
  * Created by galujanm on 5/31/17.
  */
object unsupervisedFeatureCreation {

  def run(spark: SparkSession, DF1: DataFrame): DataFrame ={

    /** These are manual cutoffs for entropy. The last bucket is a little bit larger */
    val entropyCuts20 = Array(0.0, 0.3, 0.6, 0.9, 1.2,
      1.5, 1.8, 2.1, 2.4, 2.7,
      3.0, 3.3, 3.6, 3.9, 4.2,
      4.5, 4.8, 5.1, 5.4, 10)

    val agentCounts: Map[String, Long] =
      DF1.select("useragent").rdd.map({ case Row(ua: String) => (ua, 1L) }).reduceByKey(_ + _).collect().toMap

    def addColumnIndex (df: DataFrame) = spark.createDataFrame(
      df.rdd.zipWithIndex.map{ case (row, columnindex) =>  Row.fromSeq(row.toSeq :+ columnindex) },
      StructType(df.schema.fields :+ StructField("uid", LongType, false))
    )

    val DF2 = addColumnIndex(DF1)
    //DF2.show()
    import spark.implicits._

    /** In this step we put host in propoer form */
    val DF3 = DF2.select("host", "fulluri", "resconttype","useragent","respcode" ,"reqmethod","uid").rdd.map {
      case Row (   label: Int,
      host: String,
      fulluri: String,
      resconttype: String,
      useragent: String,
      respcode: String,
      reqmethod: String,
      uid: Long  ) =>
        (label,DomainProcessor.extractDomain(host), fulluri, resconttype,useragent,respcode, reqmethod,uid )}.toDF(
      "domain","fulluri", "resconttype","useragent","respcode", "reqmethod","uid")
    //DF3.show()

    /**In this part the fixed bins are computer */
    val DF4rdd=  DF3.select("domain","fulluri", "resconttype","useragent","respcode" ,"reqmethod", "uid").rdd.map {
      case Row (
      domain: String,
      fulluri: String,
      resconttype: String,
      useragent: String,
      respcode: String,
      reqmethod: String,
      uid: Long  ) =>
        (
          if ( domain == "intel" ) 2 else if (TopDomains.TopDomains.contains(domain)) 1 else 0,
          Quantiles.bin(Entropy.stringEntropy(fulluri), entropyCuts20),
          if (resconttype.split('/').length > 0) resconttype.split('/')(0) else "unknown",
          Quantiles.logBaseXint(agentCounts(useragent), 2),
          Quantiles.logBaseXint(fulluri.length(), 2),
          if (respcode != null) respcode else "unknown",
          reqmethod,
          uid)}

    val DF4 = DF4rdd.toDF("alexa", "entropy","resconttype", "useragent","uriLen","respcode","reqmethod","uid")

    //DF4.show()

    //DF4.printSchema()
    /** In this part we check of empty strings */
    val DF5 =  DF4.select("alexa", "entropy","resconttype", "useragent","uriLen","respcode","reqmethod","uid").rdd.map {
      case Row (
      alexa: Int,
      entropy: Int,
      resconttype: String,
      useragent: Int,
      uriLen: Int,
      respcode: String,
      reqmethod: String,
      uid: Long  ) =>
        (
          alexa.toDouble,
          entropy.toDouble,
          if (resconttype == "") "empty" else resconttype,
          useragent.toDouble,
          uriLen.toDouble,
          if (respcode == "") "empty"  else respcode,
          if (reqmethod == "") "empty" else reqmethod,
          uid)}.toDF( "label","alexa", "entropy","resconttype", "useragent","uriLen","respcode","reqmethod","uid")

    //println("End of supervisedFeatureCreation")
    //DF5.show()

    return DF5

}
