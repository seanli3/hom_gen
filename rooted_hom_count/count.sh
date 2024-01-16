JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
./spark-2.4.8-bin-hadoop2.7/bin/spark-submit --class org.apache.spark.disc.SubgraphCounting ../DISC/DISC-assembly-0.1.jar -p ../DISC/disc_local.properties -d ../DISC/examples/debugData.txt -q "t50 A-D;A-E;B-C;B-D;B-E;C-D;C-E;" -e Result -u HOM -c A -o out.csv
