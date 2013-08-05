/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.clustering.conversion.meanshift;

import java.io.IOException;
import java.util.Collection;
import java.util.regex.Pattern;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.meanshift.MeanShiftCanopy;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import com.google.common.collect.Lists;

public class InputMapper extends Mapper<LongWritable,Text,Text,ClusterWritable> {
  
  private static final Pattern SPACE = Pattern.compile(" ");
  
  private int nextCanopyId;
  
  private ClusterWritable cw = new ClusterWritable();
  
  @Override
  protected void map(LongWritable key, Text values, Context context) throws IOException, InterruptedException {
    String[] numbers = SPACE.split(values.toString());
    // sometimes there are multiple separator spaces
    Collection<Double> doubles = Lists.newArrayList();
    for (String value : numbers) {
      if (!value.isEmpty()) {
        doubles.add(Double.valueOf(value));
      }
    }
    // ignore empty lines in input data
    if (!doubles.isEmpty()) {
      Vector point = new DenseVector(doubles.size());
      int index = 0;
      for (Double d : doubles) {
        point.set(index++, d);
      }
      cw.setValue(new MeanShiftCanopy(point, nextCanopyId++, new EuclideanDistanceMeasure()));
      context.write(new Text(), cw);
    }
  }
}
