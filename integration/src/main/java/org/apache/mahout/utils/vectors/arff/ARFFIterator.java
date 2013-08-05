/**
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

package org.apache.mahout.utils.vectors.arff;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.regex.Pattern;

import com.google.common.collect.AbstractIterator;
import com.google.common.io.Closeables;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

final class ARFFIterator extends AbstractIterator<Vector> {

  // This pattern will make sure a , inside a string is not a point for split.
  // Ex: "Arizona" , "0:08 PM, PDT" , 110 will be split considering "0:08 PM, PDT" as one string
  private static final Pattern COMMA_PATTERN = Pattern.compile(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");

  private final BufferedReader reader;
  private final ARFFModel model;

  ARFFIterator(BufferedReader reader, ARFFModel model) {
    this.reader = reader;
    this.model = model;
  }

  @Override
  protected Vector computeNext() {
    String line;
    try {
      while ((line = reader.readLine()) != null) {
        line = line.trim();
        if (!line.isEmpty() && !line.startsWith(ARFFModel.ARFF_COMMENT)) {
          break;
        }
      }
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
    if (line == null) {
      Closeables.closeQuietly(reader);
      return endOfData();
    }
    Vector result;
    if (line.startsWith(ARFFModel.ARFF_SPARSE)) {
      line = line.substring(1, line.length() - 1);
      String[] splits = COMMA_PATTERN.split(line);
      result = new RandomAccessSparseVector(model.getLabelSize());
      for (String split : splits) {
        split = split.trim();
        int idIndex = split.indexOf(' ');
        int idx = Integer.parseInt(split.substring(0, idIndex).trim());
        String data = split.substring(idIndex).trim();
        result.setQuick(idx, model.getValue(data, idx));
      }
    } else {
      result = new DenseVector(model.getLabelSize());
      String[] splits = COMMA_PATTERN.split(line);
      for (int i = 0; i < splits.length; i++) {
        result.setQuick(i, model.getValue(splits[i], i));
      }
    }
    //result.setLabelBindings(labelBindings);
    return result;
  }

}
