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

package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.MathHelper;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Iterator;

public class ParallelMRPJobTest extends TasteTestCase {

	private static final Logger log = LoggerFactory
			.getLogger(ParallelMRPJobTest.class);

	private File inputFile;
	private File fuFile;
	private File fgFile;
	private File outputDir;
	private File tmpDir;
	private Configuration conf;

	@Before
	@Override
	public void setUp() throws Exception {
		super.setUp();
		inputFile = getTestTempFile("prefs.txt");
		fuFile = getTestTempFile("feature-user.txt");
		fgFile = getTestTempFile("feature-geo.txt");
		outputDir = getTestTempDir("output");
		outputDir.delete();
		tmpDir = getTestTempDir("tmp");

		conf = new Configuration();
	}

	/**
	 * small integration test that runs the full job
	 * 
	 * <pre>
	 * 
	 *  user-item-matrix
	 * 
	 *          burger  hotdog  berries  icecream
	 *  dog       5       5        2        -
	 *  rabbit    2       -        3        5
	 *  cow       -       5        -        3
	 *  donkey    3       -        -        5
	 * 
	 * </pre>
	 */
	@Test
	public void completeJobToyExample() throws Exception {

		Double na = Double.NaN;
		Matrix preferences = new SparseRowMatrix(4, 5, new Vector[] {
				new DenseVector(new double[] { 1.0, 1.0, 0.3, na, 0.6 }),
				new DenseVector(new double[] { 0.3, na, 0.3, 1.0, na }),
				new DenseVector(new double[] { na, 1.0, na, 0.5, 0.2 }),
				new DenseVector(new double[] { 0.2, na, na, 1.0, 0.2 }) });

//		Matrix preferences = new SparseRowMatrix(4, 5, new Vector[] {
//				new DenseVector(new double[] { 5.0, 5.0, 2.0, na }),
//				new DenseVector(new double[] { 2.0, na, 3.0, 5.0}),
//				new DenseVector(new double[] { na, 5.0, na, 3.0}),
//				new DenseVector(new double[] { 3.0, na, na, 5.0}) });
		
		Matrix matFu = new SparseRowMatrix(4, 5, new Vector[] {
				new DenseVector(new double[] { 0.3, 0.2, 0.4, 0.7, 0.3 }),
				new DenseVector(new double[] { 0.7, 0.2, 0.4, 0.8, 0.3 }),
				new DenseVector(new double[] { 0.3, 0.2, 0.1, 0.1, 0.7 }),
				new DenseVector(new double[] { 0.9, 0.2, 0.4, 0.7, 0.3 }) });

		Matrix matFg = new SparseRowMatrix(5, 5, new Vector[] {
				new DenseVector(new double[] { 0.3, 0.1, 0.4, 0.7, 0.3 }),
				new DenseVector(new double[] { 0.7, 0.2, 0.4, 0.8, 0.3 }),
				new DenseVector(new double[] { 0.3, 0.2, 0, 0.1, 0.7 }),
				new DenseVector(new double[] { 0.9, 0.2, 0.4, 0.2, 0.3 }),
				new DenseVector(new double[] { 0.9, 0.2, 0.4, 0.2, 0.3 }) });

		writeLines(inputFile, preferencesAsText(preferences));
		writeLines(fuFile, preferencesAsText(matFu));
		writeLines(fgFile, preferencesAsText(matFg));

		ParallelMRPJob mrpFactorization = new ParallelMRPJob();
		mrpFactorization.setConf(conf);

		int numFeatures = 3;
		int numIterations = 5;
		double lambda_a = 0.4;
		double lambda_fg = 0.2;
		double lambda_fu = 0.2;
		double lambda_lg = 0.065;
		double lambda_lu = 0.065;
		double lambda_g = 0.065;
		double lambda_u = 0.065;
		double lambda_m = 0.065;
		double lr = 0.1;
		int dimFeatureGeo = 5;
		int dimFeatureUser = 5;

		mrpFactorization.run(new String[] { "--input",
				inputFile.getAbsolutePath(), "--output",
				outputDir.getAbsolutePath(), "--tempDir",
				tmpDir.getAbsolutePath(), "--lambda_a",
				String.valueOf(lambda_a), "--lambda_fg",
				String.valueOf(lambda_fg), "--lambda_fu",
				String.valueOf(lambda_fu), "--lambda_lg",
				String.valueOf(lambda_lg), "--lambda_lu",
				String.valueOf(lambda_lu), "--lambda_g",
				String.valueOf(lambda_g), "--lambda_u",
				String.valueOf(lambda_u), "--lambda_m",
				String.valueOf(lambda_m), "--numFeatures",
				String.valueOf(numFeatures), "--numIterations",
				String.valueOf(numIterations), "--lr", String.valueOf(lr),
				"--dimFeatureUser", String.valueOf(dimFeatureUser),
				"--dimFeatureGeo", String.valueOf(dimFeatureGeo),
				"--featureGeoPath", fgFile.getAbsolutePath(),
				"--featureUserPath", fuFile.getAbsolutePath() });

		Matrix Lu = MathHelper.readMatrix(conf,
				new Path(outputDir.getAbsolutePath(), "Lu/part-m-00000"),
				preferences.numRows(), numFeatures);

		Matrix Lg = MathHelper.readMatrix(conf,
				new Path(outputDir.getAbsolutePath(), "Lg/part-m-00000"),
				preferences.numCols(), numFeatures);

		Matrix m = MathHelper.readMatrix(conf,
				new Path(outputDir.getAbsolutePath(), "M/part-m-00000"),
				dimFeatureUser, dimFeatureGeo);

		StringBuilder info = new StringBuilder();
		info.append("\nA - users x items\n\n");
		info.append(MathHelper.nice(preferences));
		info.append("\nLu - users x features\n\n");
		info.append(MathHelper.nice(Lu));
		info.append("\nLg - items x features\n\n");
		info.append(MathHelper.nice(Lg));
		info.append("\nM - featuresUser x featuresGeo\n\n");
		info.append(MathHelper.nice(m));
		info.append("\nFu - users x featuresUser\n\n");
		info.append(MathHelper.nice(matFu));
		info.append("\nFg - items x featuresGeo\n\n");
		info.append(MathHelper.nice(matFg));
		Matrix res = Lu.times(Lg.transpose());
		System.err.println(res.numRows() + "\t" + res.numCols());
		Matrix res2 = (matFu.times(m).times(matFg.transpose()));
		System.err.println(res2.numRows() + "\t" + res2.numCols());
		res.plus(matFu.times(m).times(matFg.transpose()));
		info.append("\nres\n\n");
		info.append(MathHelper.nice(res));
		info.append('\n');

		log.info(info.toString());

		RunningAverage avg = new FullRunningAverage();
		Iterator<MatrixSlice> sliceIterator = preferences.iterateAll();
		while (sliceIterator.hasNext()) {
			MatrixSlice slice = sliceIterator.next();
			Iterator<Vector.Element> elementIterator = slice.vector()
					.iterateNonZero();
			while (elementIterator.hasNext()) {
				Vector.Element e = elementIterator.next();
				if (!Double.isNaN(e.get())) {
					double pref = e.get();
					double err = pref - res.get(slice.index(), e.index());
					avg.addDatum(err * err);
					log.info(
							"Comparing preference of user [{}] towards item [{}], was [{}] estimate is [{}]",
							new Object[] { slice.index(), e.index(), pref,
									res.get(slice.index(), e.index()) });
				}
			}
		}
		double rmse = Math.sqrt(avg.getAverage());
		log.info("RMSE: {}", rmse);

		assertTrue(rmse < 200);
	}

	protected static String preferencesAsText(Matrix preferences) {
		StringBuilder prefsAsText = new StringBuilder();
		String separator = "";
		Iterator<MatrixSlice> sliceIterator = preferences.iterateAll();
		while (sliceIterator.hasNext()) {
			MatrixSlice slice = sliceIterator.next();
			Iterator<Vector.Element> elementIterator = slice.vector()
					.iterateNonZero();
			while (elementIterator.hasNext()) {
				Vector.Element e = elementIterator.next();
				if (!Double.isNaN(e.get())) {
					prefsAsText.append(separator).append(slice.index())
							.append(',').append(e.index()).append(',')
							.append(e.get());
					separator = "\n";
				}
			}
		}
		return prefsAsText.toString();
	}

}
