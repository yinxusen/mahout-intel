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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.mapreduce.MergeVectorsCombiner;
import org.apache.mahout.common.mapreduce.MergeVectorsReducer;
import org.apache.mahout.common.mapreduce.TransposeMapper;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.als.AlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Mutual Reinforcement Propagation method of geographic recommendation,
 * friendship prediction and attribute inference. numIterations: the number of
 * iterations(outer) numFeatures: the number of latent features dimFeatureUser:
 * the dimension of observed features of users dimUserRelation: the dimension of
 * observed features of geography lr: learning rate of SGD for matrix M
 * lambda_a: weight of importance of matrix A (our data set) lambda_fg: weight
 * of importance of matrix Fg (Feature of geography) lambda_fu: weight of
 * importance of matrix Fu (Feature of user) lambda_lg: weight of penalty of
 * latent matrix Lg (Latent feature of geography) lambda_lu: weight of penalty
 * of latent matrix Lu (Latent feature of user) lambda_g: weight of penalty of
 * parameter matrix G lambda_u: weight of penalty of parameter matrix U
 * lambda_m: weight of penalty of parameter matrix M
 * 
 * For concrete details, please see: Regression-based latent factor models
 * (Deepak et al.) Like like alike: jointly friendship and interest propagation
 * in social networks (Shuang Hong Yang et al.) Mutual reinforcement propagation
 * on friendship prediction and attribute inference (Xusen Yin et al.)
 * 
 * For parallel implementation of SGD and others, please see: Mapreduce is good
 * enough? (Jimmy Lin)
 */
public class ParallelMRPJob extends AbstractJob {

	private static final Logger log = LoggerFactory
			.getLogger(ParallelMRPJob.class);

	static final String NUM_FEATURES = ParallelMRPJob.class.getName()
			+ ".numFeatures";
	static final String LAMBDA_PRINCIPAL = ParallelMRPJob.class.getName()
			+ ".lambdaPricipal";
	static final String LAMBDA_MINOR = ParallelMRPJob.class.getName()
			+ ".lambdaMinor";
	static final String FEATURE_MATRIX = ParallelMRPJob.class.getName()
			+ ".featureMatrix";
	static final String PATH_TO_VECTOR_SECOND = ParallelMRPJob.class.getName()
			+ ".path_to_fg_transpose";

	private int numIterations;
	private int numFeatures;
	private int dimFeatureUser;
	private int dimUserRelation;
	private double lr;
	private double lambda_a;
	private double lambda_fg;
	private double lambda_fu;
	private double lambda_lg;
	private double lambda_lu;
	private double lambda_g;
	private double lambda_u;

	private PathIndex luIndex = new PathIndex();
	private PathIndex lgIndex = new PathIndex();
	private PathIndex luIndexviaA = new PathIndex();
	private PathIndex luIndexviaF = new PathIndex();
	private PathIndex lgIndexviaA = new PathIndex();
	private PathIndex lgIndexviaF = new PathIndex();
	private PathIndex uIndex = new PathIndex();
	private PathIndex gIndex = new PathIndex();

	private String userRelationPath;
	private String featureUserPath;

	public static void main(String[] args) throws Exception {
		ToolRunner.run(new ParallelMRPJob(), args);
	}

	@Override
	public int run(String[] args) throws Exception {

		addInputOption();
		addOutputOption();

		addOption("numFeatures", null, "dimension of the feature space", true);
		addOption("numIterations", null, "number of iterations", true);
		addOption("lambda_a", null, "regularization parameter", true);
		addOption("lambda_fg", null, "regularization parameter", true);
		addOption("lambda_fu", null, "regularization parameter", true);
		addOption("lambda_lg", null, "regularization parameter", true);
		addOption("lambda_lu", null, "regularization parameter", true);
		addOption("lambda_g", null, "regularization parameter", true);
		addOption("lambda_u", null, "regularization parameter", true);
		addOption("dimFeatureUser", null, "dimension of user feautre", true);
		addOption("dimUserRelation", null, "dimension of geography feautre", true);
		addOption("userRelationPath", null, "geography features file", true);
		addOption("featureUserPath", null, "user features file", true);

		Map<String, List<String>> parsedArgs = parseArguments(args);
		if (parsedArgs == null) {
			return -1;
		}

		numFeatures = Integer.parseInt(getOption("numFeatures"));
		numIterations = Integer.parseInt(getOption("numIterations"));
		this.lambda_a = Double.parseDouble(getOption("lambda_a"));
		this.lambda_fg = Double.parseDouble(getOption("lambda_fg"));
		this.lambda_fu = Double.parseDouble(getOption("lambda_fu"));
		this.lambda_lg = Double.parseDouble(getOption("lambda_lg"));
		this.lambda_lu = Double.parseDouble(getOption("lambda_lu"));
		this.lambda_g = Double.parseDouble(getOption("lambda_g"));
		this.lambda_u = Double.parseDouble(getOption("lambda_u"));
		this.dimUserRelation = Integer.parseInt(getOption("dimUserRelation"));
		this.dimFeatureUser = Integer.parseInt(getOption("dimFeatureUser"));
		this.userRelationPath = getOption("userRelationPath");
		this.featureUserPath = getOption("featureUserPath");

		/* create A */
		Job userRatings = prepareJob(getInputPath(), pathToUserRatings(),
				TextInputFormat.class, InputVectorsMapper.class,
				IntWritable.class, VectorWritable.class,
				VectorSumReducer.class, IntWritable.class,
				VectorWritable.class, SequenceFileOutputFormat.class);
		userRatings.setCombinerClass(VectorSumReducer.class);
		boolean succeeded = userRatings.waitForCompletion(true);
		if (!succeeded)
			return -1;

		/* create A' */
		Job itemRatings = prepareJob(pathToUserRatings(), pathToItemRatings(),
				TransposeMapper.class, IntWritable.class, VectorWritable.class,
				MergeVectorsReducer.class, IntWritable.class,
				VectorWritable.class);
		itemRatings.setCombinerClass(MergeVectorsCombiner.class);
		succeeded = itemRatings.waitForCompletion(true);
		if (!succeeded)
			return -1;

		/* create U ,this is a symetric matrix, so no need to compute U' */
		Job userRelation = prepareJob(getUserRelationPath(), pathToUserRelation(),
				TextInputFormat.class, InputVectorsMapper.class,
				IntWritable.class, VectorWritable.class,
				VectorSumReducer.class, IntWritable.class,
				VectorWritable.class, SequenceFileOutputFormat.class);
		userRelation.setCombinerClass(VectorSumReducer.class);
		succeeded = userRelation.waitForCompletion(true);
		if (!succeeded)
			return -1;

		/* create Fu */
		Job featureUser = prepareJob(getFeatureUserPath(), pathToFeatureUser(),
				TextInputFormat.class, InputVectorsMapper.class,
				IntWritable.class, VectorWritable.class,
				VectorSumReducer.class, IntWritable.class,
				VectorWritable.class, SequenceFileOutputFormat.class);
		featureUser.setCombinerClass(VectorSumReducer.class);
		succeeded = featureUser.waitForCompletion(true);
		if (!succeeded)
			return -1;

		/* create Fu' */
		Job featureUserTranspose = prepareJob(pathToFeatureUser(),
				pathToFeatureUserTranspose(), TransposeMapper.class,
				IntWritable.class, VectorWritable.class,
				MergeVectorsReducer.class, IntWritable.class,
				VectorWritable.class);
		featureUserTranspose.setCombinerClass(MergeVectorsCombiner.class);
		succeeded = featureUserTranspose.waitForCompletion(true);
		if (!succeeded)
			return -1;

		/* Get some average values for initialization. */
		Job averageGeoFeatureValue = prepareJob(pathToUserRelation(),
				getTempPath("averageGeoFeatureValue"),
				AverageVectorsMapper.class, IntWritable.class,
				VectorWritable.class, MergeVectorsReducer.class,
				IntWritable.class, VectorWritable.class);
		averageGeoFeatureValue.setCombinerClass(MergeVectorsCombiner.class);
		succeeded = averageGeoFeatureValue.waitForCompletion(true);
		if (!succeeded)
			return -1;

		Job averageGeoFeatureTransposeValue = prepareJob(
				pathToUserRelationTranspose(),
				getTempPath("averageGeoFeatureValueTranspose"),
				AverageVectorsMapper.class, IntWritable.class,
				VectorWritable.class, MergeVectorsReducer.class,
				IntWritable.class, VectorWritable.class);
		averageGeoFeatureTransposeValue
				.setCombinerClass(MergeVectorsCombiner.class);
		succeeded = averageGeoFeatureTransposeValue.waitForCompletion(true);
		if (!succeeded)
			return -1;

		Job averageUserFeatureValue = prepareJob(pathToFeatureUser(),
				getTempPath("averageUserFeatureValue"),
				AverageVectorsMapper.class, IntWritable.class,
				VectorWritable.class, MergeVectorsReducer.class,
				IntWritable.class, VectorWritable.class);
		averageUserFeatureValue.setCombinerClass(MergeVectorsCombiner.class);
		succeeded = averageUserFeatureValue.waitForCompletion(true);
		if (!succeeded)
			return -1;

		Job averageUserFeatureTransposeValue = prepareJob(
				pathToFeatureUserTranspose(),
				getTempPath("averageUserFeatureValueTranspose"),
				AverageVectorsMapper.class, IntWritable.class,
				VectorWritable.class, MergeVectorsReducer.class,
				IntWritable.class, VectorWritable.class);
		averageUserFeatureTransposeValue
				.setCombinerClass(MergeVectorsCombiner.class);
		succeeded = averageUserFeatureTransposeValue.waitForCompletion(true);
		if (!succeeded)
			return -1;

		Vector averageUserValue = ALSUtils.readFirstRow(
				getTempPath("averageUserFeatureValue"), getConf());
		Vector averageGeoValue = ALSUtils.readFirstRow(
				getTempPath("averageGeoFeatureValue"), getConf());
		Vector averageUserValueTranspose = ALSUtils.readFirstRow(
				getTempPath("averageUserFeatureValueTranspose"), getConf());
		Vector averageGeoValueTranspose = ALSUtils.readFirstRow(
				getTempPath("averageGeoFeatureValueTranspose"), getConf());

		/* create an initial Lu Lg */
		initialize(averageUserValue, pathToLu(-1));
		initialize(averageGeoValue, pathToLg(-1));
		initialize(averageUserValueTranspose, pathToU(-1));
		initialize(averageGeoValueTranspose, pathToG(-1));

		for (int currentIteration = 0; currentIteration < numIterations; currentIteration++) {

			/* broadcast Lu, read A' Fu' Fg', recompute Lg */
			log.info("Recompute Lg via A (iteration {}/{})", currentIteration,
					numIterations);
			runSolver(pathToItemRatings(),
					pathToLu(currentIteration - 1),
					pathToLgviaA(currentIteration), this.lambda_a,
					this.lambda_lg);
			/* broadcast G, read Fg row-wise, recompute Lg */
			log.info("Recompute Lg via Fg (iteration {}/{})", currentIteration,
					numIterations);
			runSolver(pathToG(currentIteration - 1),
					pathToUserRelation(),
					pathToLgviaF(currentIteration), this.lambda_fg,
					this.lambda_lg);
			/* merge Lg */
			log.info("Merge Lg together (iteration {}/{})", currentIteration,
					numIterations);
			mergeLuorLg(pathToLgviaA(currentIteration),
					pathToLgviaF(currentIteration),
					pathToLg(currentIteration));

			/* broadcast Lg, read A row-wise, recompute Lu */
			log.info("Recompute Lu via A (iteration {}/{})", currentIteration,
					numIterations);
			runSolver(pathToUserRatings(),
					pathToLg(currentIteration),
					pathToLuviaA(currentIteration), this.lambda_a,
					this.lambda_lu);
			/* broadcast U, read Fu row-wise, recompute Lu */
			log.info("Recompute Lu via Fu (iteration {}/{})", currentIteration,
					numIterations);
			runSolver(pathToU(currentIteration - 1),
					pathToFeatureUser(),
					pathToLuviaF(currentIteration), this.lambda_fu,
					this.lambda_lu);
			/* merge Lu */
			log.info("Merge Lu together (iteration {}/{})", currentIteration,
					numIterations);
			mergeLuorLg(pathToLuviaA(currentIteration),
					pathToLuviaF(currentIteration),
					pathToLu(currentIteration));

			/* broadcast Lg, read Fg, recompute G */
			log.info("Recompute G via Fg' (interation {}/{})",
					currentIteration, numIterations);
			runSolver(
					pathToLg(currentIteration),
					pathToUserRelationTranspose(),
					pathToG(currentIteration), this.lambda_fg,
					this.lambda_g);

			/* broadcast Lu, read Fu, recompute U */
			log.info("Recompute U via Fu (iteration {}/{})", currentIteration,
					numIterations);
			runSolver(
					pathToLu(currentIteration),
					pathToFeatureUserTranspose(),
					pathToU(currentIteration), this.lambda_fu,
					this.lambda_u);
		}

		return 0;
	}

	private void initialize(Vector averageValue, Path averagePath)
			throws IOException {
		Random random = RandomUtils.getRandom();

		FileSystem fs = FileSystem.get(averagePath.toUri(), getConf());
		SequenceFile.Writer writer = null;
		try {
			writer = new SequenceFile.Writer(fs, getConf(), new Path(
					averagePath, "part-m-00000"), IntWritable.class,
					VectorWritable.class);

			Iterator<Vector.Element> averages = averageValue.iterateNonZero();
			while (averages.hasNext()) {
				Vector.Element e = averages.next();
				Vector row = new DenseVector(numFeatures);
				/* why initial in this way? */
				row.setQuick(0, e.get());
				for (int m = 1; m < numFeatures; m++) {
					row.setQuick(m, random.nextDouble());
				}
				writer.append(new IntWritable(e.index()), new VectorWritable(
						row));
			}
		} finally {
			Closeables.closeQuietly(writer);
		}
	}

	static class InputVectorsMapper extends
			Mapper<LongWritable, Text, IntWritable, VectorWritable> {
		protected void map(LongWritable offset, Text line, Context ctx)
				throws IOException, InterruptedException {
			String[] tokens = TasteHadoopUtils.splitPrefTokens(line.toString());
			int principalID = Integer.parseInt(tokens[0]);
			int minorID = Integer.parseInt(tokens[1]);
			float value = Float.parseFloat(tokens[2]);

			Vector values = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
			values.set(minorID, value);

			ctx.write(new IntWritable(principalID), new VectorWritable(values,
					true));
		}
	}

	private void runSolver(Path ratings, Path pathToUorI, Path output,
			double lambdaPrincipal, double lambdaMinor)
			throws ClassNotFoundException, IOException, InterruptedException {

		Job solverForUorI = prepareJob(ratings, output,
				SequenceFileInputFormat.class,
				SolveExplicitFeedbackMapper.class, IntWritable.class,
				VectorWritable.class, SequenceFileOutputFormat.class);
		Configuration solverConf = solverForUorI.getConfiguration();
		solverConf.set(LAMBDA_PRINCIPAL, String.valueOf(lambdaPrincipal));
		solverConf.set(LAMBDA_MINOR, String.valueOf(lambdaMinor));
		solverConf.setInt(NUM_FEATURES, numFeatures);
		solverConf.set(FEATURE_MATRIX, pathToUorI.toString());
		boolean succeeded = solverForUorI.waitForCompletion(true);
		if (!succeeded)
			throw new IllegalStateException("Job failed!");
	}

	/* merge 2 matrix together via plus */
	private void mergeLuorLg(Path pathFirst, Path pathSecond, Path output)
			throws IOException, ClassNotFoundException, InterruptedException {
		Job mergeTogether = prepareJob(pathFirst, output,
				SequenceFileInputFormat.class, MergeVectors.class,
				IntWritable.class, VectorWritable.class,
				SequenceFileOutputFormat.class);
		Configuration mergeTogetherConf = mergeTogether.getConfiguration();
		mergeTogetherConf.set(PATH_TO_VECTOR_SECOND, pathSecond.toString());
		boolean succeeded = mergeTogether.waitForCompletion(true);
		if (!succeeded)
			throw new IllegalStateException("Job failed!");
	}

	static class SolveExplicitFeedbackMapper extends
			Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		private double lambdaPrincipal;
		private double lambdaMinor;
		private int numFeatures;

		private OpenIntObjectHashMap<Vector> UorM;

		private AlternatingLeastSquaresSolver solver;

		@Override
		protected void setup(@SuppressWarnings("rawtypes") Mapper.Context ctx)
				throws IOException, InterruptedException {
			lambdaPrincipal = Double.parseDouble(ctx.getConfiguration().get(
					LAMBDA_PRINCIPAL));
			lambdaMinor = Double.parseDouble(ctx.getConfiguration().get(
					LAMBDA_MINOR));
			numFeatures = ctx.getConfiguration().getInt(NUM_FEATURES, -1);
			solver = new AlternatingLeastSquaresSolver();

			Path UOrIPath = new Path(ctx.getConfiguration().get(FEATURE_MATRIX));

			UorM = ALSUtils.readMatrixByRows(UOrIPath, ctx.getConfiguration());
			Preconditions.checkArgument(numFeatures > 0,
					"numFeatures was not set correctly!");
		}

		@Override
		protected void map(IntWritable userOrItemID,
				VectorWritable ratingsWritable, Context ctx)
				throws IOException, InterruptedException {
			Vector ratings = new SequentialAccessSparseVector(
					ratingsWritable.get());
			List<Vector> featureVectors = Lists.newArrayList();
			Iterator<Vector.Element> interactions = ratings.iterateNonZero();
			while (interactions.hasNext()) {
				int index = interactions.next().index();
				featureVectors.add(UorM.get(index));
			}
			Vector uiOrmj = solver.solve(featureVectors, ratings,
					lambdaPrincipal, lambdaMinor, numFeatures);
			ctx.write(userOrItemID, new VectorWritable(uiOrmj));
		}
	}

	public static class MergeVectors extends
			Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
		private OpenIntObjectHashMap<Vector> vectorSecond;

		@Override
		protected void setup(Context ctx) throws IOException,
				InterruptedException {
			Path pathToVectorSecond = new Path(ctx.getConfiguration().get(
					PATH_TO_VECTOR_SECOND));
			vectorSecond = ALSUtils.readMatrixByRows(pathToVectorSecond,
					ctx.getConfiguration());
		}

		protected void map(IntWritable userOrItemID,
				VectorWritable ratingsWritable, Context ctx)
				throws IOException, InterruptedException {
			Vector ratings = new RandomAccessSparseVector(ratingsWritable.get());
			Vector merge = new RandomAccessSparseVector(Integer.MAX_VALUE);
            Iterator<Vector.Element> elements = ratings.iterateNonZero();
            while (elements.hasNext()) {
                Vector.Element e = elements.next();
                int index = e.index();
                double part2 = 0.0;
                try {
                    part2 = vectorSecond.get(userOrItemID.get()).getQuick(index);
                } catch (Exception ecp) {
                    part2 = 0.0;
                }
                merge.setQuick(index, e.get()+part2);
            }
			ctx.write(userOrItemID, new VectorWritable(merge));
		}
	}


	static class AverageVectorsMapper extends
			Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
		@Override
		protected void map(IntWritable r, VectorWritable v, Context ctx)
				throws IOException, InterruptedException {
			RunningAverage avg = new FullRunningAverage();
			Iterator<Vector.Element> elements = v.get().iterateNonZero();
			while (elements.hasNext()) {
				avg.addDatum(elements.next().get());
			}
			Vector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
			vector.setQuick(r.get(), avg.getAverage());
			ctx.write(new IntWritable(0), new VectorWritable(vector));
		}
	}

	private Path pathToLuviaA(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("LuviaA")
				: getTempPath("LuviaA-" + iteration);
	}

	private Path pathToLuviaF(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("LuviaF")
				: getTempPath("LuviaF-" + iteration);
	}

	private Path pathToLu(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("Lu")
				: getTempPath("Lu-" + iteration);
	}

	private Path pathToLgviaA(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("LgviaA")
				: getTempPath("LgviaA-" + iteration);
	}

	private Path pathToLgviaF(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("LgviaF")
				: getTempPath("LgviaF-" + iteration);
	}

	private Path pathToLg(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("Lg")
				: getTempPath("Lg-" + iteration);
	}

	private Path pathToU(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("U")
				: getTempPath("U-" + iteration);
	}

	private Path pathToG(int iteration) {
		return iteration == numIterations - 1 ? getOutputPath("G")
				: getTempPath("G-" + iteration);
	}

	private Path pathToItemRatings() {
		return getTempPath("itemRatings");
	}

	private Path pathToUserRatings() {
		return getOutputPath("userRatings");
	}

	private Path getUserRelationPath() {
		return new Path(this.userRelationPath);
	}

	private Path pathToUserRelation() {
		return getOutputPath("userRelation");
	}

	private Path pathToUserRelationTranspose() {
		return getOutputPath("userRelationTranspose");
	}

	private Path pathToFeatureUserTranspose() {
		return getOutputPath("featureUserTranspose");
	}

	private Path pathToFeatureUser() {
		return getOutputPath("featureUser");
	}

	private Path getFeatureUserPath() {
		return new Path(this.featureUserPath);
	}
}
