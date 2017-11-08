package org.apache.nlp.sequencelearning.crf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Set;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class CRFDriver {
	private static int converge = 0;
	int maxid;

	Vector alpha;
	Vector expected;
	double obj;
	int err;
	int zeroone;

	FeatureTemplate featureTemplate;
	FeatureExpander featureExpander;
	FeatureIndexer featureIndexer;

	ArrayList<TaggerImpl> taggers;
	CRFLBFGS clbfgs;

	public CRFDriver() {
		this.clbfgs = new CRFLBFGS();
	}

	public void initializeCrfDriver() {
		this.maxid = this.featureIndexer.getMaxID();

		this.alpha = new DenseVector(this.maxid);
		for (int i = 0; i < this.alpha.size(); i++) {
			this.alpha.set(i, 0.0);
		}
		this.expected = new DenseVector(this.maxid);
		for (int i = 0; i < this.expected.size(); i++) {
			this.expected.set(i, 0.0);
		}
		this.obj = 0.0;
		this.err = 0;
		this.zeroone = 0;
	}

	@SuppressWarnings("resource")
	public CRFModel crf_learn(String templfile, String trainfile, String modelfile, boolean textmodelfile, int xsize,
			int maxitr, double freq, double eta, double C, String algorithm) throws IOException {
		this.featureTemplate = new FeatureTemplate(templfile);
		this.featureExpander = new FeatureExpander(this.featureTemplate, xsize);
		String[] setences;
		this.taggers = new ArrayList<TaggerImpl>();
		File trainDataPath = new File(trainfile); //
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(trainDataPath), "UTF8"));
		String line = null;
		ArrayList<String> token_list = new ArrayList<String>();
		while ((line = reader.readLine()) != null) {
			setences = line.split(" ");
			for (String s : setences) {
				token_list.add(s);
			}
			TaggerImpl tagger = new TaggerImpl();
			this.featureExpander.expand(token_list, tagger);
			this.taggers.add(tagger);
			token_list = new ArrayList<String>();
		}

		this.featureIndexer = new FeatureIndexer();
		this.featureIndexer.IndexingHStateIndex(this.featureExpander.getHiddenStateSet());
		for (int i = 0; i < this.taggers.size(); i++) {
			TaggerImpl tagger = this.taggers.get(i);
			this.featureIndexer.IndexingFeatureIndex(tagger);
			this.featureIndexer.Register(tagger);
		}

		this.initializeCrfDriver();
		this.iterateMR(maxitr, eta);

		CRFModel model = new CRFModel(featureTemplate, featureExpander, featureIndexer, alpha, expected, obj, err,
				zeroone);
		writeAlpha(alpha);

		return model;
	}

	private void writeAlpha(Vector alpha) {
		PrintWriter writerAlpha = null;
		try {
			writerAlpha = new PrintWriter(
					"C:\\Users\\nmhie\\Desktop\\java-crfsuite-master\\src\\test\\files\\Alpha.pos");
			writerAlpha.println(alpha.size());
			for (int i = 0; i < alpha.size(); i++) {
				writerAlpha.println(alpha.get(i));
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			writerAlpha.close();
		}
	}

	private void readAlpha() {
		BufferedReader br = null;
		try {
			int i = 0;
			String fileLine;
			br = new BufferedReader(
					new FileReader("C:\\Users\\nmhie\\Desktop\\java-crfsuite-master\\src\\test\\files\\Alpha.pos"));
			alpha = new DenseVector(Integer.parseInt(br.readLine()));
			while ((fileLine = br.readLine()) != null) {
				this.alpha.set(i, Double.parseDouble(fileLine));
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (br != null) {
					br.close();
				}
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
	}

	@SuppressWarnings("resource")
	public boolean crf_test(String templfile, String testfile, CRFModel model, int xsize) throws IOException {
		readAlpha();
		model.alpha = alpha;
		this.featureExpander = model.featureExpander;
		this.featureIndexer = model.featureIndexer;
		String[] setences;
		Set<String> hsSet = this.featureExpander.getHiddenStateSet();
		String hsArray[] = new String[hsSet.size()];
		int id = 0;
		for (String hiddenState : hsSet) {
			hsArray[id] = hiddenState;
			id++;
		}

		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(testfile), "UTF8"));
		String line = null;
		ArrayList<String> token_list = new ArrayList<String>();
		while ((line = reader.readLine()) != null) {
			TaggerImpl tagger = new TaggerImpl(model.alpha);
			setences = line.split(" ");
			for (String s : setences) {
				token_list.add(s);
			}
			this.featureExpander.expand(token_list, tagger);
			this.featureIndexer.Register(tagger);
			tagger.buildLattice();
			tagger.forwardbackward();
			ArrayList<Integer> result = tagger.viterbi();

			int tokensNum = token_list.size();
			for (int i = 0; i < tokensNum; i++) {
				System.out.println(token_list.get(i) + '\t' + hsArray[result.get(i)]);
			}

			token_list = new ArrayList<String>();
		}

		return true;
	}

	public double run() {
		double C = 4.0;
		Vector expected_current_iteration = new DenseVector(this.maxid);
		for (int i = 0; i < expected_current_iteration.size(); i++) {
			expected_current_iteration.set(i, 0.0);
		}
		this.obj = 0.0;
		this.err = 0;
		this.zeroone = 0;

		for (int i = 0; i < taggers.size(); i++) {
			TaggerImpl taggerImpl = taggers.get(i);

			taggerImpl.alpha = this.alpha;
			taggerImpl.expected = expected_current_iteration;

			this.obj += taggerImpl.gradient();

			int error_num = taggerImpl.eval();
			this.err += error_num;
			if (error_num != 0) {
				this.zeroone += 1;
			}
		}

		for (int i = 0; i < this.expected.size(); i++) {
			this.expected.set(i, 0.0);
			this.expected.set(i, this.expected.get(i) + expected_current_iteration.get(i));
		}

		int n = this.maxid;
		double x[] = new double[n];
		double f;
		double g[] = new double[n];

		for (int k = 0; k < this.alpha.size(); k++) {
			this.obj += this.alpha.get(k) * this.alpha.get(k) / (2.0 * C);
			this.expected.set(k, this.expected.get(k) + this.alpha.get(k) / C);
		}
		for (int i = 0; i < this.maxid; i++) {
			x[i] = this.alpha.get(i);
			g[i] = this.expected.get(i);
		}
		f = this.obj;

		this.clbfgs.optimize(n, x, f, g);

		for (int k = 0; k < x.length; k++) {
			this.alpha.set(k, x[k]);
		}
		this.obj = f;

		return this.obj;
	}

	public void iterateMR(int numIterations, double eta) throws IOException {
		System.out.println("Running CRF");
		Double old_obj = new Double(0.0);
		Double obj = new Double(0.0);

		int iteration = 1;
		while (iteration <= numIterations) {

			obj = this.run();
			if (isConverged(iteration, numIterations, eta, old_obj, obj)) {
				break;
			}
			old_obj = obj;
			iteration++;
		}
	}

	private boolean isConverged(int itr, int maxitr, double eta, double old_obj, double obj) throws IOException {
		double diff = (itr == 1 ? 1.0 : Math.abs(old_obj - obj) / old_obj);

		if (diff < eta) {
			converge++;
		} else {
			converge = 0;
		}
		if (itr > maxitr || converge == 3) {
			return true;
		}
		return false;
	}

}
