package org.apache.nlp.sequencelearning.crf;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

//import org.apache.mahout.classifier.sequencelearning.crf.FeatureIndexer.Pair;
import org.apache.mahout.math.Vector;

public class CRFModel {
	final int version = 100;
	final double const_factor = 1.0;

	FeatureTemplate featureTemplate;
	FeatureExpander featureExpander;
	FeatureIndexer featureIndexer;
	int maxid;
	int xsize = 2;
	Vector alpha;
	Vector expected;
	double obj;
	int err;
	int zeroone;

	public CRFModel() {
	}

	public CRFModel(FeatureTemplate featureTemplate, FeatureExpander featureExpander, FeatureIndexer featureIndexer,
			Vector alpha, Vector expected, double obj, int err, int zeroone) {
		this.featureTemplate = featureTemplate;
		this.featureExpander = featureExpander;
		this.featureIndexer = featureIndexer;
		this.alpha = alpha;
		this.expected = expected;
		this.obj = obj;
		this.err = err;
		this.zeroone = zeroone;
		this.maxid = this.featureIndexer.getMaxID();
	}

	public CRFModel load() {
		CRFModel model = new CRFModel();
		return model;
				
	}
	
	public void dump(String modelPath) throws IOException {
		String context = "";

		context += "version: " + String.valueOf(version) + "\n";
		context += "const_factor: " + String.valueOf(const_factor) + "\n";
		context += "maxid: " + String.valueOf(maxid) + "\n";
		context += "xsize: " + String.valueOf(xsize) + "\n";
		context += "\n";

		for (String hiddenState : this.featureExpander.getHiddenStateSet()) {
			context += hiddenState + "\n";
		}
		context += "\n";

		for (String template : this.featureTemplate.unigram_templs) {
			context += template + "\n";
		}
		for (String template : this.featureTemplate.bigram_templs) {
			context += template + "\n";
		}
		context += "\n";

		Map<String, Integer> featureIndexMap = this.featureIndexer.getFeatureIndexMap();
		Iterator<String> iter = featureIndexMap.keySet().iterator();
		while (iter.hasNext()) {
			String feature = iter.next();
			int id = featureIndexMap.get(feature);
			context += String.valueOf(id) + '\t' + feature + '\n';
		}
		context += "\n";

		for (int i = 0; i < this.alpha.size(); i++) {
			context += String.valueOf(this.alpha.get(i)) + "\n";
		}

		System.out.println(context);

		try {
			File file = new File(modelPath);
			if (!file.exists()) {
				file.createNewFile();
			}
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			bw.write(context);
			bw.close();
			System.out.println("Done");
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
}