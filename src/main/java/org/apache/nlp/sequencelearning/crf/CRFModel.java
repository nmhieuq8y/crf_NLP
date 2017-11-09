package org.apache.nlp.sequencelearning.crf;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.Map;

import org.apache.mahout.math.DenseVector;
//import org.apache.mahout.classifier.sequencelearning.crf.FeatureIndexer.Pair;
import org.apache.mahout.math.Vector;
import org.apache.nlp.sequencelearning.crf.FeatureIndexer.Pair;

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

	public void readModel(String modelPath) {
		String line;
		String[] lineValue;
		BufferedReader reader;
		String tab = "\t";
		int i = 0;
		try {
			reader = new BufferedReader(new InputStreamReader(new FileInputStream(modelPath), "UTF8"));
			line = reader.readLine();
			maxid = Integer.parseInt(line.split("maxid: ")[1]);
			line = reader.readLine();
			xsize = Integer.parseInt(line.split("xsize: ")[1]);
			this.alpha = new DenseVector(maxid);
			while ((line = reader.readLine()) != null) {
				if (line.contains("================featureExpander")) {
					this.featureExpander = new FeatureExpander();
					this.featureExpander.setXsize(xsize);
					line = reader.readLine();
					do {
						this.featureExpander.getHiddenStateSet().add(line);
					} while (!(line = reader.readLine()).contains("================featureTemplate.unigram_templs"));
				}

				if (line.contains("================featureTemplate.unigram_templs")) {
					this.featureTemplate = new FeatureTemplate();
					line = reader.readLine();
					do {
						this.featureTemplate.unigram_templs.add(line);
					} while (!(line = reader.readLine()).contains("================featureTemplate.bigram_templs"));
				}
				if (line.contains("================featureTemplate.bigram_templs")) {
					line = reader.readLine();
					do {
						this.featureTemplate.bigram_templs.add(line);
					} while (!(line = reader.readLine()).contains("================featureIndexer"));
				}
				this.featureExpander.setFeatureTemplate(this.featureTemplate);
				if (line.contains("================featureIndexer")) {
					this.featureIndexer = new FeatureIndexer();
					line = reader.readLine();
					this.featureIndexer.setYsize(Integer.parseInt(line));
					line = reader.readLine();
					do {
						lineValue = line.split(tab);
						this.featureIndexer.getFeatureIndexMapN().put(lineValue[0], featureIndexer.new Pair(
								Integer.parseInt(lineValue[1]), Integer.parseInt(lineValue[2])));
					} while (!(line = reader.readLine()).contains("================alpha"));
				}
				if (line.contains("================alpha")) {
					line = reader.readLine();
					do {
						this.alpha.set(i, Double.parseDouble(line));
						i++;
					} while ((line = reader.readLine()) != null);
				}
			}
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void writeModel(String modelPath) throws IOException {
		StringBuilder context = new StringBuilder();
		String downline = "\n";
		context.append("maxid: " + String.valueOf(maxid) + downline);
		context.append("xsize: " + String.valueOf(xsize) + downline);
		context.append("================featureExpander\n");

		for (String hiddenState : this.featureExpander.getHiddenStateSet()) {
			context.append(hiddenState + downline);
		}
		context.append("================featureTemplate.unigram_templs\n");

		for (String template : this.featureTemplate.unigram_templs) {
			context.append(template + downline);
		}
		context.append("================featureTemplate.bigram_templs\n");
		for (String template : this.featureTemplate.bigram_templs) {
			context.append(template + downline);
		}
		context.append("================featureIndexer\n");
		context.append(this.featureIndexer.getYsize() + downline);
		Map<String, Pair> featureIndexMap = this.featureIndexer.getFeatureIndexMapN();
		for (Map.Entry<String, Pair> entry : featureIndexMap.entrySet()) {
			context.append(entry.getKey() + "\t" + entry.getValue() + downline);
		}
		context.append("================alpha\n");

		for (int i = 0; i < this.alpha.size(); i++) {
			context.append(String.valueOf(this.alpha.get(i)) + downline);
		}

		System.out.println(context);

		try {
			File file = new File(modelPath);
			if (!file.exists()) {
				file.createNewFile();
			}
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			bw.write(context.toString());
			bw.close();
			System.out.println("Done");
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
}